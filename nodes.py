import logging

import torch
from comfy_api.latest import io

from .prompt_relay import (
    RelayConfig,
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .lora_schedule import (
    load_lora_weights,
    build_lora_assignment,
    build_blend_tensor,
    apply_lora_temporal_patches,
)

from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    config = RelayConfig(
        num_segments=len(locals_list),
        segment_lengths=effective_lengths,
        tokens_per_frame=tokens_per_frame,
        latent_frames=latent_frames,
        epsilon=epsilon,
        arch=arch,
        patch_size=patch_size,
    )

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning, config


class PromptRelayEncode(io.ComfyNode):
    """Encodes temporal local prompts and patches the model for Prompt Relay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncode",
            display_name="Prompt Relay Encode",
            category="conditioning/prompt_relay",
            description=(
                "Encodes a global prompt combined with temporal local prompts and patches the model "
                "for Prompt Relay temporal control. Local prompts are separated by |. "
                "Use a standard CLIPTextEncode for the negative prompt."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Ordered prompts for each temporal segment, separated by |",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Comma-separated pixel space frame counts per segment. Leave empty to auto-distribute evenly.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Custom("PROMPT_RELAY_CONFIG").Output(display_name="relay_config"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon) -> io.NodeOutput:
        patched, conditioning, config = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )
        return io.NodeOutput(patched, conditioning, config)


class PromptRelayEncodeTimeline(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncodeTimeline",
            display_name="Prompt Relay Encode (Timeline)",
            category="conditioning/prompt_relay",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The max_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Int.Input(
                    "max_frames", default=129, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.1, max=240.0, step=0.1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "time_units", options=["frames", "seconds"], default="frames", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Custom("PROMPT_RELAY_CONFIG").Output(display_name="relay_config"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames") -> io.NodeOutput:
        patched, conditioning, config = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )
        return io.NodeOutput(patched, conditioning, config)


class _AnyType(str):
    """String subclass where ``!=`` always returns False, so type-equality
    checks against this wildcard always succeed.  (Credit: pythongosssss)"""
    def __ne__(self, __value):
        return False


class _FlexibleOptionalInputType(dict):
    """Mirrors rgthree's FlexibleOptionalInputType: tells ComfyUI to accept
    any additional optional inputs (the dynamic JS widgets lora_1, lora_2, …)
    and pass their values through to the Python ``**kwargs``."""
    def __init__(self, data=None):
        super().__init__(data or {})
        self.type = _AnyType("*")
        self._data = data

    def __getitem__(self, key):
        if self._data is not None and key in self._data:
            return self._data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


class RelayLoraSelector:
    """Selects multiple LoRAs to stack together for a single temporal segment.
    Uses a compact dynamic UI (add / toggle / strength controls) driven by the
    frontend extension in ``web/relay_lora_selector.js``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _FlexibleOptionalInputType(),
            "hidden": {},
        }

    RETURN_TYPES = ("RELAY_LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "conditioning/prompt_relay"
    OUTPUT_NODE = False

    def build_stack(self, **kwargs):
        stack = []
        for key, value in kwargs.items():
            if (
                key.upper().startswith("LORA_")
                and isinstance(value, dict)
                and value.get("on")
                and value.get("lora")
                and value["lora"] not in ("none", "None", "")
            ):
                stack.append((value["lora"], value.get("strength", 1.0)))

        if not stack:
            log.info("[PromptRelay] RelayLoraSelector: no LoRAs selected, outputting None")
            return (None,)

        log.info("[PromptRelay] RelayLoraSelector: %d LoRAs: %s", len(stack), [n for n, _ in stack])
        return (stack,)


class PromptRelayLoraSchedule(io.ComfyNode):
    """Applies different LoRA stacks to different temporal segments defined by PromptRelayEncode."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayLoraSchedule",
            display_name="Prompt Relay LoRA Schedule",
            category="conditioning/prompt_relay",
            description=(
                "Applies different LoRA stacks to different temporal segments. Connect model and "
                "relay_config from PromptRelayEncode, then connect up to 6 Relay LoRA Selector "
                "outputs — each stack maps to a segment (1→segment 0, 2→segment 1, …). "
                "lora_global is merged into every segment with additive strength. "
                "extend_last controls whether the last assigned stack fills remaining segments."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model output from PromptRelayEncode."),
                io.Custom("PROMPT_RELAY_CONFIG").Input("relay_config"),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="LoRA blend transition sharpness. Below ~0.1 gives sharp cuts (default 0.001). Use 0.5+ for softer crossfades.",
                ),
                io.Boolean.Input("extend_last", default=True, tooltip="If true, the last assigned LoRA stack fills any remaining segments beyond the connected stacks."),
                io.Custom("RELAY_LORA_STACK").Input("lora_global", optional=True, tooltip="LoRA stack applied to ALL segments. Strengths add with per-segment stacks — e.g. global 0.3 + segment 0.6 = 0.9."),
                io.Custom("RELAY_LORA_STACK").Input("lora_stack_1", optional=True, tooltip="LoRA stack for segment 0."),
                io.Custom("RELAY_LORA_STACK").Input("lora_stack_2", optional=True, tooltip="LoRA stack for segment 1."),
                io.Custom("RELAY_LORA_STACK").Input("lora_stack_3", optional=True, tooltip="LoRA stack for segment 2."),
                io.Custom("RELAY_LORA_STACK").Input("lora_stack_4", optional=True, tooltip="LoRA stack for segment 3."),
                io.Custom("RELAY_LORA_STACK").Input("lora_stack_5", optional=True, tooltip="LoRA stack for segment 4."),
                io.Custom("RELAY_LORA_STACK").Input("lora_stack_6", optional=True, tooltip="LoRA stack for segment 5."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, relay_config, lora_global=None,
                epsilon=1e-3, extend_last=True,
                lora_stack_1=None, lora_stack_2=None,
                lora_stack_3=None, lora_stack_4=None,
                lora_stack_5=None,
                lora_stack_6=None) -> io.NodeOutput:

        per_segment = [lora_stack_1, lora_stack_2, lora_stack_3, lora_stack_4, lora_stack_5, lora_stack_6]

        # Merge global LoRAs into each segment stack (additive strength for same-name LoRAs)
        if lora_global:
            all_stacks = []
            for stack in per_segment:
                if stack is None:
                    all_stacks.append(list(lora_global))
                else:
                    combined = {}
                    for name, strength in lora_global:
                        combined[name] = combined.get(name, 0.0) + strength
                    for name, strength in stack:
                        combined[name] = combined.get(name, 0.0) + strength
                    all_stacks.append(list(combined.items()))
        else:
            all_stacks = per_segment

        # Build a flat list of unique (name, strength) entries with deduplication
        lora_entries = []       # [(name, strength), ...] in column order
        lora_entry_index = {}   # (name, strength) -> column index
        lora_slot_map = []      # per-slot: list of column indices (empty = no LoRAs)

        for stack in all_stacks:
            if stack:
                indices = []
                for name, strength in stack:
                    key = (name, strength)
                    if key not in lora_entry_index:
                        lora_entry_index[key] = len(lora_entries)
                        lora_entries.append((name, strength))
                    indices.append(lora_entry_index[key])
                lora_slot_map.append(indices)
            else:
                lora_slot_map.append([])

        if not lora_entries:
            log.info("[PromptRelay] LoRA Schedule: no LoRAs assigned, passing model through unchanged")
            return io.NodeOutput(model)

        num_loras = len(lora_entries)
        num_segments = relay_config.num_segments
        lora_assignment = build_lora_assignment(num_segments, lora_slot_map, extend_last)

        log.info(
            "[PromptRelay] LoRA Schedule: %d LoRAs for %d segments, assignment: %s",
            num_loras, num_segments, lora_assignment,
        )

        # Load all LoRA weights
        all_lora_deltas = []
        for name, s in lora_entries:
            deltas = load_lora_weights(name, model)
            # Apply strength to alpha_scale
            scaled = {k: (up, down, alpha * s) for k, (up, down, alpha) in deltas.items()}
            all_lora_deltas.append(scaled)

        # Build blend tensor [total_tokens, num_loras]
        blend = build_blend_tensor(
            relay_config.segment_lengths,
            relay_config.tokens_per_frame,
            lora_assignment,
            num_loras,
            device="cpu",
            epsilon=epsilon,
        )

        # Reorganize deltas by model key for patching
        # lora_deltas_by_key: {model_key: [(lora_up, lora_down, alpha_scale), ...]}
        lora_deltas_by_key = {}
        all_keys = set()
        for deltas in all_lora_deltas:
            all_keys.update(deltas.keys())

        for key in all_keys:
            per_lora = []
            for deltas in all_lora_deltas:
                if key in deltas:
                    per_lora.append(deltas[key])
                else:
                    per_lora.append(None)
            # Only include keys that have at least one LoRA
            if any(p is not None for p in per_lora):
                # Fill missing entries with zero-sized placeholders (no contribution)
                filled = []
                for p in per_lora:
                    if p is not None:
                        filled.append(p)
                    else:
                        # Create zero delta with same shape as a real one
                        ref = next(pp for pp in per_lora if pp is not None)
                        filled.append((
                            torch.zeros_like(ref[0]),
                            torch.zeros_like(ref[1]),
                            0.0,
                        ))
                lora_deltas_by_key[key] = filled

        # Patch model
        patched = model.clone()
        apply_lora_temporal_patches(patched, lora_deltas_by_key, blend)

        return io.NodeOutput(patched)


import folder_paths

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "RelayLoraSelector": RelayLoraSelector,
    "PromptRelayLoraSchedule": PromptRelayLoraSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "RelayLoraSelector": "Relay LoRA Selector",
    "PromptRelayLoraSchedule": "Prompt Relay LoRA Schedule",
}
