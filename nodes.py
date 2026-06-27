import logging

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, apply_patches
from .advanced_options import PromptRelayAdvancedOptions, RelayOptions

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


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

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

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


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
                RelayOptions.Input(
                    "relay_options", optional=True,
                    tooltip="Optional advanced per-stream tuning. Connect a Prompt Relay Advanced Options node.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )
        return io.NodeOutput(patched, conditioning)


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
                RelayOptions.Input(
                    "relay_options", optional=True,
                    tooltip="Optional advanced per-stream tuning. Connect a Prompt Relay Advanced Options node.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )


    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames", relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )
        # Report the fully-resolved max_frames back to the timeline editor. The
        # graph executor resolves this input no matter what feeds it (INT
        # Constant, Math Expression, a KJNodes calculator that exposes nothing to
        # the frontend, etc.), so this is the robust way to keep the gauge in
        # sync after a run when the upstream value isn't knowable in the UI.
        return io.NodeOutput(patched, conditioning, ui={"max_frames": [int(max_frames)]})


NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
}
