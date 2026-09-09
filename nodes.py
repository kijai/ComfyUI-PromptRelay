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
    # segment_lengths is optional — treat None or whitespace-only as "auto-distribute"
    if segment_lengths is None:
        segment_lengths = ""

    for name, val in (("model", model), ("clip", clip), ("latent", latent),
                      ("global_prompt", global_prompt),
                      ("local_prompts", local_prompts)):
        if val is None:
            hints = {
                "model": "Connect a WAN or LTX model loader to the 'model' input.",
                "clip": (
                    "Connect a CLIP loader to the 'clip' input. "
                    "Make sure it is the WAN text encoder (T5), not an SDXL or SD1.5 CLIP."
                ),
                "latent": "Connect an empty latent video node to the 'latent' input.",
            }
            detail = hints.get(name,
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )
            raise ValueError(f"PromptRelay: '{name}' is None. {detail}")

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
        return io.NodeOutput(patched, conditioning)


class VideoFrameCalculator:
    """
    Converts seconds to the correct frame count for LTX or Wan video models.

    LTX  requires frames = 8n + 1  (e.g. 241 for 10s at 24fps)
    Wan  requires frames = 4n + 1  (e.g. 161 for 10s at 16fps)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.1, "max": 120.0, "step": 0.5,
                    "tooltip": "Desired video duration in seconds. Type directly: 10, 20, 30 etc.",
                }),
                "model_type": (["LTX (24fps, 8n+1)", "Wan (16fps, 4n+1)"], {
                    "tooltip": "Choose the model to calculate the correct frame count.",
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "seconds_actual", "fps")
    CATEGORY = "video/utils"
    FUNCTION = "calculate"
    DESCRIPTION = (
        "Converts seconds to the nearest valid frame count for LTX or Wan models. "
        "LTX uses 8n+1 frames at 24fps. Wan uses 4n+1 frames at 16fps."
    )

    def calculate(self, seconds, model_type):
        if model_type.startswith("LTX"):
            fps = 24.0
            step = 8
        else:
            fps = 16.0
            step = 4

        raw = seconds * fps
        n = max(1, round((raw - 1) / step))
        frames = n * step + 1
        actual_seconds = round((frames - 1) / fps, 2)

        log.info(
            "[VideoFrameCalculator] %s: %.1fs → %d frames (%.2fs actual @ %.0ffps)",
            model_type, seconds, frames, actual_seconds, fps,
        )

        return (frames, actual_seconds, fps)


STYLE_PRESETS = {
    "Studio Ghibli Anime":       "Studio Ghibli anime style, soft watercolor, vibrant colors, smooth natural motion, high quality animation, detailed",
    "3D Pixar Cartoon":          "3D Pixar cartoon style, vibrant colors, smooth animation, high quality render, professional lighting, detailed textures",
    "Comic Book":                "Comic book style, bold outlines, flat colors, dynamic motion, high contrast, illustrated, graphic novel",
    "Oil Painting Impressionist":"Oil painting impressionist style, textured brushstrokes, rich colors, smooth motion, artistic, painterly",
    "Cyberpunk Neon":            "Cyberpunk neon style, dark atmosphere, glowing lights, smooth motion, cinematic, futuristic, high quality",
}


class StyleSelector:
    """
    Dropdown selector for video restyle art styles.
    Pick a style from the list and wire the STRING output
    directly into the Positive Prompt clip text encode node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (list(STYLE_PRESETS.keys()), {
                    "tooltip": "Select an art style to apply to your video.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style_prompt",)
    CATEGORY = "video/utils"
    FUNCTION = "select"
    DESCRIPTION = "Select an art style from the dropdown. Wire the output into your Positive Prompt text node."

    def select(self, style):
        prompt = STYLE_PRESETS[style]
        log.info("[StyleSelector] Selected: %s → %s", style, prompt)
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
    "VideoFrameCalculator": VideoFrameCalculator,
    "StyleSelector": StyleSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
    "VideoFrameCalculator": "Video Frame Calculator (Seconds → Frames)",
    "StyleSelector": "Style Selector 🎨",
}
