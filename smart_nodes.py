import logging
from comfy_api.latest import io
from .nodes import _encode_relay
from .prompt_relay import get_raw_tokenizer
from .parser import parse_smart_prompt

log = logging.getLogger(__name__)

class PromptRelaySmartEncode(io.ComfyNode):
    """Parses advanced syntax into Prompt Relay segments and lengths."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelaySmartEncode",
            display_name="Prompt Relay Encode (Smart)",
            category="conditioning/prompt_relay",
            description="Parses syntax like [0-50] or block headers (Second 1:) to automatically calculate segment lengths.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent"),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions entire video. Leave empty to auto-use the first parsed segment from smart_prompt as the global anchor."
                ),
                io.String.Input(
                    "smart_prompt", multiline=True, default="",
                    tooltip="Enter prompt using Smart Syntax:\\n1. Inline: 'text one [0-50] | text two [50-100]'\\n2. Block: 'Second 1:\\ntext one\\nSecond 2:\\ntext two'\\nSyntax is auto-stripped and normalized evenly or proportionally."
                ),
                io.Boolean.Input("normalize_by_tokens", default=False, tooltip="If true, scales the calculated length of each segment by its token count."),
                io.Float.Input("epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, smart_prompt, normalize_by_tokens, epsilon) -> io.NodeOutput:
        parsed = parse_smart_prompt(smart_prompt)

        valid_segments = [s for s in parsed if s["text"].strip()]
        if not valid_segments:
            valid_segments = [{"text": " ", "weight": 1.0}]

        global_prompt_str = global_prompt.strip()
        if not global_prompt_str and len(valid_segments) > 1:
            global_prompt_str = valid_segments[0]["text"]
            valid_segments = valid_segments[1:]
        elif not global_prompt_str:
            global_prompt_str = valid_segments[0]["text"]

        raw_tokenizer = get_raw_tokenizer(clip) if normalize_by_tokens else None

        local_prompts_list = []
        weights_list = []

        for seg in valid_segments:
            text = seg["text"]
            weight = seg["weight"]

            if normalize_by_tokens and raw_tokenizer:
                try:
                    tokens = raw_tokenizer(text)["input_ids"]
                    has_eos = getattr(raw_tokenizer, "add_eos", False)
                    token_count = len(tokens) - (1 if has_eos else 0)
                    token_count = max(1, token_count)
                    weight *= token_count
                except Exception as e:
                    log.warning(f"Token counting failed for segment '{text}': {e}")

            local_prompts_list.append(text)
            weights_list.append(weight)

        local_prompts_str = " | ".join(local_prompts_list)

        scale_factor = 100000.0
        segment_lengths_str = ", ".join(str(int(w * scale_factor)) for w in weights_list)

        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt_str, local_prompts_str, segment_lengths_str, epsilon
        )

        return io.NodeOutput(patched, conditioning)


class PromptRelaySmartEncodeTest(io.ComfyNode):
    """Test node for Prompt Relay Smart Encode syntax parsing."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelaySmartEncodeTest",
            display_name="Prompt Relay Smart Encode Test",
            category="conditioning/prompt_relay",
            description="Outputs the parsed syntax for testing purposes.",
            inputs=[
                io.String.Input(
                    "smart_prompt", multiline=True, default="",
                    tooltip="Enter prompt using Smart Syntax:\\n1. Inline: 'text one [0-50] | text two [50-100]'\\n2. Block: 'Second 1:\\ntext one\\nSecond 2:\\ntext two'\\nSyntax is auto-stripped and normalized evenly or proportionally."
                ),
                io.Boolean.Input("normalize_by_tokens", default=False),
                io.Clip.Input("clip", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="parsed_output"),
            ],
        )

    @classmethod
    def execute(cls, smart_prompt, normalize_by_tokens, clip=None) -> io.NodeOutput:
        parsed = parse_smart_prompt(smart_prompt)

        valid_segments = [s for s in parsed if s["text"].strip()]
        if not valid_segments:
            valid_segments = [{"text": " ", "weight": 1.0}]

        raw_tokenizer = None
        if normalize_by_tokens and clip is not None:
            from .prompt_relay import get_raw_tokenizer
            raw_tokenizer = get_raw_tokenizer(clip)

        output_lines = []
        for i, seg in enumerate(valid_segments):
            text = seg["text"]
            weight = seg["weight"]

            base_weight = weight
            token_count = None

            if normalize_by_tokens and raw_tokenizer:
                try:
                    tokens = raw_tokenizer(text)["input_ids"]
                    has_eos = getattr(raw_tokenizer, "add_eos", False)
                    token_count = len(tokens) - (1 if has_eos else 0)
                    token_count = max(1, token_count)
                    weight *= token_count
                except Exception:
                    pass

            line = f"Segment {i+1}: text='{text}', base_weight={base_weight}"
            if token_count is not None:
                line += f", tokens={token_count}, final_weight={weight}"
            output_lines.append(line)

        return io.NodeOutput("\n".join(output_lines))
