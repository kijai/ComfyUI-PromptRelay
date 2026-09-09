from .nodes import PromptRelayEncode, PromptRelayEncodeTimeline, PromptRelayAdvancedOptions, VideoFrameCalculator, StyleSelector
from .smart_nodes import PromptRelaySmartEncode, PromptRelaySmartEncodeTest
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PromptRelayEncode,
            PromptRelayEncodeTimeline,
            PromptRelaySmartEncode,
            PromptRelaySmartEncodeTest,
            PromptRelayAdvancedOptions
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelaySmartEncode": PromptRelaySmartEncode,
    "PromptRelaySmartEncodeTest": PromptRelaySmartEncodeTest,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
    "VideoFrameCalculator": VideoFrameCalculator,
    "StyleSelector": StyleSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelaySmartEncode": "Prompt Relay Encode (Smart)",
    "PromptRelaySmartEncodeTest": "Prompt Relay Smart Encode Test",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
    "VideoFrameCalculator": "Video Frame Calculator (Seconds → Frames)",
    "StyleSelector": "Style Selector 🎨",
}


WEB_DIRECTORY = "./web"

try:
    from . import storyboard_engine
    storyboard_engine.register_routes()
except Exception as _e:
    print(f"[PromptRelay] Storyboard API routes unavailable: {_e}")

try:
    from . import relay_studio
    relay_studio.register_routes()
except Exception as _e:
    print(f"[PromptRelay] Studio (Dolly) API routes unavailable: {_e}")

try:
    from . import relay_director
    relay_director.register_routes()
except Exception as _e:
    print(f"[PromptRelay] Director API routes unavailable: {_e}")

try:
    from . import seedance_studio
    seedance_studio.register_routes()
except Exception as _e:
    print(f"[PromptRelay] SeeDance studio API routes unavailable: {_e}")

try:
    from . import lena_studio
    lena_studio.register_routes()
except Exception as _e:
    print(f"[PromptRelay] Lena manager API routes unavailable: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
