"""Tool registry — the sub-agents the super-agent (Dolly) can call.

Each entry maps a tool name to its JSON schema (sent to the LLM in Ollama's
function-calling format), the Python function `fn(project, **kwargs)`, and the
pipeline task it advances (for the live task tracker)."""

from .. import voices
from .image import generate_image
from .speech import synthesize_speech
from .music import generate_music, trim_music
from .sfx import fetch_sfx
from .lipsync import lipsync
from .mix import final_mix


def list_voices(project, **_):
    """Return the available voice library so Dolly can pick voice_name."""
    return {"ok": True, "voices": voices.list_voices()}


def _tool(name, description, properties, required, fn, task=None):
    return {
        "schema": {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required}}},
        "fn": fn, "task": task,
    }


TOOL_REGISTRY = {
    "list_voices": _tool(
        "list_voices",
        "List the available voices (name, language, gender, description) so you can choose "
        "the best voice_name for the cat before synthesizing speech.",
        {}, [], list_voices, task=None),

    "synthesize_speech": _tool(
        "synthesize_speech",
        "Generate the cat's spoken voice from a line of dialogue, using a voice from the "
        "library. Writes the result to the artifact as TTS_AUDIO_URL.",
        {"text": {"type": "string", "description": "The exact words the cat says."},
         "voice_name": {"type": "string", "description": "Voice name from the library (e.g. Anthony)."},
         "voice_language": {"type": "string", "description": "Language/accent, e.g. EN_British."}},
        ["text"], synthesize_speech, task="voice"),

    "generate_music": _tool(
        "generate_music",
        "Generate ONE master BGM track (a playful theme, 30-90s). Writes BGM_URL. "
        "After this, call trim_music twice to cut the intro and outro snippets from it.",
        {"description": {"type": "string", "description": "Style/vibe, e.g. 'smooth jazz'."},
         "mood": {"type": "string", "description": "Mood keyword: jazz, upbeat, calm, tense, happy."},
         "duration": {"type": "number", "description": "Master length in seconds (e.g. 60)."}},
        [], generate_music, task="bgm"),

    "trim_music": _tool(
        "trim_music",
        "Trim a short intro or outro snippet from the master BGM track. Call once with "
        "role='intro' and once with role='outro', passing the master's media_id as source.",
        {"source": {"type": "string", "description": "media_id of the master BGM track."},
         "role": {"type": "string", "enum": ["intro", "outro"]},
         "duration": {"type": "number", "description": "Snippet length in seconds (e.g. 4)."}},
        ["role"], trim_music, task="bgm"),

    "fetch_sfx": _tool(
        "fetch_sfx",
        "Fetch a one-shot sound effect by name from the library (e.g. 'cup_falling', "
        "'drum_reverb_1', 'whoosh', 'pop', 'ding'). Writes SFX_<NAME>_URL to the artifact.",
        {"name": {"type": "string", "description": "SFX name."}},
        ["name"], fetch_sfx, task="sfx"),

    "lipsync": _tool(
        "lipsync",
        "Animate the cat photo to the voice audio, producing the main talking video "
        "(a long async render). Call AFTER synthesize_speech. Writes TALKING_VIDEO_URL.",
        {"audio": {"type": "string", "description": "media_id of the voice audio."},
         "image": {"type": "string", "description": "Optional media_id/URL of the cat photo."},
         "service": {"type": "string", "description": "Lipsync service, e.g. dlai2v_pro."}},
        [], lipsync, task="talking"),

    "final_mix": _tool(
        "final_mix",
        "Assemble the finished video: intro snippet -> talking video (with BGM bed + SFX) "
        "-> outro snippet. Call LAST. Writes FINAL_VIDEO_URL.",
        {"talking_video": {"type": "string", "description": "media_id of the talking video."},
         "intro_music": {"type": "string", "description": "media_id of the intro snippet."},
         "outro_music": {"type": "string", "description": "media_id of the outro snippet."},
         "bgm": {"type": "string", "description": "media_id of the master BGM bed (optional)."},
         "sfx_cues": {"type": "array", "description": "Optional [{media_id, at}] SFX cue list.",
                      "items": {"type": "object"}}},
        [], final_mix, task="mix"),

    "generate_image": _tool(
        "generate_image",
        "Render a NEW still image via ComfyUI from a text prompt. Only use if you need "
        "artwork beyond the provided cat photo. Needs ComfyUI running.",
        {"prompt": {"type": "string", "description": "Image description."},
         "width": {"type": "integer"}, "height": {"type": "integer"}},
        ["prompt"], generate_image, task=None),
}


def schemas():
    """List of tool schemas to hand the LLM."""
    return [t["schema"] for t in TOOL_REGISTRY.values()]


def call(name, project, args):
    """Invoke a registered tool by name. Returns (result_dict, task_id)."""
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return {"error": f"unknown tool '{name}'"}, None
    try:
        result = entry["fn"](project, **(args or {}))
    except TypeError as e:
        result = {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        result = {"error": f"{name} failed: {e}"}
    return result, entry["task"]
