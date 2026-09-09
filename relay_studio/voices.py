"""Voice library — the bank `voice_name` is chosen from (drama.land shows ~44 voice
samples in its media panel). Each entry maps a friendly name to an edge-tts voice (used
when edge-tts is installed) and a gender hint (used to pick a SAPI voice via pyttsx3)."""

VOICE_LIBRARY = [
    {"name": "Anthony", "language": "EN_British", "gender": "male",
     "description": "Decent Young Man - Magnetic, lazy, playful wit", "edge": "en-GB-RyanNeural"},
    {"name": "English_DecentYoungMan", "language": "EN_US", "gender": "male",
     "description": "Warm, easy-going young man", "edge": "en-US-GuyNeural"},
    {"name": "English_DecentBoy", "language": "EN_US", "gender": "male",
     "description": "Youthful, upbeat boy", "edge": "en-US-AndrewNeural"},
    {"name": "Robot_Armor", "language": "EN_US", "gender": "male",
     "description": "Robotic, metallic delivery", "edge": "en-US-BrianNeural"},
    {"name": "Aria", "language": "EN_US", "gender": "female",
     "description": "Bright, friendly host", "edge": "en-US-AriaNeural"},
    {"name": "Sonia", "language": "EN_British", "gender": "female",
     "description": "Crisp British presenter", "edge": "en-GB-SoniaNeural"},
    {"name": "Chinese_Mandarin_Strong", "language": "ZH_CN", "gender": "male",
     "description": "Strong, confident Mandarin", "edge": "zh-CN-YunyangNeural"},
    {"name": "Chinese_Mandarin_Wise", "language": "ZH_CN", "gender": "female",
     "description": "Warm, wise Mandarin", "edge": "zh-CN-XiaoxiaoNeural"},
]


def list_voices():
    return [{k: v[k] for k in ("name", "language", "gender", "description")}
            for v in VOICE_LIBRARY]


def get_voice(name):
    """Resolve a voice by name (case/format-insensitive); None if unknown."""
    if not name:
        return None
    key = str(name).strip().lower().replace(" ", "_")
    for v in VOICE_LIBRARY:
        if v["name"].lower().replace(" ", "_") == key:
            return v
    for v in VOICE_LIBRARY:  # loose contains match
        if key in v["name"].lower():
            return v
    return None
