"""Bundled placeholder media + lookup helpers.

The music/SFX here are synthesised tones/noise so the whole pipeline runs fully
offline with zero downloads. They are intentionally swappable: drop real royalty-free
files into assets/music/ and assets/sfx/ (keep the mood/name in the filename) and the
lookups below will prefer them automatically.
"""

import glob
import os

from . import ffmpeg

BASE = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.join(BASE, "assets", "music")
SFX_DIR = os.path.join(BASE, "assets", "sfx")
PLACEHOLDER_CAT = os.path.join(BASE, "assets", "placeholder_cat.png")

# mood keyword -> (filename, root frequency) for synthesised placeholders.
_MUSIC_BEDS = {
    "jazz":   ("jazz_bed.wav", 196.0),
    "upbeat": ("upbeat_bed.wav", 261.6),
    "calm":   ("calm_bed.wav", 174.6),
    "tense":  ("tense_bed.wav", 146.8),
    "happy":  ("happy_bed.wav", 293.7),
}
_DEFAULT_MUSIC = "jazz"

# SFX name -> (filename, noise colour, duration)
_SFX = {
    "drum_reverb_1": ("drum_reverb_1.wav", "brown", 1.0),
    "cup_falling":   ("cup_falling.wav", "white", 0.6),
    "whoosh":        ("whoosh.wav", "pink", 0.5),
    "pop":           ("pop.wav", "white", 0.2),
    "ding":          ("ding.wav", "brown", 0.4),
}


def ensure_placeholders():
    """Create any missing placeholder assets. Cheap no-op once they exist."""
    os.makedirs(MUSIC_DIR, exist_ok=True)
    os.makedirs(SFX_DIR, exist_ok=True)
    for fn, freq in _MUSIC_BEDS.values():
        path = os.path.join(MUSIC_DIR, fn)
        if not os.path.exists(path):
            ffmpeg.gen_tone(path, duration=20.0, freq=freq)
    for fn, color, dur in _SFX.values():
        path = os.path.join(SFX_DIR, fn)
        if not os.path.exists(path):
            ffmpeg.gen_noise_sfx(path, duration=dur, color=color)
    if not os.path.exists(PLACEHOLDER_CAT):
        ffmpeg.gen_color_image(PLACEHOLDER_CAT, color="0x3a506b")


def find_music(mood=""):
    """Best music file for a mood. Prefers a real drop-in file whose name contains
    the mood keyword, else the synthesised placeholder for that mood."""
    ensure_placeholders()
    mood = (mood or "").lower()
    # 1) any real file mentioning the mood
    for path in sorted(glob.glob(os.path.join(MUSIC_DIR, "*"))):
        name = os.path.basename(path).lower()
        if mood and mood in name and not name.endswith("_bed.wav"):
            return path
    # 2) placeholder for the closest mood keyword
    for key, (fn, _) in _MUSIC_BEDS.items():
        if key in mood:
            return os.path.join(MUSIC_DIR, fn)
    return os.path.join(MUSIC_DIR, _MUSIC_BEDS[_DEFAULT_MUSIC][0])


def find_sfx(name=""):
    """Best SFX file for a name. Prefers a real drop-in file, else a known
    placeholder, else a generic burst."""
    ensure_placeholders()
    name = (name or "").lower().strip().replace(" ", "_")
    for path in sorted(glob.glob(os.path.join(SFX_DIR, "*"))):
        if name and name in os.path.basename(path).lower():
            return path
    if name in _SFX:
        return os.path.join(SFX_DIR, _SFX[name][0])
    return os.path.join(SFX_DIR, _SFX["pop"][0])
