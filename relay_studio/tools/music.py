"""BGM tools — mirrors drama.land: generate ONE master track, then trim short intro
and outro snippets from it.

  generate_music(description, mood, duration)  -> master track, writes BGM_URL
  trim_music(source, role, duration)           -> intro/outro snippet from the master

Stub backend: selects a mood-matched bed from assets/music/. Swap for a real music
model (MusicGen / Stable Audio) by replacing generate_music's body."""

import os

from .. import assets, ffmpeg
from ._resolve import resolve_media


def generate_music(project, description="", mood="", duration=60.0, **_):
    """Generate the master BGM track (a 'Playful jazz theme (30-90 sec)') and write it
    to the artifact as BGM_URL."""
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 60.0
    duration = max(20.0, min(duration, 120.0))
    mood = mood or description
    src = assets.find_music(mood)

    dst = project.media_path("music_master.wav")
    fade = 1.0
    ffmpeg._run(["-stream_loop", "-1", "-i", src, "-t", f"{duration}",
                 "-filter:a", (f"afade=t=in:d={fade},"
                               f"afade=t=out:st={max(0.0, duration-fade)}:d={fade}"),
                 "-ar", str(ffmpeg.AR), "-ac", str(ffmpeg.AC), dst])
    dur = ffmpeg.media_duration(dst)
    placeholder = src.endswith("_bed.wav")
    item = project.add_media("audio", dst, via="music-stub", role="master",
                             label=f"BGM master — {mood or 'theme'}", mood=mood,
                             placeholder=placeholder, duration=round(dur, 2))
    project.write_artifact("BGM_URL", media_id=item["id"])
    return {"ok": True, "media_id": item["id"], "file": item["file"], "duration": round(dur, 2),
            "mood": mood, "placeholder": placeholder, "artifact": "BGM_URL (v%d)" % project.ver,
            "next": "call trim_music(source=this media_id, role='intro') and role='outro'"}


def trim_music(project, source="", role="intro", duration=4.0, **_):
    """Trim a short intro/outro snippet from the master BGM track."""
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 4.0
    duration = max(2.0, min(duration, 15.0))
    master = resolve_media(project, source, kind="audio") or \
        project.media_path("music_master.wav")
    if not os.path.exists(master):
        return {"error": "no master BGM track — call generate_music first"}

    role = (role or "intro").lower()
    total = ffmpeg.media_duration(master)
    start = 0.0 if role == "intro" else max(0.0, total - duration)
    dst = project.media_path(f"music_{role}.wav")
    ffmpeg.trim_audio(master, dst, start=start, duration=duration)
    dur = ffmpeg.media_duration(dst)
    item = project.add_media("audio", dst, via="ffmpeg", role=role,
                             label=f"BGM ({role}) snippet", duration=round(dur, 2))
    return {"ok": True, "media_id": item["id"], "file": item["file"], "role": role,
            "duration": round(dur, 2)}
