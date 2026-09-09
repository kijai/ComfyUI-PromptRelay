"""final_mix tool — assembles the finished video.

Real ffmpeg: intro card (cat photo + intro music) -> talking video with a BGM bed
and SFX mixed in -> outro card (cat photo + outro music), all concatenated."""

import os

from .. import ffmpeg
from ._resolve import resolve_cat_image, resolve_media


def _find(project, kind, predicate):
    for m in reversed(project.media):
        if m["kind"] == kind and predicate(m):
            return project.media_path(m["file"])
    return None


def final_mix(project, talking_video="", intro_music="", outro_music="",
              bgm="", sfx_cues=None, **_):
    """Combine intro music, the talking video (with BGM bed + SFX), and outro music
    into the final episode video. Missing pieces are auto-filled from the media library."""
    talking = resolve_media(project, talking_video, kind="video") or \
        _find(project, "video", lambda m: m["file"] == "talking.mp4")
    if not talking:
        return {"error": "no talking video found — run lipsync first"}

    intro = resolve_media(project, intro_music, kind="audio") or \
        _find(project, "audio", lambda m: m.get("meta", {}).get("role") == "intro")
    outro = resolve_media(project, outro_music, kind="audio") or \
        _find(project, "audio", lambda m: m.get("meta", {}).get("role") == "outro")
    bed = resolve_media(project, bgm, kind="audio") or \
        _find(project, "audio", lambda m: m.get("meta", {}).get("role") in ("master", "bed")) or intro

    work = talking
    steps = []

    # BGM bed under the talking audio
    if bed:
        out = project.media_path("_main_bgm.mp4")
        ffmpeg.mix_bgm(work, bed, out)
        work, _s = out, steps.append("bgm-bed")

    # SFX cues: [{name|media_id, at}] — else any sfx_* audio dropped in at 1s spacing
    cues = sfx_cues or []
    if not cues:
        for i, m in enumerate(m for m in project.media
                              if m["kind"] == "audio" and m["file"].startswith("sfx_")):
            cues.append({"media_id": m["id"], "at": 1.0 + i * 1.5})
    for i, cue in enumerate(cues):
        sfx_path = resolve_media(project, cue.get("media_id") or cue.get("name"), kind="audio")
        if not sfx_path:
            continue
        out = project.media_path(f"_main_sfx{i}.mp4")
        ffmpeg.overlay_sfx(work, sfx_path, out, at=float(cue.get("at", 1.0)))
        work = out
        steps.append(f"sfx@{cue.get('at',1.0)}")

    # Intro / outro cards
    cat_img = resolve_cat_image(project)
    clips = []
    if intro:
        intro_clip = project.media_path("_intro.mp4")
        ffmpeg.still_clip(cat_img, intro, intro_clip, motion=True)
        clips.append(intro_clip)
        steps.insert(0, "intro")
    clips.append(work)
    if outro:
        outro_clip = project.media_path("_outro.mp4")
        ffmpeg.still_clip(cat_img, outro, outro_clip, motion=True)
        clips.append(outro_clip)
        steps.append("outro")

    final = project.media_path("final.mp4")
    ffmpeg.concat_clips(clips, final)
    dur = ffmpeg.media_duration(final)
    item = project.add_media("video", final, via="ffmpeg",
                             label=f"{project.inputs.get('show_title','Final')} — final cut",
                             final=True, steps=steps, duration=round(dur, 2))
    project.write_artifact("FINAL_VIDEO_URL", media_id=item["id"])
    return {"ok": True, "media_id": item["id"], "file": item["file"],
            "duration": round(dur, 2), "steps": steps,
            "artifact": "FINAL_VIDEO_URL (v%d)" % project.ver}
