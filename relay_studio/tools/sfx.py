"""fetch_sfx tool — pulls a one-shot sound effect from the asset library and writes it
to the artifact as SFX_<NAME>_URL (e.g. SFX_CUP_FALL_URL).

Stub backend: looks up assets/sfx/ by name. Swap for a real SFX generator/library by
replacing assets.find_sfx."""

import os
import re
import shutil

from .. import assets, ffmpeg


def _field_for(name):
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "sfx").lower()).strip("_") or "sfx"
    # cup_falling -> SFX_CUP_FALL_URL  (mirror drama.land's field naming)
    slug = slug.replace("falling", "fall")
    return f"SFX_{slug.upper()}_URL"


def fetch_sfx(project, name="", **_):
    """Fetch an SFX clip by name and write it to the artifact."""
    src = assets.find_sfx(name)
    safe = (name or "sfx").lower().strip().replace(" ", "_") or "sfx"
    dst = project.media_path(f"sfx_{safe}.wav")
    shutil.copyfile(src, dst)
    dur = ffmpeg.media_duration(dst)
    item = project.add_media("audio", dst, via="library", label=f"SFX: {name or 'sound'}",
                             name=name, duration=round(dur, 2))
    field = _field_for(name)
    project.write_artifact(field, media_id=item["id"])
    return {"ok": True, "media_id": item["id"], "file": item["file"], "name": name,
            "duration": round(dur, 2), "source": os.path.basename(src),
            "artifact": "%s (v%d)" % (field, project.ver)}
