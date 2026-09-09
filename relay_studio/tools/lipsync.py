"""lipsync tool — animates the cat photo to the voice track as the main talking video.

Modelled on drama.land's behaviour: a long-running background render job that emits a
'job_started' event, then a 'notification' (系统通知 · 后台任务完成) on completion, and
writes the result to the artifact as TALKING_VIDEO_URL.

Stub backend: a ken-burns motion clip synced to the speech length (a real video, just
not mouth-synced). Swap service='dlai2v_pro' / model 'longcat-avatar' / SadTalker in here."""

import hashlib
import os
import time
import uuid

from .. import ffmpeg
from ._resolve import resolve_cat_image, resolve_media

# Optional artificial render delay (seconds) to make the async lifecycle visible.
RENDER_DELAY = float(os.environ.get("STUDIO_LIPSYNC_DELAY", "0") or 0)
# Preferred animator: auto (LTX i2v if available, else wav2lip, else ken-burns),
# or force one of: ltx | wav2lip | kenburns.
LIPSYNC_MODE = (os.environ.get("STUDIO_LIPSYNC_MODE", "auto") or "auto").lower()


def _img_key(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _cached_ltx_motion(project, img_path):
    """Return an LTX motion clip for this photo, rendering it only if the photo
    changed (LTX is the slow ~6-min step; the voice mux on top is fast)."""
    from .. import ltx_engine
    motion = project.media_path("ltx_motion.mp4")
    keyfile = project.media_path("ltx_motion.key")
    key = _img_key(img_path)
    if os.path.exists(motion) and os.path.exists(keyfile):
        try:
            if open(keyfile, encoding="utf-8").read().strip() == key:
                project.emit("status", text="Reusing cached LTX motion (photo unchanged)")
                return motion
        except OSError:
            pass
    project.emit("status", text="Rendering LTX motion video (~6 min, first time for this photo)…")
    ltx_engine.generate_i2v(img_path, motion)
    with open(keyfile, "w", encoding="utf-8") as f:
        f.write(key)
    return motion


def _mux_voice_over_motion(motion, audio_path, dst):
    """Lay the voice over the LTX motion clip, trim to voice length, normalise format."""
    vf = (f"scale={ffmpeg.WIDTH}:{ffmpeg.HEIGHT}:force_original_aspect_ratio=decrease,"
          f"pad={ffmpeg.WIDTH}:{ffmpeg.HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
          f"fps={ffmpeg.FPS},format=yuv420p")
    ffmpeg._run(["-i", motion, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                 "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(ffmpeg.FPS),
                 *ffmpeg._normalise_audio_args(), "-shortest", dst])


def lipsync(project, audio="", image="", service="", **_):
    """Combine the cat photo + voice audio into a talking video clip (async job)."""
    audio_path = resolve_media(project, audio, kind="audio")
    if not audio_path:
        out = project.outputs.get("TTS_AUDIO_URL", {})
        audio_path = resolve_media(project, out.get("media_id"), kind="audio")
    if not audio_path:
        auds = [m for m in project.media if m["kind"] == "audio" and "voice" in m["file"]]
        if auds:
            audio_path = project.media_path(auds[-1]["file"])
    if not audio_path:
        return {"error": "no voice audio found — run synthesize_speech first"}

    svc = service or project.inputs.get("lipsync_service") or "dlai2v_pro"
    job_id = uuid.uuid4().hex[:8].upper()

    img_path = resolve_cat_image(project, image)
    dst = project.media_path("talking.mp4")

    # Animate the cat. Preferred: LTX i2v (expressive motion on the real cat) with the
    # voice laid over it; fall back to Wav2Lip mouth-sync on the still, then ken-burns.
    model = "ken-burns"
    note = "ffmpeg ken-burns motion (no model installed)"
    used_real = False
    mode = "ken-burns"

    # 1. LTX i2v alone — moving, expressive cat (mouth not word-synced).
    if LIPSYNC_MODE in ("auto", "ltx"):
        try:
            from .. import ltx_engine
            if ltx_engine.available():
                project.emit("job_started", job_id=job_id,
                             label=f"lipsync ({svc} · LTX-2.3 i2v)", eta="~6 min (first render)")
                motion = _cached_ltx_motion(project, img_path)
                _mux_voice_over_motion(motion, audio_path, dst)
                model, note, used_real, mode = ("ltx-2.3-i2v",
                    "LTX i2v expressive motion (real cat motion; not word-synced)", True, "ltx")
        except Exception as e:
            project.emit("status", text=f"LTX i2v unavailable ({e}); trying wav2lip")

    # 2. Wav2Lip mouth-sync on the still.
    if not used_real and LIPSYNC_MODE in ("auto", "wav2lip"):
        try:
            from .. import wav2lip_engine
            if wav2lip_engine.available():
                project.emit("job_started", job_id=job_id,
                             label=f"lipsync ({svc} · wav2lip_gan)", eta="~30s")
                wav2lip_engine.run_wav2lip(img_path, audio_path, dst)
                model, note, used_real, mode = ("wav2lip_gan",
                    "mouth-synced lipsync (Wav2Lip)", True, "wav2lip")
        except Exception as e:
            project.emit("status", text=f"Wav2Lip unavailable ({e}); using motion fallback")

    # 3. Ken-burns motion fallback.
    if not used_real:
        project.emit("job_started", job_id=job_id,
                     label=f"lipsync ({svc} · ken-burns fallback)", eta="~10s")
        if RENDER_DELAY:
            time.sleep(RENDER_DELAY)
        ffmpeg.still_clip(img_path, audio_path, dst, motion=True)

    dur = ffmpeg.media_duration(dst)
    placeholder = img_path.endswith("placeholder_cat.png")
    item = project.add_media("video", dst, via=mode,
                             label=f"{project.inputs.get('cat_name','Cat')} talking",
                             service=svc, model=model, lipsynced=used_real,
                             placeholder_image=placeholder, duration=round(dur, 2))
    project.write_artifact("TALKING_VIDEO_URL", media_id=item["id"])
    project.emit("notification", job_id=job_id,
                 text=f"后台任务完成 · background task complete ({job_id})")
    return {"ok": True, "media_id": item["id"], "file": item["file"], "job_id": job_id,
            "duration": round(dur, 2), "service": svc, "model": model,
            "note": note, "lipsynced": used_real,
            "artifact": "TALKING_VIDEO_URL (v%d)" % project.ver}
