"""Cinematic Sequence — a Seedance-2.0-style multi-shot composer over LOCAL LTX 2.3.

Seedance 2.0's whole UI is built around ONE prompt that describes several sequenced shots:

    Cinematic Fantasy Action, Arri Alexa 65, warm museum lighting, epic and magical.
    S1: [Medium shot, static] + [A woman sits on a bench, reading.]
    S2: [Wide shot, low angle] + [A stone gargoyle comes to life.]

Your existing `reel` builder renders shots too, but each shot is an unrelated freeform
prompt — no shared style header, no structured camera grammar, and it can't keep a
consistent look across shots. This module adds that missing skeleton:

  * a GLOBAL STYLE header (genre + camera/lens + lighting) prepended to every shot, so the
    look stays consistent shot-to-shot;
  * a structured SHOT grammar (shot size, camera move, action) compiled into Seedance's
    exact `S1: [size, move] + [action]` string;
  * a compiler shared by the live-preview endpoint and the renderer, so what you see is
    what renders;
  * skeleton hooks for the two Seedance features that have no local model yet — native
    AUDIO and true 4K — which render at 1080p and tag themselves "pending" rather than
    pretending to be done.

Rendering reuses the proven local pipeline: each compiled shot -> engine.generate_t2v /
generate_i2v, all shots sharing one base seed + the style header for continuity, then
reel.stitch joins them into a single clip. Costs nothing, unlimited, NSFW-ok.
"""

import os
import time

from . import engine, reel, tasks, history, checkpoints

# ── Shot grammar ──────────────────────────────────────────────────────────────
# value -> human phrase. The UI reads these (via the grammar route) to build dropdowns,
# and the compiler turns the chosen values into the Seedance-style shot string.

SHOT_SIZES = {
    "extreme_wide": "extreme wide shot",
    "wide": "wide shot",
    "medium_wide": "medium wide shot",
    "medium": "medium shot",
    "medium_close": "medium close-up",
    "close": "close-up",
    "extreme_close": "extreme close-up",
}

MOVEMENTS = {
    "static": "static",
    "slow_push": "slow push in",
    "pull_back": "slow pull back",
    "pan_left": "pan left",
    "pan_right": "pan right",
    "tilt_up": "tilt up",
    "tilt_down": "tilt down",
    "tracking": "tracking shot",
    "handheld": "handheld",
    "crane": "crane move",
    "orbit": "orbit around subject",
    "low_angle": "low angle",
    "high_angle": "high angle",
    "overhead": "overhead top-down",
    "dutch": "dutch angle",
}

# Style-header building blocks (chips in the UI). Purely additive text — the user can also
# type freely into the style field.
CAMERAS = {
    "arri_65": "Arri Alexa 65",
    "arri_mini": "Arri Alexa Mini",
    "red_komodo": "RED Komodo",
    "film_35": "35mm film",
    "film_16": "16mm film grain",
    "anamorphic": "anamorphic lens flares",
    "wide_lens": "wide-angle lens",
    "telephoto": "telephoto lens",
    "macro": "macro lens",
}

LIGHTING = {
    "golden_hour": "warm golden-hour light",
    "soft_daylight": "soft natural daylight",
    "overcast": "soft overcast light",
    "neon_night": "neon night lighting",
    "candlelit": "warm candlelit glow",
    "high_key": "bright high-key lighting",
    "low_key": "moody low-key lighting",
    "backlit": "dramatic backlight",
}

MOODS = {
    "cinematic": "cinematic",
    "epic": "epic and magical",
    "dreamy": "soft and dreamy",
    "gritty": "gritty and raw",
    "noir": "film-noir",
    "documentary": "naturalistic documentary",
}


def grammar():
    """Everything the UI needs to build its controls (value -> label maps)."""
    return {
        "shot_sizes": SHOT_SIZES,
        "movements": MOVEMENTS,
        "cameras": CAMERAS,
        "lighting": LIGHTING,
        "moods": MOODS,
    }


def available_audio():
    """LTX 2.3's t2v/i2v graphs already decode a native synchronized audio track
    (LTXVAudioVAEDecode), so audio is real — the toggle keeps it (on) or mutes it (off)."""
    return True


# 4K target. The clip is rendered at 1080p then upscaled here (lanczos + sharpen); a true
# ESRGAN pass would OOM the 16 GB card on multi-second video, so this is a high-quality
# resample, applied non-fatally (see run()).
_UPSCALE_4K = (3840, 2160)


def status():
    return {
        "sequence": engine.available_t2v() or engine.available(),
        "t2v": engine.available_t2v(),
        "i2v": engine.available(),
        "audio": available_audio(),
        "upscale_4k": True,
        "max_native_height": 1080,  # 4K = render 1080p, then upscale post-pass
    }


# ── Compiler ──────────────────────────────────────────────────────────────────

def _phrase(mapping, key):
    """Look up a grammar value, tolerating a raw free-text phrase too."""
    if not key:
        return ""
    return mapping.get(key, str(key))


def _shot_bracket(shot):
    """The `[size, move]` half of a Seedance shot."""
    parts = [_phrase(SHOT_SIZES, shot.get("size")),
             _phrase(MOVEMENTS, shot.get("movement"))]
    return ", ".join(p for p in parts if p)


def compile_shot_prompt(style, shot):
    """The text actually fed to LTX for one shot: style header + camera framing + action,
    flattened to a single descriptive line (LTX reads prose, not the S#: syntax)."""
    bits = []
    if style:
        bits.append(style.strip().rstrip("."))
    bracket = _shot_bracket(shot)
    if bracket:
        bits.append(bracket)
    action = (shot.get("action") or "").strip()
    if action:
        bits.append(action)
    return ". ".join(b for b in bits if b).strip()


def compile_sequence(style, shots):
    """Compile a style header + shot list into (a) the human-readable Seedance-format
    script for the live preview, and (b) the per-shot render prompts.

    Returns {"script": str, "shots": [{"index","label","bracket","action","prompt"}, ...]}.
    """
    style = (style or "").strip()
    lines = []
    compiled = []
    for i, shot in enumerate(shots or [], start=1):
        label = f"S{i}"
        bracket = _shot_bracket(shot)
        action = (shot.get("action") or "").strip()
        # Seedance-format line: S1: [Medium shot, static] + [action]
        left = f"[{bracket}]" if bracket else "[shot]"
        right = f"[{action}]" if action else "[…]"
        lines.append(f"{label}: {left} + {right}")
        compiled.append({
            "index": i,
            "label": label,
            "bracket": bracket,
            "action": action,
            "prompt": compile_shot_prompt(style, shot),
        })
    header = (style + ".\n") if style else ""
    script = header + "\n".join(lines)
    return {"script": script.strip(), "shots": compiled}


# ── Renderer ──────────────────────────────────────────────────────────────────

def run(tid, style, shots, seed, width, height, duration, model,
        resolution_label=None, audio=False, transition="crossfade", xfade=0.5,
        output_dir=None, resume_done=None):
    """Blocking worker (runs in a thread): render every compiled shot on the SAME base seed
    + shared style header for continuity, then stitch into one clip. Updates the task as it
    goes. `shots` items may carry a resolved `image_path` -> that shot renders as i2v.

    Checkpointed: a checkpoint file is written at start and after every shot, and deleted on
    full success. If the render dies partway, the checkpoint (plus the shot clips already in
    ComfyUI's output dir) lets `resume()` re-run only the MISSING shots. `resume_done` maps a
    shot index (1-based, as str or int) -> an already-rendered clip path to reuse."""
    compiled = compile_sequence(style, shots)["shots"]
    n = len(compiled)
    tasks.update(tid, status="RUNNING", phase="starting", shots_total=n, shots_done=0)

    out_dir = output_dir or engine._OUTPUT_DIR
    # index(int) -> path for shots already rendered (from a prior interrupted run)
    done_map = {int(k): v for k, v in (resume_done or {}).items() if v and os.path.exists(v)}

    def _write_ckpt():
        checkpoints.save(tid, {
            "label": f"{n} shots · {(style or (compiled[0]['action'] if compiled else ''))[:40]}",
            "style": style, "shots": shots, "seed": seed, "width": width, "height": height,
            "duration": duration, "model": model, "resolution_label": resolution_label,
            "audio": audio, "transition": transition, "xfade": xfade, "output_dir": out_dir,
            "done": [{"index": i, "path": p} for i, p in sorted(done_map.items())],
            "updated": time.time()})

    paths, first_wh = [], None
    try:
        _write_ckpt()
        tasks.update(tid, shots_done=len(done_map))
        for i, cs in enumerate(compiled):
            idx = i + 1
            src = shots[i]
            if idx in done_map:  # resumed — reuse the clip already on disk, don't re-render
                tasks.update(tid, phase=f"reusing {cs['label']} ({idx}/{n})", current_shot=idx)
                paths.append(done_map[idx])
            else:
                tasks.update(tid, phase=f"rendering {cs['label']} ({idx}/{n})", current_shot=idx)
                prompt = cs["prompt"]
                if not prompt:
                    raise RuntimeError(f"{cs['label']}: add an action or style")
                img = src.get("image_path")
                if img and os.path.exists(img):
                    info = engine.generate_i2v(img, prompt, seed=seed, width=width,
                                               height=height, duration=duration, model=model,
                                               fit=bool(src.get("fit")))
                else:
                    info = engine.generate_t2v(prompt, seed=seed, width=width, height=height,
                                               duration=duration, model=model)
                p = os.path.join(out_dir, info.get("subfolder", ""), info["filename"])
                paths.append(p)
                done_map[idx] = p
                _write_ckpt()  # checkpoint after each shot so a crash resumes from here
                tasks.update(tid, shots_done=len(done_map))
            if first_wh is None:
                first_wh = (width or 1280, height or 720)

        tasks.update(tid, phase="stitching")
        w, h = first_wh or (1280, 720)
        out_name = f"sequence_{tid}.mp4"
        out_path = os.path.join(out_dir, "video", out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        reel.stitch(paths, out_path, transition=transition, xfade=xfade, width=w, height=h)

        # ── Finalize pass: 4K upscale and/or mute. NON-FATAL — if it fails we keep the
        # stitched base render rather than losing the whole sequence.
        pending = []
        want_4k = resolution_label == "4k"
        mute = not audio
        if want_4k or mute:
            tasks.update(tid, phase="upscaling to 4K" if want_4k else "finalizing")
            fin_name = f"sequence_{tid}_final.mp4"
            fin_path = os.path.join(out_dir, "video", fin_name)
            uw, uh = _UPSCALE_4K if want_4k else (None, None)
            try:
                reel.finalize(out_path, fin_path, width=uw, height=uh, mute=mute)
                out_name = fin_name  # serve the finalized clip
            except Exception as fe:  # keep the base render, just note what didn't apply
                if want_4k:
                    pending.append("4k-upscale-failed")
                if mute:
                    pending.append("mute-failed")
                tasks.update(tid, finalize_error=str(fe)[:200])

        from urllib.parse import urlencode
        q = urlencode({"filename": out_name, "subfolder": "video", "type": "output"})
        video_url = f"/view?{q}"

        tasks.update(tid, status="SUCCEEDED", phase="done", video_url=video_url,
                     seed=seed, pending=pending)
        script = compile_sequence(style, shots)["script"]
        history.record("sequence", script, video_url, seed=seed,
                       params={"shots": n, "resolution": resolution_label,
                               "duration": duration, "audio": audio, "pending": pending})
        checkpoints.delete(tid)  # finished cleanly — nothing left to resume
    except Exception as e:  # surface to the UI; KEEP the checkpoint so it can be resumed
        tasks.update(tid, status="FAILED", error=str(e)[:500], resumable=True)


def resume(tid):
    """Re-run an interrupted sequence from its checkpoint (thread worker like run()):
    reuse the shots already on disk, render only the missing ones, then stitch/finalize."""
    c = checkpoints.load(tid)
    if not c:
        tasks.update(tid, status="FAILED", error="no checkpoint to resume")
        return
    resume_done = {str(d["index"]): d["path"] for d in (c.get("done") or [])}
    run(tid, c.get("style") or "", c.get("shots") or [], c.get("seed"),
        c.get("width"), c.get("height"), c.get("duration"), c.get("model"),
        resolution_label=c.get("resolution_label"), audio=c.get("audio", False),
        transition=c.get("transition", "crossfade"), xfade=c.get("xfade", 0.5),
        output_dir=c.get("output_dir"), resume_done=resume_done)
