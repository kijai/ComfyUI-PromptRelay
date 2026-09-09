"""Generation engines for Relay Director, with graceful fallbacks.

Each function tries the real generator (ComfyUI Z-Image for stills, LTX i2v for clips,
edge-tts for voice) and degrades to an offline placeholder if the model / service isn't
available — so the full director flow is always demoable, even with nothing loaded.

We *reuse* Dolly's binaries read-only (relay_studio.ffmpeg for the bundled ffmpeg,
relay_studio.ltx_engine for the proven LTX graph) without modifying Dolly. All clip
geometry here is driven by the Story's own width/height so the director can be 16:9,
9:16 or 1:1 independently of Dolly's square pipeline.
"""

import os
import random
import time


# ---------------------------------------------------------------------------
# Lazy imports — never break package load if an engine module is missing.
# ---------------------------------------------------------------------------

def _ff():
    """Dolly's ffmpeg helper (bundled binary, _run, media_duration)."""
    from ..relay_studio import ffmpeg as _f
    return _f


def _ltx():
    try:
        from ..relay_studio import ltx_engine as _l
        return _l
    except Exception:
        return None


def _klein():
    try:
        from . import flux_klein_engine as _k
        return _k
    except Exception:
        return None


def _storyboard():
    try:
        from .. import storyboard_engine as _s
        return _s
    except Exception:
        return None


COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


# ---------------------------------------------------------------------------
# Still image — ComfyUI (Z-Image) with a gradient-card placeholder fallback.
# ---------------------------------------------------------------------------

def generate_image(story, shot, *, on_log=None, prompt_override=None):
    """Render the shot's still into story.dir. Returns (filename, via).
    `prompt_override` lets the director inject a cast/world-composed prompt for
    consistency while keeping the shot's own editable prompt intact."""
    log = on_log or (lambda *_: None)
    prompt = (prompt_override or shot.get("prompt", "")).strip() or "cinematic still, dramatic lighting"
    seed = shot.get("seed") or random.randrange(2 ** 48)
    shot["seed"] = seed
    fname = f"shot_{shot['id']}_v{shot['ver']}.png"
    dst = story.media_path(fname)

    se = _storyboard()
    if se is not None:
        try:
            log(f"ComfyUI render • {story.width}x{story.height} • seed {seed}")
            if _comfy_image(se, prompt, seed, story.width, story.height, dst):
                return fname, "comfyui"
            log("ComfyUI returned no image — using placeholder")
        except Exception as e:
            log(f"ComfyUI image failed ({e}) — using placeholder")

    _placeholder_card(dst, story.width, story.height, prompt, shot["idx"])
    return fname, "placeholder"


def generate_reference(story, character, *, on_log=None):
    """Render one canonical character portrait (locked seed) for consistency.
    Returns (filename, via, seed). Falls back to a labelled placeholder card."""
    log = on_log or (lambda *_: None)
    seed = character.get("ref_seed") or random.randrange(2 ** 48)
    w = story.world or {}
    look = ", ".join(x for x in (w.get("style"), w.get("palette")) if x)
    prompt = (f"character reference sheet, full-body and face, of {character.get('name','')}: "
              f"{character.get('description','')}. {look}. neutral studio background, "
              f"clean lighting, consistent character design, highly detailed")
    fname = f"ref_{character['id']}.png"
    dst = story.media_path(fname)
    se = _storyboard()
    if se is not None:
        try:
            log(f"character sheet • {character.get('name','')} • seed {seed}")
            if _comfy_image(se, prompt, seed, story.width, story.height, dst):
                return fname, "comfyui", seed
        except Exception as e:
            log(f"ref image failed ({e}) — placeholder")
    _placeholder_card(dst, story.width, story.height,
                      character.get("name", "?"), abs(hash(character["id"])) % 6)
    return fname, "placeholder", seed


def _comfy_image(se, prompt, seed, width, height, dst):
    import requests
    models = se.resolve_models()
    wf = se.build_workflow(prompt, seed, width=int(width), height=int(height),
                           steps=8, models=models)
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": wf, "client_id": "relay-director"}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected workflow: {r.text[:200]}")
    pid = r.json().get("prompt_id")
    deadline = time.time() + 180
    hist = None
    while time.time() < deadline:
        h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=10)
        if h.status_code == 200 and pid in h.json():
            hist = h.json()[pid]
            break
        time.sleep(1.0)
    if not hist:
        return False
    images = [im for node in hist.get("outputs", {}).values()
              for im in node.get("images", [])]
    if not images:
        return False
    info = images[0]
    view = requests.get(f"{COMFY_URL}/view", params={
        "filename": info["filename"], "subfolder": info.get("subfolder", ""),
        "type": info.get("type", "output")}, timeout=30)
    with open(dst, "wb") as fh:
        fh.write(view.content)
    return True


# A deterministic palette so placeholder shots look intentional, not broken.
_PALETTE = ["1a1f3a", "3a1a2f", "1a3a2a", "3a2f1a", "2a1a3a", "1a2a3a"]


def _placeholder_card(dst, w, h, prompt, idx):
    """A tasteful gradient card with the shot number — stands in for a real still."""
    ff = _ff()
    c1 = _PALETTE[idx % len(_PALETTE)]
    c2 = _PALETTE[(idx + 3) % len(_PALETTE)]
    # Vertical gradient via gradients source; overlay the shot index large + centered.
    vf = (f"gradients=s={w}x{h}:c0=0x{c1}:c1=0x{c2}:x0=0:y0=0:x1=0:y1={h},"
          f"drawtext=text='{idx:02d}':fontcolor=white@0.22:fontsize={int(h*0.6)}:"
          f"x=(w-text_w)/2:y=(h-text_h)/2")
    try:
        ff._run(["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}",
                 "-vf", vf, "-frames:v", "1", dst])
    except Exception:
        ff._run(["-f", "lavfi", "-i", f"color=c=0x{c1}:s={w}x{h}", "-frames:v", "1", dst])
    return dst


# ---------------------------------------------------------------------------
# Character identity-lock (Tier B) — Flux 2 Klein face swap on the final pass.
# ---------------------------------------------------------------------------

def swap_character(story, shot, subject_path, character, *, on_log=None, lora_strength=0.9):
    """Lock `character`'s identity (its reference sheet) onto the shot's base image via
    Flux Klein. Returns (filename, via) on success, or (None, None) to fall back to the
    un-swapped base image. Slow (Flux Klein) — final pass only."""
    log = on_log or (lambda *_: None)
    klein = _klein()
    ref = character.get("ref_image")
    if klein is None or not klein.available() or not ref:
        return None, None
    ref_path = story.media_path(ref)
    if not os.path.exists(ref_path):
        return None, None
    fname = f"shot_{shot['id']}_v{shot['ver']}_klein.png"
    dst = story.media_path(fname)
    try:
        log(f"Flux Klein identity-lock • {character.get('name','')} (slow)")
        klein.generate_swap(subject_path, ref_path, dst,
                            lora_strength=lora_strength, timeout=1200)
        return fname, "flux-klein"
    except Exception as e:
        log(f"identity-lock failed ({e}) — keeping base image")
        return None, None


# ---------------------------------------------------------------------------
# Clip — LTX i2v with a ken-burns placeholder fallback.
# ---------------------------------------------------------------------------

def generate_clip(story, shot, image_path, audio_path, *, on_log=None, use_ltx=True):
    """Turn the shot's still into a clip. Returns (filename, via)."""
    log = on_log or (lambda *_: None)
    fname = f"clip_{shot['id']}_v{shot['ver']}.mp4"
    dst = story.media_path(fname)
    dur = float(shot.get("duration") or 4.0)

    ltx = _ltx()
    if use_ltx and ltx is not None and ltx.available():
        try:
            log("LTX i2v render (slow on 16GB — please wait)")
            motion = shot.get("motion") or _default_motion(shot)
            raw = story.media_path(f"_ltxraw_{shot['id']}.mp4")
            ltx.generate_i2v(image_path, raw, prompt=motion, timeout=900)
            _normalize_clip(raw, dst, story.width, story.height, story.fps, audio_path)
            _safe_rm(raw)
            return fname, "ltx"
        except Exception as e:
            log(f"LTX failed ({e}) — using ken-burns motion")

    _kenburns(image_path, dst, story.width, story.height, story.fps, dur, audio_path)
    return fname, "kenburns"


def _default_motion(shot):
    return ("subtle natural motion, gentle camera drift, cinematic, "
            "consistent subject, locked composition")


def _normalize_clip(src, dst, w, h, fps, audio):
    """Scale/pad an arbitrary clip to the story geometry; attach audio (or silence)."""
    ff = _ff()
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p,fps={fps}")
    if audio and os.path.exists(audio):
        ff._run(["-i", src, "-i", audio, "-vf", vf,
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                 "-shortest", dst])
    else:
        ff._run(["-i", src, "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=44100",
                 "-vf", vf, "-map", "0:v", "-map", "1:a",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-shortest", dst])
    return dst


def _kenburns(image, dst, w, h, fps, dur, audio):
    """Slow zoom over a still — the dependable placeholder clip."""
    ff = _ff()
    frames = max(int(dur * fps), 1)
    vf = (f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
          f"crop={w*2}:{h*2},"
          f"zoompan=z='min(zoom+0.0007,1.2)':d={frames}:"
          f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps},"
          f"format=yuv420p")
    common = ["-loop", "1", "-i", image]
    if audio and os.path.exists(audio):
        ff._run([*common, "-i", audio, "-vf", vf, "-t", f"{dur}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                 "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                 "-shortest", dst])
    else:
        ff._run([*common, "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=44100",
                 "-vf", vf, "-t", f"{dur}", "-map", "0:v", "-map", "1:a",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                 "-c:a", "aac", "-b:a", "192k", "-shortest", dst])
    return dst


# ---------------------------------------------------------------------------
# Voice — edge-tts (online, good) -> silent track. Optional per shot.
# ---------------------------------------------------------------------------

def generate_voice(story, shot, *, on_log=None):
    """Synthesize the shot's dialogue. Returns filename or None (no dialogue / failed)."""
    log = on_log or (lambda *_: None)
    text = (shot.get("dialogue") or "").strip()
    if not text:
        return None
    fname = f"voice_{shot['id']}_v{shot['ver']}.mp3"
    dst = story.media_path(fname)
    try:
        import asyncio
        import edge_tts
        log("edge-tts voice")

        async def _run():
            comm = edge_tts.Communicate(text, "en-US-AriaNeural")
            await comm.save(dst)
        asyncio.new_event_loop().run_until_complete(_run())
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return fname
    except Exception as e:
        log(f"voice unavailable ({e}) — shot will be silent")
    return None


# ---------------------------------------------------------------------------
# Timeline export — concat all shot clips into one widescreen mp4.
# ---------------------------------------------------------------------------

def export_timeline(story, clip_paths, *, on_log=None):
    """Concatenate ordered shot clips (already normalised) into final.mp4."""
    log = on_log or (lambda *_: None)
    ff = _ff()
    clips = [c for c in clip_paths if c and os.path.exists(c)]
    if not clips:
        raise RuntimeError("no clips to export — render some shots first")
    dst = story.media_path("final.mp4")
    if len(clips) == 1:
        ff._run(["-i", clips[0], "-c", "copy", dst])
        return dst
    log(f"stitching {len(clips)} shots")
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(clips)))
    flt = f"{streams}concat=n={len(clips)}:v=1:a=1[v][a]"
    ff._run([*inputs, "-filter_complex", flt, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(story.fps),
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", dst],
            timeout=900)
    return dst


def _safe_rm(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def engine_status():
    """What's actually available right now — surfaced in the UI header."""
    ltx = _ltx()
    se = _storyboard()
    voice = False
    try:
        import edge_tts  # noqa: F401
        voice = True
    except Exception:
        pass
    klein = _klein()
    return {
        "image": "comfyui" if se is not None else "placeholder",
        "clip": "ltx" if (ltx is not None and ltx.available()) else "kenburns",
        "voice": "edge-tts" if voice else "silent",
        "swap": "flux-klein" if (klein is not None and klein.available()) else "none",
    }
