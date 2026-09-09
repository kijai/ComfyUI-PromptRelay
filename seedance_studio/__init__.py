"""seedance_studio — a SeeDance-style video studio wired to LOCAL ComfyUI (LTX 2.3).

A clean single-panel UI (prompt + start image -> clip) over the user's proven local LTX
image-to-video graph, so "Generate" costs nothing, is unlimited and NSFW-ok — no cloud key.
Hosted inside the ComfyUI process via PromptServer routes (see register_routes); the UI is
served at /promptrelay/seedance/ui and drives ComfyUI's own /prompt + /history + /view.
"""

import asyncio
import os
import random
import uuid as _uuid
from urllib.parse import urlencode

from . import engine, history, tasks, reel, sequence, projects, checkpoints

_BASE = os.path.dirname(os.path.abspath(__file__))
_UPLOADS = os.path.join(_BASE, "uploads")

# A few motion ideas for the "Surprise me" button.
_SURPRISE = [
    "gentle breeze moves her hair, she smiles softly and blinks, subtle natural motion",
    "slow cinematic push as golden-hour light shifts across the scene",
    "she turns her head slightly and looks toward the camera, hair catching the light",
    "steam rises from the cup, soft morning light, calm ambient motion",
    "waves roll in behind her, wind in her hair, warm sunset glow",
]


def _video_url(info):
    q = urlencode({"filename": info["filename"],
                   "subfolder": info.get("subfolder", ""),
                   "type": "output"})
    return f"/view?{q}"


# Resolution presets -> (width, height) for text-to-video. i2v ignores these (size follows
# the start image). Kept conservative for 16 GB VRAM.
_RES = {
    "draft": (640, 384),
    "720p": (1280, 720), "720p_landscape": (1280, 720),
    "720p_vertical": (720, 1280), "square": (768, 768),
    "1080p": (1920, 1080), "1080p_vertical": (1080, 1920),
}

# Quality tier -> GGUF unet. "fast" = Q4 (graph default), "hq" = Q5, "max" = Q6 (sharpest;
# exceeds 16GB VRAM so it streams from system RAM — slower but this rig has 32GB).
_MODEL = {
    "max": "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
    "hq": "LTX-2.3-22B-distilled-1.1-Q5_K_M.gguf",
    "standard": "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    # Fast tier = Q3_K_M (14.7GB) — small enough to (mostly) fit 16GB VRAM so long shots
    # avoid the heavy RAM offload the 17GB Q4 forces. Falls back to Q4 if Q3 isn't present.
    "fast": "LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf",
}


def _resolve_model(quality):
    """Map quality tier -> GGUF filename, falling back down the tiers if the file isn't
    on disk yet (e.g. Q6 still downloading)."""
    order = {"max": ["max", "hq", "standard"], "hq": ["hq", "standard"],
             "standard": ["standard"], "fast": ["fast", "standard"]}
    unet_dir = os.path.join(engine._COMFY_ROOT, "models", "unet")
    for tier in order.get(quality or "", []):
        name = _MODEL.get(tier)
        if name and os.path.isfile(os.path.join(unet_dir, name)):
            return name
    return None


def _run_task(tid, mode, image_path, prompt, negative, seed, fixed_camera,
              width, height, duration, model, fit):
    """Blocking worker (runs in a thread): drive LTX, then update the task."""
    tasks.update(tid, status="RUNNING")
    try:
        if mode == "t2v":
            info = engine.generate_t2v(prompt, negative=negative, seed=seed,
                                       width=width, height=height, duration=duration,
                                       model=model)
        else:
            info = engine.generate_i2v(image_path, prompt, negative=negative, seed=seed,
                                       fixed_camera=fixed_camera, width=width,
                                       height=height, duration=duration, model=model,
                                       fit=fit)
        tasks.update(tid, status="SUCCEEDED", video_url=_video_url(info),
                     seed=info.get("seed"))
        history.record(mode, prompt, _video_url(info), seed=info.get("seed"),
                       params={"resolution": width and height and f"{width}x{height}",
                               "duration": duration})
    except Exception as e:  # surface the failure to the UI rather than dying silently
        tasks.update(tid, status="FAILED", error=str(e)[:500])


def _run_flf(tid, first_path, last_path, prompt, negative, seed, first_strength,
             last_strength, width, height, duration, model):
    """Blocking worker: drive the first/last-frame render, then update the task."""
    tasks.update(tid, status="RUNNING")
    try:
        info = engine.generate_flf(first_path, last_path, prompt, negative=negative,
                                   seed=seed, first_strength=first_strength,
                                   last_strength=last_strength, width=width, height=height,
                                   duration=duration, model=model)
        tasks.update(tid, status="SUCCEEDED", video_url=_video_url(info),
                     seed=info.get("seed"))
        history.record("flf", prompt, _video_url(info), seed=info.get("seed"),
                       params={"duration": duration})
    except Exception as e:
        tasks.update(tid, status="FAILED", error=str(e)[:500])


def _expand_refs(prompt):
    """Turn Higgsfield-style @image1/@image2 tokens into wording Flux2 Klein understands."""
    p = prompt or ""
    for tok, repl in (("@image1", "the first reference image"),
                      ("@image2", "the second reference image"),
                      ("@image", "the first reference image")):
        p = p.replace(tok, repl)
    return p


def _run_elements(tid, element_paths, prompt, negative, seed,
                  width, height, duration, model):
    """Blocking worker: Flux2 Klein composes the elements into a still, then LTX i2v
    animates it. Two phases surfaced on the task."""
    tasks.update(tid, status="RUNNING", phase="composing still (Flux2 Klein)")
    try:
        composed = engine.compose_elements(element_paths, _expand_refs(prompt), seed=seed)
        still_path = _output_path(composed)
        still_q = urlencode({"filename": composed["filename"],
                             "subfolder": composed.get("subfolder", ""), "type": "output"})
        tasks.update(tid, phase="animating (LTX i2v)", still_url=f"/view?{still_q}")
        info = engine.generate_i2v(still_path, _expand_refs(prompt), negative=negative,
                                   seed=seed, width=width, height=height,
                                   duration=duration, model=model)
        tasks.update(tid, status="SUCCEEDED", phase="done", video_url=_video_url(info),
                     seed=info.get("seed"))
        history.record("elements", prompt, _video_url(info), seed=info.get("seed"),
                       params={"elements": len(element_paths), "duration": duration,
                               "still_url": f"/view?{still_q}"})
    except Exception as e:
        tasks.update(tid, status="FAILED", error=str(e)[:500])


def _run_motion(tid, image_path, video_path, prompt, negative, seed,
                width, height, frame_cap):
    """Blocking worker: Wan2.2 Animate — the character performs the driving video's motion."""
    tasks.update(tid, status="RUNNING", phase="pose + mask + animate (Wan 2.2)")
    try:
        info = engine.generate_motion(image_path, video_path, prompt=prompt,
                                      negative=negative, seed=seed, width=width,
                                      height=height, frame_cap=frame_cap)
        tasks.update(tid, status="SUCCEEDED", phase="done", video_url=_video_url(info),
                     seed=info.get("seed"))
        history.record("motion", prompt or "(motion control)", _video_url(info),
                       seed=info.get("seed"),
                       params={"resolution": f"{width}x{height}" if width else None,
                               "frame_cap": frame_cap})
    except Exception as e:
        tasks.update(tid, status="FAILED", error=str(e)[:500])


def _parse_duration(v):
    try:
        d = str(v or "").rstrip("s")
        return int(d) if d.isdigit() else None
    except (TypeError, ValueError):
        return None


def _output_path(info):
    """Absolute path of a ComfyUI output described by generate_*'s return dict."""
    return os.path.join(engine._OUTPUT_DIR, info.get("subfolder", ""), info["filename"])


def _run_reel(tid, shots, transition, xfade):
    """Render each shot in order, then stitch into one reel. Updates progress on the task."""
    tasks.update(tid, status="RUNNING", phase="starting",
                 shots_total=len(shots), shots_done=0)
    paths, first_wh = [], None
    try:
        for i, s in enumerate(shots):
            tasks.update(tid, phase=f"rendering shot {i + 1}/{len(shots)}", current_shot=i + 1)
            mode = "i2v" if s.get("mode") == "i2v" else "t2v"
            width, height = _RES.get(s.get("resolution"), (None, None))
            duration = _parse_duration(s.get("duration"))
            model = _resolve_model(s.get("quality"))
            prompt = (s.get("prompt") or "").strip()
            negative = (s.get("negative") or "").strip()
            if mode == "i2v":
                token = os.path.basename(s.get("image_token") or "")
                img = os.path.join(_UPLOADS, token)
                if not token or not os.path.exists(img):
                    raise RuntimeError(f"shot {i + 1}: image missing")
                info = engine.generate_i2v(img, prompt, negative=negative, width=width,
                                           height=height, duration=duration, model=model,
                                           fit=bool(s.get("fit")))
            else:
                if not prompt:
                    raise RuntimeError(f"shot {i + 1}: prompt required")
                info = engine.generate_t2v(prompt, negative=negative, width=width,
                                           height=height, duration=duration, model=model)
            paths.append(_output_path(info))
            tasks.update(tid, shots_done=i + 1)
            if first_wh is None:
                first_wh = (width or 1280, height or 720)

        tasks.update(tid, phase="stitching")
        w, h = first_wh or (1280, 720)
        out_name = f"reel_{tid}.mp4"
        out_path = os.path.join(engine._OUTPUT_DIR, "video", out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        reel.stitch(paths, out_path, transition=transition, xfade=xfade, width=w, height=h)
        q = urlencode({"filename": out_name, "subfolder": "video", "type": "output"})
        tasks.update(tid, status="SUCCEEDED", phase="done", video_url=f"/view?{q}")
        t = tasks.get(tid)
        history.record("reel", (t or {}).get("prompt"), f"/view?{q}",
                       params={"shots": len(shots), "transition": transition})
    except Exception as e:
        tasks.update(tid, status="FAILED", error=str(e)[:500])


def register_routes():
    """Mount the SeeDance studio API on ComfyUI's PromptServer (call at node load)."""
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes
    os.makedirs(_UPLOADS, exist_ok=True)

    _web = os.path.join(os.path.dirname(_BASE), "web")
    _ui_html = os.path.join(_web, "seedance.html")
    _seq_html = os.path.join(_web, "sequence.html")

    @routes.get("/promptrelay/seedance")
    async def seedance_root(request):
        raise web.HTTPFound("/promptrelay/seedance/ui")

    @routes.get("/promptrelay/seedance/ui")
    async def seedance_ui(request):
        return web.FileResponse(_ui_html)

    @routes.get("/promptrelay/sequence")
    async def sequence_root(request):
        raise web.HTTPFound("/promptrelay/sequence/ui")

    @routes.get("/promptrelay/sequence/ui")
    async def sequence_ui(request):
        return web.FileResponse(_seq_html)

    @routes.get("/promptrelay/seedance/api/status")
    async def seedance_status(request):
        return web.json_response({"engine": "LTX 2.3 (local ComfyUI)",
                                  "available": engine.available(),
                                  "i2v": engine.available(),
                                  "t2v": engine.available_t2v(),
                                  "flf": engine.available_flf(),
                                  "elements": engine.available_elements(),
                                  "motion": engine.available_motion()})

    @routes.post("/promptrelay/seedance/api/upload")
    async def seedance_upload(request):
        reader = await request.multipart()
        async for field in reader:
            if field.name == "file":
                orig = field.filename or "upload.png"
                ext = os.path.splitext(orig)[1].lower() or ".png"
                # Images for frames/elements; videos for the Motion tab's driving clip.
                if ext not in (".png", ".jpg", ".jpeg", ".webp",
                               ".mp4", ".mov", ".webm", ".mkv", ".avi"):
                    ext = ".png"
                token = f"{_uuid.uuid4().hex[:10]}{ext}"
                dst = os.path.join(_UPLOADS, token)
                with open(dst, "wb") as fh:
                    fh.write(await field.read())
                return web.json_response({"token": token})
        return web.json_response({"error": "no file received"}, status=400)

    @routes.get("/promptrelay/seedance/api/upload/{token}")
    async def seedance_upload_preview(request):
        path = os.path.join(_UPLOADS, os.path.basename(request.match_info["token"]))
        if not os.path.exists(path):
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(path)

    @routes.post("/promptrelay/seedance/api/generate")
    async def seedance_generate(request):
        data = await request.json()
        prompt = (data.get("prompt") or "").strip()
        negative = (data.get("negative") or "").strip()
        mode = "t2v" if data.get("mode") == "t2v" else "i2v"

        image_path = None
        token = data.get("image_token")
        if mode == "i2v":
            if not token:
                return web.json_response(
                    {"error": "add a start image, or switch to Text-to-Video"}, status=400)
            image_path = os.path.join(_UPLOADS, os.path.basename(token))
            if not os.path.exists(image_path):
                return web.json_response({"error": "start image not found (re-upload)"},
                                         status=400)
            if not engine.available():
                return web.json_response(
                    {"error": "LTX i2v not available — is ComfyUI up with the LTX GGUF?"},
                    status=503)
        else:  # t2v
            if not prompt:
                return web.json_response({"error": "a prompt is required for text-to-video"},
                                         status=400)
            if not engine.available_t2v():
                return web.json_response(
                    {"error": "Text-to-Video isn't wired yet — the t2v graph still needs "
                              "to be captured from ComfyUI."}, status=503)

        seed = data.get("seed")
        try:
            seed = int(seed) if seed not in (None, "", "random") else random.randrange(2 ** 53)
        except (TypeError, ValueError):
            seed = random.randrange(2 ** 53)
        fixed_camera = bool(data.get("fixed_camera"))

        # Map resolution preset -> width/height, duration -> seconds, quality -> model.
        # (i2v "follow" isn't in _RES -> (None,None), so the engine sizes from the image.)
        width, height = _RES.get(data.get("resolution"), (None, None))
        try:
            d = str(data.get("duration") or "").rstrip("s")
            duration = int(d) if d.isdigit() else None
        except (TypeError, ValueError):
            duration = None
        model = _resolve_model(data.get("quality"))

        fit = bool(data.get("fit")) and mode == "i2v"

        params = {"mode": mode, "seed": seed, "fixed_camera": fixed_camera,
                  "negative": negative, "resolution": data.get("resolution"),
                  "duration": data.get("duration"), "quality": data.get("quality"),
                  "fit": fit, "width": width, "height": height, "image_token": token}
        tid = tasks.create(prompt, params)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_task, tid, mode, image_path, prompt,
                             negative, seed, fixed_camera, width, height, duration,
                             model, fit)
        return web.json_response({"id": tid, "status": "QUEUED", "mode": mode})

    @routes.get("/promptrelay/seedance/api/tasks")
    async def seedance_tasks(request):
        return web.json_response({"tasks": tasks.listing()})

    @routes.get("/promptrelay/seedance/api/history")
    async def seedance_history(request):
        return web.json_response({"items": history.listing()})

    @routes.post("/promptrelay/seedance/api/elements")
    async def seedance_elements(request):
        data = await request.json()
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return web.json_response(
                {"error": "a prompt is required — describe the scene and mention "
                          "@image1 / @image2"}, status=400)
        if not engine.available_elements():
            return web.json_response(
                {"error": "Elements needs the Flux2 Klein model — not found."}, status=503)
        if not engine.available():
            return web.json_response({"error": "LTX i2v not available."}, status=503)

        toks = [os.path.basename(t) for t in (data.get("element_tokens") or []) if t][:2]
        paths = [os.path.join(_UPLOADS, t) for t in toks]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            return web.json_response({"error": "attach at least one element image"},
                                     status=400)

        seed = data.get("seed")
        try:
            seed = int(seed) if seed not in (None, "", "random") else random.randrange(2 ** 53)
        except (TypeError, ValueError):
            seed = random.randrange(2 ** 53)
        width, height = _RES.get(data.get("resolution"), (None, None))
        duration = _parse_duration(data.get("duration"))
        model = _resolve_model(data.get("quality"))
        negative = (data.get("negative") or "").strip()

        params = {"mode": "elements", "seed": seed, "elements": len(paths),
                  "resolution": data.get("resolution"), "duration": data.get("duration"),
                  "quality": data.get("quality")}
        tid = tasks.create(prompt, params)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_elements, tid, paths, prompt, negative, seed,
                             width, height, duration, model)
        return web.json_response({"id": tid, "status": "QUEUED", "mode": "elements"})

    @routes.post("/promptrelay/seedance/api/motion")
    async def seedance_motion(request):
        data = await request.json()
        if not engine.available_motion():
            return web.json_response(
                {"error": "Motion control needs the Wan2.2 Animate model — not found."},
                status=503)
        img_t = os.path.basename(data.get("image_token") or "")
        vid_t = os.path.basename(data.get("video_token") or "")
        img_p = os.path.join(_UPLOADS, img_t)
        vid_p = os.path.join(_UPLOADS, vid_t)
        if not (img_t and os.path.exists(img_p)):
            return web.json_response({"error": "add a character image"}, status=400)
        if not (vid_t and os.path.exists(vid_p)):
            return web.json_response({"error": "add a driving video"}, status=400)

        prompt = (data.get("prompt") or "").strip()
        negative = (data.get("negative") or "").strip()
        seed = data.get("seed")
        try:
            seed = int(seed) if seed not in (None, "", "random") else random.randrange(2 ** 53)
        except (TypeError, ValueError):
            seed = random.randrange(2 ** 53)
        # Motion output size: modest WxH presets (the graph center-crops video + character
        # to this). Kept small-ish — Wan 14B on 16GB.
        res = str(data.get("resolution") or "640x640")
        try:
            width, height = (int(v) for v in res.lower().split("x"))
        except ValueError:
            width, height = 640, 640
        try:
            frame_cap = max(0, int(data.get("frame_cap") or 0))
        except (TypeError, ValueError):
            frame_cap = 0

        params = {"mode": "motion", "seed": seed, "resolution": f"{width}x{height}",
                  "frame_cap": frame_cap}
        tid = tasks.create(prompt or "(motion control)", params)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_motion, tid, img_p, vid_p, prompt, negative,
                             seed, width, height, frame_cap)
        return web.json_response({"id": tid, "status": "QUEUED", "mode": "motion"})

    @routes.post("/promptrelay/seedance/api/reel")
    async def seedance_reel(request):
        data = await request.json()
        shots = data.get("shots") or []
        if not (2 <= len(shots) <= 10):
            return web.json_response({"error": "add between 2 and 10 shots"}, status=400)
        if not engine.available():
            return web.json_response({"error": "LTX not available (is ComfyUI up?)"},
                                     status=503)
        transition = "cut" if data.get("transition") == "cut" else "crossfade"
        try:
            xfade = min(1.5, max(0.1, float(data.get("xfade", 0.5))))
        except (TypeError, ValueError):
            xfade = 0.5
        summary = " / ".join((s.get("prompt") or "shot").strip()[:22] for s in shots[:3])
        tid = tasks.create(f"REEL · {len(shots)} shots · {summary}",
                           {"kind": "reel", "shots": len(shots), "transition": transition})
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_reel, tid, shots, transition, xfade)
        return web.json_response({"id": tid, "status": "QUEUED"})

    @routes.post("/promptrelay/seedance/api/flf")
    async def seedance_flf(request):
        data = await request.json()
        if not engine.available_flf():
            return web.json_response({"error": "First→Last Frame isn't wired yet."},
                                     status=503)
        first_t = os.path.basename(data.get("first_token") or "")
        last_t = os.path.basename(data.get("last_token") or "")
        first_p = os.path.join(_UPLOADS, first_t)
        last_p = os.path.join(_UPLOADS, last_t)
        if not (first_t and os.path.exists(first_p)):
            return web.json_response({"error": "add a start (first) image"}, status=400)
        if not (last_t and os.path.exists(last_p)):
            return web.json_response({"error": "add an end (last) image"}, status=400)

        prompt = (data.get("prompt") or "").strip()
        negative = (data.get("negative") or "").strip()
        seed = data.get("seed")
        try:
            seed = int(seed) if seed not in (None, "", "random") else random.randrange(2 ** 53)
        except (TypeError, ValueError):
            seed = random.randrange(2 ** 53)

        def _f(v, d):
            try:
                return min(1.0, max(0.0, float(v)))
            except (TypeError, ValueError):
                return d
        first_strength = _f(data.get("first_strength"), 0.5)
        last_strength = _f(data.get("last_strength"), 1.0)
        width, height = _RES.get(data.get("resolution"), (None, None))
        duration = _parse_duration(data.get("duration"))
        model = _resolve_model(data.get("quality"))

        params = {"mode": "flf", "seed": seed, "first_strength": first_strength,
                  "last_strength": last_strength, "resolution": data.get("resolution"),
                  "duration": data.get("duration"), "quality": data.get("quality")}
        tid = tasks.create(prompt or "(first→last)", params)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_flf, tid, first_p, last_p, prompt, negative, seed,
                             first_strength, last_strength, width, height, duration, model)
        return web.json_response({"id": tid, "status": "QUEUED", "mode": "flf"})

    @routes.get("/promptrelay/seedance/api/task/{id}")
    async def seedance_task(request):
        t = tasks.get(request.match_info["id"])
        if not t:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(t)

    # ── Cinematic Sequence (Seedance-2.0-style multi-shot composer) ──────────────
    # 4K is a skeleton feature: no local upscaler is wired, so it renders at 1080p and the
    # result is tagged "4k-upscale" pending (see sequence.run).
    _SEQ_RES = {"720p": (1280, 720), "720p_vertical": (720, 1280),
                "1080p": (1920, 1080), "1080p_vertical": (1080, 1920),
                "square": (1024, 1024), "4k": (1920, 1080)}

    @routes.get("/promptrelay/seedance/api/sequence/grammar")
    async def sequence_grammar(request):
        g = sequence.grammar()
        g["status"] = sequence.status()
        return web.json_response(g)

    @routes.post("/promptrelay/seedance/api/sequence/compile")
    async def sequence_compile(request):
        """Live preview: compile style + shots into the Seedance-format script. No render."""
        data = await request.json()
        result = sequence.compile_sequence(data.get("style"), data.get("shots") or [])
        return web.json_response(result)

    @routes.post("/promptrelay/seedance/api/sequence")
    async def sequence_generate(request):
        data = await request.json()
        style = (data.get("style") or "").strip()
        shots = data.get("shots") or []
        if not (1 <= len(shots) <= 10):
            return web.json_response({"error": "add between 1 and 10 shots"}, status=400)
        if not engine.available_t2v() and not engine.available():
            return web.json_response(
                {"error": "LTX not available — is ComfyUI up with the LTX GGUF?"},
                status=503)

        # Every shot needs either an action or a style header to render from.
        for i, s in enumerate(shots, start=1):
            if not (s.get("action") or "").strip() and not style:
                return web.json_response(
                    {"error": f"shot S{i}: add an action (or a global style)"}, status=400)
            # Resolve any i2v start-image token to a path for the worker.
            tok = os.path.basename(s.get("image_token") or "")
            if tok:
                p = os.path.join(_UPLOADS, tok)
                if not os.path.exists(p):
                    return web.json_response(
                        {"error": f"shot S{i}: start image not found (re-upload)"},
                        status=400)
                s["image_path"] = p

        seed = data.get("seed")
        try:
            seed = int(seed) if seed not in (None, "", "random") else random.randrange(2 ** 53)
        except (TypeError, ValueError):
            seed = random.randrange(2 ** 53)

        res_label = data.get("resolution") if data.get("resolution") in _SEQ_RES else "1080p"
        width, height = _SEQ_RES[res_label]
        duration = _parse_duration(data.get("duration"))
        model = _resolve_model(data.get("quality"))
        audio = bool(data.get("audio"))
        transition = "cut" if data.get("transition") == "cut" else "crossfade"
        try:
            xfade = min(1.5, max(0.1, float(data.get("xfade", 0.5))))
        except (TypeError, ValueError):
            xfade = 0.5

        script = sequence.compile_sequence(style, shots)["script"]
        params = {"mode": "sequence", "seed": seed, "shots": len(shots),
                  "resolution": res_label, "duration": data.get("duration"),
                  "quality": data.get("quality"), "audio": audio}
        tid = tasks.create(f"SEQUENCE · {len(shots)} shots · {(style or script)[:40]}",
                           params)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, sequence.run, tid, style, shots, seed, width, height,
                             duration, model, res_label, audio, transition, xfade,
                             engine._OUTPUT_DIR)
        return web.json_response({"id": tid, "status": "QUEUED", "mode": "sequence"})

    # ── Projects / templates (save & reuse a sequence) ──────────────────────────
    @routes.get("/promptrelay/seedance/api/projects")
    async def projects_list(request):
        return web.json_response({"items": projects.listing()})

    @routes.post("/promptrelay/seedance/api/projects")
    async def projects_save(request):
        data = await request.json()
        name = (data.get("name") or "").strip()
        if not name:
            return web.json_response({"error": "give the project a name"}, status=400)
        p = projects.save(name, data.get("style") or "", data.get("shots") or [],
                          data.get("settings") or {}, pid=data.get("id"))
        return web.json_response({"project": p})

    @routes.get("/promptrelay/seedance/api/projects/{id}")
    async def projects_get(request):
        p = projects.get(request.match_info["id"])
        if not p:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"project": p})

    @routes.delete("/promptrelay/seedance/api/projects/{id}")
    async def projects_delete(request):
        ok = projects.delete(request.match_info["id"])
        return web.json_response({"deleted": ok})

    # ── Resume an interrupted sequence from its checkpoint ───────────────────────
    @routes.get("/promptrelay/seedance/api/sequence/resumable")
    async def sequence_resumable(request):
        return web.json_response({"items": checkpoints.listing()})

    @routes.post("/promptrelay/seedance/api/sequence/resume")
    async def sequence_resume(request):
        data = await request.json()
        tid = data.get("id")
        c = checkpoints.load(tid) if tid else None
        if not c:
            return web.json_response({"error": "no resumable checkpoint for that id"},
                                     status=404)
        tasks.reopen(tid, f"RESUME · {c.get('label', 'sequence')}",
                     {"mode": "sequence", "seed": c.get("seed")})
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, sequence.resume, tid)
        return web.json_response({"id": tid, "status": "QUEUED", "resumed": True})

    @routes.get("/promptrelay/seedance/api/surprise")
    async def seedance_surprise(request):
        return web.json_response({"prompt": random.choice(_SURPRISE)})

    print("[PromptRelay] SeeDance studio API routes registered at /promptrelay/seedance/*")
