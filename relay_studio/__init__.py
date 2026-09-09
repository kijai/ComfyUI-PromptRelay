"""relay_studio — the drama.land-style agent brain for PromptRelay.

A super-agent (Dolly) reads a structured user-inputs artifact + a request, decomposes
it into a video pipeline, and drives one tool per stage (voice, BGM, lipsync, SFX, mix),
streaming thinking / task / media events. Hosted in the ComfyUI process via PromptServer
routes (see register_routes), and runnable headless via `python -m relay_studio.run`.
"""

import asyncio
import json
import os
import uuid as _uuid

from . import agent
from .session import INPUT_FIELDS, Project, get, list_projects, register


def _state(project):
    return project.to_dict()


def register_routes():
    """Mount the studio HTTP API on ComfyUI's PromptServer (call at node load)."""
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes
    from . import llm  # local import: requests only, safe here

    _web = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    _ui_html = os.path.join(_web, "studio.html")
    _home_html = os.path.join(_web, "home.html")

    @routes.get("/promptrelay/studio/ui")
    async def studio_ui(request):
        return web.FileResponse(_ui_html)

    @routes.get("/promptrelay/studio")
    async def studio_root(request):
        raise web.HTTPFound("/promptrelay/studio/home")

    @routes.get("/promptrelay/studio/home")
    async def studio_home(request):
        return web.FileResponse(_home_html)

    @routes.get("/promptrelay/studio/projects")
    async def studio_projects(request):
        return web.json_response({"projects": list_projects()})

    @routes.get("/promptrelay/studio/models")
    async def studio_models(request):
        return web.json_response({"models": llm.available_models(),
                                  "default": llm.DEFAULT_MODEL})

    @routes.get("/promptrelay/studio/voices")
    async def studio_voices(request):
        from . import voices
        return web.json_response({"voices": voices.list_voices()})

    @routes.post("/promptrelay/studio/project")
    async def studio_create(request):
        data = await request.json()
        project = register(Project(inputs=data.get("inputs") or {},
                                   request=data.get("request", "")))
        project.save()
        return web.json_response({"id": project.id, "state": _state(project)})

    @routes.get("/promptrelay/studio/project/{id}")
    async def studio_get(request):
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(_state(project))

    @routes.post("/promptrelay/studio/inputs/{id}")
    async def studio_inputs(request):
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        for k, v in (data.get("inputs") or data).items():
            if k in INPUT_FIELDS:
                project.inputs[k] = v
        project.save()
        return web.json_response(_state(project))

    @routes.get("/promptrelay/studio/media/{id}/{file}")
    async def studio_media(request):
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)
        path = project.media_path(os.path.basename(request.match_info["file"]))
        if not os.path.exists(path):
            return web.json_response({"error": "no such file"}, status=404)
        return web.FileResponse(path)

    @routes.post("/promptrelay/studio/media_delete/{id}")
    async def studio_media_delete(request):
        """Remove a media item from the library (and any artifact output that points
        at it). Deletes the underlying file unless another media item still uses it."""
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        mid = data.get("media_id")
        item = next((m for m in project.media if m["id"] == mid), None)
        if not item:
            return web.json_response({"error": "no such media"}, status=404)
        project.media = [m for m in project.media if m["id"] != mid]
        # Drop any artifact outputs referencing this media item.
        for field in [k for k, v in project.outputs.items() if v.get("media_id") == mid]:
            project.outputs.pop(field, None)
        # Delete the file only if no remaining media item references the same file.
        still_used = any(m["file"] == item["file"] for m in project.media)
        if not still_used:
            try:
                p = project.media_path(item["file"])
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        project.save()
        return web.json_response({"ok": True, "state": _state(project)})

    @routes.post("/promptrelay/studio/upload/{id}")
    async def studio_upload(request):
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)
        reader = await request.multipart()
        role = "tts"
        saved_path = None
        saved_orig = "upload.bin"
        async for field in reader:
            if field.name == "role":
                role = (await field.read(decode=True)).decode().strip()
            elif field.name == "file":
                saved_orig = field.filename or "upload.bin"
                ext = os.path.splitext(saved_orig)[1].lower() or ".bin"
                fname = f"upload_{role}_{_uuid.uuid4().hex[:6]}{ext}"
                dst = project.media_path(fname)
                data = await field.read()
                with open(dst, "wb") as fh:
                    fh.write(data)
                saved_path = dst
        if not saved_path:
            return web.json_response({"error": "no file received"}, status=400)
        labels = {"tts": "Voice Over", "bgm": "Background Music",
                  "photo": "Cat Photo", "sfx": "SFX"}
        kind = "image" if role == "photo" else "audio"
        item = project.add_media(kind, saved_path, via="upload",
                                 label=labels.get(role, saved_orig))
        if role == "tts":
            project.write_artifact("TTS_AUDIO_URL", media_id=item["id"])
            project.set_task("voice", "done")
        elif role == "bgm":
            # Convert to WAV so trim_music can slice intro/outro from music_master.wav
            from . import ffmpeg as _ff
            master = project.media_path("music_master.wav")
            _ff._run(["-i", saved_path, "-ar", str(_ff.AR), "-ac", str(_ff.AC), master])
            bgm_item = project.add_media("audio", master, via="upload",
                                         label="BGM master", role="master")
            project.write_artifact("BGM_URL", media_id=bgm_item["id"])
            project.set_task("bgm", "done")
        elif role == "photo":
            url = f"/promptrelay/studio/media/{project.id}/{os.path.basename(saved_path)}"
            project.inputs["cat_photo_url"] = url
        project.save()
        return web.json_response({"ok": True, "media_id": item["id"],
                                  "file": os.path.basename(saved_path),
                                  "state": _state(project)})

    @routes.post("/promptrelay/studio/regenerate/{id}")
    async def studio_regenerate(request):
        """Rebuild the talking video + final mix using the CURRENT inputs
        (cat photo, voice, music). Streams the same SSE events as chat."""
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)

        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def listener(ev):
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        project.on_event(listener)

        def work():
            project.emit("status", text="Re-rendering video with the current cat photo…")
            # Drop the existing videos + their artifact entries so the pipeline rebuilds them.
            project.media = [m for m in project.media if m.get("kind") != "video"]
            for f in ("TALKING_VIDEO_URL", "FINAL_VIDEO_URL"):
                project.outputs.pop(f, None)
            for tid in ("talking", "mix"):
                project.set_task(tid, "pending")
            agent.ensure_complete(project)
            project.emit("assistant", text="Done — new video rendered with your selected cat photo. 🎬")
            project.add_message("assistant", "Re-rendered the video with the new cat photo.")
            project.save()

        fut = loop.run_in_executor(None, work)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                    await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
                except asyncio.TimeoutError:
                    if fut.done() and queue.empty():
                        break
            await resp.write(b"data: {\"kind\": \"done\"}\n\n")
        finally:
            try:
                project._listeners.remove(listener)
            except ValueError:
                pass
        await resp.write_eof()
        return resp

    @routes.post("/promptrelay/studio/chat/{id}")
    async def studio_chat(request):
        project = get(request.match_info["id"])
        if not project:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        message = data.get("message", "")
        mode = data.get("mode", "agentic")

        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def listener(ev):
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        project.on_event(listener)

        def classic():
            # Classic mode: deterministic pipeline, no LLM (mirrors drama.land's Classic).
            project.add_message("user", message)
            project.emit("status", text="Classic mode — running the pipeline directly…")
            agent.ensure_complete(project, message)
            project.add_message("assistant", "Done — your video is ready. 🎬")
            project.emit("assistant", text="Done — your video is ready. 🎬")
            project.save()

        target = classic if mode == "classic" else (lambda: agent.run(project, message))
        fut = loop.run_in_executor(None, target)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                    await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
                except asyncio.TimeoutError:
                    if fut.done() and queue.empty():
                        break
            await resp.write(b"data: {\"kind\": \"done\"}\n\n")
        finally:
            try:
                project._listeners.remove(listener)
            except ValueError:
                pass
        await resp.write_eof()
        return resp

    print("[PromptRelay] Studio (Dolly) API routes registered at /promptrelay/studio/*")
