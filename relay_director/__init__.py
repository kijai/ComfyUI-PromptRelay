"""Relay Director — an OpenArt-Studio-style "vibe directing" workspace.

A director scopes a free-text vision into Scenes -> Shots, renders each shot
(image -> clip -> optional voice), iterates per-shot with @shot referencing, and
exports a widescreen timeline. Hosted on ComfyUI's PromptServer at
/promptrelay/director/*. Independent of Dolly (relay_studio); it only reuses Dolly's
engines read-only.
"""

import asyncio
import json
import os

from . import director
from .session import ASPECTS, Story, get, list_stories, register


def _stream(handler):
    """Wrap a blocking `handler(story)` that emits events into an SSE response."""
    async def run(story, request):
        from aiohttp import web
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        listener = lambda ev: loop.call_soon_threadsafe(queue.put_nowait, ev)
        story.on_event(listener)
        fut = loop.run_in_executor(None, lambda: handler(story))
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                    await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
                except asyncio.TimeoutError:
                    if fut.done() and queue.empty():
                        break
            await resp.write(b'data: {"kind": "done"}\n\n')
        finally:
            story.off_event(listener)
        await resp.write_eof()
        return resp
    return run


def register_routes():
    """Mount the director HTTP API + UI on ComfyUI's PromptServer."""
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes
    _web = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    _ui_html = os.path.join(_web, "director.html")

    @routes.get("/promptrelay/director")
    async def director_root(request):
        return web.FileResponse(_ui_html)

    @routes.get("/promptrelay/director/ui")
    async def director_ui(request):
        return web.FileResponse(_ui_html)

    @routes.get("/promptrelay/director/stories")
    async def director_stories(request):
        return web.json_response({"stories": list_stories()})

    @routes.get("/promptrelay/director/engines")
    async def director_engines(request):
        return web.json_response(director.engines.engine_status())

    @routes.post("/promptrelay/director/story")
    async def director_create(request):
        data = await request.json()
        story = register(Story(title=data.get("title", "Untitled Story"),
                               vision=data.get("vision", ""),
                               aspect=data.get("aspect", "16:9")))
        story.save()
        return web.json_response({"id": story.id, "state": story.to_dict(),
                                  "aspects": list(ASPECTS.keys())})

    @routes.get("/promptrelay/director/story/{id}")
    async def director_get(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(story.to_dict())

    @routes.post("/promptrelay/director/aspect/{id}")
    async def director_aspect(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        story.set_aspect(data.get("aspect", "16:9"))
        story.save()
        return web.json_response(story.to_dict())

    @routes.post("/promptrelay/director/shot/{id}/update")
    async def director_shot_update(request):
        """Edit a shot's prompt/motion/dialogue/duration in place (no render)."""
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        _, shot = story.get_shot(data.get("shot_id", ""))
        if not shot:
            return web.json_response({"error": "no such shot"}, status=404)
        for k in ("prompt", "motion", "dialogue"):
            if k in data:
                shot[k] = data[k]
        if "duration" in data:
            try:
                shot["duration"] = max(2.0, min(8.0, float(data["duration"])))
            except (TypeError, ValueError):
                pass
        story.save()
        return web.json_response(story.to_dict())

    @routes.get("/promptrelay/director/estimate/{id}")
    async def director_estimate(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(director.estimate(story))

    @routes.get("/promptrelay/director/media/{id}/{file}")
    async def director_media(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        path = story.media_path(os.path.basename(request.match_info["file"]))
        if not os.path.exists(path):
            return web.json_response({"error": "no such file"}, status=404)
        return web.FileResponse(path)

    def _opts(request):
        q = request.rel_url.query
        return {"use_ltx": q.get("ltx", "1") != "0",
                "voice": q.get("voice", "1") != "0"}

    @routes.post("/promptrelay/director/scope/{id}")
    async def director_scope(request):
        """Step 1: develop the vision into a script + cast + world (approval gate)."""
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        vision = data.get("vision", "")
        return await _stream(lambda s: director.develop(s, vision))(story, request)

    @routes.post("/promptrelay/director/approve/{id}")
    async def director_approve(request):
        """Step 2: lock the script -> character sheets + cast-aware storyboard."""
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        return await _stream(lambda s: director.approve(s))(story, request)

    @routes.post("/promptrelay/director/character/{id}/ref")
    async def director_char_ref(request):
        """(Re)generate one cast member's reference sheet."""
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        cid = data.get("char_id", "")
        return await _stream(lambda s: director.generate_reference(s, cid))(story, request)

    @routes.post("/promptrelay/director/character/{id}/update")
    async def director_char_update(request):
        """Edit a cast member's name/role/description in place (no render)."""
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        ch = story.get_character(data.get("char_id", ""))
        if not ch:
            return web.json_response({"error": "no such character"}, status=404)
        for k in ("name", "role", "description"):
            if k in data:
                ch[k] = data[k]
        story.save()
        return web.json_response(story.to_dict())

    @routes.post("/promptrelay/director/world/{id}")
    async def director_world(request):
        """Edit the world style/setting/palette in place."""
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        for k in ("style", "setting", "palette"):
            if k in data:
                story.world[k] = data[k]
        story.save()
        return web.json_response(story.to_dict())

    @routes.post("/promptrelay/director/chat/{id}")
    async def director_chat(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        message = data.get("message", "")
        opts = {"use_ltx": data.get("ltx", False), "voice": data.get("voice", True)}
        return await _stream(lambda s: director.chat(s, message, **opts))(story, request)

    @routes.post("/promptrelay/director/render/{id}")
    async def director_render(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json() if request.can_read_body else {}
        only = data.get("only")
        opts = {"use_ltx": data.get("ltx", False), "voice": data.get("voice", True)}
        return await _stream(lambda s: director.render_all(s, only=only, **opts))(story, request)

    @routes.post("/promptrelay/director/shot/{id}/render")
    async def director_shot_render(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        _, shot = story.get_shot(data.get("shot_id", ""))
        if not shot:
            return web.json_response({"error": "no such shot"}, status=404)
        if data.get("reroll"):
            shot["ver"] += 1
        opts = {"use_ltx": data.get("ltx", False), "voice": data.get("voice", True)}
        return await _stream(lambda s: director.render_shot(s, shot, **opts))(story, request)

    @routes.post("/promptrelay/director/export/{id}")
    async def director_export(request):
        story = get(request.match_info["id"])
        if not story:
            return web.json_response({"error": "not found"}, status=404)
        return await _stream(lambda s: director.export(s))(story, request)

    print("[PromptRelay] Director API routes registered at /promptrelay/director/*")
