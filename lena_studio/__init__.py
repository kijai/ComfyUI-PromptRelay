"""lena_studio — Claude Co-Work's manager workflow for Lena AI-influencer content.

A **Batch** (one content request) moves through named handoff points: intake -> plan ->
generate (dispatched to registered sub-agents) -> review (approval gate) -> advance
(QA + publish) -> archive. See README.md for the full architecture, the sub-agent
registry, and how to add a new sub-agent. Hosted on ComfyUI's PromptServer at
/promptrelay/lena/*, independent of the other studios (only reuses relay_studio.llm
read-only, same as channel_studio).
"""

from . import agents, manager
from .agents.base import AgentError
from .session import get, list_batches


def register_routes():
    """Mount the Lena manager HTTP API on ComfyUI's PromptServer (call at node load)."""
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    def _err(e, status=400):
        return web.json_response({"error": str(e)}, status=status)

    @routes.get("/promptrelay/lena/agents")
    async def lena_agents(request):
        return web.json_response({"agents": agents.list_agents()})

    @routes.get("/promptrelay/lena/batches")
    async def lena_batches(request):
        return web.json_response({"batches": list_batches()})

    @routes.post("/promptrelay/lena/batch")
    async def lena_create(request):
        """Handoff 1+2: intake a brief and auto-plan it into tasks."""
        data = await request.json()
        brief = data.get("brief", "")
        channel = data.get("channel", "lena")
        try:
            batch = manager.intake(brief, channel=channel)
            manager.plan(batch)
        except Exception as e:
            return _err(e, 500)
        return web.json_response(batch.to_dict())

    @routes.get("/promptrelay/lena/batch/{id}")
    async def lena_get(request):
        batch = get(request.match_info["id"])
        if not batch:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(batch.to_dict())

    @routes.post("/promptrelay/lena/batch/{id}/dispatch")
    async def lena_dispatch(request):
        """Handoff 3: run QUEUED task(s) through their registered sub-agent."""
        batch = get(request.match_info["id"])
        if not batch:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json() if request.can_read_body else {}
        manager.dispatch(batch, task_id=data.get("task_id"))
        return web.json_response(batch.to_dict())

    @routes.post("/promptrelay/lena/batch/{id}/review")
    async def lena_review(request):
        """Handoff 4: approve / reject / regenerate a NEEDS_REVIEW task."""
        batch = get(request.match_info["id"])
        if not batch:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        try:
            manager.review(batch, data["task_id"], data["decision"], data.get("notes", ""))
        except (AgentError, KeyError) as e:
            return _err(e)
        return web.json_response(batch.to_dict())

    @routes.post("/promptrelay/lena/batch/{id}/advance")
    async def lena_advance(request):
        """Handoff 5: QA + publish once every task has cleared review, then archive."""
        batch = get(request.match_info["id"])
        if not batch:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json() if request.can_read_body else {}
        try:
            manager.advance(batch, dry_run=data.get("dry_run", True))
        except AgentError as e:
            return _err(e)
        return web.json_response(batch.to_dict())

    @routes.post("/promptrelay/lena/batch/{id}/retry")
    async def lena_retry(request):
        """Manual override: force a FAILED task back into the queue."""
        batch = get(request.match_info["id"])
        if not batch:
            return web.json_response({"error": "not found"}, status=404)
        data = await request.json()
        try:
            manager.retry(batch, data["task_id"])
        except (AgentError, KeyError) as e:
            return _err(e)
        return web.json_response(batch.to_dict())

    print("[PromptRelay] Lena manager API routes registered at /promptrelay/lena/*")
