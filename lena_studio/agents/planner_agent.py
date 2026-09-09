"""PLAN stage — turns a free-text content brief into a list of concrete tasks for the
registered "generate" sub-agents.

Uses relay_studio's Ollama brain when available; falls back to fanning the brief out to
every registered generate-stage sub-agent, so a batch is never left with nothing to do
even with no LLM running (same degrade-gracefully philosophy as every other studio).
"""

import json
import re

try:
    from relay_studio import llm  # reuse Dolly's Ollama transport, read-only
except Exception:  # pragma: no cover - allows standalone import without relay_studio
    llm = None

from .base import by_stage, register_agent

SYSTEM = """You are a content production planner for an AI influencer's social content.
Given a brief and a list of available sub-agent names, output STRICT JSON (no prose):
{"tasks": [{"agent": "<one of the available names>", "stage": "generate", "input": {"brief": "<...>"}}]}
Only use sub-agent names from the provided list. Keep it to the sub-agents actually
useful for this brief."""


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run(batch, task):
    brief = task["input"].get("brief") or batch.brief
    available = [a["name"] for a in by_stage("generate")]

    tasks = None
    if llm is not None and available:
        try:
            user = f"Brief: {brief}\nAvailable generate sub-agents: {available}"
            resp = llm.chat(messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}])
            parsed = _extract_json(resp.get("content") or "")
            if parsed and parsed.get("tasks"):
                tasks = [t for t in parsed["tasks"] if t.get("agent") in available]
        except Exception:
            tasks = None

    if not tasks:
        # Deterministic fallback: one task per registered generate sub-agent.
        tasks = [{"agent": a, "stage": "generate", "input": {"brief": brief}} for a in available]

    if not tasks:
        # No generate sub-agents registered at all yet — still return something so
        # intake never silently produces an empty batch.
        tasks = [{"agent": "draft_copy", "stage": "generate", "input": {"brief": brief}}]

    return {"tasks": tasks}


register_agent(
    "plan_batch", "plan",
    "Break a content brief into concrete tasks for the registered generate sub-agents.",
    run, needs_review=False, max_attempts=1,
)
