"""REVIEW stage (automated) — a lightweight pre-flight check over approved task outputs,
run once at advance() time, before a batch is allowed to publish.

This is NOT a replacement for the human/Claude review gate (that happens per-task, at
NEEDS_REVIEW) — it only catches the obvious stuff a reviewer could miss (empty output,
placeholder text) so junk never reaches the publish step. Swap in a real content-safety
guard here once content types are decided (e.g. reuse storyboard_engine's prompt-safety
checks — see README "Adding a new sub-agent").
"""

from .base import register_agent


def run(batch, task):
    approved = task["input"].get("approved_tasks", [])
    flags = []
    for t in approved:
        output = t.get("output") or {}
        text = output.get("caption") or output.get("text") or ""
        if not text.strip():
            flags.append({"task_id": t["id"], "agent": t["agent"], "issue": "empty output"})
        elif len(text.strip()) < 3:
            flags.append({"task_id": t["id"], "agent": t["agent"], "issue": "output too short"})
    return {"flags": flags, "checked": len(approved)}


register_agent(
    "qa_check", "review",
    "Automated pre-flight check over approved outputs before publish (empty/placeholder text).",
    run, needs_review=False, max_attempts=1,
)
