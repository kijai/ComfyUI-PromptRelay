"""PUBLISH stage (stub) — no real posting destination is wired up yet; Lena's platforms
and content types are still TBD (per the brief for this scaffold).

Logs what WOULD be published so the manager's handoff chain (intake -> plan -> generate
-> review -> advance -> archive) is complete and testable end to end today. Replace with
a real uploader later — see channel_studio/publisher.py for the pattern this should
follow once a platform is chosen (typed PublishError with setup instructions, credentials
in a gitignored .secrets/ folder).
"""

import os

from .base import register_agent

MODE = os.environ.get("LENA_PUBLISH_MODE", "log")


def run(batch, task):
    approved = task["input"].get("approved_tasks", [])
    dry_run = task["input"].get("dry_run", True)
    payload = [{"task_id": t["id"], "agent": t["agent"], "output": t.get("output")}
               for t in approved]
    if MODE == "log" or dry_run:
        return {"mode": "log", "dry_run": dry_run, "would_publish": payload}
    # Real publishing hooks go here once a destination is chosen.
    return {"mode": MODE, "dry_run": dry_run, "published": payload}


register_agent(
    "publish_stub", "publish",
    "Stand-in for the real publish destination (platform TBD). Logs what would ship; "
    "replace with a real uploader when a platform is chosen.",
    run, needs_review=False, max_attempts=1,
)
