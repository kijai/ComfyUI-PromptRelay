"""The Lena manager — the orchestration Claude Co-Work drives as the manager overseeing
Lena's content sub-agents.

A **Batch** (one content request) moves through named handoff points:

    intake -> plan -> generate (dispatched, per task, to a registered sub-agent)
           -> review (human/Claude approval gate, per task)
           -> advance (QA + publish, once nothing is pending) -> archive

Every transition is persisted (`Batch.save()`) so a crash/restart can resume from disk,
and every sub-agent call is wrapped so a failure marks/retries that ONE task instead of
crashing the batch. See README.md for the full architecture, the sub-agent registry, and
how to add a new sub-agent.
"""

import json
import os
import threading
import time

from . import agents
from .agents.base import AgentError
from .session import Batch, new_task, register

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "history.jsonl")

# One lock per batch id — dispatch()/advance() hold it so two callers can't race the
# same batch's GPU/backend calls. Mirrors relay_director's per-story render lock.
_locks = {}
_locks_guard = threading.Lock()


def _lock_for(bid):
    with _locks_guard:
        if bid not in _locks:
            _locks[bid] = threading.Lock()
        return _locks[bid]


def load_character(name="lena"):
    """Load a character profile JSON (mirrors channel_studio's channel profiles)."""
    fname = "character.json" if name == "lena" else f"{name}.json"
    path = os.path.join(BASE_DIR, fname)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --- handoff 1: intake -------------------------------------------------------------

def intake(brief, channel="lena"):
    """A content request enters the system."""
    character = load_character(channel)
    batch = register(Batch(brief=brief, channel=channel, character=character))
    batch.emit("intake", brief=brief)
    batch.save()
    return batch


# --- handoff 2: plan -----------------------------------------------------------------

def plan(batch):
    """The planner sub-agent breaks the brief into concrete generate-stage tasks."""
    batch.stage = "planning"
    planners = agents.by_stage("plan")
    task_specs = None
    if planners:
        out = _call(planners[0], batch, {"brief": batch.brief})
        if out:
            task_specs = out.get("tasks")
    if not task_specs:
        # Deterministic fallback: never leave a batch with nothing to do.
        task_specs = [{"agent": "draft_copy", "stage": "generate", "input": {"brief": batch.brief}}]
    for spec in task_specs:
        batch.add_task(spec["agent"], spec.get("stage", "generate"), spec.get("input"),
                        max_attempts=spec.get("max_attempts", 2))
    batch.stage = "generating"
    batch.emit("planned", task_count=len(task_specs))
    batch.save()
    return batch


# --- handoff 3: generate (dispatch) --------------------------------------------------

def dispatch(batch, task_id=None):
    """Run every QUEUED task (or one specific task) through its registered sub-agent."""
    lock = _lock_for(batch.id)
    with lock:
        if task_id:
            targets = [t for t in [batch.get_task(task_id)] if t]
        else:
            targets = batch.tasks_by_status("QUEUED")
        for task in targets:
            _run_task(batch, task)
        batch.save()
    return batch


def _run_task(batch, task):
    agent = agents.get(task["agent"])
    if not agent:
        task["status"] = "FAILED"
        task["error"] = f"unknown sub-agent '{task['agent']}'"
        batch.emit("task_failed", task_id=task["id"], error=task["error"])
        return
    task["status"] = "RUNNING"
    task["attempts"] += 1
    task["updated"] = time.time()
    batch.emit("task_started", task_id=task["id"], agent=agent["name"])
    try:
        output = agent["fn"](batch, task)
        task["output"] = output
        task["error"] = None
        task["status"] = "NEEDS_REVIEW" if agent["needs_review"] else "DONE"
        batch.emit("task_completed", task_id=task["id"], status=task["status"])
    except AgentError as e:
        _fail_task(batch, task, str(e))
    except Exception as e:  # never let one bad sub-agent crash the batch
        _fail_task(batch, task, f"unexpected error: {e}")
    task["updated"] = time.time()


def _fail_task(batch, task, message):
    task["error"] = message
    if task["attempts"] < task["max_attempts"]:
        task["status"] = "QUEUED"  # auto-retry on the next dispatch() call
        batch.emit("task_retry", task_id=task["id"], attempt=task["attempts"], error=message)
    else:
        task["status"] = "FAILED"  # exhausted retries — needs a human (see retry())
        batch.emit("task_failed", task_id=task["id"], error=message)


# --- handoff 4: review (human / Claude approval gate) ---------------------------------

def review(batch, task_id, decision, notes=""):
    """decision: approve | reject | regenerate. regenerate requeues the task with the
    reviewer's notes attached as feedback for the sub-agent's next attempt."""
    task = batch.get_task(task_id)
    if not task:
        raise AgentError(f"no such task '{task_id}'")
    if task["status"] != "NEEDS_REVIEW":
        raise AgentError(f"task '{task_id}' is {task['status']}, not awaiting review")
    task["notes"] = notes
    if decision == "approve":
        task["status"] = "APPROVED"
    elif decision == "regenerate":
        task["input"]["feedback"] = notes
        task["status"] = "QUEUED"
    elif decision == "reject":
        task["status"] = "FAILED"
        task["error"] = notes or "rejected on review"
    else:
        raise AgentError(f"unknown decision '{decision}' (want approve|reject|regenerate)")
    task["updated"] = time.time()
    batch.emit("reviewed", task_id=task_id, decision=decision, notes=notes)
    batch.save()
    return batch


def retry(batch, task_id):
    """Manual override: force a FAILED task back into the queue with a clean slate."""
    task = batch.get_task(task_id)
    if not task:
        raise AgentError(f"no such task '{task_id}'")
    task["status"] = "QUEUED"
    task["attempts"] = 0
    task["error"] = None
    task["updated"] = time.time()
    batch.emit("manual_retry", task_id=task_id)
    batch.save()
    return batch


# --- handoff 5: advance (assemble / publish / archive) --------------------------------

def ready_to_advance(batch):
    blocking = batch.tasks_by_status("QUEUED", "RUNNING", "NEEDS_REVIEW")
    return not blocking


def advance(batch, dry_run=True):
    """Once nothing is pending generation/review, run the review-stage (QA) and
    publish-stage sub-agents over the approved outputs, then archive the batch.
    dry_run=True (default) logs what WOULD publish without needing real credentials —
    same convention as channel_studio's dry_run."""
    if not ready_to_advance(batch):
        raise AgentError("batch still has tasks pending generation/review")
    lock = _lock_for(batch.id)
    with lock:
        batch.stage = "assembling"
        approved = batch.tasks_by_status("APPROVED", "DONE")

        qa_flags = []
        for agent in agents.by_stage("review"):
            out = _call(agent, batch, {"approved_tasks": approved})
            if out:
                qa_flags.extend(out.get("flags", []))

        publish_result = None
        for agent in agents.by_stage("publish"):
            publish_result = _call(agent, batch, {"approved_tasks": approved, "dry_run": dry_run})

        batch.stage = "published" if (publish_result and not dry_run) else "archived"
        batch.emit("advanced", stage=batch.stage, qa_flags=qa_flags, publish_result=publish_result)
        batch.save()
        _append_history(batch, qa_flags, publish_result)
    return batch


def _call(agent, batch, input):
    """Invoke a batch-level sub-agent (plan/review/publish) via a synthetic task, the
    same way dispatch() invokes a per-content generate task — one call path, one error
    contract, for every stage."""
    pseudo = new_task(agent["name"], agent["stage"], input, max_attempts=1)
    try:
        output = agent["fn"](batch, pseudo)
        batch.emit("agent_ran", agent=agent["name"], stage=agent["stage"])
        return output
    except Exception as e:
        batch.emit("agent_error", agent=agent["name"], stage=agent["stage"], error=str(e))
        return None


def _append_history(batch, qa_flags, publish_result):
    record = {
        "t": time.time(), "batch_id": batch.id, "brief": batch.brief,
        "channel": batch.channel, "stage": batch.stage, "counts": batch.counts(),
        "qa_flags": qa_flags, "publish_result": publish_result,
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# --- monitoring -------------------------------------------------------------------

def status(batch):
    return batch.to_dict()
