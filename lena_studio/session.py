"""Batch state for the Lena content manager.

The core unit is a **Batch** — one content request. It carries a free-text brief, the
character profile it's for, and a list of **Task** dicts, each one unit of work handed
to exactly one registered sub-agent. A batch moves through named stages (see
BATCH_STAGES) and each task through its own lifecycle (see TASK_STATES); every
transition is persisted so the manager can resume after a restart. Mirrors the proven
Story/event pattern from relay_director/session.py.
"""

import json
import os
import time
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

# Batch lifecycle — the handoff points the manager moves a batch through.
BATCH_STAGES = ("intake", "planning", "generating", "assembling", "published", "archived")

# Task lifecycle. A task advances QUEUED -> RUNNING -> (NEEDS_REVIEW -> APPROVED) -> DONE,
# with REJECTED/FAILED as terminal error states and QUEUED re-entered on retry.
TASK_STATES = ("QUEUED", "RUNNING", "NEEDS_REVIEW", "APPROVED", "REJECTED", "FAILED", "DONE")


def _nid(n=8):
    return uuid.uuid4().hex[:n]


def new_task(agent, stage, input=None, max_attempts=2):
    """A fresh task record (plain dict so it serialises cleanly)."""
    return {
        "id": _nid(),
        "agent": agent,            # sub-agent name in agents.AGENT_REGISTRY
        "stage": stage,            # plan | generate | review | publish
        "status": "QUEUED",
        "input": input or {},
        "output": None,
        "error": None,
        "attempts": 0,
        "max_attempts": max_attempts,
        "notes": "",               # reviewer feedback, carried into a "regenerate" retry
        "created": time.time(),
        "updated": time.time(),
    }


class Batch:
    def __init__(self, *, brief="", channel="lena", character=None, bid=None):
        self.id = bid or _nid(12)
        self.created = time.time()
        self.brief = brief
        self.channel = channel
        self.character = character or {}   # loaded once at intake, carried with the batch
        self.stage = "intake"
        self.tasks = []
        self.events = []
        self.ver = 1
        self._listeners = []

    # --- task helpers --------------------------------------------------------
    def add_task(self, agent, stage, input=None, max_attempts=2):
        t = new_task(agent, stage, input, max_attempts)
        self.tasks.append(t)
        return t

    def get_task(self, task_id):
        return next((t for t in self.tasks if t["id"] == task_id), None)

    def tasks_by_status(self, *statuses):
        return [t for t in self.tasks if t["status"] in statuses]

    def counts(self):
        out = {s: 0 for s in TASK_STATES}
        for t in self.tasks:
            out[t["status"]] = out.get(t["status"], 0) + 1
        return out

    # --- persistence -----------------------------------------------------------
    @property
    def dir(self):
        d = os.path.join(PROJECTS_DIR, self.id)
        os.makedirs(d, exist_ok=True)
        return d

    def save(self):
        with open(os.path.join(self.dir, "batch.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, bid):
        path = os.path.join(PROJECTS_DIR, bid, "batch.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        b = cls(brief=d.get("brief", ""), channel=d.get("channel", "lena"),
                 character=d.get("character"), bid=d["id"])
        b.created = d.get("created", b.created)
        b.stage = d.get("stage", "intake")
        b.tasks = d.get("tasks", [])
        b.events = d.get("events", [])
        b.ver = d.get("ver", 1)
        return b

    def to_dict(self):
        return {
            "id": self.id, "created": self.created, "brief": self.brief,
            "channel": self.channel, "character": self.character, "stage": self.stage,
            "tasks": self.tasks, "events": self.events, "ver": self.ver,
            "counts": self.counts(),
        }

    # --- events / live updates --------------------------------------------------
    def on_event(self, cb):
        self._listeners.append(cb)

    def off_event(self, cb):
        try:
            self._listeners.remove(cb)
        except ValueError:
            pass

    def emit(self, kind, **data):
        ev = {"t": time.time(), "kind": kind, **data}
        self.events.append(ev)
        for cb in list(self._listeners):
            try:
                cb(ev)
            except Exception:
                pass
        return ev


# In-process registry so HTTP routes / manager calls share one live Batch.
_REGISTRY = {}


def register(batch):
    _REGISTRY[batch.id] = batch
    return batch


def get(bid):
    if bid in _REGISTRY:
        return _REGISTRY[bid]
    b = Batch.load(bid)
    if b:
        _REGISTRY[bid] = b
    return b


def list_batches():
    """Summaries for a batch list/dashboard (newest first)."""
    out = []
    if not os.path.isdir(PROJECTS_DIR):
        return out
    for bid in os.listdir(PROJECTS_DIR):
        path = os.path.join(PROJECTS_DIR, bid, "batch.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        out.append({
            "id": d.get("id", bid),
            "brief": (d.get("brief", "") or "")[:120],
            "channel": d.get("channel", "lena"),
            "stage": d.get("stage", "intake"),
            "created": d.get("created", 0),
            "tasks": len(d.get("tasks", [])),
        })
    out.sort(key=lambda x: x["created"], reverse=True)
    return out
