"""In-memory task registry for the SeeDance studio.

One process, one GPU — a simple thread-safe dict is plenty. Tasks mirror the SeeDance
"Tasks" list: QUEUED -> RUNNING -> SUCCEEDED / FAILED, each carrying its prompt, params,
resulting video URL and any error.
"""

import threading
import time
import uuid

_LOCK = threading.Lock()
_TASKS = {}  # id -> task dict


def create(prompt, params):
    tid = uuid.uuid4().hex[:12]
    with _LOCK:
        _TASKS[tid] = {
            "id": tid,
            "status": "QUEUED",
            "prompt": prompt,
            "params": params,
            "seed": params.get("seed"),
            "video_url": None,
            "error": None,
            "created": time.time(),
            "updated": time.time(),
        }
    return tid


def reopen(tid, prompt, params):
    """Recreate a task entry under an EXISTING id — used to resume a checkpointed render
    whose in-memory task was lost (e.g. after a ComfyUI restart)."""
    with _LOCK:
        _TASKS[tid] = {
            "id": tid,
            "status": "QUEUED",
            "prompt": prompt,
            "params": params,
            "seed": params.get("seed"),
            "video_url": None,
            "error": None,
            "created": time.time(),
            "updated": time.time(),
        }
    return tid


def update(tid, **kw):
    with _LOCK:
        t = _TASKS.get(tid)
        if t:
            t.update(kw)
            t["updated"] = time.time()


def get(tid):
    with _LOCK:
        t = _TASKS.get(tid)
        return dict(t) if t else None


def listing(limit=50):
    with _LOCK:
        items = sorted(_TASKS.values(), key=lambda t: t["created"], reverse=True)
        return [dict(t) for t in items[:limit]]
