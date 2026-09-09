"""Persistent render history for the SeeDance studio.

One JSONL line per *finished* render, so the Higgsfield-style date-grouped history
survives ComfyUI restarts (the in-memory tasks registry does not). Append-only; the
video files themselves already live in ComfyUI's output dir, we only keep pointers.
"""

import json
import os
import threading
import time
import uuid

_LOCK = threading.Lock()
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.jsonl")


def record(mode, prompt, video_url, seed=None, params=None):
    """Append one finished render. Returns the stored entry."""
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "mode": mode,
        "prompt": (prompt or "")[:500],
        "video_url": video_url,
        "seed": seed,
        "params": params or {},
    }
    with _LOCK:
        with open(_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def listing(limit=500):
    """All recorded renders, newest first. Malformed lines are skipped."""
    try:
        with _LOCK:
            with open(_PATH, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
    except FileNotFoundError:
        return []
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except ValueError:
            continue
    items.reverse()
    return items[:limit]
