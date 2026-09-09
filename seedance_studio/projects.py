"""Saved projects / templates for the Cinematic Sequence studio.

Seedance's "client projects / content pipelines" pitch is really just: save a sequence and
reuse it. This stores each saved sequence (style header + shot list + output settings) as a
named project in one JSON file, so you can reload it for the next client or batch instead of
rebuilding it by hand. Read-modify-write on a small file — plenty for one local user.
"""

import json
import os
import threading
import time
import uuid

_LOCK = threading.Lock()
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects.json")


def _load():
    try:
        with open(_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, ValueError):
        return []


def _write(items):
    tmp = _PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, _PATH)  # atomic-ish swap so a crash mid-write can't truncate the store


def listing():
    """All saved projects, newest-updated first (without echoing huge fields twice)."""
    with _LOCK:
        items = _load()
    items.sort(key=lambda p: p.get("updated", 0), reverse=True)
    return items


def get(pid):
    with _LOCK:
        for p in _load():
            if p.get("id") == pid:
                return p
    return None


def save(name, style, shots, settings, pid=None):
    """Create a new project, or update the one with `pid`. Returns the stored project."""
    name = (name or "Untitled").strip()[:80]
    now = time.time()
    entry = {
        "name": name,
        "style": style or "",
        "shots": shots or [],
        "settings": settings or {},
        "updated": now,
    }
    with _LOCK:
        items = _load()
        if pid:
            for p in items:
                if p.get("id") == pid:
                    p.update(entry)
                    _write(items)
                    return p
        entry["id"] = uuid.uuid4().hex[:12]
        entry["created"] = now
        items.append(entry)
        _write(items)
        return entry


def delete(pid):
    with _LOCK:
        items = _load()
        kept = [p for p in items if p.get("id") != pid]
        if len(kept) != len(items):
            _write(kept)
            return True
    return False
