"""Per-sequence render checkpoints, so an interrupted multi-shot render can resume instead
of losing every shot already rendered.

One JSON file per in-flight sequence (`seedance_studio/checkpoints/{tid}.json`) holding
everything needed to finish it: the style/shots/settings AND the list of shots already
rendered (index -> output file path). The file is written when the render starts, appended
to after each shot completes, and DELETED on full success — so any file left on disk is by
definition an interrupted render that can be resumed. Shot output files live in ComfyUI's
output dir and survive restarts, so resume just re-stitches them + renders the missing ones.
"""

import json
import os
import threading

_LOCK = threading.Lock()
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def _path(tid):
    return os.path.join(_DIR, f"{tid}.json")


def save(tid, data):
    """Write/overwrite the checkpoint for `tid` (atomic-ish swap)."""
    os.makedirs(_DIR, exist_ok=True)
    with _LOCK:
        tmp = _path(tid) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _path(tid))


def load(tid):
    with _LOCK:
        try:
            with open(_path(tid), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, ValueError):
            return None


def delete(tid):
    with _LOCK:
        try:
            os.remove(_path(tid))
        except OSError:
            pass


def listing():
    """All interrupted (still-on-disk) checkpoints, newest first, with resume progress."""
    try:
        names = [n for n in os.listdir(_DIR) if n.endswith(".json")]
    except FileNotFoundError:
        return []
    items = []
    for n in names:
        tid = n[:-5]
        c = load(tid)
        if not c:
            continue
        done = c.get("done") or []
        # only surface genuinely-resumable ones (at least one shot rendered, some remaining)
        total = len(c.get("shots") or [])
        items.append({"id": tid, "label": c.get("label") or c.get("style") or "sequence",
                      "done": len(done), "total": total, "updated": c.get("updated", 0)})
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items
