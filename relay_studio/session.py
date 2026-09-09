"""Project state for the studio: the user-inputs artifact, the task list, the
media library and an event log. Mirrors a drama.land "project" — everything for one
video lives under projects/<id>/ (state json + generated media files)."""

import json
import os
import time
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

# The structured artifact, modelled on drama.land's "user-inputs" panel.
INPUT_FIELDS = [
    "cat_name", "show_title", "episode", "cat_photo_url",
    "voice_name", "voice_description", "voice_language",
    "lipsync_service", "requested_output",
]

# Fixed pipeline the super-agent works through (id -> human title).
PIPELINE_TASKS = [
    ("script", "Write the episode script"),
    ("voice", "Generate the cat's voice (TTS)"),
    ("bgm", "Compose BGM (intro + outro)"),
    ("talking", "Lipsync the talking video"),
    ("sfx", "Add sound effects"),
    ("mix", "Final mix — combine everything"),
]


class Project:
    def __init__(self, inputs=None, request="", pid=None):
        self.id = pid or uuid.uuid4().hex[:12]
        self.created = time.time()
        self.request = request
        self.inputs = {k: "" for k in INPUT_FIELDS}
        self.inputs.update({k: v for k, v in (inputs or {}).items() if k in INPUT_FIELDS})
        self.tasks = [{"id": tid, "title": t, "status": "pending"} for tid, t in PIPELINE_TASKS]
        self.media = []          # [{id, kind, file, label, meta, via, created}]
        self.messages = []       # chat transcript [{role, content}]
        self.events = []         # full event log (thinking/tool/task/media/...)
        self.ver = 1             # artifact version (drama.land bumps v1 -> v2 as outputs land)
        self.outputs = {}        # written-back artifact fields: FIELD -> {media_id|value, kind}
        self._listeners = []     # live SSE callbacks

    # --- persistence -------------------------------------------------------
    @property
    def dir(self):
        d = os.path.join(PROJECTS_DIR, self.id)
        os.makedirs(d, exist_ok=True)
        return d

    def media_path(self, filename):
        return os.path.join(self.dir, filename)

    def save(self):
        with open(os.path.join(self.dir, "project.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, pid):
        path = os.path.join(PROJECTS_DIR, pid, "project.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        p = cls(inputs=data.get("inputs"), request=data.get("request", ""), pid=data["id"])
        p.created = data.get("created", p.created)
        p.tasks = data.get("tasks", p.tasks)
        p.media = data.get("media", [])
        p.messages = data.get("messages", [])
        p.events = data.get("events", [])
        p.ver = data.get("ver", 1)
        p.outputs = data.get("outputs", {})
        return p

    def to_dict(self):
        return {
            "id": self.id, "created": self.created, "request": self.request,
            "inputs": self.inputs, "tasks": self.tasks, "media": self.media,
            "messages": self.messages, "events": self.events,
            "ver": self.ver, "outputs": self.outputs,
        }

    # --- versioned artifact writeback -------------------------------------
    def write_artifact(self, field, *, media_id=None, value=None):
        """Append/overwrite an output field on the user-inputs artifact and bump the
        version (drama.land's v1 -> v2 behaviour). Returns the new version."""
        kind = None
        if media_id:
            for m in self.media:
                if m["id"] == media_id:
                    kind = m["kind"]
                    break
        self.outputs[field] = {"media_id": media_id, "value": value, "kind": kind}
        self.ver += 1
        self.emit("artifact", field=field, media_id=media_id, value=value,
                  mkind=kind, ver=self.ver)
        return self.ver

    # --- events / live updates --------------------------------------------
    def on_event(self, cb):
        self._listeners.append(cb)

    def emit(self, kind, **data):
        ev = {"t": time.time(), "kind": kind, **data}
        self.events.append(ev)
        for cb in list(self._listeners):
            try:
                cb(ev)
            except Exception:
                pass
        return ev

    # --- task helpers ------------------------------------------------------
    def set_task(self, tid, status):
        for t in self.tasks:
            if t["id"] == tid:
                t["status"] = status
                done = sum(1 for x in self.tasks if x["status"] == "done")
                self.emit("task", task_id=tid, title=t["title"], status=status,
                          done=done, total=len(self.tasks))
                return t
        return None

    # --- media library -----------------------------------------------------
    def add_media(self, kind, file, *, label="", via="ffmpeg", **meta):
        item = {"id": uuid.uuid4().hex[:8], "kind": kind,
                "file": os.path.basename(file), "label": label or os.path.basename(file),
                "via": via, "meta": meta, "created": time.time()}
        self.media.append(item)
        self.emit("media", item=item)
        return item

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})


# In-process registry so the HTTP routes can find live projects between requests.
_REGISTRY = {}


def register(project):
    _REGISTRY[project.id] = project
    return project


def get(pid):
    if pid in _REGISTRY:
        return _REGISTRY[pid]
    p = Project.load(pid)
    if p:
        _REGISTRY[pid] = p
    return p


def list_projects():
    """Lightweight summary of every saved project for the Explore gallery."""
    out = []
    if not os.path.isdir(PROJECTS_DIR):
        return out
    for pid in os.listdir(PROJECTS_DIR):
        path = os.path.join(PROJECTS_DIR, pid, "project.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        media = d.get("media", [])
        final = next((m for m in media if m.get("meta", {}).get("final")), None)
        talking = next((m for m in media if m.get("kind") == "video"
                        and not m.get("meta", {}).get("final")), None)
        image = next((m for m in media if m.get("kind") == "image"), None)
        vid = final or talking
        ins = d.get("inputs", {})
        out.append({
            "id": d.get("id", pid),
            "title": ins.get("show_title") or (d.get("request", "") or "Untitled")[:48],
            "subtitle": ins.get("episode") or "",
            "tag": ins.get("requested_output") or "video",
            "video": vid["file"] if vid else None,
            "image": image["file"] if image else None,
            "done": bool(final),
            "created": d.get("created", 0),
        })
    out.sort(key=lambda x: x["created"], reverse=True)
    return out
