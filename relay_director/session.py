"""Story state for Relay Director — an OpenArt-Studio-style director workspace.

The core unit is a **Story**, decomposed into **Scenes**, each holding ordered
**Shots**. A shot is the atomic generation unit (the OpenArt "Shot"): it owns its own
image prompt, motion prompt, optional dialogue/narration, a still image, a video clip,
audio, a status and a version. Everything for one story lives under projects/<id>/
(story.json + generated media). Mirrors the proven event/listener pattern from Dolly
(relay_studio) but is otherwise independent — Dolly is never touched.
"""

import json
import os
import time
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

# Shot lifecycle. A shot advances empty -> planned -> image -> clip -> done (or error).
SHOT_STATES = ("empty", "planned", "image", "clip", "done", "error")

# Aspect presets the director offers (label -> WxH, kept light for a 16GB rig).
ASPECTS = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (768, 768),
}


def _nid(n=8):
    return uuid.uuid4().hex[:n]


def new_shot(idx, *, prompt="", motion="", dialogue="", duration=4.0):
    """A fresh shot record (plain dict so it serialises cleanly)."""
    return {
        "id": _nid(),
        "idx": idx,                # order within the whole story
        "prompt": prompt,          # what the still image shows
        "motion": motion,          # how it moves (LTX i2v motion prompt)
        "dialogue": dialogue,      # optional spoken line / narration
        "duration": duration,      # target clip length (s)
        "cast_ids": [],            # which cast members appear (character ids)
        "status": "planned" if prompt else "empty",
        "image": None,             # filename of the still
        "clip": None,              # filename of the rendered clip
        "audio": None,             # filename of the narration
        "seed": None,
        "ver": 1,                  # bumps each regenerate (OpenArt-style re-roll)
        "error": None,
    }


class Story:
    def __init__(self, *, title="Untitled Story", vision="", aspect="16:9", sid=None):
        self.id = sid or _nid(12)
        self.created = time.time()
        self.title = title
        self.vision = vision           # the director's free-text "vibe"
        self.aspect = aspect if aspect in ASPECTS else "16:9"
        self.width, self.height = ASPECTS[self.aspect]
        self.fps = 24
        # Script-first / consistency model (OpenArt-style):
        self.logline = ""              # one-line pitch
        self.script = ""               # the screenplay / treatment text
        self.world = {"style": "", "setting": "", "palette": ""}  # shared look anchor
        self.cast = []                 # [{id,name,role,description,ref_image,ref_seed}]
        self.approved = False          # script approved -> storyboard built
        self.stage = "empty"           # empty -> script -> approved
        self.scenes = []               # [{id, title, summary, shots:[shot,...]}]
        self.messages = []             # director chat transcript
        self.events = []               # full event log
        self.ver = 1                   # story version (bumps as the plan changes)
        self._listeners = []

    # --- scene / shot helpers ---------------------------------------------
    def add_scene(self, title="", summary=""):
        scene = {"id": _nid(), "title": title or f"Scene {len(self.scenes)+1}",
                 "summary": summary, "shots": []}
        self.scenes.append(scene)
        return scene

    def all_shots(self):
        """Every shot across all scenes, in story order."""
        out = []
        for sc in self.scenes:
            out.extend(sc["shots"])
        return out

    def get_shot(self, shot_id):
        for sc in self.scenes:
            for sh in sc["shots"]:
                if sh["id"] == shot_id:
                    return sc, sh
        return None, None

    # --- cast helpers ------------------------------------------------------
    def add_character(self, name="", role="", description=""):
        ch = {"id": _nid(), "name": name or f"Character {len(self.cast)+1}",
              "role": role, "description": description,
              "ref_image": None, "ref_seed": None}
        self.cast.append(ch)
        return ch

    def get_character(self, cid):
        return next((c for c in self.cast if c["id"] == cid), None)

    def character_by_name(self, name):
        n = (name or "").strip().lower()
        return next((c for c in self.cast if c["name"].strip().lower() == n), None)

    def shot_by_number(self, n):
        """1-based lookup so the chat can resolve '@shot3'."""
        shots = self.all_shots()
        if 1 <= n <= len(shots):
            return shots[n - 1]
        return None

    def reindex(self):
        for i, sh in enumerate(self.all_shots(), start=1):
            sh["idx"] = i

    def set_aspect(self, aspect):
        if aspect in ASPECTS:
            self.aspect = aspect
            self.width, self.height = ASPECTS[aspect]

    # --- persistence -------------------------------------------------------
    @property
    def dir(self):
        d = os.path.join(PROJECTS_DIR, self.id)
        os.makedirs(d, exist_ok=True)
        return d

    def media_path(self, filename):
        return os.path.join(self.dir, os.path.basename(filename))

    def media_url(self, filename):
        return f"/promptrelay/director/media/{self.id}/{os.path.basename(filename)}"

    def save(self):
        with open(os.path.join(self.dir, "story.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, sid):
        path = os.path.join(PROJECTS_DIR, sid, "story.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        s = cls(title=d.get("title", "Untitled Story"), vision=d.get("vision", ""),
                aspect=d.get("aspect", "16:9"), sid=d["id"])
        s.created = d.get("created", s.created)
        s.fps = d.get("fps", 24)
        s.logline = d.get("logline", "")
        s.script = d.get("script", "")
        s.world = d.get("world") or {"style": "", "setting": "", "palette": ""}
        s.cast = d.get("cast", [])
        s.approved = d.get("approved", False)
        s.stage = d.get("stage", "approved" if d.get("scenes") else "empty")
        s.scenes = d.get("scenes", [])
        s.messages = d.get("messages", [])
        s.events = d.get("events", [])
        s.ver = d.get("ver", 1)
        return s

    def to_dict(self):
        return {
            "id": self.id, "created": self.created, "title": self.title,
            "vision": self.vision, "aspect": self.aspect,
            "width": self.width, "height": self.height, "fps": self.fps,
            "logline": self.logline, "script": self.script,
            "world": self.world, "cast": self.cast,
            "approved": self.approved, "stage": self.stage,
            "scenes": self.scenes, "messages": self.messages,
            "events": self.events, "ver": self.ver,
            "final": self.final_url(),
            "counts": self.counts(),
        }

    def counts(self):
        shots = self.all_shots()
        return {
            "scenes": len(self.scenes),
            "shots": len(shots),
            "cast": len(self.cast),
            "refs": sum(1 for c in self.cast if c.get("ref_image")),
            "images": sum(1 for s in shots if s.get("image")),
            "clips": sum(1 for s in shots if s.get("clip")),
            "done": sum(1 for s in shots if s.get("status") == "done"),
        }

    def final_url(self):
        path = self.media_path("final.mp4")
        return self.media_url("final.mp4") if os.path.exists(path) else None

    # --- events / live updates --------------------------------------------
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

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})


# In-process registry so HTTP routes share one live Story between requests.
_REGISTRY = {}


def register(story):
    _REGISTRY[story.id] = story
    return story


def get(sid):
    if sid in _REGISTRY:
        return _REGISTRY[sid]
    s = Story.load(sid)
    if s:
        _REGISTRY[sid] = s
    return s


def list_stories():
    """Summaries for the Explore gallery (newest first)."""
    out = []
    if not os.path.isdir(PROJECTS_DIR):
        return out
    for sid in os.listdir(PROJECTS_DIR):
        path = os.path.join(PROJECTS_DIR, sid, "story.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        shots = [sh for sc in d.get("scenes", []) for sh in sc.get("shots", [])]
        thumb = next((sh["image"] for sh in shots if sh.get("image")), None)
        out.append({
            "id": d.get("id", sid),
            "title": d.get("title", "Untitled Story"),
            "vision": (d.get("vision", "") or "")[:120],
            "aspect": d.get("aspect", "16:9"),
            "shots": len(shots),
            "thumb": f"/promptrelay/director/media/{sid}/{thumb}" if thumb else None,
            "final": (f"/promptrelay/director/media/{sid}/final.mp4"
                      if os.path.exists(os.path.join(PROJECTS_DIR, sid, "final.mp4")) else None),
            "created": d.get("created", 0),
        })
    out.sort(key=lambda x: x["created"], reverse=True)
    return out
