"""The Director brain — "vibe directing" for Relay Director.

Given a free-text vision, the Director *scopes* a story into Scenes -> Shots (the
OpenArt plan-before-you-spend step), then drives a per-shot pipeline: still image ->
clip (-> optional voice). Shots can be re-rolled individually (@shot N), and the whole
thing exports to one widescreen mp4.

Planning uses a local Ollama model when available (relay_studio.llm, read-only) and
falls back to a deterministic planner so the director always produces a sensible
storyboard offline. Generation goes through engines.py, which itself degrades to
placeholders — so the entire flow works on any rig.
"""

import json
import math
import re
import threading
import time

from . import engines
from .session import new_shot

# Storyboard size caps. Raised from the original tiny limits so a longer ask (e.g. a
# "5-minute" piece) produces a real storyboard rather than 2 shots.
MAX_SCENES = 6
MAX_SHOTS_PER_SCENE = 6
MAX_SHOTS = 24            # hard ceiling on total shots (keeps LTX render time sane)

# Per-story render locks: one render at a time per story, so a double-clicked "Render"
# (or a second "Render all") can't spawn overlapping renders that fight over the GPU.
_RENDER_LOCKS = {}
BUSY_MSG = ("A render is already running for this story — let it finish before starting "
            "another. (One render at a time keeps the GPU sane.)")


def _lock_for(story):
    return _RENDER_LOCKS.setdefault(story.id, threading.Lock())


# ---------------------------------------------------------------------------
# Planning (scope the vision into scenes & shots).
# ---------------------------------------------------------------------------

DEVELOP_SYSTEM = """You are a film director's development AI. Turn the user's vision into
a short treatment with a consistent cast and world. Return ONLY JSON, no prose:
{"title": "...",
 "logline": "<one sentence pitch>",
 "script": "<a short screenplay/treatment, 1-3 short paragraphs of beats and key moments>",
 "world": {"style": "<visual style, e.g. gritty sci-fi, 35mm, teal-orange grade>",
            "setting": "<where/when it takes place>",
            "palette": "<dominant colors / lighting>"},
 "cast": [{"name": "<character name>", "role": "<their role>",
            "description": "<FIXED physical appearance to keep them identical across shots: "
                           "age, build, hair, face, signature wardrobe, distinguishing marks>"}]}
Rules: 1-5 named cast members. Each description is a concrete, reusable visual spec (no plot).
No commentary outside the JSON."""

PLAN_SYSTEM = """You are a film director's AI. Break the approved script into a shot list,
reusing the given cast and world for consistency. Return ONLY JSON, no prose:
{"title": "...", "scenes": [
  {"title": "...", "summary": "...", "shots": [
     {"prompt": "<vivid visual description of the still frame; describe the action/framing, "
                "and name which cast members appear by name>",
      "cast": ["<names of cast members visible in this shot>"],
      "motion": "<how it moves: camera + subject motion>",
      "dialogue": "<spoken line or empty>",
      "duration": <seconds 3-6>}
  ]}
]}
Rules: split the story across 2-6 scenes with 2-6 shots each. Prompts are concrete and
cinematic (lighting, lens, composition) and consistent with the world. Reference cast by
their exact names. Aim for the requested total shot count; if a duration is named, cover it
at ~one shot per 10-12 seconds. No commentary."""


def _llm():
    try:
        from ..relay_studio import llm as _l
        return _l
    except Exception:
        return None


# --- Step 1: develop the script + cast + world (no storyboard yet) ---------

def develop(story, vision):
    """Turn the vision into a treatment: title, logline, script, world and a named cast.
    Stops at an approval gate (stage='script') — the storyboard is built on approve()."""
    vision = (vision or story.vision or "").strip()
    story.vision = vision
    story.emit("status", text="Director is developing your story…")
    story.emit("thinking", text=f"Shaping a screenplay + cast from: “{vision[:100]}”")

    data = _llm_develop(story, vision)
    if data is None:
        story.emit("thinking", text="No local model — using the built-in developer.")
        data = _fallback_develop(vision)

    story.title = data.get("title") or story.title
    story.logline = (data.get("logline") or "").strip()
    story.script = (data.get("script") or "").strip()
    w = data.get("world") or {}
    story.world = {"style": (w.get("style") or "").strip(),
                   "setting": (w.get("setting") or "").strip(),
                   "palette": (w.get("palette") or "").strip()}
    story.cast = []
    for c in (data.get("cast") or [])[:5]:
        story.add_character(name=(c.get("name") or "").strip(),
                            role=(c.get("role") or "").strip(),
                            description=(c.get("description") or "").strip())
    story.stage = "script"
    story.approved = False
    story.ver += 1
    story.emit("script", title=story.title, logline=story.logline,
               script=story.script, world=story.world, cast=story.cast)
    story.emit("assistant",
               text=(f"Developed **{story.title}** — {len(story.cast)} cast members and a "
                     f"locked world look. Review the script & cast on the **Story** tab. "
                     f"Say **“looks good”** to build the storyboard, or tell me what to change."))
    story.save()
    return data


def _llm_develop(story, vision):
    llm = _llm()
    if llm is None:
        return None
    try:
        if not llm.available_models():
            return None
        msg = llm.chat(
            [{"role": "system", "content": DEVELOP_SYSTEM},
             {"role": "user", "content": f"Vision: {vision}\nAspect: {story.aspect}"}],
            think=False, timeout=180)
        th = (msg.get("thinking") or "").strip()
        if th:
            story.emit("thinking", text=th[:400])
        return _extract_json((msg.get("content") or "").strip())
    except Exception as e:
        story.emit("thinking", text=f"Developer model error ({e}); using built-in developer.")
        return None


def _fallback_develop(vision):
    """Deterministic treatment when no model is available."""
    vision = vision or "a cinematic short"
    title = " ".join(vision.split()[:5]).title() or "Untitled Story"
    return {
        "title": title,
        "logline": vision[:140],
        "script": (f"{vision}. We open on the world, establish the stakes, follow the lead "
                   f"through a rising conflict to a climactic turn, and close on the "
                   f"aftermath."),
        "world": {"style": "cinematic, 35mm, shallow depth of field, filmic color grade",
                  "setting": vision, "palette": "moody contrast, rich shadows"},
        "cast": [{"name": "The Lead", "role": "protagonist",
                  "description": "the central figure, distinctive silhouette and wardrobe, "
                                 "kept identical in every shot"}],
    }


def revise(story, message):
    """Adjust the in-progress treatment per the user's instruction (pre-approval)."""
    llm = _llm()
    story.emit("status", text="Revising the treatment…")
    data = None
    if llm is not None:
        try:
            if llm.available_models():
                ctx = (f"Current treatment JSON:\n{json.dumps(_treatment(story))}\n\n"
                       f"Apply this change and return the FULL updated JSON: {message}")
                msg = llm.chat([{"role": "system", "content": DEVELOP_SYSTEM},
                                {"role": "user", "content": ctx}], think=False, timeout=180)
                data = _extract_json((msg.get("content") or "").strip())
        except Exception:
            data = None
    if not data:
        # Lightweight fallback: fold the instruction into the world style.
        story.world["style"] = (story.world.get("style", "") + ", " + message).strip(", ")
        story.emit("script", title=story.title, logline=story.logline,
                   script=story.script, world=story.world, cast=story.cast)
        story.emit("assistant", text="Noted — folded that into the world look. "
                   "Say “looks good” when you're ready to build the storyboard.")
        story.save()
        return
    story.title = data.get("title") or story.title
    story.logline = (data.get("logline") or story.logline).strip()
    story.script = (data.get("script") or story.script).strip()
    w = data.get("world") or story.world
    story.world = {"style": w.get("style", ""), "setting": w.get("setting", ""),
                   "palette": w.get("palette", "")}
    if data.get("cast"):
        story.cast = []
        for c in data["cast"][:5]:
            story.add_character(name=c.get("name", ""), role=c.get("role", ""),
                                description=c.get("description", ""))
    story.ver += 1
    story.emit("script", title=story.title, logline=story.logline,
               script=story.script, world=story.world, cast=story.cast)
    story.emit("assistant", text="Updated the treatment. Say “looks good” to build the storyboard.")
    story.save()


def _treatment(story):
    return {"title": story.title, "logline": story.logline, "script": story.script,
            "world": story.world,
            "cast": [{"name": c["name"], "role": c["role"], "description": c["description"]}
                     for c in story.cast]}


# --- Step 2: approve -> reference sheets + cast-aware storyboard ------------

def approve(story):
    """Lock the script: generate a reference sheet per cast member, then build a
    cast-aware storyboard. Guarded by the render lock (it generates images)."""
    lock = _lock_for(story)
    if not lock.acquire(blocking=False):
        story.emit("assistant", text=BUSY_MSG)
        return story
    try:
        if not story.script and not story.cast:
            story.emit("assistant", text="Nothing to approve yet — describe your story first.")
            return story
        story.emit("status", text="Approved — generating character sheets…")
        for ch in story.cast:
            _do_generate_reference(story, ch)
        story.emit("status", text="Breaking the script into a storyboard…")
        build_storyboard(story)
        story.approved = True
        story.stage = "approved"
        story.save()
        return story
    finally:
        lock.release()


def build_storyboard(story, *, target=None):
    """Break the (approved) script into a cast-aware shot list. Maps cast names to ids
    and tags each shot with the cast_ids it features."""
    target = int(_clamp(target or _target_shots(story.vision + " " + story.script),
                        1, MAX_SHOTS))
    story.emit("thinking", text=f"Scoping ~{target} shots across the script…")
    plan = _llm_plan(story, target)
    if plan is None:
        story.emit("thinking", text="No local model — using the built-in planner.")
        plan = _fallback_plan(story.vision or story.logline, target)
    _apply_plan(story, plan)
    story.ver += 1
    c = story.counts()
    story.emit("plan", title=story.title, scenes=story.scenes, cast=story.cast,
               counts=c, est=estimate(story))
    story.emit("assistant",
               text=(f"Storyboard ready — {c['scenes']} scenes, {c['shots']} shots, "
                     f"{c['cast']} cast locked. Render previews are fast; tick "
                     f"**🎬 Final (LTX)** for the high-quality pass."))
    story.save()
    return plan


def _llm_plan(story, target=6):
    llm = _llm()
    if llm is None:
        return None
    try:
        if not llm.available_models():
            return None
        cast_lines = "\n".join(f"- {c['name']} ({c.get('role','')}): {c['description']}"
                               for c in story.cast) or "(none)"
        world = story.world
        user = (f"SCRIPT:\n{story.script or story.vision}\n\n"
                f"WORLD: style={world.get('style','')}; setting={world.get('setting','')}; "
                f"palette={world.get('palette','')}\n\n"
                f"CAST:\n{cast_lines}\n\n"
                f"Aspect: {story.aspect}. Produce about {target} shots total.")
        msg = llm.chat([{"role": "system", "content": PLAN_SYSTEM},
                        {"role": "user", "content": user}], think=False, timeout=180)
        th = (msg.get("thinking") or "").strip()
        if th:
            story.emit("thinking", text=th[:400])
        return _extract_json((msg.get("content") or "").strip())
    except Exception as e:
        story.emit("thinking", text=f"Planner model error ({e}); using built-in planner.")
        return None


def _extract_json(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _apply_plan(story, plan):
    """Replace the storyboard with the plan's scenes/shots (preserves nothing — this is
    a fresh scope; use chat/edit for incremental changes)."""
    story.title = plan.get("title") or story.title
    story.scenes = []
    idx = 0
    for sc in (plan.get("scenes") or [])[:MAX_SCENES]:
        scene = story.add_scene(sc.get("title", ""), sc.get("summary", ""))
        for sh in (sc.get("shots") or [])[:MAX_SHOTS_PER_SCENE]:
            if idx >= MAX_SHOTS:
                break
            idx += 1
            shot = new_shot(
                idx,
                prompt=(sh.get("prompt") or "").strip(),
                motion=(sh.get("motion") or "").strip(),
                dialogue=(sh.get("dialogue") or "").strip(),
                duration=_clamp(sh.get("duration", 4), 2, 8))
            shot["cast_ids"] = _resolve_cast(story, sh.get("cast") or [])
            scene["shots"].append(shot)
    story.reindex()


def _resolve_cast(story, names):
    """Map a list of cast names from the plan to character ids on the story."""
    ids = []
    for n in names:
        ch = story.character_by_name(n if isinstance(n, str) else n.get("name", ""))
        if ch and ch["id"] not in ids:
            ids.append(ch["id"])
    return ids


def _fallback_plan(vision, target=6):
    """A deterministic storyboard when no model is available. Splits the vision into a
    classic arc, scaling scene/shot counts to hit roughly `target` total shots."""
    vision = vision or "a cinematic short"
    target = int(_clamp(target, 1, MAX_SHOTS))
    n_scenes = int(_clamp(math.ceil(target / 4), 1, MAX_SCENES))
    shots_per = int(_clamp(math.ceil(target / n_scenes), 1, MAX_SHOTS_PER_SCENE))
    beats = ["establishing wide shot", "medium shot, the subject in focus",
             "close-up, emotional detail", "dynamic angle, motion and energy",
             "over-the-shoulder, tension building", "low-angle hero shot"]
    moods = ["golden hour, soft light", "moody low-key lighting",
             "bright high-key, airy", "neon night, cinematic"]
    scene_titles = ["Opening", "Rising Action", "Turn", "Climax", "Fallout", "Resolution"]
    scenes = []
    idx = 0
    for s in range(n_scenes):
        shots = []
        for k in range(shots_per):
            if idx >= target:
                break
            idx += 1
            beat = beats[(idx - 1) % len(beats)]
            mood = moods[(idx - 1) % len(moods)]
            shots.append({
                "prompt": f"{vision}, {beat}, {mood}, 35mm, shallow depth of field, "
                          f"highly detailed, cinematic color grade",
                "motion": "slow cinematic camera move, natural subject motion",
                "dialogue": "",
                "duration": 4,
            })
        if shots:
            scenes.append({"title": scene_titles[s % len(scene_titles)],
                           "summary": f"{vision} — {scene_titles[s % len(scene_titles)].lower()}",
                           "shots": shots})
    title = " ".join(vision.split()[:5]).title() or "Untitled Story"
    return {"title": title, "scenes": scenes}


def _target_shots(vision):
    """Infer a sensible total shot count from the vision. A named duration scales the
    storyboard (~5 shots/min); an explicit "N shots" wins; otherwise a richer default."""
    text = (vision or "").lower()
    m = re.search(r"(\d+)\s*shots?\b", text)
    if m:
        return int(_clamp(int(m.group(1)), 1, MAX_SHOTS))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b", text)
    if m:
        return int(_clamp(round(float(m.group(1)) * 5), 4, MAX_SHOTS))
    m = re.search(r"(\d+)\s*(?:seconds?|secs?|s)\b", text)
    if m:
        return int(_clamp(round(float(m.group(1)) / 10), 3, MAX_SHOTS))
    return 6


# ---------------------------------------------------------------------------
# Character reference sheets (Phase 3) + Tier A consistency (Phase 4).
# ---------------------------------------------------------------------------

def _do_generate_reference(story, character):
    """Render one canonical portrait for a cast member (locked seed) and store it on the
    character. Internal: assumes the render lock is held."""
    story.emit("status", text=f"Character sheet: {character['name']}")
    img, via, seed = engines.generate_reference(
        story, character, on_log=lambda t: story.emit("log", shot=None, text=t))
    character["ref_image"] = img
    character["ref_seed"] = seed
    story.emit("character", id=character["id"], name=character["name"],
               file=img, url=story.media_url(img), via=via)
    return character


def generate_reference(story, char_id):
    """Public: (re)generate a single character's reference sheet. Lock-guarded."""
    ch = story.get_character(char_id)
    if not ch:
        story.emit("error", text="No such character.")
        return None
    lock = _lock_for(story)
    if not lock.acquire(blocking=False):
        story.emit("assistant", text=BUSY_MSG)
        return ch
    try:
        ch["ref_seed"] = None  # fresh look on explicit regenerate
        _do_generate_reference(story, ch)
        story.save()
        return ch
    finally:
        lock.release()


def _compose_shot_prompt(story, shot):
    """Tier A consistency: build the effective image prompt from the shot, the locked
    descriptions of the cast members in it, and the world style/palette — so characters
    and look stay consistent across every shot."""
    parts = [shot.get("prompt", "").strip()]
    cast_ids = shot.get("cast_ids") or []
    for cid in cast_ids:
        ch = story.get_character(cid)
        if ch and ch.get("description"):
            parts.append(f"{ch['name']}: {ch['description']}")
    w = story.world or {}
    look = ", ".join(x for x in (w.get("style"), w.get("palette")) if x)
    if look:
        parts.append(look)
    return ". ".join(p for p in parts if p)


def _primary_cast(story, shot):
    """The lead cast member in a shot that has a generated reference sheet — the one
    whose identity Flux Klein locks onto the shot (Tier B). First listed wins."""
    for cid in (shot.get("cast_ids") or []):
        ch = story.get_character(cid)
        if ch and ch.get("ref_image"):
            return ch
    return None


def _shot_seed(story, shot):
    """Reuse a character's locked seed when a single cast member dominates the shot —
    a gentle nudge toward identity consistency (Tier A)."""
    if shot.get("seed"):
        return shot["seed"]
    cast_ids = shot.get("cast_ids") or []
    if len(cast_ids) == 1:
        ch = story.get_character(cast_ids[0])
        if ch and ch.get("ref_seed"):
            return ch["ref_seed"]
    return None


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def _do_render_shot(story, shot, *, use_ltx=False, voice=True):
    """Run image -> (voice) -> clip for one shot. Emits granular events.
    Internal: assumes the caller already holds the story's render lock."""
    sc, _ = story.get_shot(shot["id"])
    log = lambda t: story.emit("log", shot=shot["id"], text=t)
    story.emit("shot_status", shot=shot["id"], status="image",
               text=f"Shot {shot['idx']}: rendering still")
    shot["status"] = "image"
    shot["error"] = None
    try:
        shot["seed"] = _shot_seed(story, shot)
        composed = _compose_shot_prompt(story, shot)
        img, via = engines.generate_image(story, shot, on_log=log, prompt_override=composed)
        shot["image"] = img
        story.emit("shot_image", shot=shot["id"], file=img, url=story.media_url(img), via=via)

        # Tier B identity-lock: on the FINAL (LTX) pass, swap the shot's primary cast
        # member's locked face onto the base image via Flux Klein, then animate THAT.
        if use_ltx:
            primary = _primary_cast(story, shot)
            if primary:
                story.emit("shot_status", shot=shot["id"], status="image",
                           text=f"Shot {shot['idx']}: identity-lock ({primary['name']})")
                swapped, svia = engines.swap_character(
                    story, shot, story.media_path(img), primary, on_log=log)
                if swapped:
                    img = swapped
                    shot["image"] = img
                    story.emit("shot_image", shot=shot["id"], file=img,
                               url=story.media_url(img), via=svia)

        audio = None
        if voice and shot.get("dialogue"):
            story.emit("shot_status", shot=shot["id"], status="image", text="voice")
            v = engines.generate_voice(story, shot, on_log=log)
            if v:
                shot["audio"] = v
                audio = story.media_path(v)

        story.emit("shot_status", shot=shot["id"], status="clip",
                   text=f"Shot {shot['idx']}: animating")
        shot["status"] = "clip"
        clip, cvia = engines.generate_clip(
            story, shot, story.media_path(img), audio, on_log=log, use_ltx=use_ltx)
        shot["clip"] = clip
        shot["status"] = "done"
        story.emit("shot_clip", shot=shot["id"], file=clip,
                   url=story.media_url(clip), via=cvia)
        story.emit("shot_status", shot=shot["id"], status="done",
                   text=f"Shot {shot['idx']}: done")
    except Exception as e:
        shot["status"] = "error"
        shot["error"] = str(e)
        story.emit("shot_status", shot=shot["id"], status="error", text=str(e))
    story.save()
    return shot


def render_shot(story, shot, *, use_ltx=False, voice=True):
    """Public single-shot render. Guarded by the story's render lock so a stray
    double-click (or a second request) can't start an overlapping render."""
    lock = _lock_for(story)
    if not lock.acquire(blocking=False):
        story.emit("assistant", text=BUSY_MSG)
        return shot
    try:
        return _do_render_shot(story, shot, use_ltx=use_ltx, voice=voice)
    finally:
        lock.release()


def render_all(story, *, use_ltx=False, voice=True, only=None):
    """Render every shot (or only the given shot ids). Emits progress + a final summary.
    Guarded by the render lock — a second 'Render all' while one is running is ignored."""
    lock = _lock_for(story)
    if not lock.acquire(blocking=False):
        story.emit("assistant", text=BUSY_MSG)
        return story
    try:
        shots = story.all_shots()
        if only:
            shots = [s for s in shots if s["id"] in set(only)]
        total = len(shots)
        if not total:
            story.emit("assistant", text="No shots to render — scope a story first.")
            return story
        for i, sh in enumerate(shots, start=1):
            story.emit("progress", done=i - 1, total=total,
                       text=f"Rendering shot {i}/{total}")
            _do_render_shot(story, sh, use_ltx=use_ltx, voice=voice)
        story.emit("progress", done=total, total=total, text="All shots rendered")
        c = story.counts()
        mode = "final (LTX)" if use_ltx else "preview"
        story.emit("assistant",
                   text=f"Rendered {c['clips']}/{c['shots']} shots ({mode}). "
                        f"Hit Export to stitch the timeline.")
        story.save()
        return story
    finally:
        lock.release()


def export(story):
    """Stitch all rendered shot clips into final.mp4 (in story order)."""
    clips = [story.media_path(s["clip"]) for s in story.all_shots() if s.get("clip")]
    if not clips:
        story.emit("error", text="Nothing to export yet — render shots first.")
        return None
    story.emit("status", text="Exporting timeline…")
    try:
        dst = engines.export_timeline(
            story, clips, on_log=lambda t: story.emit("status", text=t))
        story.emit("final", url=story.media_url("final.mp4"))
        story.emit("assistant", text="Exported! Your film is ready in the preview. 🎬")
        story.save()
        return dst
    except Exception as e:
        story.emit("error", text=f"Export failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Director chat — natural-language edits, @shot referencing, re-scoping.
# ---------------------------------------------------------------------------

_SHOT_REF = re.compile(r"@?shot\s*#?\s*(\d+)", re.IGNORECASE)
_APPROVE = ("looks good", "approve", "approved", "build it", "build the storyboard",
            "go ahead", "lgtm", "ship it", "perfect", "love it", "let's go", "lets go",
            "make it", "do it", "yes build", "looks great")


def _is_approval(low):
    return any(k in low for k in _APPROVE) or low.strip() in ("yes", "y", "ok", "okay", "👍")


def chat(story, message, *, use_ltx=False, voice=True):
    """Route one director message by stage:
    empty -> develop a treatment; script -> approve or revise; approved -> @shot edits,
    render/export, or a storyboard rework. Lightweight intent routing, no tool-calling."""
    story.add_message("user", message)
    msg = message.strip()
    low = msg.lower()

    # Stage 1 — nothing developed yet: treat the message as the vision.
    if story.stage == "empty" and not story.script and not story.all_shots():
        return develop(story, msg)

    # Stage 2 — treatment awaiting approval: approve or revise.
    if story.stage == "script" and not story.approved:
        if _is_approval(low):
            return approve(story)
        return revise(story, msg)

    # Stage 3 — approved storyboard.
    m = _SHOT_REF.search(low)
    if m:
        n = int(m.group(1))
        shot = story.shot_by_number(n)
        if not shot:
            story.emit("assistant", text=f"There's no shot {n}. You have "
                       f"{len(story.all_shots())} shots.")
            return story
        instruction = _SHOT_REF.sub("", msg).strip(" ,.:")
        if instruction:
            shot["prompt"] = (shot["prompt"] + ", " + instruction).strip(", ")
        shot["ver"] += 1
        story.emit("assistant",
                   text=f"Re-rolling shot {n}" + (f" with: “{instruction}”." if instruction else "."))
        render_shot(story, shot, use_ltx=use_ltx, voice=voice)
        story.emit("assistant", text=f"Shot {n} updated.")
        story.save()
        return story

    if any(k in low for k in ("render all", "render everything", "generate all", "make the video")):
        return render_all(story, use_ltx=use_ltx, voice=voice)
    if "export" in low or "stitch" in low or "final cut" in low:
        export(story)
        return story

    # Freeform post-approval: rework the storyboard with the new direction, keeping the
    # locked cast + world so characters stay consistent.
    story.emit("assistant", text="Reworking the storyboard with your new direction (cast kept)…")
    story.vision = (story.vision + ". " + msg).strip(". ")
    build_storyboard(story)
    return story


# ---------------------------------------------------------------------------
# Cost / time estimate — the "scope before you spend" preview.
# ---------------------------------------------------------------------------

def estimate(story):
    """Rough render-time estimate so the director knows the cost before committing.
    LTX is the dominant term on this rig (~6 min/shot); placeholder is seconds."""
    status = engines.engine_status()
    per_clip = 360 if status["clip"] == "ltx" else 4
    per_img = 12 if status["image"] == "comfyui" else 1
    shots = story.all_shots()
    n = len(shots)
    secs = n * (per_clip + per_img)
    # Final pass also runs a Flux-Klein face-lock (~80s) on each shot that has a cast
    # member with a reference sheet.
    if status.get("swap") == "flux-klein":
        secs += 80 * sum(1 for s in shots if _primary_cast(story, s))
    return {"shots": n, "seconds": secs, "human": _human_time(secs),
            "engines": status}


def _human_time(secs):
    if secs < 90:
        return f"{int(secs)}s"
    m = secs / 60
    if m < 60:
        return f"{m:.0f} min"
    return f"{m/60:.1f} hr"


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo
