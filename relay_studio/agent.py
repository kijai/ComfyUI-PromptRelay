"""The super-agent ("Dolly") — plans the video and drives the pipeline tools.

Mirrors drama.land: read the user-inputs artifact + request, decompose into the fixed
pipeline (script -> voice -> BGM master+trim -> lipsync -> SFX -> final mix), and call a
tool per stage via Ollama function-calling. Tools write their outputs back into the
versioned artifact (TTS_AUDIO_URL / BGM_URL / SFX_*_URL / TALKING_VIDEO_URL /
FINAL_VIDEO_URL, bumping v1 -> v2 -> ...). Every step emits an event so the UI can show
thinking, BASH-style tool blocks with timing, task progress, async job notifications and
the growing media library.

If the local model stalls before the final cut, `ensure_complete` finishes the
remaining stages deterministically so a video is always produced."""

import json
import time

from . import llm, tools

MAX_ITERS = 20

SYSTEM_PROMPT = """You are Dolly, a playful AI video producer (you're a cat — fun, upbeat tone). \
You have access to a set of tools. ALWAYS call tools using the tool_calls mechanism — \
NEVER output JSON, code blocks, or tool syntax as plain text.

Read the user request carefully and decide which pipeline to run:

=== IMAGE REQUEST (user asks for portrait, image, photo, picture) ===
Call: generate_image(prompt="<vivid description of the character>", width=768, height=768)
Then stop and report the result warmly.

=== VIDEO REQUEST (default — talking influencer video) ===
Call these tools in order, one at a time:
1. synthesize_speech(text="<1-3 sentence fun cat script>", voice_name, voice_language)
2. generate_music(mood="<jazz|upbeat|calm|happy>", duration=60)
3. trim_music(role="intro", duration=4)
4. trim_music(role="outro", duration=4)
5. lipsync(service="<from inputs>")
6. fetch_sfx(name="cup_falling")
7. final_mix()
Then give a short upbeat wrap-up and stop.

Rules:
- ONLY use tool_calls. Never write JSON or pseudocode — call the actual tool.
- Call ONE tool per turn; wait for the result before calling the next.
- Reuse media_id values from tool results when calling later tools.
- Be concise between tool calls."""


def _emit_thinking(project, msg):
    th = (msg.get("thinking") or "").strip()
    if th:
        project.emit("thinking", text=th)


def _dispatch_tool(project, name, args):
    """Run one tool, manage task state, emit events (with timing). Returns the result."""
    project.emit("tool_call", name=name, args=args)
    if name == "synthesize_speech":
        project.set_task("script", "done")
    t0 = time.time()
    result, task = tools.call(name, project, args)
    elapsed = round(time.time() - t0, 2)
    if task:
        project.set_task(task, "error" if result.get("error") else "done")
    project.emit("tool_result", name=name, result=result, elapsed=elapsed, steps=1)
    return result


def run(project, message):
    """Drive the pipeline for one user message. Streams events via project.emit."""
    project.add_message("user", message)
    project.set_task("script", "running")
    project.emit("status", text="Dolly is planning your video…")

    inputs_block = json.dumps(project.inputs, indent=2)
    user_content = (f"User request: {message}\n\n"
                    f"user-inputs artifact (v{project.ver}):\n{inputs_block}\n\n"
                    f"Produce the video by calling the pipeline tools in order.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}]

    final_text = ""
    for _ in range(MAX_ITERS):
        try:
            msg = llm.chat(messages, tools=tools.schemas())
        except Exception as e:
            project.emit("error", text=f"LLM error: {e}")
            break
        _emit_thinking(project, msg)
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            final_text = (msg.get("content") or "").strip()
            if final_text:
                project.emit("assistant", text=final_text)
            break

        for tc in calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            result = _dispatch_tool(project, name, args)
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result)[:2000]})

    fb = ensure_complete(project, message)
    if not final_text and not fb:
        final_text = "Your video is ready! 🎬"
        project.emit("assistant", text=final_text)
    if final_text:
        project.add_message("assistant", final_text)
    project.save()
    return project


# ----------------------------------------------------------------------------
# Deterministic completion — fills any pipeline stage the model skipped so a final
# video is always produced.
# ----------------------------------------------------------------------------

def _has(project, predicate):
    return any(predicate(m) for m in project.media)


def _role(project, role):
    return _has(project, lambda m: m.get("meta", {}).get("role") == role)


def _wants_image(message, project):
    req = ((message or "") + " " + (project.inputs.get("requested_output") or "")).lower()
    return any(k in req for k in ("image", "portrait", "photo", "picture", "generate image", "draw"))


def ensure_complete(project, message=""):
    """Run any missing pipeline stages with sensible defaults. Returns True if it
    had to step in."""
    stepped = False
    ins = project.inputs
    cat = ins.get("cat_name") or "Whiskers"
    show = ins.get("show_title") or "the show"
    episode = ins.get("episode") or "today's episode"

    # Image request — skip the video pipeline and generate an image instead.
    if _wants_image(message, project):
        already = _has(project, lambda m: m["kind"] == "image" and m.get("via") == "comfyui")
        if not already:
            stepped = True
            project.emit("status", text="Auto-completing: generating portrait image")
            prompt = (message or f"portrait photo of {cat}, influencer style, soft studio lighting")
            _dispatch_tool(project, "generate_image", {"prompt": prompt, "width": 768, "height": 768})
        return stepped

    if "TTS_AUDIO_URL" not in project.outputs:
        stepped = True
        project.emit("status", text="Auto-completing: voice")
        line = f"Hi, I'm {cat}! Welcome to {show}. {episode} — let's get into it!"
        _dispatch_tool(project, "synthesize_speech",
                       {"text": line, "voice_name": ins.get("voice_name", ""),
                        "voice_language": ins.get("voice_language", "")})

    mood = _guess_mood(message)
    if "BGM_URL" not in project.outputs:
        stepped = True
        project.emit("status", text="Auto-completing: BGM master")
        _dispatch_tool(project, "generate_music", {"mood": mood, "duration": 60})
    if not _role(project, "intro"):
        stepped = True
        project.emit("status", text="Auto-completing: BGM intro snippet")
        _dispatch_tool(project, "trim_music", {"role": "intro", "duration": 4})
    if not _role(project, "outro"):
        stepped = True
        project.emit("status", text="Auto-completing: BGM outro snippet")
        _dispatch_tool(project, "trim_music", {"role": "outro", "duration": 4})

    if "TALKING_VIDEO_URL" not in project.outputs:
        stepped = True
        project.emit("status", text="Auto-completing: lipsync")
        _dispatch_tool(project, "lipsync", {"service": ins.get("lipsync_service", "")})

    if not _has(project, lambda m: m["kind"] == "audio" and m["file"].startswith("sfx_")):
        stepped = True
        project.emit("status", text="Auto-completing: SFX (cup_falling)")
        _dispatch_tool(project, "fetch_sfx", {"name": "cup_falling"})

    if "FINAL_VIDEO_URL" not in project.outputs:
        stepped = True
        project.emit("status", text="Auto-completing: final mix")
        _dispatch_tool(project, "final_mix", {})

    final = next((m for m in project.media if m.get("meta", {}).get("final")), None)
    if final:
        project.emit("final", media_id=final["id"], file=final["file"], label=final["label"])
    return stepped


def _guess_mood(text):
    text = (text or "").lower()
    for kw in ("jazz", "upbeat", "calm", "tense", "happy"):
        if kw in text:
            return kw
    return "jazz"
