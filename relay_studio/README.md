# relay_studio — the "Dolly" agent brain

A drama.land-style agent pipeline for PromptRelay. A **super-agent (Dolly)** reads a
structured *user-inputs* artifact + a free-text request, decomposes it into a fixed
video pipeline, and drives **one tool per stage** via local LLM function-calling —
streaming its thinking, task progress and a growing media library, then ffmpeg-mixes
a final talking-influencer video.

```
request + user-inputs ─► Dolly (qwen3, tool-calling) ─► tools ─► media library ─► final.mp4
                              │
   script ─ synthesize_speech ─ generate_music(intro/outro) ─ lipsync ─ fetch_sfx ─ final_mix
```

## Pieces

| File | Role |
|------|------|
| `agent.py` | The super-agent: system prompt, tool-calling loop, event emission, **deterministic completion fallback** so a video is always produced. |
| `llm.py` | Ollama transport (`/api/chat` with `tools`). Model via `STUDIO_MODEL` env (default `qwen3`). |
| `tools/` | One module per pipeline stage; `TOOL_REGISTRY` exposes JSON schemas to the LLM. |
| `session.py` | Project state: user-inputs, task list, media library, event log; persisted to `projects/<id>/`. |
| `ffmpeg.py` | ffmpeg helpers via the `imageio_ffmpeg` bundled binary (no system ffmpeg needed). |
| `assets.py` | Bundled **placeholder** music/SFX/photo, synthesised offline on first run. |
| `__init__.py` | `register_routes()` — mounts the HTTP API + UI on ComfyUI's PromptServer. |
| `run.py` | Headless CLI runner. |
| `_standalone.py` | Dev server that runs the API + UI **without** ComfyUI. |

## Run it

**Headless (no ComfyUI, no UI):**
```
python -m relay_studio.run relay_studio/sample_inputs.json          # full LLM-driven run
python -m relay_studio.run relay_studio/sample_inputs.json --no-llm # deterministic pipeline
```

**Inside ComfyUI:** the node auto-registers routes. Open
`http://127.0.0.1:8188/promptrelay/studio/ui`.

**Standalone UI dev server (no ComfyUI):**
```
python -m relay_studio._standalone        # http://localhost:8189/promptrelay/studio/ui
```

## HTTP API (`/promptrelay/studio/*`)
- `GET  /home` — the Explore landing (brand shell, masonry gallery, global composer)
- `GET  /projects` — JSON list of saved projects (for the gallery)
- `GET  /ui` — the 3-panel workspace (`?id=<id>` to open one, `&autostart=1` to auto-run)
- `GET  /voices` — voice library
- `POST /project` `{inputs, request}` → `{id, state}`
- `GET  /project/{id}` — full state
- `POST /inputs/{id}` `{inputs}` — update the artifact
- `POST /chat/{id}` `{message}` — **SSE** stream of agent events
- `GET  /media/{id}/{file}` — serve generated media
- `GET  /models` — available Ollama models

## What's real vs. stub (v1)
- **Real:** the agent brain (qwen3 tool-calling), the task/media/event engine, the full
  ffmpeg assembly (intro → talking clip w/ BGM bed + SFX → outro), image-gen via ComfyUI.
- **Stub, swappable:** `lipsync` (ken-burns motion, not mouth-synced), `generate_music`
  (mood-matched tone bed), `fetch_sfx` (noise burst), `synthesize_speech`
  (edge-tts → pyttsx3 → tone fallback).

## Upgrading the stubs
- **Voice:** `pip install edge-tts` (online, high quality) — used automatically before pyttsx3.
- **Music/SFX:** drop real royalty-free files into `assets/music/` and `assets/sfx/`. Keep the
  mood keyword (e.g. `jazz`) or SFX name (e.g. `cup_falling`) in the filename — the lookups
  in `assets.py` prefer real files over the synthesised placeholders.
- **Lipsync:** replace the body of `tools/lipsync.py` with a SadTalker / ComfyUI lipsync
  call (keep the `(project, audio, image, service)` signature and `add_media("video", …)`).
- **Brain:** point `OLLAMA_URL` / `STUDIO_MODEL` at another model, or swap `llm.chat` for a
  cloud client (keep the returned `message` shape: `content` / `thinking` / `tool_calls`).
```
