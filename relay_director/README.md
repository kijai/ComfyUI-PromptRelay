# relay_director — OpenArt-style "vibe directing" studio

A director workspace inspired by OpenArt's AI Creator Studio. You describe a **vision**
in natural language; the Director scopes it into **Scenes → Shots**, you refine each
shot, render, and export a widescreen timeline. Independent of Dolly (`relay_studio`) —
it only *reuses Dolly's engines read-only* and never modifies it.

```
vision ─► develop ─► script + CAST (locked looks) + WORLD ─► [approve] ─► reference sheets
                                                                  │
                                                                  ▼
                          cast-aware Scenes → Shots ─► per-shot: image(+cast+world) → clip → (voice) ─► Timeline → final.mp4
                                    ▲ refine any shot with @shot N
```

**Consistency model:**
- **Tier A (every render):** each shot's image prompt is composed from the shot + the
  locked descriptions of the cast members in it + the world style
  (`director._compose_shot_prompt`), so characters/look stay consistent.
- **Tier B (Final pass only):** when **🎬 Final (LTX + face-lock)** is on, each shot's
  primary cast member's reference sheet is swapped onto the base image via **Flux 2 Klein**
  (`flux_klein_engine.py` + `flux_klein_api.json`), locking true identity, then LTX animates
  the result. Slow (~80s/shot swap + LTX). Falls back to the un-swapped base image if Flux
  Klein is unavailable. LoRA strength (turn2real, default 0.9) is the fidelity lever.

## Pieces
| File | Role |
|------|------|
| `session.py` | `Story → Scene → Shot` data model, persistence (`projects/<id>/story.json` + media), event/listener stream. Self-contained (stdlib only). |
| `director.py` | The brain: `scope` (Ollama planner + deterministic fallback), `render_shot`/`render_all`, `export`, `chat` (intent routing + `@shot N` re-roll), `estimate` (render-time preview). |
| `engines.py` | Generation with graceful fallback: image (ComfyUI Z-Image → gradient card), clip (LTX i2v → ken-burns), voice (edge-tts → silent), timeline concat. 16:9/9:16/1:1 aware. |
| `__init__.py` | `register_routes()` — mounts the HTTP API + UI on ComfyUI's PromptServer. |
| `../web/director.html` | The studio UI: director chat, Shot/Scene/Timeline views, per-shot re-roll, render-plan badges, export. |

## Run it
Inside ComfyUI (auto-registers on node load — **restart ComfyUI to pick up the routes**):
```
http://127.0.0.1:8188/promptrelay/director
```

## HTTP API (`/promptrelay/director/*`)
- `GET  /` , `/ui` — the studio
- `GET  /stories` — gallery summaries · `GET /engines` — what's available now
- `POST /story` `{vision, aspect}` → `{id, state}`
- `GET  /story/{id}` · `POST /aspect/{id}` `{aspect}`
- `POST /scope/{id}` `{vision}` — **SSE** develop the script + cast + world (approval gate)
- `POST /approve/{id}` — **SSE** generate cast reference sheets + build the storyboard
- `POST /character/{id}/ref` `{char_id}` — **SSE** (re)generate one cast reference sheet
- `POST /character/{id}/update` `{char_id, name?/role?/description?}` · `POST /world/{id}` `{style?/setting?/palette?}`
- `POST /chat/{id}` `{message}` — **SSE** (handles `@shot N`, render/export intents, re-scope)
- `POST /render/{id}` `{only?}` — **SSE** render all (or a subset)
- `POST /shot/{id}/render` `{shot_id, reroll?}` — **SSE** one shot
- `POST /shot/{id}/update` `{shot_id, prompt?/motion?/dialogue?/duration?}`
- `GET  /estimate/{id}` — render-time preview
- `POST /export/{id}` — **SSE** stitch timeline → `final.mp4`
- `GET  /media/{id}/{file}` — serve generated media

## Real vs. fallback
- **Real when available:** Ollama story planning, ComfyUI Z-Image stills, LTX i2v clips,
  edge-tts voice, bundled-ffmpeg timeline export.
- **Always-works fallback:** built-in beat planner, gradient placeholder stills,
  ken-burns clips, silent audio — so the full flow runs on any rig.

> Note: the engine badges/estimate detect availability by *module import*, not live
> server reachability — if ComfyUI/Ollama aren't serving, rendering still succeeds via
> the fallbacks. Swap engines via `engines.py` (keep the function signatures).
