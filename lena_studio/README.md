# lena_studio — Claude Co-Work as manager, overseeing Lena's content sub-agents

Turns a **content brief** into finished, reviewed, publish-ready output for the Lena AI
influencer, by having a **manager** delegate each piece of work to a registered
**sub-agent**, track it through a review gate, and hand off to the next stage. Built on
top of this repo's existing engines (relay_studio's Ollama brain, the ComfyUI/LTX stack
used by the other studios) using the same conventions as `relay_director` and
`channel_studio` — plain-dict state, JSON persistence, degrade-gracefully fallbacks.

**Content types are intentionally not decided yet.** This scaffold ships one working
example sub-agent (a caption/copy drafter) end to end, to prove the manager/registry/
review loop for real — swapping in image, video, or other content agents later is a
one-file change (see "Adding a new sub-agent" below).

```
brief ──► manager.intake() ──► manager.plan()  ──► manager.dispatch() ──► manager.review() ──► manager.advance()
          (Batch created)      (plan_batch        (each QUEUED task    (human/Claude       (QA sub-agents, then
                                 sub-agent breaks    runs its             approve /           publish sub-agents,
                                 brief into tasks)   registered agent)    reject /            over every APPROVED
                                                                          regenerate)         task) ──► archived
```

## Roles

**Claude Co-Work (the manager):** drives `manager.py` — creates batches, dispatches
work, reviews sub-agent outputs against the brief and the character profile, decides
approve/reject/regenerate, and calls `advance()` once a batch is ready. This module
*is* what "Claude Co-Work as manager" means operationally: a structured API to delegate
through, not ad hoc generation.

**Sub-agents (workers):** plain Python functions in `agents/`, each responsible for one
capability. They don't know about each other or about review — they take a task's
input, do the work (or degrade to a deterministic fallback if a backend is down), and
return an output dict.

**Dan (owner):** approves what actually ships (publish credentials, brand calls),
same as every other studio in this repo.

## Pieces

| File | Role |
|------|------|
| `character.json` | Lena's profile — identity/LoRA fields filled in from her trained LoRA (`lena_klein4b_v1`); editorial fields (`niche`, `brand_voice`, `platforms`, `content_pillars`, `safe_mode`, `themes`) are placeholders marked `"TBD"` — content types come later. |
| `session.py` | `Batch` + `Task` data model: stages, states, JSON persistence to `projects/<id>/batch.json`, an `emit()`/`on_event()` pub-sub (same pattern as `relay_director/session.py`), and an in-process registry so repeated calls share one live batch. |
| `manager.py` | The orchestrator. `intake()`, `plan()`, `dispatch()`, `review()`, `retry()`, `advance()` — one function per handoff point, each documented below. |
| `agents/base.py` | `AGENT_REGISTRY`, `register_agent()`, `AgentError`. The extension point — see "Adding a new sub-agent". |
| `agents/planner_agent.py` | PLAN stage — breaks a brief into tasks for the registered generate-stage sub-agents. LLM + deterministic fallback (fans out to every generate agent). |
| `agents/copy_agent.py` | GENERATE stage — the one working example: drafts a caption in Lena's brand voice. LLM + deterministic fallback. |
| `agents/qa_agent.py` | REVIEW stage (automated) — pre-flight check over approved outputs (empty/placeholder text) before `advance()` publishes. Not a substitute for the human/Claude per-task review gate. |
| `agents/publish_agent.py` | PUBLISH stage (stub) — logs what would ship; no real destination wired up yet (`LENA_PUBLISH_MODE=log` by default). |
| `projects/<id>/batch.json` | Per-batch state, written after every transition. |
| `history.jsonl` | Append-only log, one line per archived/published batch (mirrors `seedance_studio/history.py`). |

## The state machine (handoff points)

**Batch stage:** `intake → planning → generating → assembling → published | archived`

**Task status** (one per sub-agent call):
```
QUEUED ──dispatch──► RUNNING ──ok, needs_review──► NEEDS_REVIEW ──approve──► APPROVED ──► (counts toward advance)
                        │                               │──reject──► FAILED (terminal, notes attached)
                        │                               └─regenerate─► QUEUED (feedback attached to input)
                        └──ok, no review──► DONE
                        └──error──► QUEUED (auto-retry, attempts < max_attempts)
                                  └► FAILED (attempts exhausted — needs manager.retry())
```

Every arrow above is one `manager.py` function call. Nothing moves a batch or a task
except an explicit call — there's no background poller, so the manager (Claude, or a
scheduled job later) decides the pace.

## Error handling

- Every sub-agent call is wrapped in try/except inside `manager._run_task` /
  `manager._call` — a sub-agent raising `AgentError` (expected failure) or any other
  exception (bug, backend down) marks **that task** FAILED/retried; it never crashes
  the batch or other tasks.
- Auto-retry: a failed task requeues itself (`QUEUED`) up to `max_attempts` (default 2,
  set per sub-agent in `register_agent(...)`); once exhausted it's `FAILED` and needs a
  human/Claude to call `manager.retry()` after fixing the underlying cause.
- `manager.review(decision="regenerate")` is the *intentional* retry path — it attaches
  the reviewer's notes to `task["input"]["feedback"]` so the sub-agent's next attempt
  can address them (see `copy_agent.run` reading `task["input"].get("feedback")`).
- `manager.advance()` refuses to run (raises `AgentError`) while any task is
  `QUEUED`/`RUNNING`/`NEEDS_REVIEW` — a batch can't skip review on the way to publish.
- Every backend a sub-agent might call (Ollama via `relay_studio.llm`, later ComfyUI/LTX)
  is behind a lazy import + try/except with a deterministic fallback, same philosophy as
  every other studio in this repo — the manager always has *something* to review, even
  with no LLM/GPU running.
- Per-batch `threading.Lock` (`manager._lock_for`) serializes `dispatch()`/`advance()`
  calls on the same batch, so two callers can't race a shared GPU/backend.

## Adding a new sub-agent

No changes to `manager.py` are needed. To add, say, an image-generation sub-agent:

```python
# lena_studio/agents/image_agent.py
from .base import register_agent

def run(batch, task):
    brief = task["input"].get("brief") or batch.brief
    # ... call ComfyUI, or degrade to a placeholder if it's not running ...
    return {"image_path": "...", "brief": brief}

register_agent("draft_image", "generate", "Render a still image for the brief.",
               run, needs_review=True, max_attempts=2)
```

Then add one line to `agents/__init__.py`: `from . import image_agent  # noqa: F401`.

The planner will start including it in fan-out plans automatically (it reads
`by_stage("generate")` at plan time), and the manager will dispatch/review/retry it
exactly like `draft_copy` — same registry, same state machine, same error handling.

## Monitoring / review, in practice (what Claude does each session)

```python
from lena_studio import manager

batch = manager.intake("cozy morning routine, soft light", channel="lena")
manager.plan(batch)                       # -> tasks queued for generate-stage agents
manager.dispatch(batch)                   # -> runs them, most land at NEEDS_REVIEW

for t in batch.tasks_by_status("NEEDS_REVIEW"):
    print(t["agent"], t["output"])        # Claude reviews against the brief/character
    manager.review(batch, t["id"], "approve")   # or "reject" / "regenerate" with notes

manager.advance(batch, dry_run=True)      # QA + publish-stub, then archived
print(batch.stage, batch.counts())
```

`manager.status(batch)` / `GET /promptrelay/lena/batch/{id}` return the full batch
(tasks, outputs, errors, event log) for monitoring at any point.

## HTTP API (mounted at `/promptrelay/lena/*` once ComfyUI loads this node pack)

| Method + path | Does |
|---|---|
| `GET /promptrelay/lena/agents` | List the sub-agent registry. |
| `GET /promptrelay/lena/batches` | List batches (newest first). |
| `POST /promptrelay/lena/batch` `{brief, channel}` | Intake + auto-plan a new batch. |
| `GET /promptrelay/lena/batch/{id}` | Full batch state. |
| `POST /promptrelay/lena/batch/{id}/dispatch` `{task_id?}` | Run QUEUED task(s). |
| `POST /promptrelay/lena/batch/{id}/review` `{task_id, decision, notes?}` | approve\|reject\|regenerate. |
| `POST /promptrelay/lena/batch/{id}/advance` `{dry_run?}` | QA + publish + archive. |
| `POST /promptrelay/lena/batch/{id}/retry` `{task_id}` | Force a FAILED task back to QUEUED. |

Handlers are synchronous JSON today (the shipped agents are fast/text-only). Once a slow
sub-agent (image/video render) is added, wrap its dispatch route in the same SSE
`_stream()` helper `relay_director/__init__.py` uses — the `Batch.emit()`/`on_event()`
pub-sub is already wired for it.

## Run it

```bash
# no ComfyUI, no Ollama needed — deterministic fallbacks the whole way:
python -c "
from lena_studio import manager
b = manager.intake('a quiet morning routine', channel='lena')
manager.plan(b); manager.dispatch(b)
for t in b.tasks_by_status('NEEDS_REVIEW'):
    manager.review(b, t['id'], 'approve')
manager.advance(b, dry_run=True)
print(b.stage, b.counts())
"

# check the registry:
python -c "from lena_studio import agents; import json; print(json.dumps(agents.list_agents(), indent=2))"
```

Restart ComfyUI to pick up the HTTP routes (mounted at `/promptrelay/lena/*`, printed to
console on load, same as the other studios).

## Status

- ✅ **Manager state machine** (intake → plan → generate → review → advance → archive) — working end to end, verified with no ComfyUI/Ollama running (deterministic fallbacks)
- ✅ **Sub-agent registry + extension pattern** — working; adding a sub-agent is a one-file, zero-manager-change addition
- ✅ **Error handling** (per-task retry, review-driven regenerate, manual retry, batch lock) — implemented and exercised by the smoke test above
- ✅ **HTTP API** — routes registered at `/promptrelay/lena/*`; not yet exercised against a live ComfyUI/PromptServer process
- ✅ **One working example sub-agent** (`draft_copy`) — proves the loop; not a committed content type
- ⬜ **Real content sub-agents** (image/video/etc.) — deliberately deferred, per the brief this scaffold was built from
- ⬜ **Real publish destination** — `publish_stub` only logs; needs a platform decision + credentials, same pattern as `channel_studio/publisher.py`
- ⬜ **Dashboard UI** — Phase 2, same as `channel_studio`'s
- ✅ **`character.json` technical fields** (`base_model`/`text_encoder`/`vae`/`lora`) — confirmed against Lena's own AI-Toolkit training config (`ai-toolkit/app/output/lena_klein4b_v1/config.yaml`, base model `black-forest-labs/FLUX.2-klein-base-4B`) and her validated ComfyUI workflow (`workflows/Lena/Lena_Character.json`, which loads `flux-2-klein-4b-fp8.safetensors` / `qwen_3_4b.safetensors` / `flux2-vae.safetensors` / `lena_klein4b_v1.safetensors` at strength 1.0) — same Flux2 Klein stack as Mira's, but verified from Lena's own files, not assumed from the pattern
