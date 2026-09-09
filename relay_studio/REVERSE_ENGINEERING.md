# drama.land "Dolly" — Reverse-Engineering Spec

**Goal:** capture the *observable contract* of drama.land's hosted agent ("Dolly") so
`relay_studio` can replicate its behaviour faithfully.

**Method / honesty caveat:** drama.land is a closed hosted SaaS. We cannot recover its
source, prompts, model weights, or infra. This spec is built from (a) two user screenshots
of a live "talking influencer" project, and (b) drama.land's public marketing/FAQ/tools
pages. Items are tagged **[obs]** = directly observed, **[doc]** = from public docs,
**[inf]** = inferred. "100%" here means 100% of the *externally observable* behaviour.

---

## 1. Product context [doc]
drama.land is an AI video platform (music videos, explainer, cinema). Documented tools:
- **Lip Sync** — "make any portrait talk or sing naturally"
- **AI Voices** — "natural AI voices in multiple languages"
- **Music Studio** — AI music generation; **Sound Design** — "SFX, ambience, music"
- **Image Studio** + **GPT-2-Image**
- Video models (Premium): **Seedance 2.0**, **Kling 3.0 Pro**, **HappyHorse**, **Grok Imagine**
- **Motion Sync** — recreate movement from reference video
- "Closed-loop workflow: Idea → shared generation (music+video) → automatic sync → finished MV"

The **"hosted"** experience (URL `drama.land/hosted?projectId=<uuid>`) is an **agentic
orchestration layer** ("Dolly") that calls these tools for you. That agent layer is what
we clone.

## 2. UI structure [obs]
Three columns:
1. **Chat (Dolly)** — conversation, plan/TODO, collapsible thinking/step blocks, a task
   counter (`Task (2/3)`), composer ("Got any ideas, talk to Dolly….", with +attach, quote, send).
2. **Media** — tabs `All 46 / Videos 1 / Images 1 / Audio 44`; each asset card has an inline
   player/thumbnail, name, `type: text`, provenance `via bash`, timestamp, bookmark + download.
3. **user-inputs artifact** — a typed, **versioned** key/value panel (`VER v1`, later `VER v2`),
   with an add-field (`+`) control and a comment affordance.
Top bar: Home, credit balance (`168 STARTER`), Share.

## 3. The agent ("Dolly") [obs]
- **Persona:** playful cat; casual quips ("Like dumplings — done when they float.",
  "Feeling great — could hop all day.", "BGM's locked in.").
- **Planning:** emits an emoji TODO and narrates intent
  ("here's the plan for what happens next" → 1. BGM 2. Lipsync 3. SFX 4. Final mix),
  then **reports incrementally** as each piece lands ("I'll report back as each piece lands").
- **Tool transparency:** collapsible blocks `1 steps` and `Worked for 1.1s · 1 steps`
  (shows wall-clock duration + step count per tool turn). [obs]
- **Thinking:** a THINKING block is shown, incl. self-notes like *"I should update the
  user-inputs artifact with the BGM URL and update the todo."* [obs, screenshot 1]
- **Bash tool:** assets are tagged `via bash`; ffmpeg assembly is described as the final
  mix → Dolly runs shell/ffmpeg via a bash tool. [obs/inf]

## 4. Data model

### 4.1 user-inputs artifact (versioned, mutable, input **and** output manifest) [obs]
Initial fields (v1):
```
cat_name        = Whiskers
show_title      = The Cat Chat Show
episode         = Why Humans Are So Weird
cat_photo_url   = <image>            (typed: image, with thumbnail + download)
voice_name      = Anthony
voice_description = Decent Young Man - Magnetic, lazy, playful wit
voice_language  = EN_British
lipsync_service = dlai2v_pro
requested_output = talking influencer video
```
As the pipeline runs, Dolly **writes outputs back into the same artifact** and bumps the
version (v1 → **v2**), adding typed **audio** fields rendered as inline players:
```
BGM_URL          = <audio 2:39>
SFX_CUP_FALL_URL = <audio 0:05>
TTS_AUDIO_URL    = <audio 2:23>
```
**Key insight:** the artifact is the single source of truth and a living manifest — not a
static input form. Field types are inferred from value (text / image-url / audio-url) and
audio/image render with players/thumbnails + bookmark + download.

### 4.2 Media library [obs]
Every asset (generated or fetched) is registered with: kind (video/image/audio), name,
provenance (`via bash`), a `type: text` tag [inf: tool/source that produced it], timestamp,
inline player, bookmark, download. The 44-strong audio bank (e.g. `Chinese_Mandarin_*`,
`English_DecentBoy.mp3`, `Robot_Armor.mp3`, `English_DecentYoung*`) is the **voice/TTS
sample library** the `voice_name` is chosen from.

### 4.3 Storage [obs]
Assets live in cloud storage: `https://storage.googleapis.com/dramaland-public/ugc_media/<date>/<id>.mp3`.
Asset ids are short hex (e.g. `e8a43fce`) and referenced inline in chat (``BGM `e8a43fce` ``).

## 5. The pipeline — "talking influencer" recipe [obs]
1. **TTS** — synthesize the cat's narration → `TTS_AUDIO_URL` (~2:23). Voice chosen from the
   library by `voice_name`/`voice_language`/`voice_description`.
2. **BGM** — generate a "Playful jazz theme (30–90 sec)" (asset `e8a43fce`) → `BGM_URL` (2:39),
   then **trim a short intro snippet + outro snippet from that one track**.
3. **Lipsync** — `cat_photo_url` + TTS audio → talking video via the lipsync service
   (`lipsync_service = dlai2v_pro`, model **"longcat-avatar"**). **Long async render, ~1–3 min.**
4. **SFX** — fetch `drum_reverb_1` from the asset library → `SFX_CUP_FALL_URL` (0:05) for the
   comedic "cup falling / oops" moment.
5. **Final mix (ffmpeg)** — exact recipe, verbatim:
   > 3s jazz intro → Whiskers talking with BGM underneath → comedic drum hit for the "oops"
   > cup moment → jazz outro fade
   producing a single `video.mp4`.

## 6. Async task / notification model [obs]
- Long tools run as **background jobs**; chat continues meanwhile ("the lipsync job is still
  rendering … Once that lands, I'll assemble the final video").
- On completion a **system notification** fires: `系统通知 · 后台任务完成 (E8A43FCE…)`
  ("System notification · background task completed", with the job id).
- A **task counter** (`Task (2/3)`) tracks overall progress; each completed job updates it
  and triggers the next dependent step + an artifact writeback.

## 7. Models / services / infra [obs/doc]
- Lipsync: `dlai2v_pro` service / `longcat-avatar` model [obs]
- Video (platform): Seedance 2.0, Kling 3.0 Pro, HappyHorse, Grok Imagine [doc]
- Image: GPT-2-Image / Image Studio [doc]
- Voices: multi-language TTS library [doc/obs]
- Music + Sound Design (SFX) generators [doc]
- Infra: GCS storage, async job queue + notifications, credit system (`STARTER` tier, credits per gen) [obs]

---

## 8. Gap analysis — relay_studio v1 vs drama.land

| Capability | drama.land | relay_studio v1 | Action |
|---|---|---|---|
| 3-panel UI (chat/media/artifact) | ✓ | ✓ | polish |
| Dolly persona + emoji plan + incremental reporting | ✓ | partial (prompt) | tighten prompt |
| **Versioned artifact (v1→v2)** | ✓ | ✗ (static inputs) | **ADD** |
| **Artifact writeback** of output URLs (BGM_URL/TTS_AUDIO_URL/SFX_CUP_FALL_URL) | ✓ | ✗ (media library only) | **ADD** |
| Typed artifact fields w/ inline audio players + download | ✓ | ✗ | **ADD** |
| **Async background jobs + completion notifications** | ✓ | ✗ (synchronous) | **ADD (simulated)** |
| Task (n/N) counter | ✓ | ✓ | align labels |
| Tool "Worked for Xs · N steps" timing | ✓ | tool blocks, no timing | **ADD timing** |
| Media provenance + download + bookmark | ✓ | cards only | **ADD download/provenance** |
| **Voice library** to choose voice_name | ✓ (44 samples) | ✗ | **ADD picker** |
| BGM = one track trimmed into intro+outro | ✓ | two separate gens | align to single-source trim |
| Lipsync (long async, longcat-avatar) | ✓ | stub ken-burns (sync) | keep stub, make async |
| Final-mix recipe (3s intro→talk+bed→drum hit→outro fade) | ✓ | ✓ (equivalent) | match timing/fade |
| Cloud storage URLs | GCS | local files | local equivalent OK |
| Credits / tiers | ✓ | ✗ | optional |

> **Status (implemented + verified):** gaps 1–6 below are all DONE — versioned artifact
> writeback (v1→v6), async job model + 系统通知 notifications, tool step-timing, voice
> library, single-source BGM trim, and media download/provenance. Verified headless and
> through the browser UI. Remaining optional: cloud-storage URLs and credits/tiers.

## 8b. Home / Explore shell + design system [obs, 3rd screenshot]
The product has TWO surfaces: the **Explore/Home** landing and the **project workspace**
(the 3-panel Dolly view in §2). Home page observations:
- **Brand:** `Drama.Land` wordmark (stacked logo mark + italic serif wordmark, red underline).
- **Top bar:** left logo; right = row of red circular quick-action icons, credit chip
  `🚀 203 STARTER`, `Pricing` pill, user avatar. Invite toast: "Share invite links, earn
  credits! Task Center → Invite".
- **Left nav rail (vertical, pill container):** Home (active, red), Create/Image, Avatar,
  Tools (`NEW` badge).
- **Hero:** large cream serif title ("The Drama.Lands").
- **Masonry gallery** of creations: rounded thumbnail cards, translucent dark pill tags
  top-left (`Music MV`, `Story`, `anime stories`, `Old Stories`, `Sci-Fi Stories`,
  `Cat Tax`, `Performance`, `SINGLE`), `Show More` button.
- **Global composer** (bottom-center pill): `+ Generate a viral TikTok-style outfit` + send.
- **Footer:** About / Blog (left); X / YouTube / Discord + Share (right).

**Design tokens [obs/inf]:** bg `#000`; heading = cream serif (~Instrument Serif look),
`#EDE7D9`; accent red `~#E11D2A`; body sans `#E7E7EA` / dim `#8A8A93`; cards radius ~16px,
translucent dark tag pills; generous black negative space.

## 9. Recommended changes to reach fidelity (priority order)
1. **Versioned artifact + writeback** — make the artifact mutable; after each stage Dolly
   writes `TTS_AUDIO_URL` / `BGM_URL` / `SFX_*_URL` / `FINAL_VIDEO_URL` and bumps `ver`.
   Add an `update_artifact(field, media_id)` tool the agent must call. Render typed fields
   (text/image/audio) with players + download in the right panel.
2. **Async job model** — run lipsync/music/TTS as background jobs that emit a
   `job_started` → later `job_done` (system-notification) event; agent proceeds when each lands.
3. **Tool step timing** — wrap each tool turn with start/stop and emit `Worked for Xs · N steps`.
4. **Voice library** — ship a small voice-sample bank; `voice_name` selects from it; expose a
   `list_voices`/`pick_voice` affordance.
5. **BGM single-source trim** — generate one BGM track, then trim intro+outro snippets from it.
6. **Media polish** — per-asset download + provenance (`via <tool>`) in the media panel.
7. (Optional) credit counter + share/export.
