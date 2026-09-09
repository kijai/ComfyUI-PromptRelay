"""GENERATE stage — drafts a short caption/copy blurb for a piece of Lena content.

This is the one fully-wired example sub-agent in the scaffold: it proves the manager ->
registry -> agent -> review loop end to end without committing to any specific content
type (image/video/etc — TBD). Uses relay_studio's Ollama brain when available; falls
back to a deterministic template so a batch never stalls waiting on the LLM.
"""

try:
    from relay_studio import llm  # reuse Dolly's Ollama transport, read-only
except Exception:  # pragma: no cover - allows standalone import without relay_studio
    llm = None

from .base import register_agent

SYSTEM = """You write short social captions for an AI influencer character. Given the
character's brand voice and a content brief, write ONE caption (1-3 sentences, no
hashtags unless the brief asks for them) in her voice. Return plain text only — no
quotes, no markdown, no preamble."""


def run(batch, task):
    brief = task["input"].get("brief") or batch.brief
    feedback = task["input"].get("feedback")
    character = batch.character or {}
    name = character.get("display_name") or character.get("name") or batch.channel
    voice = character.get("brand_voice") or character.get("visual_style") or "warm, natural, first person"

    text = None
    if llm is not None:
        try:
            user = f"Character: {name}\nBrand voice: {voice}\nBrief: {brief}"
            if feedback:
                user += f"\nReviewer feedback on the last draft — address this: {feedback}"
            resp = llm.chat(messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}])
            text = (resp.get("content") or "").strip()
        except Exception:
            text = None

    if not text:
        base = brief.strip() or "a quiet moment, just for you."
        text = f"{name}: {base}" if not feedback else f"{name} (revised): {base}"

    return {"caption": text, "brief": brief}


register_agent(
    "draft_copy", "generate",
    "Draft a short caption/copy blurb in the character's brand voice.",
    run, needs_review=True, max_attempts=2,
)
