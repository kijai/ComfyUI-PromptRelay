"""Storyboard backend: character continuity engine, script parser + HTTP API.

Registers routes on ComfyUI's PromptServer:

    GET    /promptrelay/storyboard/state         current engine state + style presets
    POST   /promptrelay/storyboard/character     lock a character reference
    DELETE /promptrelay/storyboard/character     clear the lock
    POST   /promptrelay/storyboard/compile       compile (and optionally queue) shot workflows
    POST   /promptrelay/storyboard/parse_script  raw script text/file -> character shot payloads

The engine and parser are pure Python — run standalone without ComfyUI:

    python storyboard_engine.py                          # demo
    python storyboard_engine.py script.txt --aliases Maya --description "a woman in a yellow raincoat"
"""

import html
import json
import random
import re

# Each preset is a prompt prefix; "lora" is applied via LoraLoaderModelOnly when set.
STYLE_PRESETS = {
    "cinematic_hyper_realistic": {
        "prompt": "cinematic film still, hyper realistic, dramatic lighting, shallow depth of field, 35mm"},
    "storyboard_sketch": {
        "prompt": "storyboard sketch, rough pencil shading, monochrome with one accent color"},
    "concept_art": {
        "prompt": "detailed concept art, painterly, dramatic composition"},
    "noir": {
        "prompt": "film noir, high contrast black and white, hard shadows, venetian blind lighting"},
    "anime": {
        "prompt": "anime keyframe, cel shading, clean lineart, vivid colors"},
    "pixel_art": {
        "prompt": "pixel art style",
        "lora": "pixel_art_style_z_image_turbo.safetensors"},
}

# Used when folder_paths is unavailable (standalone run) or detection finds nothing.
FALLBACK_MODELS = {
    "unet": "z_image_turbo_bf16.safetensors",
    "clip": "qwen_3_4b.safetensors",
    "vae": "ae.safetensors",
    "dtype": "fp8_e5m2",
}

MAX_REFERENCE_INFLUENCE = 0.8  # below denoise 0.2 the output is just the reference

# Hard cap on the prompt handed to the text encoder. A pasted HTML page or other
# huge blob can otherwise blow past the encoder's token limit (e.g. 131072) and
# OOM the GPU. Storyboard shot prompts are single sentences, so this is generous.
MAX_PROMPT_CHARS = 2000


def _clamp_prompt(text):
    """Truncate an over-long prompt on a word boundary. Returns (text, truncated)."""
    if len(text) <= MAX_PROMPT_CHARS:
        return text, False
    cut = text[:MAX_PROMPT_CHARS]
    sp = cut.rfind(" ")  # back off to the last space so we don't slice mid-word
    if sp > MAX_PROMPT_CHARS // 2:
        cut = cut[:sp]
    return cut.rstrip(), True


def _detect_model(kind, prefer, fallback):
    try:
        import folder_paths
        names = folder_paths.get_filename_list(kind)
        for n in names:
            if prefer in n.lower():
                return n
        return names[0] if names else fallback
    except Exception:
        return fallback


def resolve_models(overrides=None):
    models = {
        "unet": _detect_model("diffusion_models", "z_image", FALLBACK_MODELS["unet"]),
        "clip": _detect_model("text_encoders", "qwen_3_4b", FALLBACK_MODELS["clip"]),
        "vae": _detect_model("vae", "ae.safetensors", FALLBACK_MODELS["vae"]),
        "dtype": FALLBACK_MODELS["dtype"],
        "lora": None,
        "lora_strength": 1.0,
    }
    for k, v in (overrides or {}).items():
        if v not in (None, "", "none"):
            models[k] = v
    return models


def build_workflow(prompt_text, seed, *, width, height, steps, models,
                   reference_image=None, denoise=1.0):
    """Z-Image Turbo text-to-image graph (API format), mirroring the workspace page."""
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": models["unet"], "weight_dtype": models["dtype"]}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": models["clip"], "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt_text}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "storyboard/shot"}},
    }
    if models.get("lora"):
        wf["11"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": models["lora"],
            "strength_model": float(models.get("lora_strength") or 1.0)}}
        wf["8"]["inputs"]["model"] = ["11", 0]
    if reference_image and denoise < 1.0:
        wf["12"] = {"class_type": "LoadImage", "inputs": {"image": reference_image}}
        wf["13"] = {"class_type": "ImageScale", "inputs": {
            "image": ["12", 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        wf["14"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}}
        wf["8"]["inputs"]["latent_image"] = ["14", 0]
        wf["8"]["inputs"]["denoise"] = round(denoise, 2)
    return wf


# ======================================================================
# Script parser: raw text -> sentences featuring the locked character
# ======================================================================

def _looks_like_html(text):
    head = (text or "").lstrip()[:512].lower()
    if head.startswith(("<!doctype", "<html", "<head")):
        return True
    # lots of angle-bracket tags relative to size -> treat as markup, not prose
    return text.count("<") > 20 and ("</" in text or "/>" in text)


def strip_html(text):
    """Reduce pasted HTML (e.g. a whole web page) to readable text so markup never
    reaches the prompt. <script>/<style>/<head> contents are dropped, tags removed,
    entities decoded, whitespace collapsed. Plain prose passes through untouched."""
    if not text or not _looks_like_html(text):
        return text
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", text)  # drop code blocks
    text = re.sub(r"(?s)<!--.*?-->", " ", text)                        # comments
    text = re.sub(r"(?s)<[^>]+>", " ", text)                           # remaining tags
    text = html.unescape(text)
    text = re.sub(r"[^\S\n]+", " ", text)         # collapse spaces/tabs, keep newlines
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)    # collapse blank-line runs
    return text.strip()


# "Scene 1:" / "Shot 3-5:" header lines (PromptRelay smart syntax) and screenplay slug lines.
_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z ]*?\s+\d+(?:\s*-\s*\d+)?\s*:\s*$")
_SLUG_RE = re.compile(r"^(INT|EXT|INT/EXT|EST)[.\s]", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_ABBREVS = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "no",
            "col", "gen", "lt", "sgt", "capt", "e.g", "i.e", "approx"}
# Person pronouns that can open a follow-up sentence about the character
# ("it" is deliberately excluded — it usually refers to objects).
_LEAD_PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs"}
# Ignored when deriving match keywords from a character description.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "with", "without", "in", "on", "of", "at",
    "to", "from", "for", "by", "as", "is", "are", "was", "were", "has", "have", "had",
    "this", "that", "these", "those", "its", "it", "very", "who", "whom", "whose",
    "wearing", "wears", "dressed", "style", "styled", "look", "looking", "looks",
    "he", "she", "they", "him", "her", "them", "his", "hers", "their",
    "year", "years", "old", "aged", "tall", "short", "small", "large", "big",
    "same", "recurring", "character", "every", "shot", "person", "one",
}


def split_into_sentences(text):
    """Split raw script text into sentences. Blank lines, `Scene N:` headers and
    screenplay slug lines act as hard breaks; common abbreviations don't split."""
    paragraphs, current = [], []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or _HEADER_RE.match(stripped) or _SLUG_RE.match(stripped):
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    sentences = []
    for para in paragraphs:
        for part in _SENT_SPLIT.split(para):
            part = part.strip()
            if not part:
                continue
            if sentences:
                tail = sentences[-1].rstrip(".").split()
                last = tail[-1].lower() if tail else ""
                # rejoin false splits after abbreviations and single initials ("J. Smith")
                if (last in _ABBREVS or re.search(r"(?:^|\s)[A-Z]\.$", sentences[-1])) \
                        and not _SENT_SPLIT.split(para)[0] == part == sentences[-1]:
                    sentences[-1] += " " + part
                    continue
            sentences.append(part)
    return sentences


def extract_match_terms(aliases=None, character=None):
    """Terms that identify the character in script text. Explicit aliases win;
    otherwise keywords are derived from the locked character description."""
    terms = {t.strip().lower() for t in (aliases or []) if t and t.strip()}
    if not terms and character:
        desc = character.get("character_description", "")
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", desc.lower()):
            if word not in _STOPWORDS:
                terms.add(word)
    return terms


def _sentence_matches(sentence, terms):
    low = sentence.lower()
    words = set(re.findall(r"[a-z'-]+", low))
    hits = []
    for t in terms:
        if (" " in t and t in low) or t in words:
            hits.append(t)
    return sorted(hits)


def isolate_character_sentences(text, terms, follow_pronouns=True, pronoun_window=2):
    """Return [{sentence, sentence_index, matched_terms, via_pronoun}] for sentences
    featuring the character. A sentence that opens with a person pronoun is kept if
    a character sentence occurred within the last `pronoun_window` sentences."""
    results = []
    last_included = None
    for i, sentence in enumerate(split_into_sentences(text)):
        matched = _sentence_matches(sentence, terms)
        via_pronoun = False
        if not matched and follow_pronouns and last_included is not None \
                and (i - last_included) <= pronoun_window:
            lead = re.findall(r"[a-z']+", sentence.lower())[:4]
            via_pronoun = any(w in _LEAD_PRONOUNS for w in lead)
        if matched or via_pronoun:
            results.append({"sentence": sentence, "sentence_index": i,
                            "matched_terms": matched, "via_pronoun": via_pronoun})
            last_included = i
    return results


class CharacterContinuityEngine:
    def __init__(self):
        self.character = None
        self.active_style_preset = "cinematic_hyper_realistic"

    def set_character_reference(self, reference_image=None, character_description="",
                                weight_modifier=0.85, aliases=None):
        """Lock the character used for every compiled shot.

        reference_image: filename in ComfyUI's input dir (from /upload/image) —
        rendered over via img2img. weight_modifier 0..1: how strongly shots stick
        to the reference (mapped to inverse sampler denoise). aliases: names the
        script parser uses to find this character in raw text.
        """
        self.character = {
            "reference_image": reference_image or None,
            "character_description": (character_description or "").strip(),
            "weight_modifier": min(max(float(weight_modifier), 0.0), 1.0),
            "aliases": [a.strip() for a in (aliases or []) if a and a.strip()],
        }
        return {"status": "Character properties locked successfully",
                "character": self.character}

    def clear_character(self):
        self.character = None
        return {"status": "Character cleared"}

    def compile_shot_payload(self, user_prompt, frame_number=1, *, seed=None,
                             width=1024, height=576, steps=8,
                             style_preset=None, style_text=None,
                             character=None, models=None):
        """Merge the scene prompt with the locked character + style and build the
        full ComfyUI workflow for one shot."""
        character = character if character is not None else self.character
        preset_name = style_preset or self.active_style_preset
        preset = STYLE_PRESETS.get(preset_name, {})
        style = style_text if style_text is not None else preset.get("prompt", "")

        parts = [style.strip(), user_prompt.strip()]
        reference_image, denoise = None, 1.0
        if character:
            desc = character.get("character_description", "")
            if desc:
                parts.append(f"same recurring character in every shot: {desc}")
            reference_image = character.get("reference_image")
            influence = min(float(character.get("weight_modifier", 0.0)),
                            MAX_REFERENCE_INFLUENCE)
            if reference_image and influence > 0:
                denoise = 1.0 - influence
        injected_prompt = ". ".join(p.strip().rstrip(".") for p in parts if p.strip())
        injected_prompt, prompt_truncated = _clamp_prompt(injected_prompt)

        if seed is None:
            seed = random.randrange(2 ** 48)
        models = resolve_models(models)
        if not models.get("lora") and preset.get("lora") and style_text is None:
            models["lora"] = preset["lora"]

        return {
            "frame_number": frame_number,
            "injected_prompt": injected_prompt,
            "prompt_truncated": prompt_truncated,
            "style_preset": preset_name if style_text is None else None,
            "seed": seed,
            "generation_parameters": {
                "width": width, "height": height, "steps": steps,
                "character_reference": reference_image,
                "denoise": round(denoise, 2),
                "models": models,
            },
            "workflow": build_workflow(injected_prompt, seed, width=width,
                                       height=height, steps=steps, models=models,
                                       reference_image=reference_image,
                                       denoise=denoise),
        }

    def parse_script_payloads(self, raw_text, *, aliases=None, character=None,
                              follow_pronouns=True, pronoun_window=2, max_shots=24,
                              seed=None, seed_lock=True, **gen):
        """The automated parser: raw script text -> array of formatted shot payloads.

        Isolates sentences featuring the locked character (or `character`/`aliases`
        overrides) and compiles each into a full shot payload. Extra keyword args
        (width/height/steps/style_preset/style_text/models) pass through to
        compile_shot_payload.
        """
        raw_text = strip_html(raw_text)  # a pasted web page -> readable text
        character = character if character is not None else self.character
        terms = extract_match_terms(aliases or (character or {}).get("aliases"),
                                    character)
        if not terms:
            raise ValueError("No character to match — lock a character "
                             "(or pass aliases) before parsing a script.")
        scenes = isolate_character_sentences(raw_text, terms,
                                             follow_pronouns=follow_pronouns,
                                             pronoun_window=pronoun_window)
        scenes = scenes[:max(1, int(max_shots))]
        if seed is None:
            seed = random.randrange(2 ** 48)
        payloads = []
        for i, scene in enumerate(scenes):
            payload = self.compile_shot_payload(
                scene["sentence"], frame_number=i + 1,
                seed=seed if seed_lock else seed + i,
                character=character, **gen)
            payload["source"] = {k: scene[k] for k in
                                 ("sentence_index", "matched_terms", "via_pronoun")}
            payloads.append(payload)
        return {"match_terms": sorted(terms), "scenes": scenes, "shots": payloads}


ENGINE = CharacterContinuityEngine()


def register_routes():
    from aiohttp import ClientSession, web
    from server import PromptServer

    routes = PromptServer.instance.routes

    async def queue_shots(request, shots, client_id):
        origin = str(request.url.origin())
        async with ClientSession() as session:
            for shot in shots:
                async with session.post(origin + "/prompt", json={
                        "prompt": shot["workflow"],
                        "client_id": client_id}) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        shot["prompt_id"] = result.get("prompt_id")
                    else:
                        shot["error"] = result.get("error", result)

    def gen_kwargs(data):
        return {
            "width": int(data.get("width", 1024)),
            "height": int(data.get("height", 576)),
            "steps": int(data.get("steps", 8)),
            "style_preset": data.get("style_preset"),
            "style_text": data.get("style_text"),
            "models": data.get("models"),
        }

    @routes.get("/promptrelay/storyboard/state")
    async def storyboard_state(request):
        return web.json_response({
            "character": ENGINE.character,
            "active_style_preset": ENGINE.active_style_preset,
            "style_presets": {k: v["prompt"] for k, v in STYLE_PRESETS.items()},
            "models": resolve_models(),
        })

    @routes.post("/promptrelay/storyboard/character")
    async def storyboard_set_character(request):
        data = await request.json()
        if "style_preset" in data and data["style_preset"] in STYLE_PRESETS:
            ENGINE.active_style_preset = data["style_preset"]
        aliases = data.get("aliases")
        if isinstance(aliases, str):
            aliases = aliases.split(",")
        result = ENGINE.set_character_reference(
            reference_image=data.get("reference_image"),
            character_description=data.get("character_description",
                                           data.get("clothing_description", "")),
            weight_modifier=data.get("weight_modifier", 0.85),
            aliases=aliases)
        return web.json_response(result)

    @routes.delete("/promptrelay/storyboard/character")
    async def storyboard_clear_character(request):
        return web.json_response(ENGINE.clear_character())

    @routes.post("/promptrelay/storyboard/compile")
    async def storyboard_compile(request):
        data = await request.json()
        scenes = data.get("scenes") or []
        if not scenes:
            return web.json_response({"error": "no scenes provided"}, status=400)
        base_seed = data.get("seed")
        if base_seed is None:
            base_seed = random.randrange(2 ** 48)
        seed_lock = bool(data.get("seed_lock", True))
        character = data.get("character", None)  # None -> use locked state
        kwargs = gen_kwargs(data)

        shots = []
        for i, scene in enumerate(scenes):
            if isinstance(scene, str):
                scene = {"prompt": scene}
            seed = scene.get("seed")
            if seed is None:
                seed = base_seed if seed_lock else base_seed + i
            shots.append(ENGINE.compile_shot_payload(
                scene.get("prompt", ""),
                frame_number=scene.get("frame_number", i + 1),
                seed=seed, character=character, **kwargs))

        if data.get("queue"):
            await queue_shots(request, shots, data.get("client_id", "storyboard-engine"))
        return web.json_response({"shots": shots})

    @routes.post("/promptrelay/storyboard/parse_script")
    async def storyboard_parse_script(request):
        data = await request.json()
        raw_text = data.get("script_text")
        if not raw_text and data.get("script_file"):
            try:
                with open(data["script_file"], "r", encoding="utf-8", errors="replace") as fh:
                    raw_text = fh.read()
            except OSError as e:
                return web.json_response({"error": f"cannot read script_file: {e}"}, status=400)
        if not raw_text or not raw_text.strip():
            return web.json_response({"error": "no script text provided"}, status=400)
        aliases = data.get("aliases")
        if isinstance(aliases, str):
            aliases = aliases.split(",")
        try:
            result = ENGINE.parse_script_payloads(
                raw_text,
                aliases=aliases,
                character=data.get("character", None),
                follow_pronouns=bool(data.get("follow_pronouns", True)),
                pronoun_window=int(data.get("pronoun_window", 2)),
                max_shots=int(data.get("max_shots", 24)),
                seed=data.get("seed"),
                seed_lock=bool(data.get("seed_lock", True)),
                **gen_kwargs(data))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        if not data.get("compile", True):
            result.pop("shots", None)
        elif data.get("queue"):
            await queue_shots(request, result["shots"],
                              data.get("client_id", "storyboard-engine"))
        return web.json_response(result)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Storyboard script parser / continuity engine")
    ap.add_argument("script_file", nargs="?", help="raw text script file to parse")
    ap.add_argument("--aliases", default="", help="comma-separated character names/aliases")
    ap.add_argument("--description", default="", help="character description (continuity prompt)")
    ap.add_argument("--reference", default=None, help="reference image filename (ComfyUI input dir)")
    ap.add_argument("--weight", type=float, default=0.4, help="reference strictness 0..1")
    ap.add_argument("--style", default=None, help="style preset name")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-shots", type=int, default=24)
    ap.add_argument("--full", action="store_true", help="print full payloads incl. workflows")
    args = ap.parse_args()

    engine = CharacterContinuityEngine()
    if args.description or args.reference or args.aliases:
        engine.set_character_reference(
            reference_image=args.reference,
            character_description=args.description,
            weight_modifier=args.weight,
            aliases=args.aliases.split(",") if args.aliases else None)

    if args.script_file:
        with open(args.script_file, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        result = engine.parse_script_payloads(
            raw, max_shots=args.max_shots, seed=args.seed,
            width=args.width, height=args.height, steps=args.steps,
            style_preset=args.style)
        if not args.full:
            for shot in result["shots"]:
                shot.pop("workflow", None)
        print(json.dumps(result, indent=2))
    else:
        # demo
        engine.set_character_reference(
            reference_image="face_ref_98213.png",
            character_description="blue leather jacket and silver spectacles",
            aliases=["Maya"])
        demo_script = (
            "INT. CANYON BASE - DAWN\n\n"
            "Maya checks the seals on her suit. The wind howls across the dunes. "
            "She climbs the ridge with slow, deliberate steps. A distant satellite "
            "blinks overhead. Maya kneels beside a half-buried antenna. It hums "
            "with a faint signal. The storm front swallows the horizon.\n")
        result = engine.parse_script_payloads(demo_script, seed=424242)
        for shot in result["shots"]:
            shot.pop("workflow", None)
        print(json.dumps(result, indent=2))
