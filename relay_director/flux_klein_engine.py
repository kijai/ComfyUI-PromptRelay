"""Flux 2 Klein character-swap via ComfyUI's /prompt API — Tier B identity-lock.

Drives the user's working "Flux2-Klein turn to real face reference" workflow (captured to
flux_klein_api.json in flat API format) to lock a character's identity onto a shot:
takes the shot's base/scene image as the SUBJECT and the character's locked reference
sheet as the FACE REFERENCE, and regenerates the subject as a photo wearing that face.

We only swap the two input images, the prompt, the seed, the turn2real LoRA strength
(the fidelity lever) and the output name — everything else is the proven-working graph.

Flux Klein 9B is gated/heavy (18GB, RAM-offloaded on this 16GB rig), so swaps are slow:
this is the FINAL pass only, never the fast preview.
"""

import json
import os
import random
import shutil
import time

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")

_BASE = os.path.dirname(os.path.abspath(__file__))
_GRAPH = os.path.join(_BASE, "flux_klein_api.json")

# This file lives in custom_nodes/ComfyUI-PromptRelay/relay_director
_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_BASE)))
_INPUT_DIR = os.path.join(_COMFY_ROOT, "input")
_OUTPUT_DIR = os.path.join(_COMFY_ROOT, "output")

# Node ids in flux_klein_api.json (stable; verified against the captured graph).
N_SUBJECT = "4"      # LoadImage — the base/scene image to keep composition of
N_FACE = "5"         # LoadImage — the character reference (identity to lock)
N_PROMPT = "11"      # CLIPTextEncode positive
N_LORA = "13"        # Power Lora Loader (rgthree) — turn2real strength
N_SEED = "18"        # RandomNoise
N_SAVE = "25"        # SaveImage

DEFAULT_PROMPT = ("turn this image into a realistic photo, keep the composition and pose, "
                  "use the face and identity from the face reference image")


def available():
    """True if the captured graph + the Flux Klein model are present."""
    if not os.path.isfile(_GRAPH):
        return False
    dm = os.path.join(_COMFY_ROOT, "models", "diffusion_models")
    try:
        return any("flux-2-klein" in f.lower() for f in os.listdir(dm))
    except OSError:
        return False


def _stage_image(image_path, name):
    """Copy a source image into ComfyUI's input dir; return the bare filename."""
    os.makedirs(_INPUT_DIR, exist_ok=True)
    dst = os.path.join(_INPUT_DIR, name)
    shutil.copyfile(image_path, dst)
    return name


def _patch_graph(graph, subject_name, face_name, prompt, prefix, seed, lora_strength):
    graph[N_SUBJECT]["inputs"]["image"] = subject_name
    graph[N_FACE]["inputs"]["image"] = face_name
    graph[N_PROMPT]["inputs"]["text"] = prompt
    graph[N_SEED]["inputs"]["noise_seed"] = seed
    graph[N_SAVE]["inputs"]["filename_prefix"] = prefix
    try:
        graph[N_LORA]["inputs"]["lora_1"]["strength"] = float(lora_strength)
    except (KeyError, TypeError, ValueError):
        pass
    return graph


def _queue(graph):
    import requests
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": graph, "client_id": "relay-director-klein"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected Flux-Klein workflow: {r.text[:400]}")
    return r.json().get("prompt_id")


def _await_output(prompt_id, timeout):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                outs = data[prompt_id].get("outputs", {})
                for node_out in outs.values():
                    for f in node_out.get("images", []):
                        return f
        time.sleep(3.0)
    return None


def generate_swap(subject_path, face_path, out_path, *, prompt=None, seed=None,
                  lora_strength=0.9, timeout=900):
    """Lock `face_path`'s identity onto `subject_path`; write the result to `out_path`.

    Raises on failure so the caller can fall back to the un-swapped base image.
    Returns out_path on success."""
    if not available():
        raise RuntimeError("Flux Klein not available (graph or model missing)")

    with open(_GRAPH, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    tag = random.randrange(10 ** 6)
    subject_name = _stage_image(subject_path, f"klein_subj_{tag}.png")
    face_name = _stage_image(face_path, f"klein_face_{tag}.png")
    prefix = f"director_klein_{tag}"
    if seed is None:
        seed = random.randrange(2 ** 53)
    _patch_graph(graph, subject_name, face_name, prompt or DEFAULT_PROMPT,
                 prefix, seed, lora_strength)

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout)
    if not info:
        raise RuntimeError(f"Flux-Klein render timed out after {timeout}s")

    src = os.path.join(_OUTPUT_DIR, info.get("subfolder", ""), info["filename"])
    if not os.path.exists(src):
        raise RuntimeError(f"Flux-Klein output not found on disk: {src}")
    shutil.copyfile(src, out_path)
    return out_path
