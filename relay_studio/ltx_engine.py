"""LTX 2.3 image-to-video via ComfyUI's /prompt API.

Drives the user's working LTX23_GGUF_I2V workflow (captured to ltx_i2v_api.json in flat
API format) to animate a still cat photo into a moving, expressive clip. We only swap the
input image, the motion prompt, the seeds and the output name — everything else is left
exactly as the proven-working graph, so we don't fight the subgraph internals.

LTX renders are slow (~5-7 min on 16 GB), so lipsync.py caches the result per photo.
"""

import json
import os
import random
import shutil
import time

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")

_BASE = os.path.dirname(os.path.abspath(__file__))
_GRAPH = os.path.join(_BASE, "ltx_i2v_api.json")

# ComfyUI's input/output dirs (this file lives in custom_nodes/ComfyUI-PromptRelay/relay_studio)
_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_BASE)))
_INPUT_DIR = os.path.join(_COMFY_ROOT, "input")
_OUTPUT_DIR = os.path.join(_COMFY_ROOT, "output")

# Motion ONLY — never re-describe the scene (that makes LTX rebuild it and the cat drifts /
# the camera pans). Identity is held by the i2v strength + sigma tweaks in _patch_graph.
DEFAULT_PROMPT = ("the same cat stays in place and talks, mouth opening and closing naturally "
                  "as it speaks, small gentle head movements, occasional natural blinks, "
                  "locked-off static camera")

NEGATIVE_EXTRA = ("camera movement, camera pan, camera zoom, dolly, push in, different cat, "
                  "changing appearance, morphing, identity change, warping, distortion")


def available():
    """True if the captured graph + at least one LTX GGUF are present."""
    if not os.path.isfile(_GRAPH):
        return False
    unet = os.path.join(_COMFY_ROOT, "models", "unet")
    try:
        return any("ltx" in f.lower() and f.endswith(".gguf") for f in os.listdir(unet))
    except OSError:
        return False


def _stage_image(image_path):
    """Copy the source photo into ComfyUI's input dir; return the bare filename."""
    os.makedirs(_INPUT_DIR, exist_ok=True)
    name = "dolly_ltx_src.png"
    dst = os.path.join(_INPUT_DIR, name)
    shutil.copyfile(image_path, dst)
    return name


def _patch_graph(graph, image_name, prompt, prefix, seed=None):
    """Set image, prompt, seed and output prefix, and apply the I2V_CONSISTENT
    character-hold tweaks (high i2v strength + low stage-2 sigmas) so the cat doesn't
    drift into a different cat or trigger a camera pan."""
    if seed is None:
        seed = random.randrange(2 ** 53)
    for nid, n in graph.items():
        ct = n.get("class_type")
        ins = n.get("inputs", {})
        if ct == "LoadImage":
            ins["image"] = image_name
        elif ct == "PrimitiveStringMultiline":
            ins["value"] = prompt
        elif ct == "RandomNoise":
            ins["noise_seed"] = seed
        elif ct == "SaveVideo":
            ins["filename_prefix"] = prefix
        elif ct == "LTXVImgToVideoInplace":
            # Raise the stage-2 image strength to 1.0 to hold the input identity.
            if isinstance(ins.get("strength"), (int, float)) and ins["strength"] < 1.0:
                ins["strength"] = 1.0
        elif ct == "ManualSigmas":
            # The 4-step stage-2 schedule: lower it so stage 2 regenerates less (less drift).
            s = ins.get("sigmas")
            if isinstance(s, str) and len([x for x in s.split(",") if x.strip()]) == 4:
                ins["sigmas"] = "0.5, 0.4265, 0.2482, 0.0"
        elif ct == "CLIPTextEncode":
            # Append anti-drift / anti-camera terms to the negative prompt.
            t = ins.get("text")
            if isinstance(t, str) and t.strip():
                ins["text"] = t + ", " + NEGATIVE_EXTRA
    return graph


def _queue(graph):
    import requests
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": graph, "client_id": "relay-ltx"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected LTX workflow: {r.text[:400]}")
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
                    for key in ("gifs", "videos", "images"):
                        for f in node_out.get(key, []):
                            return f
        time.sleep(3.0)
    return None


def generate_i2v(image_path, out_path, *, prompt=None, timeout=900):
    """Animate `image_path` into a moving clip via LTX i2v; write it to `out_path`.

    Raises on failure so the caller can fall back. Returns out_path on success."""
    if not available():
        raise RuntimeError("LTX i2v not available (graph or model missing)")

    with open(_GRAPH, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    image_name = _stage_image(image_path)
    prefix = f"video/dolly_ltx_{random.randrange(10**6)}"
    _patch_graph(graph, image_name, prompt or DEFAULT_PROMPT, prefix)

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout)
    if not info:
        raise RuntimeError(f"LTX render timed out after {timeout}s")

    src = os.path.join(_OUTPUT_DIR, info.get("subfolder", ""), info["filename"])
    if not os.path.exists(src):
        raise RuntimeError(f"LTX output not found on disk: {src}")
    shutil.copyfile(src, out_path)
    return out_path
