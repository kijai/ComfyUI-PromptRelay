"""LTX 2.3 image-to-video for the SeeDance-style studio, via ComfyUI's /prompt API.

Reuses the *proven* LTX i2v graph captured for relay_studio (relay_studio/ltx_i2v_api.json)
so we inherit a known-good render. v1 patches only the input image, the motion prompt, the
seeds and the output name — everything else (samplers, sigmas, upscaler, audio VAE) is left
exactly as the working graph, so we never fight the subgraph internals.

Output resolution/aspect follow the *input image* (the graph auto-resizes by longer edge),
so the studio's resolution/aspect controls are advisory in v1; real size/length control lands
with the text-to-video pass. LTX renders are slow (~5-7 min on 16 GB).
"""

import json
import os
import random
import shutil
import time

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")

_BASE = os.path.dirname(os.path.abspath(__file__))
# i2v graph: prefer the studio-owned v2 (parameterized LTX23_GGUF_I2V — titled Width/Height/
# Duration nodes, so we control size + length), falling back to the legacy relay_studio graph.
_GRAPH_V2 = os.path.join(_BASE, "ltx_i2v_v2_api.json")
_GRAPH_LEGACY = os.path.join(os.path.dirname(_BASE), "relay_studio", "ltx_i2v_api.json")
_GRAPH = _GRAPH_V2 if os.path.isfile(_GRAPH_V2) else _GRAPH_LEGACY
# Text-to-video graph: captured from the user's LTX23_GGUF t2v workflow (flat API format)
# via ComfyUI's graphToPrompt. Absent until captured; t2v is gated on it existing.
_GRAPH_T2V = os.path.join(_BASE, "ltx_t2v_api.json")
# First/last-frame graph: captured from LTX23_FLF_First_Last_Frame_32GB (enhancer rewired out).
_GRAPH_FLF = os.path.join(_BASE, "ltx_flf_api.json")
# Elements graph: Flux2 Klein 2-reference-image compose (converted from the UI workflow
# "Flux2-Klein style-identity blend (2 image)"). Fixed node ids — see _ELEM_* below.
_GRAPH_ELEMENTS = os.path.join(_BASE, "flux2_elements_api.json")
# Motion-control graph: native Wan2.2 Animate character replace (converted + trimmed from
# "Wan22_Animate_Character_Replace.json": debug SaveVideos removed, final SaveVideo rewired).
_GRAPH_MOTION = os.path.join(_BASE, "wan_animate_api.json")

# This file lives in custom_nodes/ComfyUI-PromptRelay/seedance_studio → up 3 = ComfyUI root.
_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_BASE)))
_INPUT_DIR = os.path.join(_COMFY_ROOT, "input")
_OUTPUT_DIR = os.path.join(_COMFY_ROOT, "output")

# Appended to the motion prompt when "fixed camera" is on — biases LTX toward a locked shot
# without touching the negative-conditioning nodes (safer than guessing pos/neg in flat API).
_FIXED_CAMERA_SUFFIX = "locked-off static camera, no camera movement"


def available():
    """True if the captured graph + at least one LTX GGUF unet are present."""
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
    ext = os.path.splitext(image_path)[1].lower() or ".png"
    name = f"seedance_src_{random.randrange(10 ** 8)}{ext}"
    shutil.copyfile(image_path, os.path.join(_INPUT_DIR, name))
    return name


def _letterbox_stage(image_path, target_w, target_h):
    """Pad the image with black bars onto a target_w:target_h-aspect canvas so the WHOLE
    image survives the graph's center-crop (no cropping). Saves into ComfyUI's input dir
    and returns the bare filename. Falls back to a plain stage if PIL/aspect fails."""
    try:
        from PIL import Image
        os.makedirs(_INPUT_DIR, exist_ok=True)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            iw, ih = im.size
        if not iw or not ih:
            return _stage_image(image_path)
        target_ratio = target_w / target_h
        if (iw / ih) > target_ratio:          # image wider than target -> pad top/bottom
            cw, ch = iw, int(round(iw / target_ratio))
        else:                                  # image taller -> pad left/right
            cw, ch = int(round(ih * target_ratio)), ih
        canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
        canvas.paste(im, ((cw - iw) // 2, (ch - ih) // 2))
        name = f"seedance_fit_{random.randrange(10 ** 8)}.png"
        canvas.save(os.path.join(_INPUT_DIR, name))
        return name
    except Exception:
        return _stage_image(image_path)


def _neg_encode_targets(graph):
    """Ids of CLIPTextEncode nodes fed by a PrimitiveStringMultiline (the POSITIVE branch),
    so callers can target the *other* encode for the negative prompt."""
    pos_str_ids = [nid for nid, n in graph.items()
                   if n.get("class_type") == "PrimitiveStringMultiline"]
    targets = set()
    for nid, n in graph.items():
        if n.get("class_type") == "CLIPTextEncode":
            src = n.get("inputs", {}).get("text")
            if isinstance(src, list) and src and str(src[0]) in pos_str_ids:
                targets.add(nid)
    return targets


def _patch(graph, image_name, prompt, negative, seed, prefix,
           width=None, height=None, duration=None, model=None):
    """Patch the flat i2v graph: input image, motion prompt, negative, seeds, output name,
    and (optionally) width/height/duration via titled PrimitiveInt nodes and the GGUF unet
    (quality tier)."""
    pos_targets = _neg_encode_targets(graph)
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
        elif ct == "PrimitiveInt":
            title = _meta_title(n)
            if title == "Width" and width:
                ins["value"] = int(width)
            elif title == "Height" and height:
                ins["value"] = int(height)
            elif title == "Duration" and duration:
                ins["value"] = int(duration)
        elif ct == "UnetLoaderGGUF" and model:
            ins["unet_name"] = model
        elif ct == "CLIPTextEncode" and negative and nid not in pos_targets:
            if isinstance(ins.get("text"), str):
                ins["text"] = negative
    return graph


def image_dims(image_path, longer=768, mult=32, cap=1280):
    """Return (w, h) matching the image's aspect, longer edge ~`longer`, rounded to `mult`,
    capped at `cap`. Used for i2v 'follow start image' sizing."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        return None, None
    if not w or not h:
        return None, None
    scale = longer / max(w, h)
    w2, h2 = int(round(w * scale / mult)) * mult, int(round(h * scale / mult)) * mult
    w2 = max(mult, min(cap, w2))
    h2 = max(mult, min(cap, h2))
    return w2, h2


def _queue(graph):
    import requests
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": graph, "client_id": "seedance-studio"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the LTX workflow: {r.text[:400]}")
    return r.json().get("prompt_id")


def _await_output(prompt_id, timeout, on_progress=None):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                entry = data[prompt_id]
                outs = entry.get("outputs", {})
                for node_out in outs.values():
                    for key in ("videos", "gifs", "images"):
                        for f in node_out.get(key, []):
                            return f
                # No outputs: if the prompt ERRORED, fail now instead of timing out.
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    detail = "workflow errored"
                    for msg in status.get("messages", []):
                        if isinstance(msg, (list, tuple)) and len(msg) > 1 \
                                and msg[0] == "execution_error":
                            d = msg[1] or {}
                            detail = (f"{d.get('node_type', 'node')}: "
                                      f"{d.get('exception_message', 'error')}")[:300]
                    raise RuntimeError(f"ComfyUI error — {detail}")
        if on_progress:
            on_progress()
        time.sleep(3.0)
    return None


def available_t2v():
    """True once the text-to-video graph has been captured and an LTX GGUF is present."""
    if not os.path.isfile(_GRAPH_T2V):
        return False
    unet = os.path.join(_COMFY_ROOT, "models", "unet")
    try:
        return any("ltx" in f.lower() and f.endswith(".gguf") for f in os.listdir(unet))
    except OSError:
        return False


def _meta_title(n):
    return (n.get("_meta") or {}).get("title", "")


def _patch_t2v(graph, prompt, negative, seed, prefix,
               width=None, height=None, duration=None, model=None):
    """Patch the flat t2v graph: positive prompt, seeds, output name, and (optionally)
    width/height/duration via the titled PrimitiveInt nodes, plus the GGUF unet (quality
    tier). The negative prompt is applied to the CLIPTextEncode that is NOT fed by the
    PrimitiveStringMultiline (the negative branch), when present."""
    # Find which CLIPTextEncode node id the positive prompt string flows into, so we can
    # target the *other* one for the negative prompt.
    pos_str_ids = [nid for nid, n in graph.items()
                   if n.get("class_type") == "PrimitiveStringMultiline"]
    pos_encode_targets = set()
    for nid, n in graph.items():
        if n.get("class_type") == "CLIPTextEncode":
            src = n.get("inputs", {}).get("text")
            if isinstance(src, list) and src and str(src[0]) in pos_str_ids:
                pos_encode_targets.add(nid)

    for nid, n in graph.items():
        ct = n.get("class_type")
        ins = n.get("inputs", {})
        if ct == "PrimitiveStringMultiline":
            ins["value"] = prompt
        elif ct == "RandomNoise":
            ins["noise_seed"] = seed
        elif ct == "SaveVideo":
            ins["filename_prefix"] = prefix
        elif ct == "PrimitiveInt":
            title = _meta_title(n)
            if title == "Width" and width:
                ins["value"] = int(width)
            elif title == "Height" and height:
                ins["value"] = int(height)
            elif title == "Duration" and duration:
                ins["value"] = int(duration)
        elif ct == "UnetLoaderGGUF" and model:
            ins["unet_name"] = model
        elif ct == "CLIPTextEncode" and negative and nid not in pos_encode_targets:
            # Only override a literal (non-linked) negative text field.
            if isinstance(ins.get("text"), str):
                ins["text"] = negative
    return graph


def available_flf():
    """True once the first/last-frame graph is captured and an LTX GGUF is present."""
    if not os.path.isfile(_GRAPH_FLF):
        return False
    unet = os.path.join(_COMFY_ROOT, "models", "unet")
    try:
        return any("ltx" in f.lower() and f.endswith(".gguf") for f in os.listdir(unet))
    except OSError:
        return False


def _patch_flf(graph, first_name, last_name, prompt, negative, seed, prefix,
               first_strength=None, last_strength=None,
               width=None, height=None, duration=None, model=None):
    """Patch the flat FLF graph: the two titled LoadImage nodes, prompt, negative, seeds,
    the two titled strength floats, WIDTH/HEIGHT/LENGTH INTConstants, GGUF unet and output
    name. (This graph uses INTConstant for size/length and VHS_VideoCombine for output.)"""
    pos_targets = _neg_encode_targets(graph)
    for nid, n in graph.items():
        ct = n.get("class_type")
        ins = n.get("inputs", {})
        title = _meta_title(n)
        if ct == "LoadImage":
            if title == "LAST FRAME":
                ins["image"] = last_name
            else:  # FIRST FRAME (default)
                ins["image"] = first_name
        elif ct == "PrimitiveStringMultiline" and title == "PROMPT":
            ins["value"] = prompt
        elif ct == "RandomNoise":
            ins["noise_seed"] = seed
        elif ct in ("VHS_VideoCombine", "SaveVideo"):
            ins["filename_prefix"] = prefix
        elif ct == "INTConstant":
            if title == "WIDTH" and width:
                ins["value"] = int(width)
            elif title == "HEIGHT" and height:
                ins["value"] = int(height)
            elif "LENGTH" in title and duration:
                ins["value"] = int(duration)
        elif ct == "PrimitiveFloat":
            if title == "FIRST FRAME STRENGTH" and first_strength is not None:
                ins["value"] = float(first_strength)
            elif title == "LAST FRAME STRENGTH" and last_strength is not None:
                ins["value"] = float(last_strength)
        elif ct == "UnetLoaderGGUF" and model:
            ins["unet_name"] = model
        elif ct == "CLIPTextEncode" and negative and nid not in pos_targets:
            if isinstance(ins.get("text"), str):
                ins["text"] = negative
    return graph


def generate_flf(first_path, last_path, prompt, *, negative=None, seed=None,
                 first_strength=0.5, last_strength=1.0, width=None, height=None,
                 duration=None, model=None, timeout=2400, on_progress=None):
    """First/last-frame video: animate from `first_path` to `last_path`.
    Returns {filename, subfolder, seed}."""
    if not available_flf():
        raise RuntimeError("FLF graph not captured yet (ltx_flf_api.json missing)")

    with open(_GRAPH_FLF, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    if seed is None:
        seed = random.randrange(2 ** 53)
    duration = int(duration) if duration else 5
    first_name = _stage_image(first_path)
    last_name = _stage_image(last_path)
    prefix = f"video/seedance_flf_{random.randrange(10 ** 6)}"
    _patch_flf(graph, first_name, last_name, (prompt or "").strip(),
               (negative or "").strip(), seed, prefix,
               first_strength=first_strength, last_strength=last_strength,
               width=width, height=height, duration=duration, model=model)

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout, on_progress=on_progress)
    if not info:
        raise RuntimeError(f"LTX FLF render timed out after {timeout}s")
    return {"filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "seed": seed}


def generate_t2v(prompt, *, negative=None, seed=None, width=None, height=None,
                 duration=None, model=None, timeout=1800, on_progress=None):
    """Text-to-video via the captured LTX t2v graph. Returns {filename, subfolder, seed}."""
    if not available_t2v():
        raise RuntimeError("t2v graph not captured yet (ltx_t2v_api.json missing)")

    with open(_GRAPH_T2V, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    if seed is None:
        seed = random.randrange(2 ** 53)
    prefix = f"video/seedance_t2v_{random.randrange(10 ** 6)}"
    _patch_t2v(graph, (prompt or "").strip(), (negative or "").strip(), seed, prefix,
               width=width, height=height, duration=duration, model=model)

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout, on_progress=on_progress)
    if not info:
        raise RuntimeError(f"LTX t2v render timed out after {timeout}s")
    return {"filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "seed": seed}


def available_elements():
    """True when the Flux2 Klein compose graph + the Klein model are present."""
    if not os.path.isfile(_GRAPH_ELEMENTS):
        return False
    return os.path.isfile(os.path.join(_COMFY_ROOT, "models", "diffusion_models",
                                       "flux-2-klein-9b.safetensors"))


def available_motion():
    """True when the Wan2.2 Animate graph + the Wan Animate model are present."""
    if not os.path.isfile(_GRAPH_MOTION):
        return False
    return os.path.isfile(os.path.join(_COMFY_ROOT, "models", "diffusion_models",
                                       "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"))


def _stage_video(video_path):
    """Copy a driving video into ComfyUI's input dir; return the bare filename."""
    os.makedirs(_INPUT_DIR, exist_ok=True)
    ext = os.path.splitext(video_path)[1].lower() or ".mp4"
    name = f"seedance_drv_{random.randrange(10 ** 8)}{ext}"
    shutil.copyfile(video_path, os.path.join(_INPUT_DIR, name))
    return name


def compose_elements(element_paths, prompt, *, seed=None, timeout=1200, on_progress=None):
    """Flux2 Klein: compose 1-2 reference "element" images + a prompt into ONE still.
    Output size/aspect follows element 1 (~1MP). Returns {filename, subfolder, seed}
    of the resulting IMAGE (not video) — feed it to generate_i2v to animate.

    Fixed node ids in flux2_elements_api.json:
      "1"/"2" LoadImage (element 1/2) · "12" CLIPTextEncode prompt · "9" RandomNoise ·
      "3" SaveImage · "6" CFGGuider (positive/negative rewired for the 1-element case:
      ReferenceLatent "22"/"20" are the image-1-only conditioning, "25"/"23" chain in
      image 2 on top).
    """
    if not available_elements():
        raise RuntimeError("Elements not available (Flux2 Klein graph or model missing)")
    with open(_GRAPH_ELEMENTS, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    if seed is None:
        seed = random.randrange(2 ** 53)
    names = [_stage_image(p) for p in element_paths[:2]]
    graph["1"]["inputs"]["image"] = names[0]
    if len(names) > 1:
        graph["2"]["inputs"]["image"] = names[1]
    else:
        # Single element: point the guider at the image-1-only ReferenceLatent chain so
        # the unused image-2 branch is never executed (ComfyUI prunes unreached nodes).
        graph["2"]["inputs"]["image"] = names[0]
        graph["6"]["inputs"]["positive"] = ["22", 0]
        graph["6"]["inputs"]["negative"] = ["20", 0]
    graph["12"]["inputs"]["text"] = (prompt or "").strip()
    graph["9"]["inputs"]["noise_seed"] = seed
    graph["3"]["inputs"]["filename_prefix"] = f"seedance_elem_{random.randrange(10 ** 6)}"

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout, on_progress=on_progress)
    if not info:
        raise RuntimeError(f"Flux2 Klein compose timed out after {timeout}s")
    return {"filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "seed": seed}


def generate_motion(image_path, video_path, *, prompt=None, negative=None, seed=None,
                    width=None, height=None, frame_cap=0, timeout=3600, on_progress=None):
    """Wan2.2 Animate motion control: the character in `image_path` performs the motion
    from the driving `video_path` (auto character replace — SAM2 mask + ViTPose). Length
    and fps follow the driving video; `frame_cap` (frames, 0 = whole video) trims it to
    keep VRAM sane on 16GB. Returns {filename, subfolder, seed}.

    Fixed node ids in wan_animate_api.json:
      "10" LoadImage character · "301" VHS_LoadVideo driving video · "374"/"349" pos/neg
      CLIPTextEncode · "377" PrimitiveInt seed · "159"/"160" Width/Height · "19" SaveVideo.
    """
    if not available_motion():
        raise RuntimeError("Motion control not available (Wan Animate graph or model missing)")
    with open(_GRAPH_MOTION, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    if seed is None:
        seed = random.randrange(2 ** 53)
    graph["10"]["inputs"]["image"] = _stage_image(image_path)
    graph["301"]["inputs"]["video"] = _stage_video(video_path)
    graph["301"]["inputs"]["frame_load_cap"] = max(0, int(frame_cap or 0))
    if prompt and prompt.strip():
        graph["374"]["inputs"]["text"] = prompt.strip()
    if negative and negative.strip():
        graph["349"]["inputs"]["text"] = negative.strip()
    graph["377"]["inputs"]["value"] = seed
    if width:
        graph["159"]["inputs"]["value"] = int(width)
    if height:
        graph["160"]["inputs"]["value"] = int(height)
    graph["19"]["inputs"]["filename_prefix"] = \
        f"video/seedance_motion_{random.randrange(10 ** 6)}"

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout, on_progress=on_progress)
    if not info:
        raise RuntimeError(f"Wan Animate render timed out after {timeout}s")
    return {"filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "seed": seed}


def generate_i2v(image_path, prompt, *, negative=None, seed=None, fixed_camera=False,
                 width=None, height=None, duration=None, model=None, fit=False,
                 timeout=1800, on_progress=None):
    """Animate `image_path` into a clip via the LTX i2v graph, with adjustable size/length/
    quality. When `fit` is set (with an explicit width/height), the image is letterboxed so
    the whole frame is preserved instead of centre-cropped. Returns {filename, subfolder,
    seed}. Raises on failure.
    """
    if not available():
        raise RuntimeError("LTX i2v not available (graph or GGUF model missing)")

    with open(_GRAPH, "r", encoding="utf-8") as fh:
        graph = json.load(fh)

    if seed is None:
        seed = random.randrange(2 ** 53)
    # Default to a SHORT clip so i2v is fast — never the graph's slow ~13s default.
    duration = int(duration) if duration else 5
    prompt = (prompt or "").strip()
    if fixed_camera:
        prompt = (prompt + ", " + _FIXED_CAMERA_SUFFIX).strip(", ")

    # 'Follow start image': derive output dims from the image when caller didn't set them.
    if (not width or not height) and os.path.isfile(image_path):
        iw, ih = image_dims(image_path)
        width = width or iw
        height = height or ih

    # 'Fit' pads to the target aspect so nothing is cropped; otherwise stage as-is.
    if fit and width and height:
        image_name = _letterbox_stage(image_path, width, height)
    else:
        image_name = _stage_image(image_path)
    prefix = f"video/seedance_{random.randrange(10 ** 6)}"
    _patch(graph, image_name, prompt, (negative or "").strip(), seed, prefix,
           width=width, height=height, duration=duration, model=model)

    prompt_id = _queue(graph)
    info = _await_output(prompt_id, timeout, on_progress=on_progress)
    if not info:
        raise RuntimeError(f"LTX render timed out after {timeout}s")

    return {"filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "seed": seed}
