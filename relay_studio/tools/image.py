"""generate_image tool — renders a new still via ComfyUI (Z-Image Turbo).

Reuses storyboard_engine.build_workflow / resolve_models, queues to ComfyUI's /prompt,
polls /history for the result and pulls the file via /view. If ComfyUI isn't running it
returns a clear error the agent can surface (the cat pipeline can proceed on the photo)."""

import os
import random
import time

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def _get_storyboard_engine():
    """Lazy import so a missing storyboard_engine doesn't break the whole package load."""
    try:
        # Three dots: relay_studio/tools/image.py → relay_studio/ → ComfyUI-PromptRelay/
        from ... import storyboard_engine as _se
        return _se
    except (ImportError, ValueError):
        pass
    try:
        import storyboard_engine as _se  # standalone / sys.path
        return _se
    except ImportError:
        return None


def _poll_history(prompt_id, timeout=180):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                return data[prompt_id]
        time.sleep(1.0)
    return None


def generate_image(project, prompt="", width=768, height=768, steps=8, seed=None, **_):
    """Render a still image from a text prompt via ComfyUI. Returns the media file."""
    import requests
    if not prompt.strip():
        return {"error": "empty prompt"}
    storyboard_engine = _get_storyboard_engine()
    if storyboard_engine is None:
        return {"error": "storyboard_engine not available — ComfyUI image gen is disabled"}
    if seed is None:
        seed = random.randrange(2 ** 48)
    models = storyboard_engine.resolve_models()
    wf = storyboard_engine.build_workflow(prompt, seed, width=int(width), height=int(height),
                                          steps=int(steps), models=models)
    try:
        r = requests.post(f"{COMFY_URL}/prompt",
                          json={"prompt": wf, "client_id": "relay-studio"}, timeout=15)
    except Exception as e:
        return {"error": f"ComfyUI not reachable at {COMFY_URL}: {e}"}
    if r.status_code != 200:
        return {"error": f"ComfyUI rejected workflow: {r.text[:300]}"}
    prompt_id = r.json().get("prompt_id")

    hist = _poll_history(prompt_id)
    if not hist:
        return {"error": "timed out waiting for ComfyUI to render"}
    images = []
    for node in hist.get("outputs", {}).values():
        images += node.get("images", [])
    if not images:
        return {"error": "ComfyUI produced no image"}

    info = images[0]
    view = requests.get(f"{COMFY_URL}/view", params={
        "filename": info["filename"], "subfolder": info.get("subfolder", ""),
        "type": info.get("type", "output")}, timeout=30)
    dst = project.media_path(f"image_{seed}.png")
    with open(dst, "wb") as fh:
        fh.write(view.content)
    item = project.add_media("image", dst, via="comfyui", label=prompt[:60],
                             prompt=prompt, seed=seed)
    return {"ok": True, "media_id": item["id"], "file": item["file"], "seed": seed}
