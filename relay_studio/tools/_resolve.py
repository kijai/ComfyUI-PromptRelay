"""Helpers for tools to resolve a media reference (a media_id, a filename already in
the project dir, a URL to download, or an inputs field) to a real file path."""

import os

from .. import assets


def _download(url, dst):
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dst, "wb") as fh:
        fh.write(r.content)
    return dst


def resolve_media(project, ref, kind=None):
    """Resolve `ref` to an absolute file path, or None.

    ref may be: a media_id, a bare filename in the project dir, an absolute path,
    or an http(s) URL (downloaded into the project dir)."""
    if not ref:
        return None
    ref = str(ref).strip()
    # media_id
    for m in project.media:
        if m["id"] == ref and (kind is None or m["kind"] == kind):
            return project.media_path(m["file"])
    # url
    if ref.startswith("http://") or ref.startswith("https://"):
        ext = os.path.splitext(ref.split("?")[0])[1] or ".bin"
        dst = project.media_path(f"download_{abs(hash(ref)) % 10**8}{ext}")
        if not os.path.exists(dst):
            try:
                _download(ref, dst)
            except Exception:
                return None
        return dst
    # bare filename in project dir
    cand = project.media_path(os.path.basename(ref))
    if os.path.exists(cand):
        return cand
    # absolute / relative path
    if os.path.exists(ref):
        return ref
    return None


def resolve_cat_image(project, image_ref=""):
    """The photo to animate: an explicit ref, else the cat_photo_url input, else a
    placeholder image so the pipeline still runs."""
    path = resolve_media(project, image_ref, kind="image") or resolve_media(project, image_ref)
    if path:
        return path
    path = resolve_media(project, project.inputs.get("cat_photo_url", ""))
    if path:
        return path
    assets.ensure_placeholders()
    return assets.PLACEHOLDER_CAT
