"""Stitch multiple rendered clips into one video (the multi-shot 'reel builder').

Clips are normalised to a common size/fps, then joined either with hard cuts (concat) or
short crossfades (ffmpeg xfade + acrossfade). Audio is crossfaded/concatenated to match.
"""

import os
import shutil
import subprocess

# Reuse the same ffmpeg the rest of PromptRelay uses.
_FFMPEG = os.environ.get("PROMPTRELAY_FFMPEG",
                         r"C:\pinokio\bin\ffmpeg-env\Library\bin\ffmpeg.exe")
_FFPROBE = os.environ.get("PROMPTRELAY_FFPROBE",
                          r"C:\pinokio\bin\ffmpeg-env\Library\bin\ffprobe.exe")


def _ffmpeg():
    return _FFMPEG if os.path.isfile(_FFMPEG) else (shutil.which("ffmpeg") or "ffmpeg")


def _ffprobe():
    return _FFPROBE if os.path.isfile(_FFPROBE) else (shutil.which("ffprobe") or "ffprobe")


def _duration(path):
    try:
        out = subprocess.run([_ffprobe(), "-v", "error", "-show_entries",
                              "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _run(args, timeout=1200):
    p = subprocess.run([_ffmpeg(), "-y", *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-600:]}")


def stitch(clips, out_path, *, transition="crossfade", xfade=0.5, width=1280, height=720, fps=25):
    """Join `clips` (list of file paths, in order) into `out_path`.

    transition: "crossfade" (dissolve) or "cut" (hard). Every clip is scaled/padded to
    width x height @ fps first so the filters line up. Returns out_path.
    """
    clips = [c for c in clips if c and os.path.isfile(c)]
    if not clips:
        raise RuntimeError("no clips to stitch")
    if len(clips) == 1:
        shutil.copyfile(clips[0], out_path)
        return out_path

    # Normalise every clip to a common canvas (letterbox-pad to keep aspect) + fps + audio.
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps},setsar=1")
    norm = []
    for i, c in enumerate(clips):
        n = os.path.join(os.path.dirname(out_path), f"_norm_{i}.mp4")
        _run(["-i", c, "-vf", vf, "-af", "aformat=sample_rates=48000:channel_layouts=stereo",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
              "-c:a", "aac", "-r", str(fps), n])
        norm.append(n)

    try:
        if transition == "cut":
            _concat_cut(norm, out_path)
        else:
            _concat_xfade(norm, out_path, xfade, fps)
    finally:
        for n in norm:
            try:
                os.remove(n)
            except OSError:
                pass
    return out_path


def finalize(in_path, out_path, *, width=None, height=None, mute=False):
    """Optional post-pass over a finished clip: high-quality upscale to width x height
    (lanczos + light sharpen) and/or strip the audio track. Used to wire the '4K' and
    'audio off' options.

    IMPORTANT: the caller must treat a raised exception as NON-FATAL — on failure keep the
    original `in_path` so a heavy 4K pass that OOMs never loses the base render.
    Returns out_path on success.
    """
    args = ["-i", in_path]
    if width and height:  # real upscale — must re-encode video
        args += ["-vf", f"scale={width}:{height}:flags=lanczos,unsharp=5:5:0.8:3:3:0.4",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    else:                 # no resize — keep the video stream as-is
        args += ["-c:v", "copy"]
    args += ["-an"] if mute else ["-c:a", "copy"]
    args += [out_path]
    _run(args)
    return out_path


def _concat_cut(clips, out_path):
    """Hard-cut concat via the concat demuxer (clips already normalised, so codecs match)."""
    listfile = os.path.join(os.path.dirname(out_path), "_concat.txt")
    with open(listfile, "w", encoding="utf-8") as fh:
        for c in clips:
            fh.write(f"file '{c.replace(os.sep, '/')}'\n")
    try:
        _run(["-f", "concat", "-safe", "0", "-i", listfile,
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", out_path])
    finally:
        try:
            os.remove(listfile)
        except OSError:
            pass


def _concat_xfade(clips, out_path, xfade, fps):
    """Chain xfade (video) + acrossfade (audio). Offset for each transition is the running
    total of prior clip durations minus the accumulated overlap."""
    durs = [_duration(c) for c in clips]
    inputs = []
    for c in clips:
        inputs += ["-i", c]

    v_parts, a_parts = [], []
    v_prev, a_prev = "[0:v]", "[0:a]"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += durs[i - 1] - xfade
        vo = f"[vx{i}]"
        ao = f"[ax{i}]"
        v_parts.append(f"{v_prev}[{i}:v]xfade=transition=fade:duration={xfade}:"
                       f"offset={offset:.3f}{vo}")
        a_parts.append(f"{a_prev}[{i}:a]acrossfade=d={xfade}{ao}")
        v_prev, a_prev = vo, ao

    filtergraph = ";".join(v_parts + a_parts)
    _run([*inputs, "-filter_complex", filtergraph,
          "-map", v_prev, "-map", a_prev,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
          "-c:a", "aac", "-r", str(fps), out_path])
