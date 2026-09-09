"""Thin ffmpeg wrapper for the studio pipeline.

Uses the ffmpeg binary bundled with imageio_ffmpeg (ships inside ComfyUI's
python_embeded), so no system ffmpeg install is required. Every clip the pipeline
produces is normalised to the same WxH / fps / audio layout so they concat cleanly.
"""

import json
import os
import re
import subprocess

# Uniform output format — all generated clips share these so concat/mix "just works".
WIDTH = 768
HEIGHT = 768
FPS = 30
AR = 44100          # audio sample rate
AC = 2              # audio channels

_FFMPEG = None


def ffmpeg_exe():
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def _run(args, *, timeout=300):
    """Run ffmpeg with the given args (binary auto-prepended). Raises on failure."""
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc.stdout


def media_duration(path):
    """Duration of an audio/video file in seconds (via ffmpeg, no ffprobe needed)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not m:
        return 0.0
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


# ----------------------------------------------------------------------------
# Placeholder asset synthesis (tones / noise / solid image).
# These stand in for real music / SFX / a real photo so the pipeline runs fully
# offline. Replace the files under assets/ with real media to upgrade quality.
# ----------------------------------------------------------------------------

def gen_tone(dst, *, duration=8.0, freq=220.0, volume=0.25, harmony=True):
    """A simple musical-ish tone bed (root + fifth) — placeholder 'music'."""
    if harmony:
        src = (f"sine=frequency={freq}:duration={duration},"
               f"aevalsrc=0.5*sin(2*PI*{freq*1.5}*t):d={duration}")
        flt = (f"[0:a]volume={volume}[a0];[1:a]volume={volume*0.6}[a1];"
               f"[a0][a1]amix=inputs=2,afade=t=in:d=0.4,"
               f"afade=t=out:st={max(0.0,duration-0.5)}:d=0.5[a]")
        _run(["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
              "-f", "lavfi", "-i", f"sine=frequency={freq*1.5}:duration={duration}",
              "-filter_complex", flt, "-map", "[a]",
              "-ar", str(AR), "-ac", str(AC), dst])
    else:
        _run(["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
              "-filter:a", f"volume={volume}", "-ar", str(AR), "-ac", str(AC), dst])
    return dst


def gen_silence(dst, *, duration=1.0):
    """A silent audio track — used as the placeholder track under a ken-burns clip
    before narration/bgm gets mixed in on top of it."""
    _run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{duration}", dst])
    return dst


def gen_noise_sfx(dst, *, duration=0.8, color="brown", volume=0.7):
    """A short percussive noise burst — placeholder SFX."""
    _run(["-f", "lavfi", "-i", f"anoisesrc=d={duration}:c={color}:a=0.5",
          "-filter:a", f"volume={volume},afade=t=out:st=0:d={duration}",
          "-ar", str(AR), "-ac", str(AC), dst])
    return dst


def trim_audio(src, dst, *, start=0.0, duration=5.0, fade=0.6):
    """Cut a snippet (with in/out fades) from a longer track — used to carve the
    intro/outro snippets out of one master BGM track, like drama.land does."""
    fade = min(fade, duration / 3)
    _run(["-ss", f"{start}", "-i", src, "-t", f"{duration}",
          "-filter:a", (f"afade=t=in:d={fade},"
                        f"afade=t=out:st={max(0.0, duration-fade)}:d={fade}"),
          "-ar", str(AR), "-ac", str(AC), dst])
    return dst


def gen_color_image(dst, *, w=WIDTH, h=HEIGHT, color="0x2b2d42"):
    """A single solid-colour PNG — placeholder 'photo' for ken-burns clips."""
    _run(["-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}", "-frames:v", "1", dst])
    return dst


# ----------------------------------------------------------------------------
# Clip builders.
# ----------------------------------------------------------------------------

def _normalise_audio_args():
    return ["-ar", str(AR), "-ac", str(AC), "-c:a", "aac", "-b:a", "192k"]


def still_clip(image, audio, dst, *, motion=True):
    """Image + audio -> mp4 whose length matches the audio. With `motion`, a slow
    ken-burns zoom; falls back to a static frame if the zoompan filter errors."""
    dur = max(media_duration(audio), 0.5)
    if motion:
        try:
            vf = (f"scale={WIDTH*2}:{HEIGHT*2}:force_original_aspect_ratio=increase,"
                  f"crop={WIDTH*2}:{HEIGHT*2},"
                  f"zoompan=z='min(zoom+0.0008,1.25)':d={int(dur*FPS)}:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
                  f"format=yuv420p")
            _run(["-loop", "1", "-i", image, "-i", audio,
                  "-vf", vf, "-t", f"{dur}",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
                  *_normalise_audio_args(), "-shortest", dst])
            return dst
        except RuntimeError:
            pass  # fall through to static
    vf = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
          f"crop={WIDTH}:{HEIGHT},format=yuv420p")
    _run(["-loop", "1", "-i", image, "-i", audio,
          "-vf", vf, "-t", f"{dur}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
          *_normalise_audio_args(), "-shortest", dst])
    return dst


def replace_audio(video, audio, dst):
    """Swap a silent (or scratch) video's audio track for a real one, e.g. laying
    narration onto a concatenated ken-burns visual timeline."""
    _run(["-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
          "-c:v", "copy", *_normalise_audio_args(), "-shortest", dst])
    return dst


def mix_bgm(video, bgm, dst, *, volume=0.18):
    """Lay `bgm` quietly under a video's existing audio, trimmed to the video length."""
    dur = media_duration(video)
    flt = (f"[1:a]volume={volume},aloop=loop=-1:size=2e9,atrim=0:{dur},"
           f"afade=t=out:st={max(0.0,dur-1.0)}:d=1.0[bg];"
           f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]")
    _run(["-i", video, "-i", bgm, "-filter_complex", flt,
          "-map", "0:v", "-map", "[a]", "-c:v", "copy",
          *_normalise_audio_args(), dst])
    return dst


def overlay_sfx(video, sfx, dst, *, at=0.0, volume=0.9):
    """Mix a one-shot SFX into a video's audio starting at `at` seconds."""
    delay = int(max(at, 0.0) * 1000)
    flt = (f"[1:a]volume={volume},adelay={delay}|{delay}[s];"
           f"[0:a][s]amix=inputs=2:duration=first:dropout_transition=0[a]")
    _run(["-i", video, "-i", sfx, "-filter_complex", flt,
          "-map", "0:v", "-map", "[a]", "-c:v", "copy",
          *_normalise_audio_args(), dst])
    return dst


def concat_clips(clips, dst):
    """Concatenate normalised clips (each with 1 video + 1 audio stream)."""
    clips = [c for c in clips if c]
    if not clips:
        raise ValueError("no clips to concat")
    if len(clips) == 1:
        _run(["-i", clips[0], "-c", "copy", dst])
        return dst
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(clips)))
    flt = f"{streams}concat=n={len(clips)}:v=1:a=1[v][a]"
    _run([*inputs, "-filter_complex", flt, "-map", "[v]", "-map", "[a]",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
          *_normalise_audio_args(), dst])
    return dst
