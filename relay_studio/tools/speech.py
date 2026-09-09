"""synthesize_speech tool — turns the cat's lines into an audio file, writing the
result back to the artifact as TTS_AUDIO_URL.

Backend order (swappable stub): edge-tts (online, high quality) -> pyttsx3 (offline
Windows SAPI) -> a tone placeholder so the pipeline never hard-fails. The chosen
`voice_name` is resolved against the voice library (voices.py)."""

import os

from .. import ffmpeg, voices


def _edge(text, dst, voice):
    import asyncio
    import edge_tts  # type: ignore
    v = (voice or {}).get("edge") or "en-US-GuyNeural"
    mp3 = os.path.splitext(dst)[0] + ".mp3"

    async def go():
        await edge_tts.Communicate(text, v).save(mp3)
    asyncio.run(go())
    return mp3


def _pyttsx3(text, dst, voice):
    import pyttsx3  # type: ignore
    wav = os.path.splitext(dst)[0] + ".wav"
    engine = pyttsx3.init()
    want = (voice or {}).get("gender", "")
    if want:
        for sv in engine.getProperty("voices"):
            blob = f"{sv.id} {getattr(sv, 'name', '')} {getattr(sv, 'gender', '')}".lower()
            if want in blob or (want == "female" and "zira" in blob) or \
               (want == "male" and "david" in blob):
                engine.setProperty("voice", sv.id)
                break
    engine.save_to_file(text, wav)
    engine.runAndWait()
    if not os.path.exists(wav) or os.path.getsize(wav) == 0:
        raise RuntimeError("pyttsx3 produced no audio")
    return wav


def _placeholder(text, dst):
    dur = max(2.0, min(len(text.split()) * 0.4, 30.0))
    wav = os.path.splitext(dst)[0] + ".wav"
    ffmpeg.gen_tone(wav, duration=dur, freq=160.0, volume=0.2, harmony=False)
    return wav


def synthesize_speech(project, text, voice_name="", voice_language="", **_):
    """Generate spoken audio for `text` and write it to the artifact as TTS_AUDIO_URL."""
    text = (text or "").strip()
    if not text:
        return {"error": "no text to speak"}
    voice = voices.get_voice(voice_name) or voices.get_voice(project.inputs.get("voice_name"))
    base = project.media_path(f"voice_{len([m for m in project.media if m['kind']=='audio'])+1}")

    backend, path = None, None
    for name, fn in (("edge-tts", lambda: _edge(text, base, voice)),
                     ("pyttsx3", lambda: _pyttsx3(text, base, voice)),
                     ("placeholder", lambda: _placeholder(text, base))):
        try:
            path = fn()
            backend = name
            break
        except Exception:
            continue

    dur = ffmpeg.media_duration(path)
    vname = (voice or {}).get("name", voice_name or "voice")
    item = project.add_media("audio", path, via=backend,
                             label=f"{project.inputs.get('cat_name','Cat')} voice — {vname}",
                             voice=vname, duration=round(dur, 2))
    project.write_artifact("TTS_AUDIO_URL", media_id=item["id"])
    return {"ok": True, "media_id": item["id"], "file": item["file"], "voice": vname,
            "duration": round(dur, 2), "backend": backend, "spoken_text": text,
            "artifact": "TTS_AUDIO_URL (v%d)" % project.ver}
