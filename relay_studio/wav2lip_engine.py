"""Real lip-sync via Wav2Lip (custom_nodes/ComfyUI_wav2lip).

Calls the node's core `wav2lip_(frames, audio, batch, mode, model)` function directly
(no ComfyUI graph) with a single still image in "repetitive" mode, so the one photo's
mouth is animated to the voice track. Output is normalised to the studio's clip format
(768x768, 30fps, h264/yuv420p, AAC stereo) so the final mix concatenates cleanly.

Wav2Lip is trained on human faces and needs a detectable face; if face detection fails
(e.g. a stylised animal face) run_wav2lip raises, and lipsync.py falls back to ken-burns.
"""

import os
import sys
import tempfile

from . import ffmpeg

_NODE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ComfyUI_wav2lip",
)
_MODEL = os.path.join(_NODE_DIR, "Wav2Lip", "checkpoints", "wav2lip_gan.pth")

_fn = None


def available():
    """True if the node + GAN weights are present."""
    return os.path.isfile(_MODEL)


def _get_fn():
    """Import and cache the node's core wav2lip_ inference function."""
    global _fn
    if _fn is not None:
        return _fn
    if _NODE_DIR not in sys.path:
        sys.path.insert(0, _NODE_DIR)
    from Wav2Lip.wav2lip_node import wav2lip_  # namespace package, relative imports inside
    _fn = wav2lip_
    return _fn


_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif")


def _read_frames(path):
    """Return a list of BGR frames. A still image -> [frame]; a video -> all frames."""
    import cv2

    if path.lower().endswith(_VIDEO_EXTS):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError(f"could not read frames from video: {path}")
        return frames
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"could not read image: {path}")
    return [img]


def _robust_infer(frames, audio_path, face_detect_batch=8):
    """Wav2Lip inference that never drops frames.

    The stock node skips any frame where face detection fails, which truncates the
    output and desyncs audio on a moving (animated) subject. Here we hold the last
    valid face box (or a centred default) so every audio frame gets rendered — the
    output length always matches the audio."""
    import numpy as np
    import cv2
    import torch

    if _NODE_DIR not in sys.path:
        sys.path.insert(0, _NODE_DIR)
    from Wav2Lip import audio as w2l_audio
    from Wav2Lip import face_detection
    from Wav2Lip.wav2lip_node import load_model, mel_step_size, device

    img_size = 96
    wav = w2l_audio.load_wav(audio_path, 16000)
    mel = w2l_audio.melspectrogram(wav)

    mel_chunks = []
    mel_idx_multiplier = 80. / 30  # output is 30 fps
    i = 0
    while True:
        start = int(i * mel_idx_multiplier)
        if start + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start: start + mel_step_size])
        i += 1
    n_out = len(mel_chunks)

    # Detect a face box per frame, holding the previous box when detection fails.
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
    preds = []
    for i in range(0, len(frames), face_detect_batch):
        preds.extend(detector.get_detections_for_batch(np.array(frames[i:i + face_detect_batch])))
    del detector

    pady1, pady2, padx1, padx2 = 0, 10, 0, 0
    boxes, last = [], None
    for rect, image in zip(preds, frames):
        h, w = image.shape[:2]
        if rect is None:
            box = last if last is not None else [w // 4, h // 5, w * 3 // 4, h * 4 // 5]
        else:
            box = [max(0, rect[0] - padx1), max(0, rect[1] - pady1),
                   min(w, rect[2] + padx2), min(h, rect[3] + pady2)]
            last = box
        boxes.append(box)

    # Map each output frame to a source frame: sequential ride over the motion,
    # stretched/compressed so the whole motion clip spans the audio length.
    frame_size = len(frames)
    model = load_model(str(_MODEL))
    out_images = []
    batch_img, batch_mel, batch_frame, batch_coord = [], [], [], []

    def flush():
        if not batch_img:
            return
        ib = np.asarray(batch_img)
        mb = np.asarray(batch_mel)
        masked = ib.copy()
        masked[:, img_size // 2:] = 0
        ib = np.concatenate((masked, ib), axis=3) / 255.
        mb = np.reshape(mb, [len(mb), mb.shape[1], mb.shape[2], 1])
        it = torch.FloatTensor(np.transpose(ib, (0, 3, 1, 2))).to(device)
        mt = torch.FloatTensor(np.transpose(mb, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mt, it)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        for p, fr, c in zip(pred, batch_frame, batch_coord):
            x1, y1, x2, y2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            fr[y1:y2, x1:x2] = p
            out_images.append(fr)
        batch_img.clear(); batch_mel.clear(); batch_frame.clear(); batch_coord.clear()

    for i, m in enumerate(mel_chunks):
        idx = 0 if frame_size == 1 else min(frame_size - 1, int(i * frame_size / n_out))
        x1, y1, x2, y2 = boxes[idx]
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = boxes[idx] = [0, 0, frames[idx].shape[1], frames[idx].shape[0]]
        face = cv2.resize(frames[idx][y1:y2, x1:x2], (img_size, img_size))
        batch_img.append(face); batch_mel.append(m)
        batch_frame.append(frames[idx].copy()); batch_coord.append((x1, y1, x2, y2))
        if len(batch_img) >= 128:
            flush()
    flush()
    return out_images


def run_wav2lip(image_path, audio_path, dst, *, face_detect_batch=8):
    """Animate the mouth on `image_path` to `audio_path`, writing `dst` (mp4).

    `image_path` may be a still image (mouth animated on one frame) OR a video
    (e.g. an LTX motion clip — mouth synced across the moving frames).

    Raises on any failure (missing model, no face detected, etc.) so the caller can
    fall back to the ken-burns stub. Returns dst on success."""
    if not available():
        raise RuntimeError("wav2lip model not installed")

    import cv2

    frames = _read_frames(image_path)

    tmp_wav = tmp_silent = None
    try:
        # Wav2Lip wants a clean 16 kHz mono wav; librosa can choke on arbitrary mp3.
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        ffmpeg._run(["-i", audio_path, "-ar", "16000", "-ac", "1", tmp_wav])

        out_frames = _robust_infer(frames, tmp_wav, int(face_detect_batch))
        if not out_frames:
            raise RuntimeError("wav2lip produced no frames (face not detected?)")

        h, w = out_frames[0].shape[:2]
        tmp_silent = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        vw = cv2.VideoWriter(tmp_silent, cv2.VideoWriter_fourcc(*"mp4v"),
                             ffmpeg.FPS, (w, h))
        if not vw.isOpened():
            raise RuntimeError("could not open VideoWriter for wav2lip frames")
        for f in out_frames:
            vw.write(f)
        vw.release()

        # Normalise to the studio clip format and attach the original (full-quality) audio.
        vf = (f"scale={ffmpeg.WIDTH}:{ffmpeg.HEIGHT}:force_original_aspect_ratio=decrease,"
              f"pad={ffmpeg.WIDTH}:{ffmpeg.HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
              f"fps={ffmpeg.FPS},format=yuv420p")
        ffmpeg._run(["-i", tmp_silent, "-i", audio_path,
                     "-map", "0:v:0", "-map", "1:a:0", "-vf", vf,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(ffmpeg.FPS),
                     *ffmpeg._normalise_audio_args(), "-shortest", dst])
        return dst
    finally:
        for p in (tmp_wav, tmp_silent):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
