"""Hand crop front-end for the rock-paper-scissors skill.

The classifier never sees the full camera frame: MediaPipe finds the hand and we cut a
square around it. Training clips (person ~2-3 m away, 4K) and a laptop webcam (hand at
arm's length) then produce the same kind of image, so the model doesn't care about distance.

Extract training crops (one sub-folder per video = one Reflex episode):
    python examples/rps/handcrop.py data/rock-paper-scissors data/rps-hands --fps 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
DETECT_WIDTH = 960   # MediaPipe runs on a downscaled copy; the crop is cut from full-res
CROP_SIZE = 224
MARGIN = 0.35        # extra border around the landmark box, as a fraction of its side


class HandCropper:
    def __init__(self, video: bool = False, min_conf: float = 0.4):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), num_hands=1,
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf)
        self.det = vision.HandLandmarker.create_from_options(opts)
        self.video = video

    def box(self, frame_bgr: np.ndarray, t_ms: int = 0) -> tuple[int, int, int, int] | None:
        """Square (x0, y0, x1, y1) around the hand in full-res pixels, or None if no hand."""
        h, w = frame_bgr.shape[:2]
        s = min(1.0, DETECT_WIDTH / w)
        small = cv2.resize(frame_bgr, (round(w * s), round(h * s))) if s < 1 else frame_bgr
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        res = self.det.detect_for_video(img, t_ms) if self.video else self.det.detect(img)
        if not res.hand_landmarks:
            return None
        xs = np.array([p.x for p in res.hand_landmarks[0]]) * w
        ys = np.array([p.y for p in res.hand_landmarks[0]]) * h
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * MARGIN) / 2
        return round(cx - half), round(cy - half), round(cx + half), round(cy + half)

    def crop(self, frame_bgr: np.ndarray, t_ms: int = 0) -> tuple[np.ndarray | None, tuple | None]:
        b = self.box(frame_bgr, t_ms)
        if b is None:
            return None, None
        x0, y0, x1, y1 = b
        h, w = frame_bgr.shape[:2]
        # Pad with black where the square runs off the frame edge, so the hand stays centred.
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        if pad:
            frame_bgr = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        patch = frame_bgr[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
        return cv2.resize(patch, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA), b


def extract(src: Path, dst: Path, fps: float) -> None:
    for video in sorted(p for p in src.iterdir() if p.suffix.lower() in {".mov", ".mp4"}):
        out = dst / video.stem
        out.mkdir(parents=True, exist_ok=True)
        # AVFoundation = macOS hardware HEVC decode (~7x faster than ffmpeg on 4K iPhone clips)
        cap = cv2.VideoCapture(str(video), cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(video))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cropper = HandCropper(video=True)
        i = kept = seen = 0
        next_t = 0.0
        while cap.grab():  # grab() skips decoding-to-BGR for frames we don't sample
            t = i / native
            i += 1
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / fps
            ok, frame = cap.retrieve()
            if not ok:
                break
            seen += 1
            patch, _ = cropper.crop(frame, round(t * 1000))
            if patch is not None:
                # zero-padded timestamp in the name keeps file order == time order
                cv2.imwrite(str(out / f"{round(t * 1000):07d}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 90])
                kept += 1
        cap.release()
        print(f"{video.name}: {kept}/{seen} sampled frames had a hand -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--fps", type=float, default=6.0)
    a = ap.parse_args()
    extract(a.src, a.dst, a.fps)
