"""Reflex runtime: `Skill` (on-device student) and `Teacher` (cloud Decisions API).

Standalone on purpose: only numpy + onnxruntime + opencv (+ stdlib). This file is
copied into every exported skill folder as `predict.py`, so a skill folder works
without the rest of Reflex installed.

    from reflex.runtime import Skill, Teacher      # or: from predict import Skill, Teacher
    skill = Skill("skills/rps")
    skill.predict(frame_bgr)  # {"label": "rock", "probs": {...}, "ms": 3.2}
    Teacher("skills/rps").predict(frame_bgr)       # same dict, "ms" ~ 300+

CLI (from inside a skill folder):
    python predict.py image.jpg [--teacher]
    python predict.py --webcam [--teacher]
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DECISIONS_URL = "https://api.openai.com/v1/decisions"
DEFAULT_TEACHER_MODEL = "gpt-6-luna"
TEACHER_IMAGE_SIZE = 384  # long side, px — same as ingest
BACKGROUND_NAMES = ("none", "unclear", "other", "background", "nothing")


# ---------------------------------------------------------------- spec helpers
# Shared with reflex.teacher (async labeller) so the live Teacher and the
# training labels are produced by exactly the same request/parse code.

def spec_classes(spec: dict) -> list[str]:
    q = spec["question"]
    if q["type"] == "predicate":
        return ["no", "yes"]
    return [c["value"] for c in q["choices"]]


def background_class(spec: dict) -> str | None:
    """The 'nothing happening' class: target of hindsight propagation, excluded from cues."""
    classes = spec_classes(spec)
    if spec.get("background_class") in classes:
        return spec["background_class"]
    if spec["question"]["type"] == "predicate":
        return "no"
    for c in classes:
        if c.lower() in BACKGROUND_NAMES:
            return c
    return None


def encode_jpeg_b64(frame_bgr: np.ndarray, long_side: int = TEACHER_IMAGE_SIZE, quality: int = 85) -> str:
    h, w = frame_bgr.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode()


def build_decisions_body(spec: dict, image_b64: str, model: str | None = None) -> dict:
    return {
        "model": model or os.environ.get("REFLEX_TEACHER_MODEL", DEFAULT_TEACHER_MODEL),
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": spec["task_summary"]},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ]}],
        "questions": [spec["question"]],
    }


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(vars(obj))


def parse_decision(resp, spec: dict) -> tuple[dict[str, float], dict]:
    """Decisions API response -> ({class: prob}, usage dict)."""
    resp = _as_dict(resp)
    classes = spec_classes(spec)
    a = _as_dict(resp["answers"][0])
    refused = a.get("type") == "refusal"
    if refused:
        # No answer: uniform (so it is down-weighted as low-confidence), ties go to background.
        bg = background_class(spec)
        probs = {c: (1.01 if c == bg else 1.0) / (len(classes) + 0.01) for c in classes}
    elif spec["question"]["type"] == "predicate":
        p = float(a.get("probability", 1.0 if a.get("value") in (True, "yes") else 0.0))
        probs = {"no": 1 - p, "yes": p}
    else:
        probs = {c: 0.0 for c in classes}
        plist = a.get("probabilities") or []
        for item in plist:
            item = _as_dict(item)
            if item.get("value") in probs:
                probs[item["value"]] = float(item.get("probability", 0.0))
        if not plist or sum(probs.values()) <= 0:
            # Only a top choice + confidence came back: spread the rest evenly.
            conf = float(a.get("confidence", 1.0))
            rest = (1 - conf) / max(1, len(classes) - 1)
            probs = {c: (conf if c == a.get("choice") else rest) for c in classes}
        total = sum(probs.values()) or 1.0
        probs = {c: v / total for c, v in probs.items()}
    usage = _as_dict(resp.get("usage") or {}) if resp.get("usage") is not None else {}
    return probs, {"input_tokens": int(usage.get("input_tokens") or 0),
                   "output_tokens": int(usage.get("output_tokens") or 0), "refused": refused}


def _result(classes, probs, ms) -> dict:
    probs = {c: float(p) for c, p in zip(classes, probs)} if not isinstance(probs, dict) else probs
    label = max(probs, key=probs.get)
    return {"label": label, "probs": probs, "ms": round(ms, 2)}


# ---------------------------------------------------------------- student

def preprocess(frame_bgr: np.ndarray, input_size, mean, std) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 1x3xHxW. Plain resize (no crop), same as training."""
    h, w = input_size
    img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _providers(device: str | None) -> list[str]:
    """'auto' (default): Apple's CoreML (Neural Engine/GPU) when onnxruntime has it, else CPU.
    CoreML keeps the student fast even when the CPU is busy. Override: device='cpu' or REFLEX_DEVICE."""
    import onnxruntime as ort

    device = (device or os.environ.get("REFLEX_DEVICE") or "auto").lower()
    if device in ("auto", "coreml") and "CoreMLExecutionProvider" in ort.get_available_providers():
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class Skill:
    """On-device student: model.onnx + skill.json."""

    def __init__(self, path: str | os.PathLike, threads: int | None = None, device: str | None = None):
        import onnxruntime as ort

        self.path = Path(path)
        self.meta = json.loads((self.path / "skill.json").read_text())
        self.name = self.meta["name"]
        self.classes = self.meta["classes"]
        self.input_size = tuple(self.meta["input_size"])  # (H, W)
        self.mean, self.std = self.meta["mean"], self.meta["std"]
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path / "model.onnx"), so, providers=_providers(device))
        self.device = "coreml" if "CoreMLExecutionProvider" in self.session.get_providers() else "cpu"
        self.input_name = self.session.get_inputs()[0].name
        self.predict(np.zeros((*self.input_size, 3), np.uint8))  # warm-up

    def predict_probs(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = preprocess(frame_bgr, self.input_size, self.mean, self.std)
        return _softmax(self.session.run(None, {self.input_name: x})[0][0])

    def predict(self, frame_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        p = self.predict_probs(frame_bgr)
        return _result(self.classes, p, (time.perf_counter() - t0) * 1000)

    def predict_zones(self, frame_bgr: np.ndarray, zones: dict) -> dict:
        """Run the skill on fixed crops {name: (x, y, w, h)} -> {name: result}. Poor-man's
        localisation: train the skill on footage cropped the same way for best results."""
        return {k: self.predict(frame_bgr[y:y + h, x:x + w]) for k, (x, y, w, h) in zones.items()}


# ---------------------------------------------------------------- teacher

class Teacher:
    """Cloud brain: same interface as Skill, but asks the Decisions API live."""

    def __init__(self, path: str | os.PathLike, model: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0):
        _load_dotenv()
        self.path = Path(path)
        self.meta = json.loads((self.path / "skill.json").read_text())
        self.spec = self.meta["spec"]
        self.classes = spec_classes(self.spec)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("Teacher needs OPENAI_API_KEY (env or .env)")

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(5):
            req = urllib.request.Request(DECISIONS_URL, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(min(8, 0.5 * 2 ** attempt) + random.random() * 0.3)
                    continue
                raise RuntimeError(f"Decisions API {e.code}: {e.read().decode(errors='replace')[:500]}") from e
        raise RuntimeError("Decisions API: retries exhausted")

    def predict(self, frame_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        body = build_decisions_body(self.spec, encode_jpeg_b64(frame_bgr), self.model)
        probs, _ = parse_decision(self._post(body), self.spec)
        return _result(self.classes, probs, (time.perf_counter() - t0) * 1000)


def _load_dotenv():
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass


# ---------------------------------------------------------------- CLI

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Run a Reflex skill on an image or webcam.")
    ap.add_argument("image", nargs="?", help="image path (omit with --webcam)")
    ap.add_argument("--skill", default=str(Path(__file__).resolve().parent), help="skill folder")
    ap.add_argument("--teacher", action="store_true", help="use the cloud teacher instead")
    ap.add_argument("--webcam", type=int, nargs="?", const=0, default=None, help="camera index")
    args = ap.parse_args()
    brain = (Teacher if args.teacher else Skill)(args.skill)

    if args.webcam is None:
        if not args.image:
            ap.error("give an image path or --webcam")
        print(json.dumps(brain.predict(cv2.imread(args.image)), indent=2))
        return

    cap = cv2.VideoCapture(args.webcam)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = brain.predict(frame)
        txt = f"{out['label']} {out['probs'][out['label']]:.2f}  {out['ms']:.1f} ms"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("reflex", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break


if __name__ == "__main__":
    _main()
