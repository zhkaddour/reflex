"""Synthetic test footage for smoke-testing Reflex without real data.

    python examples/make_synthetic.py data/shapes            # 4 videos: circle/square/triangle/none
    python examples/make_synthetic.py data/shapes --fake-labels runs/shapes
        # after `reflex ingest data/shapes --run shapes`: writes spec.json + labels from ground
        # truth (no API calls), so train/eval/export can be tested offline.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["circle", "square", "triangle", "none"]
W, H, FPS, SECONDS, SEG = 640, 480, 30, 16, 1.6


def draw(frame, cls, cx, cy, r, color):
    if cls == "circle":
        cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
    elif cls == "square":
        cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    elif cls == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)


def make_video(path: Path, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    segs, t = [], 0.0
    while t < SECONDS:
        segs.append({"start": t, "end": t + SEG, "class": CLASSES[rng.integers(len(CLASSES))],
                     "color": tuple(int(c) for c in rng.integers(40, 255, 3)), "r": int(rng.integers(40, 90)),
                     "x0": int(rng.integers(120, W - 120)), "y0": int(rng.integers(120, H - 120)),
                     "vx": float(rng.uniform(-60, 60)), "vy": float(rng.uniform(-40, 40))})
        t += SEG
    bg = np.tile(np.linspace(60, 160, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
    bg[..., 0] = (bg[..., 0] * rng.uniform(0.6, 1.0)).astype(np.uint8)
    for i in range(int(SECONDS * FPS)):
        ts = i / FPS
        s = next(s for s in segs if s["start"] <= ts < s["end"])
        frame = bg.copy()
        frame = np.clip(frame + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
        dt = ts - s["start"]
        draw(frame, s["class"], int(s["x0"] + s["vx"] * dt), int(s["y0"] + s["vy"] * dt), s["r"], s["color"])
        out.write(frame)
    out.release()
    return [{k: s[k] for k in ("start", "end", "class")} for s in segs]


def fake_labels(data_dir: Path, run: Path):
    import csv

    gt = json.loads((data_dir / "ground_truth.json").read_text())
    spec = {
        "name": run.name, "query": "what shape is on screen?",
        "task_summary": "Identify the single filled shape on the screen.",
        "visual_cues": ["circle: round", "square: four right angles", "triangle: three corners", "none: empty"],
        "question": {"type": "choice", "name": "shape", "instructions": "Which shape is visible?",
                     "choices": [{"value": c, "description": c} for c in CLASSES]},
        "edge_cases": [], "min_conf": 0.6, "propagate_back": 0, "hflip_ok": True, "answers": {}, "status": "ready",
    }
    (run / "spec.json").write_text(json.dumps(spec, indent=2))
    rng = np.random.default_rng(0)
    rows = []
    for f in csv.DictReader(open(run / "frames.csv")):
        segs = gt[f["episode"]]
        t = float(f["t_sec"])
        cls = next((s["class"] for s in segs if s["start"] <= t < s["end"]), "none")
        conf = float(rng.uniform(0.75, 0.99))
        probs = {c: (conf if c == cls else (1 - conf) / 3) for c in CLASSES}
        rows.append({"frame_id": f["frame_id"], "probs": probs, "top": cls, "conf": conf,
                     "latency_ms": float(rng.normal(420, 60)), "input_tokens": 600, "output_tokens": 20,
                     "source": "teacher", "spec_hash": "synthetic", "weight": conf})
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (run / "labels.jsonl").write_text(text)
    (run / "labels_final.jsonl").write_text(text)
    print(f"wrote spec + {len(rows)} fake labels into {run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--fake-labels", type=Path, help="run folder to fill with ground-truth labels")
    a = ap.parse_args()
    if a.fake_labels:
        fake_labels(a.out, a.fake_labels)
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        gt = {f"shapes_{i}": make_video(a.out / f"shapes_{i}.mp4", seed=i) for i in range(a.videos)}
        (a.out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
        print(f"wrote {a.videos} videos to {a.out}")
