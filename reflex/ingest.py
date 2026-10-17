"""Stage 1: videos/images -> frames/*.jpg + frames.csv, plus a measured time/cost estimate."""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import cv2

from .utils import ReflexError, console, load_frames, pctl, price_usd, read_json, write_json

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Generic question used for the estimate when no spec exists yet (similar token cost).
PLACEHOLDER_SPEC = {
    "task_summary": "Look at the camera frame.",
    "question": {"type": "predicate", "name": "has_person",
                 "instructions": "Is any person or part of a person (e.g. a hand) visible in the image?"},
}


def _slug(p: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(p.with_suffix(""))).strip("_") or "root"


def _resize_long(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def _video_info(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def ingest(data_dir: Path, run: Path, fps: float = 5.0, max_frames: int = 1500, size: int = 384,
           estimate_n: int = 20, concurrency: int = 16) -> dict:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ReflexError(f"data dir {data_dir} not found")
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    if not videos and not images:
        raise ReflexError(f"no videos or images in {data_dir}")

    # Lower the sampling rate if the full set would exceed max_frames.
    durations = {v: (lambda f, n: n / f)(*_video_info(v)) for v in videos}
    total_dur = sum(durations.values())
    budget = max(1, max_frames - len(images))
    eff_fps = min(fps, budget / total_dur) if total_dur > 0 else fps
    if eff_fps < fps:
        console.print(f"[yellow]capping at {max_frames} frames: sampling {eff_fps:.2f} fps instead of {fps}[/yellow]")

    frames_dir = run / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for v in videos:
        ep = _slug(v.relative_to(data_dir))
        cap = cv2.VideoCapture(str(v))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        next_t, i, k = 0.0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / native
            i += 1
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / eff_fps
            fid = f"{ep}__{k:05d}"
            rel = f"frames/{fid}.jpg"
            cv2.imwrite(str(run / rel), _resize_long(frame, size), [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append({"frame_id": fid, "path": rel, "episode": ep, "t_sec": round(t, 3), "idx": k})
            k += 1
        cap.release()

    # Images: each sub-folder is an episode, file order is time.
    by_folder: dict[Path, list[Path]] = {}
    for p in images:
        by_folder.setdefault(p.parent, []).append(p)
    for folder, imgs in by_folder.items():
        ep = "img_" + _slug(folder.relative_to(data_dir)) if folder != data_dir else "images"
        for k, p in enumerate(sorted(imgs)):
            img = cv2.imread(str(p))
            if img is None:
                continue
            fid = f"{ep}__{k:05d}"
            rel = f"frames/{fid}.jpg"
            cv2.imwrite(str(run / rel), _resize_long(img, size), [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append({"frame_id": fid, "path": rel, "episode": ep, "t_sec": float(k), "idx": k})

    with open(run / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "path", "episode", "t_sec", "idx"])
        w.writeheader()
        w.writerows(rows)

    episodes = sorted({r["episode"] for r in rows})
    out = {"run": run.name, "n_frames": len(rows), "n_episodes": len(episodes), "episodes": episodes,
           "videos": len(videos), "images": len(images), "fps": round(eff_fps, 3),
           "video_seconds": round(total_dur, 1), "frames_csv": str(run / "frames.csv")}
    console.print(f"[green]{len(rows)} frames[/green] from {len(videos)} videos + {len(images)} images, "
                  f"{len(episodes)} episodes")
    if estimate_n > 0:
        try:
            out["estimate"] = estimate(run, n=estimate_n, concurrency=concurrency)
        except ReflexError as e:  # frames are still usable; the estimate can be redone later
            console.print(f"[yellow]estimate skipped: {e}[/yellow]")
            out["estimate"] = {"error": str(e)}
    return out


def estimate(run: Path, n: int = 20, concurrency: int = 16) -> dict:
    """Label n random frames, extrapolate time and cost for the full set."""
    from .teacher import label_frames

    frames = load_frames(run)
    spec = read_json(run / "spec.json", default={})
    use_spec = bool(spec.get("question"))
    sample = random.Random(0).sample(frames, min(n, len(frames)))
    rows, _ = label_frames(spec if use_spec else PLACEHOLDER_SPEC, sample, run,
                           concurrency=concurrency, cache=use_spec, desc="estimate")
    lat = [r["latency_ms"] for r in rows]
    tin = sum(r["input_tokens"] for r in rows) / len(rows)
    tout = sum(r["output_tokens"] for r in rows) / len(rows)
    total = len(frames)
    minutes = total * pctl(lat, 50) / 1000 / concurrency / 60
    usd = price_usd(tin, tout) * total
    est = {"sampled": len(rows), "used_spec": use_spec, "latency_p50_ms": round(pctl(lat, 50), 1),
           "latency_p95_ms": round(pctl(lat, 95), 1), "input_tokens_per_frame": round(tin, 1),
           "output_tokens_per_frame": round(tout, 1), "total_frames": total,
           "est_minutes": round(minutes, 1), "est_usd": round(usd, 3), "concurrency": concurrency}
    write_json(run / "estimate.json", est)
    console.print(f"[bold]{total:,} frames ≈ {minutes:.1f} min at concurrency {concurrency}, ≈ ${usd:.2f}[/bold]"
                  f" [dim](p50 {est['latency_p50_ms']:.0f} ms, {tin:.0f} in-tokens/frame)[/dim]")
    return est
