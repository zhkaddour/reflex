"""Run brains against belts in real time, record training footage, sweep belt speed, score vs ground truth."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..utils import ReflexError, console, read_json, read_jsonl, write_json
from .brains import Brain
from .sim import PX_PER_M, Belt, Clock, random_light


class Runner:
    """`brain.workers` threads, each: grab the next camera frame (shared camera clock, `cam_fps`),
    ask the brain, hand the answer back to the belt. API brains block on the network, so several
    workers keep several requests in flight — this raises the decision *rate*, not the latency."""

    def __init__(self, belt: Belt, brain: Brain, cam_fps: float = 30.0):
        self.belt, self.brain, self.cam_fps = belt, brain, cam_fps
        belt.min_votes = brain.min_votes
        self.last: dict | None = None            # last answered snapshot, for the web view
        self._next_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.paused = False
        self.threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(max(1, brain.workers))]

    def start(self):
        for t in self.threads:
            t.start()
        return self

    def stop(self):
        self._stop.set()

    def _wait_frame(self):
        with self._lock:
            now = time.perf_counter()
            t = max(now, self._next_t)
            self._next_t = t + 1 / self.cam_fps
        time.sleep(max(0.0, t - time.perf_counter()))

    def _loop(self):
        while not self._stop.is_set():
            if self.paused:
                time.sleep(0.05)
                continue
            self._wait_frame()
            snap = self.belt.capture()
            try:
                out = self.brain.predict(snap["frame"], snap["gt"])
            except Exception as e:  # noqa: BLE001 — e.g. wifi off: show it on the belt, keep retrying
                self.belt.decide(snap, False, 0.0, error=f"{type(e).__name__}: {e}")
                time.sleep(1.0)
                continue
            self.belt.decide(snap, out["push"], out["ms"], out.get("usd", 0.0))
            self.last = {**snap, **out}


# ---------------------------------------------------------------- training footage

def record(out: Path, sprites: dict, episodes: int = 12, seconds: float = 60.0, fps: float = 10.0,
           speed_range: tuple[float, float] = (0.1, 2.5), seed: int = 1000) -> dict:
    """Headless (faster than real time): one mp4 of camera frames per episode, plus
    ground_truth.json (per frame: is a target in the zone). Speed, lighting and object density vary
    per episode so the student sees the conditions it will be deployed in."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gt_all, params, dt, pos = {}, {}, 1 / 60, 0
    for e in range(episodes):
        ep = f"ep{e:03d}"
        speed = float(np.exp(rng.uniform(*np.log(speed_range)))) * PX_PER_M
        belt = Belt(sprites, seed=seed + e, speed=speed, p_target=float(rng.uniform(0.2, 0.45)),
                    gap=(float(rng.uniform(130, 190)), float(rng.uniform(260, 380))), light=random_light(rng))
        while belt.t < 1300 / speed:          # pre-roll until the belt is full
            belt.step(dt)
        writer, rows, t0 = None, [], belt.t
        n = int(seconds * fps)
        for i in range(n):
            while belt.t - t0 < i / fps:
                belt.step(dt)
            snap = belt.capture(in_flight=False)
            if writer is None:
                h, w = snap["frame"].shape[:2]
                writer = cv2.VideoWriter(str(out / f"{ep}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(snap["frame"])
            rows.append({"i": i, "t": round(i / fps, 3), "target_in_zone": snap["gt"]["target_in_zone"],
                         "kinds": snap["gt"]["kinds_in_zone"], "ambiguous": snap["gt"]["ambiguous"]})
        writer.release()
        gt_all[ep] = rows
        pos += sum(r["target_in_zone"] for r in rows)
        params[ep] = {"speed_mps": round(speed / PX_PER_M, 3), "light": belt.light, "p_target": belt.p_target}
        console.print(f"  {ep}: {speed / PX_PER_M:.2f} m/s, {sum(r['target_in_zone'] for r in rows)}/{n} frames with a target")
    write_json(out / "ground_truth.json", {"fps": fps, "target": "plastic_bottle", "episodes": gt_all, "params": params})
    total = episodes * int(seconds * fps)
    return {"dir": str(out), "episodes": episodes, "frames": total, "positive_frac": round(pos / total, 3),
            "ground_truth": str(out / "ground_truth.json")}


def score(run: Path, data: Path) -> dict:
    """Teacher labels and student predictions vs the simulator's ground truth."""
    gt = read_json(Path(data) / "ground_truth.json")
    fps = gt["fps"]
    truth, ambiguous = {}, set()
    with open(run / "frames.csv") as f:
        for r in csv.DictReader(f):
            rows = gt["episodes"].get(r["episode"])
            if rows:
                i = min(len(rows) - 1, round(float(r["t_sec"]) * fps))
                truth[r["frame_id"]] = "yes" if rows[i]["target_in_zone"] else "no"
                if rows[i].get("ambiguous"):
                    ambiguous.add(r["frame_id"])
    if not truth:
        raise ReflexError(f"no frames of {run} match episodes in {data}/ground_truth.json")

    def _acc(pairs, clear_only=False):
        pairs = [(p, truth[i]) for i, p in pairs if i in truth and not (clear_only and i in ambiguous)]
        if not pairs:
            return None
        tp = sum(p == g == "yes" for p, g in pairs)
        fp = sum(p == "yes" != g for p, g in pairs)
        fn = sum(g == "yes" != p for p, g in pairs)
        return {"n": len(pairs), "accuracy": round(sum(p == g for p, g in pairs) / len(pairs), 4),
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None}

    labels = read_jsonl(run / "labels_final.jsonl")
    preds = read_jsonl(run / "predictions.jsonl")
    teacher = [(r["frame_id"], r["top"]) for r in labels if r.get("source") == "teacher"]
    student = [(r["frame_id"], r["student"]) for r in preds]
    teacher_test = [(r["frame_id"], r["teacher"]) for r in preds]
    out = {"teacher_vs_truth": _acc(teacher), "teacher_vs_truth_clear": _acc(teacher, True),
           "teacher_on_test_vs_truth": _acc(teacher_test), "teacher_on_test_vs_truth_clear": _acc(teacher_test, True),
           "student_vs_truth": _acc(student), "student_vs_truth_clear": _acc(student, True),
           "ambiguous_frac": round(len(ambiguous) / len(truth), 3)}
    write_json(run / "conveyor_score.json", out)
    for k, v in out.items():
        if isinstance(v, dict):
            console.print(f"  {k:26s} acc {v['accuracy']:.3f}  P {v['precision']}  R {v['recall']}  (n={v['n']})")
    return out


# ---------------------------------------------------------------- speed sweep

def bench(brains: list[Brain], sprites: dict, speeds_mps: list[float], objects: int = 30,
          max_seconds: float = 150.0, seed: int = 7, out: Path | None = None, cam_fps: float = 30.0) -> dict:
    """Real time, all brains side by side on identical belts, one speed at a time. A speed is
    'sustained' when recall and precision on the target are both >= 0.9."""
    belts = [Belt(sprites, seed=seed) for _ in brains]
    clock = Clock(belts).start()
    runners = [Runner(b, br, cam_fps).start() for b, br in zip(belts, brains)]
    results = {br.name: [] for br in brains}
    try:
        for v in speeds_mps:
            for r in runners:
                r.paused = True
            clock.paused = True
            time.sleep(0.3)
            for b in belts:
                b.speed = v * PX_PER_M
                b.reset()
            clock.paused = False
            for r in runners:
                r.paused = False
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < max_seconds and min(b.stats.tp + b.stats.fp + b.stats.fn + b.stats.tn
                                                                  for b in belts) < objects:
                time.sleep(0.25)
            line = []
            for b, br in zip(belts, brains):
                s = b.stats.as_dict(b.t)
                s["speed_mps"] = v
                s["sustained"] = bool(s["recall"] is not None and s["recall"] >= 0.9 and (s["precision"] or 0) >= 0.9)
                results[br.name].append(s)
                line.append(f"{br.name.split(' ')[0]} R={s['recall']} P={s['precision']} p50={s['p50_ms']}ms")
            console.print(f"[bold]{v:.2f} m/s[/bold]  " + "   ".join(line))
    finally:
        for r in runners:
            r.stop()
        clock.stop()
    summary = {name: {"max_sustained_mps": max([s["speed_mps"] for s in rows if s["sustained"]], default=0.0),
                      "p50_ms": rows[-1]["p50_ms"], "usd_per_hour": max(s["usd_per_hour"] for s in rows)}
               for name, rows in results.items()}
    res = {"speeds_mps": speeds_mps, "objects_per_speed": objects, "results": results, "summary": summary}
    if out:
        write_json(out / "conveyor_bench.json", res)
        _bench_plot(res, out / "conveyor_bench.png")
        res["plot"] = str(out / "conveyor_bench.png")
    for name, s in summary.items():
        console.print(f"  {name:40s} max sustained {s['max_sustained_mps']:.2f} m/s   p50 {s['p50_ms']} ms   ${s['usd_per_hour']}/h")
    return res


def _bench_plot(res: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, rows in res["results"].items():
        sp = [r["speed_mps"] for r in rows]
        ax[0].plot(sp, [r["recall"] or 0 for r in rows], marker="o", label=name)
        ax[1].plot(sp, [r["precision"] or 0 for r in rows], marker="o", label=name)
    for a, t in zip(ax, ["recall (bottles pushed)", "precision (pushes that were bottles)"]):
        a.set_xscale("log")
        a.set_xlabel("belt speed (m/s)")
        a.set_title(t)
        a.set_ylim(-0.05, 1.05)
        a.axhline(0.9, color="#999", lw=0.8, ls="--")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
