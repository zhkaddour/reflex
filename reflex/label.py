"""Stage 3: label every frame with the teacher, then hindsight propagation.

labels.jsonl        raw teacher cache (append-only, resumable, tagged with spec hash)
labels_final.jsonl  training labels: teacher soft labels + propagated ones + sample weights"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .runtime import background_class
from .utils import console, load_frames, load_spec, pctl, price_usd, write_json

LOW_CONF_WEIGHT = 0.5   # multiplier for frames under min_conf (down-weighted, not dropped)
PROPAGATED_WEIGHT = 0.8


def hindsight_propagate(frames: list[dict], rows: dict[str, dict], n: int, min_conf: float, bg: str | None) -> dict:
    """Within each episode copy each confident non-background label to the previous n frames
    that are background or low-confidence. Lets the student anticipate (e.g. an RPS throw)."""
    final = {fid: {**r, "source": "teacher"} for fid, r in rows.items()}
    if n <= 0:
        return final
    episodes: dict[str, list[dict]] = {}
    for f in frames:
        if f["frame_id"] in rows:
            episodes.setdefault(f["episode"], []).append(f)
    weak = lambda r: r["top"] == bg or r["conf"] < min_conf  # noqa: E731
    for eps in episodes.values():
        ids = [f["frame_id"] for f in sorted(eps, key=lambda f: f["idx"])]
        for i, fid in enumerate(ids):
            src = rows[fid]
            if weak(src):
                continue
            for j in range(i - 1, max(-1, i - n - 1), -1):
                if not weak(rows[ids[j]]):
                    break
                final[ids[j]] = {**rows[ids[j]], "probs": src["probs"], "top": src["top"], "conf": src["conf"],
                                 "source": "propagated", "propagated_from": fid}
    return final


def label(run, concurrency: int = 16, propagate_back: int | None = None, min_conf: float | None = None) -> dict:
    from .teacher import label_frames

    s = load_spec(run)
    pb = s["propagate_back"] if propagate_back is None else propagate_back
    mc = s["min_conf"] if min_conf is None else min_conf
    frames = load_frames(run)
    rows, failed = label_frames(s, frames, run, concurrency=concurrency, desc="labelling")
    by_id = {r["frame_id"]: r for r in rows}
    final = hindsight_propagate(frames, by_id, pb, mc, background_class(s))
    for r in final.values():
        w = r["conf"]
        if r["conf"] < mc:
            w *= LOW_CONF_WEIGHT
        if r["source"] == "propagated":
            w *= PROPAGATED_WEIGHT
        r["weight"] = round(w, 4)
    ordered = [final[f["frame_id"]] for f in frames if f["frame_id"] in final]
    (run / "labels_final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ordered))

    lat = [r["latency_ms"] for r in rows]
    tin = np.mean([r["input_tokens"] for r in rows]) if rows else 0
    tout = np.mean([r["output_tokens"] for r in rows]) if rows else 0
    low = [r["frame_id"] for r in ordered if r["conf"] < mc]
    teacher_dist = Counter(r["top"] for r in rows)
    final_dist = Counter(r["top"] for r in ordered)
    n_prop = sum(r["source"] == "propagated" for r in ordered)
    summary = {
        "labelled": len(rows), "failed": failed, "total_frames": len(frames),
        "teacher_distribution": dict(teacher_dist), "final_distribution": dict(final_dist),
        "propagate_back": pb, "propagated": n_prop, "min_conf": mc, "low_conf": len(low),
        "low_conf_frames": low[:100],
        "teacher_p50_ms": round(pctl(lat, 50), 1), "teacher_p95_ms": round(pctl(lat, 95), 1),
        "input_tokens_per_frame": round(float(tin), 1), "output_tokens_per_frame": round(float(tout), 1),
        "usd_per_1000_frames": round(price_usd(tin, tout) * 1000, 3),
        "usd_total_est": round(price_usd(tin, tout) * len(rows), 3),
        "labels": str(run / "labels_final.jsonl"),
    }
    write_json(run / "label_summary.json", summary)
    console.print(f"[green]{len(rows)}/{len(frames)} labelled[/green], {n_prop} propagated, {len(low)} low-conf. "
                  f"teacher p50 {summary['teacher_p50_ms']:.0f} ms. final: {dict(final_dist)}")
    return summary
