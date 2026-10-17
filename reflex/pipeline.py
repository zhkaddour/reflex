"""`reflex run`: every stage end to end. Stops (exit 2) at spec when input is needed."""

from __future__ import annotations

from pathlib import Path

from .utils import console, read_json, run_dir, write_json


def _stage(run: Path, name: str, fn, **kw) -> dict:
    console.rule(f"[bold]{name}")
    result = {"stage": name, "status": "ok", **fn(**kw)}
    write_json(run / f"{name}.json", result)
    return result


def run_all(data_dir: Path, name: str, query: str, draft: Path | None = None, accept: bool = False,
            backbone: str = "mobilenetv3_small_100", fps: float = 5.0, max_frames: int = 1500,
            concurrency: int = 16, with_stress: bool = False) -> dict:
    from .evaluate import evaluate
    from .export import export_skill
    from .ingest import ingest
    from .label import label
    from .spec import spec
    from .train import train

    run = run_dir(name, create=True)
    if not (run / "frames.csv").exists():
        _stage(run, "ingest", ingest, data_dir=data_dir, run=run, fps=fps, max_frames=max_frames,
               estimate_n=0, concurrency=concurrency)
    else:
        console.print("[dim]frames already extracted — skipping ingest[/dim]")
    existing = read_json(run / "spec.json", default={})
    if draft or existing.get("status") != "ready" or existing.get("query") != query:
        _stage(run, "spec", spec, run=run, query=query, draft=draft, accept=accept)  # may raise NeedsInput
    out = {"run": name}
    out["label"] = _stage(run, "label", label, run=run, concurrency=concurrency)
    out["train"] = _stage(run, "train", train, run=run, backbone=backbone)
    out["eval"] = _stage(run, "eval", evaluate, run=run)
    if with_stress:
        from .stress import stress

        out["stress"] = _stage(run, "stress", stress, run=run, use_default=True)
    out["export"] = _stage(run, "export", export_skill, run=run)
    m = out["eval"]
    return {"skill_dir": out["export"]["skill_dir"], "agreement": m["agreement"], "macro_f1": m["macro_f1"],
            "p50_ms": m["p50_ms"], "teacher_p50_ms": m["teacher_p50_ms"], "speedup": m["speedup"],
            "size_mb": m["size_mb"], "report": m["report"]}
