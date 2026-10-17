"""`reflex` CLI. Every command is non-interactive, prints a short summary, writes a JSON
result into the run folder, and exits 0 (ok), 2 (needs user/agent input) or 1 (error)."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.table import Table

from .utils import NeedsInput, ReflexError, console, load_env, run_dir, write_json

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Reflex: big models teach, small models react.")


def _guard(stage: str, run: Optional[str], fn: Callable[[], dict]):
    """Run a stage, write runs/<run>/<stage>.json, map errors to exit codes."""
    load_env()
    out_path = run_dir(run, create=True) / f"{stage}.json" if run else None
    try:
        result = {"stage": stage, "status": "ok", **fn()}
        code = 0
    except NeedsInput as e:
        result, code = {"stage": stage, "status": "needs_input", "message": str(e), **e.info}, 2
        console.print(f"[yellow]NEEDS INPUT:[/yellow] {e}")
    except ReflexError as e:
        result, code = {"stage": stage, "status": "error", "message": str(e)}, 1
        console.print(f"[red]ERROR:[/red] {e}")
    except Exception as e:  # noqa: BLE001 — surface anything to the calling agent
        result, code = {"stage": stage, "status": "error", "message": f"{type(e).__name__}: {e}"}, 1
        console.print(traceback.format_exc(), style="red")
    if out_path:
        write_json(out_path, result)
    _print_summary(result, out_path)
    raise typer.Exit(code)


def _print_summary(result: dict, out_path: Optional[Path]):
    short = {k: v for k, v in result.items() if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 300}
    console.print_json(json.dumps(short, default=str))
    if out_path:
        console.print(f"[dim]result → {out_path}[/dim]")


# ---------------------------------------------------------------- stages

@app.command()
def ingest(
    data_dir: Path = typer.Argument(..., help="folder of videos (.mp4/.mov) and/or images"),
    run: str = typer.Option(..., "--run", help="run name (folder under runs/)"),
    fps: float = typer.Option(5.0, help="frames per second to sample from videos"),
    max_frames: int = typer.Option(1500, help="cap on total frames (fps is lowered to fit)"),
    size: int = typer.Option(384, help="long side of saved frames, px"),
    estimate: int = typer.Option(20, help="frames to label for the time/cost estimate (0 = skip)"),
    concurrency: int = typer.Option(16, help="assumed labelling concurrency for the estimate"),
):
    """Extract frames → frames.csv, then measure time/cost on a small sample."""
    from .ingest import ingest as _ingest

    _guard("ingest", run, lambda: _ingest(data_dir, run_dir(run, create=True), fps=fps,
                                          max_frames=max_frames, size=size,
                                          estimate_n=estimate, concurrency=concurrency))


@app.command()
def spec(
    run: str = typer.Option(..., "--run"),
    query: Optional[str] = typer.Option(None, help="the vague request, e.g. 'is the hand showing rock, paper or scissors?'"),
    draft: Optional[Path] = typer.Option(None, help="spec draft JSON written by the calling agent (agent mode)"),
    answers: Optional[Path] = typer.Option(None, help="answers.json for questions.json"),
    samples: int = typer.Option(8, help="sample frames shown to the LLM/agent"),
    llm: bool = typer.Option(False, "--llm", help="force headless mode (needs REFLEX_LLM_MODEL)"),
    yes: bool = typer.Option(False, "--yes", help="accept the draft: use defaults for blocking questions"),
):
    """Turn the vague query into spec.json (+ clarifying questions). Exit 2 = input needed."""
    from .spec import spec as _spec

    _guard("spec", run, lambda: _spec(run_dir(run), query=query, draft=draft, answers=answers,
                                      n_samples=samples, force_llm=llm, accept=yes))


@app.command("spec-check")
def spec_check(
    run: str = typer.Option(..., "--run"),
    n: int = typer.Option(24, help="frames to label"),
    concurrency: int = typer.Option(16),
):
    """Label a few frames with the spec → contact sheet + class distribution to eyeball."""
    from .spec import spec_check as _check

    _guard("spec_check", run, lambda: _check(run_dir(run), n=n, concurrency=concurrency))


@app.command()
def label(
    run: str = typer.Option(..., "--run"),
    concurrency: int = typer.Option(16),
    propagate_back: Optional[int] = typer.Option(None, help="hindsight propagation window (frames); default from spec"),
    min_conf: Optional[float] = typer.Option(None, help="below this the frame is down-weighted; default from spec"),
):
    """Label every frame with the Decisions API (cached + resumable in labels.jsonl)."""
    from .label import label as _label

    _guard("label", run, lambda: _label(run_dir(run), concurrency=concurrency,
                                        propagate_back=propagate_back, min_conf=min_conf))


@app.command()
def models():
    """List the backbone menu."""
    from .backbones import BACKBONES, DEFAULT_BACKBONE

    t = Table("backbone", "params", "use when")
    for k, v in BACKBONES.items():
        t.add_row(k + (" (default)" if k == DEFAULT_BACKBONE else ""), v["params"], v["use"])
    console.print(t)


@app.command()
def train(
    run: str = typer.Option(..., "--run"),
    backbone: str = typer.Option("mobilenetv3_small_100"),
    epochs: int = typer.Option(300, help="head training epochs"),
    hidden: int = typer.Option(0, help="hidden units in the head (0 = linear)"),
    aug: int = typer.Option(2, help="augmented copies per training frame"),
    finetune: bool = typer.Option(False, help="also unfreeze the last backbone blocks (slower)"),
    ft_epochs: int = typer.Option(3),
):
    """Cache embeddings, train the head on soft labels (time-based split), build ONNX."""
    from .train import train as _train

    _guard("train", run, lambda: _train(run_dir(run), backbone=backbone, epochs=epochs, hidden=hidden,
                                        aug=aug, finetune=finetune, ft_epochs=ft_epochs))


@app.command("eval")
def eval_(
    run: str = typer.Option(..., "--run"),
    human_check: int = typer.Option(0, help="write N test frames for a human to label (re-run to score)"),
):
    """Student vs teacher report: agreement, speed, size, cost, disagreements."""
    from .evaluate import evaluate

    _guard("eval", run, lambda: evaluate(run_dir(run), human_check=human_check))


@app.command()
def stress(
    run: str = typer.Option(..., "--run"),
    plan: Optional[Path] = typer.Option(None, help="perturbation plan JSON written by the calling agent"),
    teacher_sample: int = typer.Option(0, help="also re-label N perturbed frames per perturbation with the teacher"),
    llm: bool = typer.Option(False, "--llm", help="let REFLEX_LLM_MODEL pick the perturbations"),
    default: bool = typer.Option(False, "--default", help="use the built-in default perturbation plan"),
):
    """Robustness test under realistic perturbations."""
    from .stress import stress as _stress

    _guard("stress", run, lambda: _stress(run_dir(run), plan=plan, teacher_sample=teacher_sample,
                                          force_llm=llm, use_default=default))


@app.command()
def export(
    run: str = typer.Option(..., "--run"),
    name: Optional[str] = typer.Option(None, help="skill name (default: run name)"),
):
    """Write skills/<name>/ (model.onnx, skill.json, predict.py, reports)."""
    from .export import export_skill

    _guard("export", run, lambda: export_skill(run_dir(run), name=name))


@app.command("run")
def run_all(
    data_dir: Path = typer.Argument(...),
    name: str = typer.Option(..., "--name"),
    query: str = typer.Option(..., "--query"),
    draft: Optional[Path] = typer.Option(None, help="agent-written spec draft (agent mode)"),
    yes: bool = typer.Option(False, "--yes", help="accept the spec draft without answering questions"),
    backbone: str = typer.Option("mobilenetv3_small_100"),
    fps: float = typer.Option(5.0),
    max_frames: int = typer.Option(1500),
    concurrency: int = typer.Option(16),
    with_stress: bool = typer.Option(False, "--stress", help="also run stress with the default plan"),
):
    """Everything end to end. Stops (exit 2) at spec if input is needed."""
    from .pipeline import run_all as _run_all

    _guard("run", name, lambda: _run_all(data_dir, name, query, draft=draft, accept=yes,
                                         backbone=backbone, fps=fps, max_frames=max_frames,
                                         concurrency=concurrency, with_stress=with_stress))


@app.command()
def status(run: str = typer.Option(..., "--run")):
    """Which stages are done for a run (reads the per-stage JSON files)."""
    def _status():
        d = run_dir(run)
        stages = {}
        for s in ["ingest", "spec", "spec_check", "label", "train", "eval", "stress", "export"]:
            p = d / f"{s}.json"
            stages[s] = json.loads(p.read_text())["status"] if p.exists() else "todo"
        return {"run": run, "stages": stages}

    _guard("status", None, _status)


@app.command()
def dummy(
    name: str = typer.Option(..., "--name"),
    classes: str = typer.Option(..., help="comma-separated, e.g. rock,paper,scissors,none"),
):
    """Placeholder skill (random weights, real interface) so demos can integrate now."""
    from .export import dummy_skill

    cls = [c.strip() for c in classes.split(",") if c.strip()]
    _guard("dummy", None, lambda: dummy_skill(name, cls))


if __name__ == "__main__":
    app()
