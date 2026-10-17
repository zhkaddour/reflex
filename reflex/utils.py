"""Shared helpers: run folders, JSON/JSONL IO, env, pricing, errors."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()


class ReflexError(Exception):
    """Expected failure: printed without a traceback, exit code 1."""


class NeedsInput(Exception):
    """User/agent input required (e.g. blocking spec questions): exit code 2.
    `info` (kind, files to read, next command) is merged into the stage's JSON result."""

    def __init__(self, message: str, **info):
        super().__init__(message)
        self.info = info


def load_env():
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def runs_dir() -> Path:
    return Path(os.environ.get("REFLEX_RUNS_DIR", "runs"))


def skills_dir() -> Path:
    return Path(os.environ.get("REFLEX_SKILLS_DIR", "skills"))


def run_dir(name: str, create: bool = False) -> Path:
    d = runs_dir() / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    elif not d.exists():
        raise ReflexError(f"run '{name}' not found at {d} — run `reflex ingest` first")
    return d


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise ReflexError(f"missing {path}")
    return json.loads(path.read_text())


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict]):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def load_frames(run: Path) -> list[dict]:
    p = run / "frames.csv"
    if not p.exists():
        raise ReflexError(f"missing {p} — run `reflex ingest` first")
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t_sec"] = float(r["t_sec"])
        r["idx"] = int(r["idx"])
    return rows


def load_spec(run: Path, require_ready: bool = True) -> dict:
    spec = read_json(run / "spec.json", default={})
    if not spec:
        raise ReflexError(f"no spec for run — run `reflex spec --run {run.name} --query ...` first")
    if require_ready and spec.get("status") != "ready":
        raise NeedsInput(f"spec has blocking questions: answer {run / 'questions.json'} "
                         f"then `reflex spec --run {run.name} --answers answers.json`")
    return spec


def spec_hash(spec: dict) -> str:
    """Hash of what the teacher sees — labels are cached per hash."""
    key = {"task_summary": spec["task_summary"], "question": spec["question"]}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def price_usd(input_tokens: float, output_tokens: float) -> float:
    pin = float(os.environ.get("REFLEX_PRICE_IN_PER_M", "1.0"))
    pout = float(os.environ.get("REFLEX_PRICE_OUT_PER_M", "4.0"))
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def pctl(values, q) -> float:
    return float(np.percentile(np.asarray(values, float), q)) if len(values) else float("nan")


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
