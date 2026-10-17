"""Brains that decide "push or not" from one camera frame. Same interface for all:

    brain.predict(frame_bgr, gt) -> {"push": bool, "p": float, "ms": float, "usd": float}

- student:   the trained Reflex skill (ONNX, on-device)
- decisions: the teacher, OpenAI Decisions API, same request/parse code as labelling
- vlm:       a general vision LLM via the Responses API (REFLEX_VLM_MODEL, default gpt-5.5)
- oracle:    reads the sim's ground truth after a fixed delay — `oracle@800` is a perfect
             classifier with 800 ms latency, to isolate the effect of latency (no API, free)

`gt` is only used by the oracle."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

from ..runtime import (DECISIONS_URL, Skill, background_class, build_decisions_body, encode_jpeg_b64,
                       parse_decision, spec_classes)
from ..utils import ReflexError, price_usd, read_json

DEFAULT_VLM_MODEL = "gpt-5.5"
DEFAULT_SPEC = Path(__file__).resolve().parents[2] / "examples" / "conveyor" / "spec_draft.json"


def positive_class(spec: dict) -> str:
    """The class that means 'push': 'yes' for a predicate, else the first non-background class."""
    classes = spec_classes(spec)
    if spec.get("push_class") in classes:
        return spec["push_class"]
    if spec["question"]["type"] == "predicate":
        return "yes"
    bg = background_class(spec)
    return next(c for c in classes if c != bg)


def resolve_spec(spec: str | Path | None = None, skill: str | Path | None = None) -> dict:
    """--spec file > the skill's spec > runs/conveyor/spec.json > the bundled draft."""
    from ..spec import normalize

    for p in [spec, skill and Path(skill) / "skill.json", Path("runs/conveyor/spec.json"), DEFAULT_SPEC]:
        if p and Path(p).exists():
            raw = read_json(Path(p))
            raw = raw.get("spec", raw)
            if "question" in raw and "query" in raw and "min_conf" in raw:
                return raw
            return normalize(raw, "conveyor", raw.get("query", raw.get("task_summary", "")))[0]
    raise ReflexError("no conveyor spec found")


def model_size(onnx_path: Path) -> str:
    """'4.0M params · 16 MB' from the ONNX weights."""
    import onnx

    g = onnx.load(str(onnx_path), load_external_data=False).graph
    n = sum(int(np.prod(t.dims)) for t in g.initializer)
    return f"{n / 1e6:.1f}M params · {onnx_path.stat().st_size / 1e6:.0f} MB"


class Brain:
    name = "brain"       # unique key (bench results, logs)
    title = "brain"      # big display name
    detail = ""          # model · size · where it runs
    price_note = ""      # pricing basis for the $/hour figure
    where = "cloud"      # cloud | device
    workers = 1          # concurrent requests in flight
    min_votes = 1        # yes-answers per object before pushing; >1 debounces fast, noisy brains

    def predict(self, frame: np.ndarray, gt: dict) -> dict:
        raise NotImplementedError

    def close(self):
        pass


class StudentBrain(Brain):
    def __init__(self, skill: str | Path, min_votes: int = 2):
        if not (Path(skill) / "model.onnx").exists():
            raise ReflexError(f"no trained skill at {skill} — train one first (see examples/conveyor/README.md)")
        self.skill = Skill(skill, threads=2)
        self.min_votes = min_votes
        self.name = f"student ({self.skill.meta.get('backbone', 'onnx')} · {model_size(Path(skill) / 'model.onnx')}, {self.skill.device})"
        self.pos = positive_class(self.skill.meta["spec"]) if "spec" in self.skill.meta else "yes"
        hw = "Apple Neural Engine / GPU" if self.skill.device == "coreml" else "CPU"
        self.title, self.where = "Reflex student", "device"
        self.detail = f"{self.skill.meta.get('backbone', 'onnx')} · {model_size(Path(skill) / 'model.onnx')} · on this laptop ({hw})"
        self.price_note = "runs locally: no per-call cost"

    def predict(self, frame, gt):
        out = self.skill.predict(frame)
        p = out["probs"][self.pos]
        return {"push": p >= 0.5, "p": p, "ms": out["ms"], "usd": 0.0}


class DecisionsBrain(Brain):
    def __init__(self, spec: dict, model: str | None = None, workers: int = 2):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ReflexError("OPENAI_API_KEY is not set")
        self.spec, self.model, self.workers = spec, model, workers
        self.pos = positive_class(spec)
        m = model or os.environ.get("REFLEX_TEACHER_MODEL", "gpt-6-luna")
        self.name = f"decisions ({m} · size undisclosed)"
        self.title, self.detail = "Decisions API", f"OpenAI {m} · cloud · size undisclosed"
        self.price_note = (f"${float(os.environ.get('REFLEX_PRICE_IN_PER_M', '0.10')):g} / 1M input tokens, "
                           f"output ${float(os.environ.get('REFLEX_PRICE_OUT_PER_M', '0')):g}")
        self.url = DECISIONS_URL
        self.http = httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0), headers={"Authorization": f"Bearer {key}"},
                                 limits=httpx.Limits(max_keepalive_connections=workers + 2))

    def predict(self, frame, gt):
        t0 = time.perf_counter()
        body = build_decisions_body(self.spec, encode_jpeg_b64(frame), self.model)
        r = self.http.post(self.url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"Decisions API {r.status_code}: {r.text[:200]}")
        probs, usage = parse_decision(r.json(), self.spec)
        p = probs[self.pos]
        return {"push": p >= 0.5, "p": p, "ms": (time.perf_counter() - t0) * 1000,
                "usd": price_usd(usage["input_tokens"], usage["output_tokens"])}

    def close(self):
        self.http.close()


VLM_INSTRUCTIONS = """You are the vision system of a recycling sorter. You see one top-down camera frame.
{task}
Question: {question}
Reply with JSON only: {{"answer": "yes" or "no", "confidence": number between 0 and 1}}."""


class VLMBrain(Brain):
    def __init__(self, spec: dict, model: str | None = None, workers: int = 2, effort: str | None = None):
        from openai import OpenAI

        self.model = model or os.environ.get("REFLEX_VLM_MODEL", DEFAULT_VLM_MODEL)
        self.effort = effort or os.environ.get("REFLEX_VLM_EFFORT", "low")
        self.workers, self.name = workers, f"vlm ({self.model} · size undisclosed, {self.effort})"
        self.client = OpenAI(max_retries=0, timeout=httpx.Timeout(20.0, connect=3.0))
        self.instructions = VLM_INSTRUCTIONS.format(task=spec["task_summary"], question=spec["question"]["instructions"])
        # gpt-5.5 list price (Oct 2026): $5 / 1M input, $30 / 1M output (reasoning tokens bill as output).
        self.pin = float(os.environ.get("REFLEX_VLM_PRICE_IN_PER_M", "5.0"))
        self.pout = float(os.environ.get("REFLEX_VLM_PRICE_OUT_PER_M", "30.0"))
        self.title = self.model.upper()
        self.detail = f"general vision LLM · reasoning {self.effort} · cloud · size undisclosed"
        self.price_note = f"${self.pin:g} / 1M input, ${self.pout:g} / 1M output tokens"

    def predict(self, frame, gt):
        t0 = time.perf_counter()
        img = f"data:image/jpeg;base64,{encode_jpeg_b64(frame)}"
        resp = self.client.responses.create(
            model=self.model, instructions=self.instructions, reasoning={"effort": self.effort},
            input=[{"role": "user", "content": [{"type": "input_text", "text": "Camera frame (answer in JSON):"},
                                                 {"type": "input_image", "image_url": img}]}],
            text={"format": {"type": "json_object"}})
        try:
            a = json.loads(resp.output_text)
            yes = str(a.get("answer", "")).lower().startswith("y")
            conf = float(a.get("confidence", 1.0))
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"VLM gave no JSON: {resp.output_text[:120]}") from e
        u = resp.usage
        usd = (u.input_tokens * self.pin + u.output_tokens * self.pout) / 1e6 if u else 0.0
        return {"push": yes, "p": conf if yes else 1 - conf, "ms": (time.perf_counter() - t0) * 1000, "usd": usd}


class OracleBrain(Brain):
    def __init__(self, latency_ms: float = 0.0, workers: int = 1):
        self.latency, self.workers = latency_ms, workers
        self.name = f"oracle @ {latency_ms:.0f} ms"
        self.title, self.detail = f"Oracle · {latency_ms:.0f} ms", "perfect answers (sim ground truth) after a fixed delay"
        self.price_note, self.where = "free (simulated)", "device"

    def predict(self, frame, gt):
        t0 = time.perf_counter()
        time.sleep(self.latency / 1000)
        return {"push": gt["target_in_zone"], "p": float(gt["target_in_zone"]),
                "ms": (time.perf_counter() - t0) * 1000, "usd": 0.0}


def make_brain(name: str, spec: dict, skill: str | Path = "skills/conveyor", workers: int | None = None) -> Brain:
    """'student' | 'decisions' | 'vlm' | 'oracle[@ms]', optionally ':<workers>' (e.g. 'vlm:4')
    and '~<votes>' (e.g. 'student~3': 3 yes-answers per object before pushing)."""
    name, _, votes = name.partition("~")
    name, _, w = name.partition(":")
    w = int(w) if w else workers
    kind, _, arg = name.partition("@")
    if kind == "student":
        b = StudentBrain(skill)
    elif kind == "decisions":
        b = DecisionsBrain(spec, model=arg or None, workers=w or 2)
    elif kind == "vlm":
        b = VLMBrain(spec, model=arg or None, workers=w or 3)
    elif kind == "oracle":
        b = OracleBrain(float(arg or 0), workers=w or 1)
    else:
        raise ReflexError(f"unknown brain {name!r} (student, decisions, vlm, oracle@ms)")
    if w:
        b.workers = w
    if votes:
        b.min_votes = int(votes)
    if b.min_votes > 1:
        b.name += f" ~{b.min_votes} votes"
    return b
