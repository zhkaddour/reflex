"""Stage 2: vague query -> spec.json (+ questions.json), and spec-check.

Agent mode (default): writes spec_request.md + sample frames and exits 2; the calling agent
looks at the frames, writes a draft, then runs `reflex spec --draft draft.json`.
Headless mode (REFLEX_LLM_MODEL set or --llm): the engine asks the LLM itself."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .llm import llm_json, llm_model, prompt
from .runtime import background_class, spec_classes
from .utils import NeedsInput, ReflexError, console, load_frames, load_spec, read_json, write_json


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(s).strip().lower()).strip("_") or "x"


def _even(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, n)]


def sample_frames(run: Path, n: int) -> list[dict]:
    frames = sorted(load_frames(run), key=lambda f: (f["episode"], f["idx"]))
    return _even(frames, n)


def normalize(raw: dict, name: str, query: str) -> tuple[dict, list[str]]:
    """Validate/clean a spec draft (from the LLM or an agent). -> (spec, warnings)."""
    warnings = []
    q = raw.get("question") or {}
    qtype = q.get("type", "choice")
    if qtype not in ("choice", "predicate"):
        raise ReflexError(f"question.type must be 'choice' or 'predicate', got {qtype!r}")
    question = {"type": qtype, "name": _slug(q.get("name") or name),
                "instructions": q.get("instructions") or query}
    if qtype == "choice":
        choices, seen = [], set()
        for c in q.get("choices") or []:
            c = {"value": c, "description": c} if isinstance(c, str) else c
            v = _slug(c.get("value", ""))
            if v in seen:
                raise ReflexError(f"duplicate class '{v}'")
            seen.add(v)
            choices.append({"value": v, "description": c.get("description") or v})
        if not 2 <= len(choices) <= 6:
            raise ReflexError(f"choice question needs 2–6 classes, got {len(choices)}")
        question["choices"] = choices
    spec = {
        "name": name,
        "query": query,
        "task_summary": raw.get("task_summary") or query,
        "visual_cues": list(raw.get("visual_cues") or []),
        "question": question,
        "edge_cases": list(raw.get("edge_cases") or []),
        "min_conf": float(np.clip(float(raw.get("min_conf", 0.6)), 0.0, 1.0)),
        "propagate_back": int(np.clip(int(raw.get("propagate_back", 0)), 0, 30)),
        "hflip_ok": bool(raw.get("hflip_ok", True)),
    }
    if raw.get("background_class"):
        spec["background_class"] = _slug(raw["background_class"])
    if qtype == "choice" and background_class(spec) is None:
        warnings.append("no 'none'/'unclear' class — fine only if every frame clearly shows one class")
    return spec, warnings


def _normalize_questions(raw_qs) -> list[dict]:
    out = []
    for i, q in enumerate(raw_qs or []):
        q = {"question": q} if isinstance(q, str) else dict(q)
        q.setdefault("id", f"q{i + 1}")
        q.setdefault("why", "")
        q["blocking"] = bool(q.get("blocking", False))
        q.setdefault("default", None)
        out.append(q)
    return out[:3]


def _load_answers(path: Path | None) -> dict:
    if not path:
        return {}
    a = read_json(path)
    if isinstance(a, list):  # [{id, answer}] form
        a = {x["id"]: x.get("answer") for x in a}
    return {str(k): v for k, v in a.items()}


def _write_request(run: Path, query: str, n: int) -> list[Path]:
    req = run / "spec_request"
    if req.exists():
        shutil.rmtree(req)
    req.mkdir(parents=True)
    paths = []
    for i, f in enumerate(sample_frames(run, n)):
        dst = req / f"sample_{i + 1:02d}__{f['episode']}.jpg"
        shutil.copy(run / f["path"], dst)
        paths.append(dst)
    ingest = read_json(run / "ingest.json", default={})
    (req / "README.md").write_text(
        f"# Reflex spec request — run `{run.name}`\n\n"
        f"**User request:** {query}\n\n"
        f"Data: {ingest.get('n_frames', '?')} frames, {ingest.get('n_episodes', '?')} episodes.\n"
        f"Sample frames (look at every one): {', '.join(p.name for p in paths)}\n\n"
        f"Write the draft JSON described below to a file, then run:\n\n"
        f"    reflex spec --run {run.name} --draft <file>\n\n---\n\n" + prompt("spec"))
    return paths


def _fold_answers(spec: dict, questions: list[dict], answers: dict) -> dict:
    """No LLM available to revise: append the Q&A to the teacher instructions."""
    qa = [f"- {q['question']} → {answers[q['id']]}" for q in questions if q["id"] in answers]
    if qa:
        spec["question"]["instructions"] += "\nClarifications:\n" + "\n".join(qa)
        spec["edge_cases"] += qa
    return spec


def spec(run: Path, query: str | None = None, draft: Path | None = None, answers: Path | None = None,
         n_samples: int = 8, force_llm: bool = False, accept: bool = False) -> dict:
    prev = read_json(run / "spec.json", default={})
    qfile = run / "query.txt"
    query = query or prev.get("query") or (qfile.read_text().strip() if qfile.exists() else None)
    if not query:
        raise ReflexError("--query is required the first time")
    qfile.write_text(query)
    ans = {**prev.get("answers", {}), **_load_answers(answers)}
    use_llm = force_llm or bool(llm_model())

    if draft:
        raw = read_json(draft)
        questions = _normalize_questions(raw.get("questions"))
        if answers and not raw.get("questions"):
            questions = read_json(run / "questions.json", default=[])
    elif answers:
        if not prev:
            raise ReflexError("--answers given but there is no spec.json yet")
        old_qs = read_json(run / "questions.json", default=[])
        if use_llm:
            samples = sorted((run / "spec_request").glob("*.jpg")) or \
                [run / f["path"] for f in sample_frames(run, n_samples)]
            text = (f"User request: {query}\n\nCurrent spec:\n{json.dumps(prev, indent=2)}\n\n"
                    f"The user answered your clarifying questions:\n"
                    + "\n".join(f"- {q['question']} → {ans.get(q['id'], q.get('default'))}" for q in old_qs)
                    + "\n\nRevise the spec accordingly. Only ask new questions if still essential.")
            raw = llm_json(prompt("spec"), text, samples)
            questions = old_qs + _normalize_questions(raw.get("questions"))
        else:
            raw, questions = _fold_answers(dict(prev), old_qs, ans), old_qs
    else:
        samples = _write_request(run, query, n_samples)
        if not use_llm:
            req = run / "spec_request" / "README.md"
            raise NeedsInput(
                f"agent mode: read {req}, view the sample frames in {req.parent}, write a spec draft "
                f"JSON, then run `reflex spec --run {run.name} --draft <file>`",
                kind="agent_draft", request=str(req), frames=[str(p) for p in samples],
                next=f"reflex spec --run {run.name} --draft <draft.json>")
        console.print(f"[dim]asking {llm_model()} for a spec...[/dim]")
        raw = llm_json(prompt("spec"), f"User request: {query}", samples)
        questions = _normalize_questions(raw.get("questions"))

    # Re-number new question ids that collide with old ones.
    seen = set()
    for i, q in enumerate(questions):
        if q["id"] in seen:
            q["id"] = f"q{len(seen) + i + 1}"
        seen.add(q["id"])

    s, warnings = normalize(raw, run.name, query)
    for q in questions:
        q["answer"] = ans.get(q["id"])
    pending = [q for q in questions if q["blocking"] and q["answer"] is None]
    if accept and pending:
        for q in pending:
            q["answer"] = ans[q["id"]] = q.get("default")
        s = _fold_answers(s, pending, ans)
        pending = []
    s["answers"] = ans
    s["status"] = "needs_input" if pending else "ready"
    write_json(run / "spec.json", s)
    write_json(run / "questions.json", questions)

    summary = {"spec": str(run / "spec.json"), "spec_status": s["status"], "task_summary": s["task_summary"],
               "classes": spec_classes(s), "visual_cues": s["visual_cues"], "edge_cases": s["edge_cases"],
               "propagate_back": s["propagate_back"], "min_conf": s["min_conf"], "warnings": warnings,
               "questions": questions}
    console.print(f"[bold]{s['task_summary']}[/bold]  classes: {', '.join(spec_classes(s))}")
    for c in s["visual_cues"]:
        console.print(f"  • {c}")
    for w in warnings:
        console.print(f"[yellow]warning: {w}[/yellow]")
    if pending:
        raise NeedsInput(
            f"{len(pending)} blocking question(s) in {run / 'questions.json'}: ask the user, write "
            f"answers.json as {{\"<id>\": \"<answer>\"}}, then `reflex spec --run {run.name} --answers answers.json`",
            kind="user_questions", questions=pending, spec=str(run / "spec.json"),
            next=f"reflex spec --run {run.name} --answers answers.json")
    return summary


# ---------------------------------------------------------------- spec-check

def contact_sheet(items: list[tuple[Path, str]], out: Path, cols: int = 6, tile: int = 240):
    """items: (image path, caption) -> grid PNG."""
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * tile, cols * tile, 3), 30, np.uint8)
    for i, (p, cap) in enumerate(items):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = tile / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        r, c = divmod(i, cols)
        y0, x0 = r * tile + (tile - img.shape[0]) // 2, c * tile + (tile - img.shape[1]) // 2
        sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.rectangle(sheet, (c * tile, r * tile + tile - 26), ((c + 1) * tile - 1, (r + 1) * tile - 1), (0, 0, 0), -1)
        cv2.putText(sheet, cap[:34], (c * tile + 4, (r + 1) * tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet)


def spec_check(run: Path, n: int = 24, concurrency: int = 16) -> dict:
    from .teacher import label_frames

    s = load_spec(run)
    frames = sample_frames(run, n)
    rows, failed = label_frames(s, frames, run, concurrency=concurrency, desc="spec-check")
    by_id = {r["frame_id"]: r for r in rows}
    items = [(run / f["path"], f"{by_id[f['frame_id']]['top']} {by_id[f['frame_id']]['conf']:.2f}")
             for f in frames if f["frame_id"] in by_id]
    contact_sheet(items, run / "spec_check.png")
    dist = Counter(r["top"] for r in rows)
    classes = spec_classes(s)
    low = [r["frame_id"] for r in rows if r["conf"] < s["min_conf"]]
    warnings = []
    if missing := [c for c in classes if dist[c] == 0]:
        warnings.append(f"classes never seen in the sample: {missing} — record more of them or check definitions")
    if rows and max(dist.values()) / len(rows) > 0.8:
        warnings.append("one class dominates (>80%) — the data may be unbalanced")
    if rows and len(low) / len(rows) > 0.3:
        warnings.append(f"{len(low)}/{len(rows)} low-confidence labels — classes may be hard to see; clarify the spec")
    for w in warnings:
        console.print(f"[yellow]warning: {w}[/yellow]")
    console.print(f"distribution: {dict(dist)}  → look at {run / 'spec_check.png'}")
    return {"labelled": len(rows), "failed": failed, "distribution": {c: dist[c] for c in classes},
            "mean_conf": round(float(np.mean([r["conf"] for r in rows])), 3) if rows else None,
            "low_conf_frames": low, "contact_sheet": str(run / "spec_check.png"), "warnings": warnings}
