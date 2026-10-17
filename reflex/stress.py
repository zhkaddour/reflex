"""Stage 6 (optional): robustness under realistic perturbations, student vs teacher.

Plan source: --plan (agent-written JSON), --default, or the headless LLM (REFLEX_LLM_MODEL /
--llm). Without any of those it writes stress_request/ and exits 2 for the calling agent."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .llm import llm_json, llm_model, prompt
from .runtime import Skill
from .spec import contact_sheet
from .train import load_dataset
from .utils import NeedsInput, ReflexError, console, load_spec, read_json, write_json

MAX_EVAL_FRAMES = 300

MENU = {  # kind: {param: (default, lo, hi)}
    "brightness": {"factor": (0.5, 0.2, 2.0)},
    "contrast": {"factor": (0.5, 0.2, 2.0)},
    "gaussian_blur": {"sigma": (2.0, 0.3, 8.0)},
    "motion_blur": {"length": (15, 3, 40), "angle": (0, 0, 180)},
    "noise": {"std": (20, 1, 60)},
    "occlusion_patch": {"size": (0.25, 0.05, 0.6)},
    "rotation": {"degrees": (10, -45, 45)},
    "jpeg_quality": {"quality": (20, 5, 95)},
    "crop_shift": {"shift": (0.1, 0.0, 0.3)},
    "color_shift": {"hue": (15, -90, 90), "saturation": (1.0, 0.0, 2.0)},
}

DEFAULT_PLAN = [
    {"kind": "brightness", "params": {"factor": 0.5}, "why": "dimmer venue lighting"},
    {"kind": "motion_blur", "params": {"length": 15, "angle": 0}, "why": "fast movement"},
    {"kind": "noise", "params": {"std": 20}, "why": "cheap webcam sensor noise in low light"},
    {"kind": "occlusion_patch", "params": {"size": 0.25}, "why": "something partly in the way"},
    {"kind": "rotation", "params": {"degrees": 10}, "why": "camera mounted slightly differently"},
    {"kind": "color_shift", "params": {"hue": 15, "saturation": 1.0}, "why": "different light colour"},
]

ADVICE = {
    "brightness": "record extra footage in dimmer and brighter light",
    "contrast": "record with backlight / flat lighting",
    "gaussian_blur": "record some slightly out-of-focus footage, or fix the camera focus",
    "motion_blur": "record fast motions; consider a higher shutter speed",
    "noise": "record in low light with the demo camera",
    "occlusion_patch": "record with partial occlusions (hands, objects in the way)",
    "rotation": "record with the camera tilted a few degrees",
    "jpeg_quality": "record with the same compression as the live stream",
    "crop_shift": "record with the camera moved/zoomed slightly",
    "color_shift": "record under different light colours (warm/cool)",
}


def validate_plan(raw) -> list[dict]:
    items = raw.get("perturbations", raw) if isinstance(raw, dict) else raw
    plan = []
    for it in items or []:
        kind = it.get("kind")
        if kind not in MENU:
            console.print(f"[yellow]skipping unknown perturbation {kind!r}[/yellow]")
            continue
        params = {}
        for k, (d, lo, hi) in MENU[kind].items():
            params[k] = float(np.clip(float((it.get("params") or {}).get(k, d)), lo, hi))
        plan.append({"kind": kind, "params": params, "why": it.get("why", "")})
    if not plan:
        raise ReflexError("empty perturbation plan")
    return plan[:8]


def apply(img: np.ndarray, kind: str, p: dict, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    if kind == "brightness":
        f = f * p["factor"]
    elif kind == "contrast":
        f = (f - f.mean()) * p["factor"] + f.mean()
    elif kind == "gaussian_blur":
        return cv2.GaussianBlur(img, (0, 0), p["sigma"])
    elif kind == "motion_blur":
        n = max(3, int(p["length"]))
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), p["angle"], 1), (n, n))
        return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))
    elif kind == "noise":
        f = f + rng.normal(0, p["std"], f.shape)
    elif kind == "occlusion_patch":
        s = int(min(h, w) * p["size"])
        y, x = rng.integers(0, h - s + 1), rng.integers(0, w - s + 1)
        out = img.copy()
        out[y:y + s, x:x + s] = 128
        return out
    elif kind == "rotation":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), p["degrees"], 1)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif kind == "jpeg_quality":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p["quality"])])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif kind == "crop_shift":
        s = p["shift"]
        dx, dy = int(w * s * rng.choice([-1, 1])), int(h * s * rng.choice([-1, 1]))
        m = np.float32([[1 + s, 0, dx - w * s / 2], [0, 1 + s, dy - h * s / 2]])
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif kind == "color_shift":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + p["hue"] / 2) % 180  # OpenCV hue is 0–180
        hsv[..., 1] *= p["saturation"]
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(f, 0, 255).astype(np.uint8)


def _get_plan(run, spec, plan, force_llm, use_default) -> tuple[list[dict], str]:
    if plan:
        return validate_plan(read_json(plan)), "agent"
    if use_default:
        return validate_plan(DEFAULT_PLAN), "default"
    samples = sorted((run / "spec_request").glob("*.jpg"))[:6]
    if force_llm or llm_model():
        text = f"Spec:\n{json.dumps({k: spec[k] for k in ('query', 'task_summary', 'visual_cues', 'question')}, indent=2)}"
        return validate_plan(llm_json(prompt("stress"), text, samples)), "llm"
    req = run / "stress_request"
    if req.exists():
        shutil.rmtree(req)
    req.mkdir()
    for p in samples:
        shutil.copy(p, req / p.name)
    (req / "README.md").write_text(
        f"# Reflex stress plan request — run `{run.name}`\n\nSpec:\n```json\n{json.dumps(spec, indent=2)}\n```\n\n"
        f"Sample frames in this folder. Write the plan JSON, then run:\n\n"
        f"    reflex stress --run {run.name} --plan <file>\n\n(or `--default` for the built-in plan)\n\n---\n\n"
        + prompt("stress"))
    raise NeedsInput(f"agent mode: read {req / 'README.md'}, write a perturbation plan JSON, then "
                     f"`reflex stress --run {run.name} --plan <file>` (or --default)",
                     kind="agent_plan", request=str(req / "README.md"),
                     next=f"reflex stress --run {run.name} --plan <plan.json>")


def _chart(results, clean_s, clean_t, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["kind"] for r in results]
    x = np.arange(len(names))
    has_t = any(r.get("teacher_agreement") is not None for r in results)
    wd = 0.38 if has_t else 0.6
    fig, ax = plt.subplots(figsize=(1.5 + 1.3 * len(names), 4))
    ax.bar(x - (wd / 2 if has_t else 0), [r["student_agreement"] for r in results], wd, label="student (Reflex)")
    if has_t:
        ax.bar(x + wd / 2, [r.get("teacher_agreement") or 0 for r in results], wd, label="teacher (cloud)")
    ax.axhline(clean_s, color="C0", ls="--", lw=1, label=f"student clean {clean_s:.0%}")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("agreement with clean teacher label")
    ax.set_title("Who breaks first?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def stress(run: Path, plan: Path | None = None, teacher_sample: int = 0, force_llm: bool = False,
           use_default: bool = False) -> dict:
    spec = load_spec(run)
    plan_items, plan_source = _get_plan(run, spec, plan, force_llm, use_default)
    data, classes = load_dataset(run)
    sp = read_json(run / "split.json")
    by_id = {d["frame_id"]: d for d in data}
    test = [by_id[i] for i in sp["test"] if i in by_id]
    if len(test) > MAX_EVAL_FRAMES:
        test = [test[int(i)] for i in np.linspace(0, len(test) - 1, MAX_EVAL_FRAMES)]
    skill = Skill(run / "student")
    imgs = [cv2.imread(str(run / d["path"])) for d in test]
    t_top = np.array([d["top"] for d in test])
    clean = np.array([skill.predict_probs(im).argmax() for im in imgs])
    clean_acc = float((clean == t_top).mean())

    results, examples = [], []
    for it in plan_items:
        rng = np.random.default_rng(0)
        pert = [apply(im, it["kind"], it["params"], rng) for im in imgs]
        pred = np.array([skill.predict_probs(im).argmax() for im in pert])
        acc = float((pred == t_top).mean())
        r = {**it, "student_agreement": round(acc, 4), "student_drop": round(clean_acc - acc, 4)}
        ex = run / "stress_frames" / f"{it['kind']}.jpg"
        ex.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(ex), pert[len(pert) // 2])
        examples.append((ex, f"{it['kind']} {acc:.0%}"))

        if teacher_sample > 0:
            from .teacher import label_frames

            idx = np.linspace(0, len(test) - 1, min(teacher_sample, len(test))).astype(int)
            frames = []
            for i in idx:
                rel = f"stress_frames/{it['kind']}/{test[i]['frame_id']}.jpg"
                (run / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(run / rel), pert[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append({"frame_id": f"{test[i]['frame_id']}@{it['kind']}", "path": rel})
            rows, _ = label_frames(spec, frames, run, cache=False, desc=f"teacher {it['kind']}")
            t_pred = {r_["frame_id"]: classes.index(r_["top"]) for r_ in rows}
            ok = [t_pred[f["frame_id"]] == t_top[i] for f, i in zip(frames, idx) if f["frame_id"] in t_pred]
            r["teacher_agreement"] = round(float(np.mean(ok)), 4) if ok else None
            r["student_agreement_on_teacher_sample"] = round(float((pred[idx] == t_top[idx]).mean()), 4)
            r["teacher_n"] = len(ok)
        results.append(r)
        console.print(f"  {it['kind']:<16} student {acc:.1%} (drop {clean_acc - acc:+.1%})"
                      + (f"  teacher {r['teacher_agreement']:.1%}" if r.get("teacher_agreement") is not None else ""))

    _chart(results, clean_acc, None, run / "stress.png")
    contact_sheet(examples, run / "stress_examples.png", cols=3)
    worst = sorted(results, key=lambda r: -r["student_drop"])
    suggestions = [f"{ADVICE[r['kind']]} (agreement drops {r['student_drop']:.0%} under {r['kind']})"
                   for r in worst if r["student_drop"] > 0.10]
    out = {"plan_source": plan_source, "n_frames": len(test), "clean_agreement": round(clean_acc, 4),
           "results": results, "suggestions": suggestions, "chart": str(run / "stress.png"),
           "examples": str(run / "stress_examples.png")}
    write_json(run / "stress_report.json", out)
    lines = [f"# Reflex stress test — `{run.name}`", "", f"Clean agreement: **{clean_acc:.1%}** on {len(test)} test frames. "
             f"Plan: {plan_source}.", "", "| perturbation | params | why | student | drop | teacher |", "|---|---|---|---|---|---|"]
    for r in results:
        t = f"{r['teacher_agreement']:.0%}" if r.get("teacher_agreement") is not None else "—"
        lines.append(f"| {r['kind']} | {r['params']} | {r['why']} | {r['student_agreement']:.0%} | "
                     f"{r['student_drop']:+.0%} | {t} |")
    lines += ["", "![stress](stress.png)", "", "![examples](stress_examples.png)", "", "## Suggestions", ""]
    lines += [f"- {s}" for s in suggestions] or ["- Robust to all tested perturbations (drop ≤ 10%)."]
    (run / "stress.md").write_text("\n".join(lines) + "\n")
    return out
