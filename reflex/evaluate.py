"""Stage 5: student vs teacher on the held-out split -> report.md / report.json / confusion.png.

Uses the exported ONNX through the real runtime `Skill`, so the numbers are what demos get."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .runtime import Skill
from .spec import contact_sheet
from .train import load_dataset
from .utils import NeedsInput, ReflexError, console, load_spec, pctl, price_usd, read_json, read_jsonl, write_json

N_WORST = 12


def _confusion_png(cm: np.ndarray, classes: list[str], out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(classes), 1 + 0.8 * len(classes)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("student")
    ax.set_ylabel("teacher")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title("student vs teacher (test split)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _annotate(img: np.ndarray, classes, p_student, p_teacher, ms) -> np.ndarray:
    """Overlay student vs teacher on a held-out frame (green = agree, red = disagree)."""
    s, t = int(np.argmax(p_student)), int(np.argmax(p_teacher))
    img = cv2.resize(img, (img.shape[1] * 480 // img.shape[0], 480)) if img.shape[0] < 480 else img.copy()
    color = (60, 200, 60) if s == t else (40, 40, 230)
    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(img, f"Reflex: {classes[s]} {p_student[s]:.2f}  {ms:.1f} ms", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"Teacher: {classes[t]} {p_teacher[t]:.2f}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 3)
    return img


def _demo_frames(run: Path, test: list[dict], classes, P, T, ms) -> dict:
    """eval_frames/*.jpg for every held-out frame + eval_clip.mp4 of the longest held-out episode."""
    out_dir = run / "eval_frames"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    by_ep: dict[str, list[int]] = {}
    for i, d in enumerate(test):
        img = _annotate(cv2.imread(str(run / d["path"])), classes, P[i], T[i], ms[i])
        cv2.imwrite(str(out_dir / f"{d['frame_id']}.jpg"), img)
        by_ep.setdefault(d["episode"], []).append(i)
    ep, idx = max(by_ep.items(), key=lambda kv: len(kv[1]))
    idx.sort(key=lambda i: test[i]["t_sec"])
    fps = read_json(run / "ingest.json", default={}).get("fps") or 5
    first = cv2.imread(str(out_dir / f"{test[idx[0]]['frame_id']}.jpg"))
    clip = run / "eval_clip.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (first.shape[1], first.shape[0]))
    for i in idx:
        vw.write(cv2.resize(cv2.imread(str(out_dir / f"{test[i]['frame_id']}.jpg")), (first.shape[1], first.shape[0])))
    vw.release()
    return {"eval_frames": str(out_dir), "eval_clip": str(clip), "eval_clip_episode": ep}


def _human_check(run: Path, n: int, test: list[dict], classes, t_top, s_top) -> dict:
    """Write human_check.csv for a person to fill in; score it once filled."""
    path = run / "human_check.csv"
    if not path.exists():
        idx = np.linspace(0, len(test) - 1, min(n, len(test))).astype(int)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "frame_id", "path", "label"])
            for k, i in enumerate(idx):
                w.writerow([k + 1, test[i]["frame_id"], str(run / test[i]["path"]), ""])
        contact_sheet([(run / test[i]["path"], f"#{k + 1}") for k, i in enumerate(idx)], run / "human_check.png")
        return {"status": "pending", "file": str(path), "sheet": str(run / "human_check.png")}
    rows = [r for r in csv.DictReader(open(path)) if r["label"].strip()]
    if not rows:
        return {"status": "pending", "file": str(path), "sheet": str(run / "human_check.png")}
    pos = {d["frame_id"]: i for i, d in enumerate(test)}
    rows = [r for r in rows if r["frame_id"] in pos]
    h = np.array([r["label"].strip().lower() for r in rows])
    t = np.array([classes[t_top[pos[r["frame_id"]]]] for r in rows])
    s = np.array([classes[s_top[pos[r["frame_id"]]]] for r in rows])
    return {"status": "done", "n": len(rows), "teacher_vs_human": round(float((t == h).mean()), 4),
            "student_vs_human": round(float((s == h).mean()), 4)}


def evaluate(run: Path, human_check: int = 0) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    spec = load_spec(run)
    data, classes = load_dataset(run)
    if not (run / "split.json").exists():
        raise ReflexError("no split.json — run `reflex train` first (eval only scores held-out frames)")
    sp = read_json(run / "split.json")
    if set(sp["train"]) & set(sp["test"]):
        raise ReflexError("train and test splits overlap — re-run `reflex train`")
    by_id = {d["frame_id"]: d for d in data}
    test = [by_id[i] for i in sp["test"] if i in by_id]
    skill = Skill(run / "student")
    if skill.classes != classes:
        raise RuntimeError("student classes differ from spec — re-run `reflex train`")

    probs, ms = [], []
    for d in test:
        out = skill.predict(cv2.imread(str(run / d["path"])))
        probs.append([out["probs"][c] for c in classes])
        ms.append(out["ms"])
    P = np.array(probs)
    s_top = P.argmax(1)
    t_top = np.array([d["top"] for d in test])
    T = np.stack([d["y"] for d in test])
    labels = list(range(len(classes)))

    acc = float((s_top == t_top).mean())
    present = sorted(set(t_top.tolist()) | set(s_top.tolist()))  # classes absent from test don't count
    f1 = float(f1_score(t_top, s_top, labels=present, average="macro", zero_division=0))
    prec, rec, _, sup = precision_recall_fscore_support(t_top, s_top, labels=labels, zero_division=0)
    cm = confusion_matrix(t_top, s_top, labels=labels)
    teacher_rows = [r for r in read_jsonl(run / "labels.jsonl") if r.get("source") == "teacher"]
    conf_mask = T.max(1) >= spec["min_conf"]
    acc_conf = float((s_top[conf_mask] == t_top[conf_mask]).mean()) if conf_mask.any() else None

    lab = read_json(run / "label_summary.json", default={})
    t_lat = [r["latency_ms"] for r in teacher_rows]
    t_p50, t_p95 = (pctl(t_lat, 50), pctl(t_lat, 95)) if t_lat else (lab.get("teacher_p50_ms"), lab.get("teacher_p95_ms"))
    s_p50, s_p95 = pctl(ms, 50), pctl(ms, 95)
    size_mb = (run / "student" / "model.onnx").stat().st_size / 1e6
    usd_1000 = lab.get("usd_per_1000_frames")
    if usd_1000 is None and teacher_rows:
        usd_1000 = round(price_usd(np.mean([r.get("input_tokens", 0) for r in teacher_rows]),
                                   np.mean([r.get("output_tokens", 0) for r in teacher_rows])) * 1000, 3)

    # Worst disagreements: student confidently wrong where the teacher was confident.
    wrong = np.where(s_top != t_top)[0]
    score = T[wrong, t_top[wrong]] * P[wrong, s_top[wrong]]
    worst = wrong[np.argsort(-score)][:N_WORST]
    disagreements = [{"frame_id": test[i]["frame_id"], "path": str(run / test[i]["path"]),
                      "teacher": classes[t_top[i]], "teacher_conf": round(float(T[i, t_top[i]]), 3),
                      "student": classes[s_top[i]], "student_conf": round(float(P[i, s_top[i]]), 3)} for i in worst]
    (run / "disagreements.png").unlink(missing_ok=True)
    if disagreements:
        contact_sheet([(Path(d["path"]), f"T:{d['teacher']} S:{d['student']} {d['student_conf']:.2f}")
                       for d in disagreements], run / "disagreements.png", cols=4)
    _confusion_png(cm, classes, run / "confusion.png")
    demo = _demo_frames(run, test, classes, P, T, ms)

    metrics = {
        "agreement": round(acc, 4), "macro_f1": round(f1, 4),
        "agreement_on_confident": round(acc_conf, 4) if acc_conf is not None else None,
        "p50_ms": round(s_p50, 2), "p95_ms": round(s_p95, 2),
        "teacher_p50_ms": round(t_p50, 1) if t_p50 else None, "teacher_p95_ms": round(t_p95, 1) if t_p95 else None,
        "speedup": round(t_p50 / s_p50, 1) if t_p50 else None,
        "size_mb": round(size_mb, 2), "teacher_usd_per_1000": usd_1000, "student_usd_per_1000": 0.0,
        "n_test": len(test), "split_method": sp["method"],
    }
    per_class = {c: {"precision": round(float(prec[i]), 3), "recall": round(float(rec[i]), 3), "support": int(sup[i])}
                 for i, c in enumerate(classes)}
    report = {"run": run.name, "classes": classes, "metrics": metrics, "per_class": per_class,
              "confusion": cm.tolist(), "disagreements": disagreements,
              "train": read_json(run / "train.json", default={}), "labels": lab}
    if human_check:
        report["human_check"] = _human_check(run, human_check, test, classes, t_top, s_top)
    write_json(run / "report.json", report)
    (run / "report.md").write_text(_markdown(report, spec))

    meta = read_json(run / "student" / "skill.json")
    meta["metrics"] = {k: metrics[k] for k in ["agreement", "macro_f1", "p50_ms", "teacher_p50_ms", "size_mb", "speedup"]}
    write_json(run / "student" / "skill.json", meta)

    console.print(f"[bold green]{acc:.1%} agreement with the teacher[/bold green] (macro-F1 {f1:.2f}, n={len(test)}), "
                  f"student {s_p50:.1f} ms vs teacher {t_p50 or float('nan'):.0f} ms"
                  + (f" → {metrics['speedup']}× faster" if metrics["speedup"] else "") + f", {size_mb:.1f} MB")
    summary = {"report": str(run / "report.md"), "report_json": str(run / "report.json"),
               "confusion": str(run / "confusion.png"), "disagreements_png": str(run / "disagreements.png") if disagreements else None,
               **demo,
               **metrics, "per_class": per_class}
    hc = report.get("human_check")
    if hc:
        summary["human_check"] = hc
        if hc["status"] == "pending":
            raise NeedsInput(f"fill the 'label' column of {hc['file']} (see {hc['sheet']}), then re-run eval",
                             kind="human_check", **summary)
    return summary


def _markdown(r: dict, spec: dict) -> str:
    m, c = r["metrics"], r["classes"]
    lines = [
        f"# Reflex report — `{r['run']}`", "",
        f"**Task:** {spec['task_summary']}  ", f"**Query:** {spec['query']}  ",
        f"**Classes:** {', '.join(c)}  ", f"**Backbone:** {r['train'].get('backbone', '?')}", "",
        "## Student vs teacher", "",
        "| | Student (Reflex) | Teacher (cloud) |", "|---|---|---|",
        f"| Agreement with teacher | **{m['agreement']:.1%}** (macro-F1 {m['macro_f1']:.2f}) | — |",
        f"| Latency p50 / p95 | **{m['p50_ms']:.1f} / {m['p95_ms']:.1f} ms** (CPU, onnxruntime) | "
        f"{m['teacher_p50_ms']} / {m['teacher_p95_ms']} ms (+ network) |",
        f"| Speed-up | **{m['speedup']}×** | 1× |",
        f"| Size | **{m['size_mb']} MB** | — |",
        f"| Cost per 1,000 frames | **$0** | ${m['teacher_usd_per_1000']} |", "",
        f"Test set: {m['n_test']} frames, split by **{m['split_method']}** (no random frame split → no leakage).",
    ]
    if m.get("agreement_on_confident") is not None:
        lines.append(f"Agreement on frames where the teacher was confident (≥ {spec['min_conf']}): "
                     f"{m['agreement_on_confident']:.1%}.")
    lines += ["", "## Per class", "", "| class | precision | recall | support |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['support']} |" for k, v in r["per_class"].items()]
    lines += ["", "![confusion](confusion.png)", ""]
    if r["disagreements"]:
        lines += ["## Worst disagreements", "", "![disagreements](disagreements.png)", "",
                  "| frame | teacher | student |", "|---|---|---|"]
        lines += [f"| {d['frame_id']} | {d['teacher']} ({d['teacher_conf']:.2f}) | {d['student']} ({d['student_conf']:.2f}) |"
                  for d in r["disagreements"]]
    hc = r.get("human_check")
    if hc and hc["status"] == "done":
        lines += ["", "## Human check", "", f"On {hc['n']} human-labelled test frames: teacher "
                  f"{hc['teacher_vs_human']:.1%}, student {hc['student_vs_human']:.1%} correct."]
    lab = r.get("labels") or {}
    if lab:
        lines += ["", "## Labels", "", f"{lab.get('labelled')} frames labelled by the teacher, "
                  f"{lab.get('propagated', 0)} propagated back, {lab.get('low_conf', 0)} low-confidence (down-weighted)."]
    return "\n".join(lines) + "\n"
