"""Build the student (backbone + head) as ONNX + skill.json, and package skills/<name>/."""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .utils import ReflexError, console, read_json, skills_dir, write_json

SKILL_VERSION = 1


def _torch_modules():
    import torch
    from torch import nn

    class FeatNorm(nn.Module):
        """Fixed feature standardisation folded into the head (keeps training stable)."""

        def __init__(self, mu, sigma):
            super().__init__()
            self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
            self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))

        def forward(self, x):
            return (x - self.mu) / self.sigma

    class StudentNet(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            return self.head(self.backbone(x))

    return FeatNorm, StudentNet


def make_head(dim: int, n_classes: int, hidden: int = 0, mu=None, sigma=None):
    from torch import nn

    FeatNorm, _ = _torch_modules()
    mu = np.zeros(dim, np.float32) if mu is None else mu
    sigma = np.ones(dim, np.float32) if sigma is None else sigma
    layers = [FeatNorm(mu, sigma)]
    if hidden:
        layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_classes)]
    else:
        layers += [nn.Dropout(0.1), nn.Linear(dim, n_classes)]
    return nn.Sequential(*layers)


def make_student(backbone, head):
    _, StudentNet = _torch_modules()
    return StudentNet(backbone, head).eval()


def export_onnx(model, input_size, path: Path) -> float:
    """torch -> ONNX (opset 17, dynamic batch). Returns max |onnx - torch| on random input."""
    import onnxruntime as ort
    import torch

    model = model.cpu().eval()
    h, w = input_size
    x = torch.randn(2, 3, h, w)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=17,
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, (x,), str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, (x,), str(path), **kw)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    if diff > 1e-3:
        raise ReflexError(f"ONNX output differs from torch by {diff:.2e}")
    return diff


def write_skill(out_dir: Path, *, name, classes, cfg, backbone, spec, metrics=None, extra=None) -> Path:
    meta = {
        "name": name,
        "classes": classes,
        "input_size": cfg["input_size"],  # [H, W]; model input is 1x3xHxW float32
        "mean": cfg["mean"],
        "std": cfg["std"],
        "backbone": backbone,
        "spec": spec,
        "metrics": metrics or {},
        "version": SKILL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    return write_json(out_dir / "skill.json", meta)


def copy_runtime(out_dir: Path):
    shutil.copy(Path(__file__).with_name("runtime.py"), out_dir / "predict.py")


def build_student(run: Path) -> dict:
    """Trained head (+ optional fine-tuned backbone) -> run/student/{model.onnx, skill.json}."""
    import torch

    from .backbones import create_backbone
    from .utils import load_spec

    train = read_json(run / "train.json")
    spec = load_spec(run)
    backbone, cfg = create_backbone(train["backbone"])
    if (run / "backbone_ft.pt").exists():
        backbone.load_state_dict(torch.load(run / "backbone_ft.pt", map_location="cpu"))
    ck = torch.load(run / "head.pt", map_location="cpu")
    head = make_head(cfg["dim"], len(train["classes"]), ck["hidden"])
    head.load_state_dict(ck["state_dict"])
    out = run / "student"
    t0 = time.time()
    diff = export_onnx(make_student(backbone, head), cfg["input_size"], out / "model.onnx")
    write_skill(out, name=run.name, classes=train["classes"], cfg=cfg, backbone=train["backbone"], spec=spec)
    return {"onnx_max_abs_diff": diff, "export_s": round(time.time() - t0, 1),
            "size_mb": round((out / "model.onnx").stat().st_size / 1e6, 2)}


def export_skill(run: Path, name: str | None = None) -> dict:
    """Package run/student + reports into skills/<name>/."""
    student = run / "student"
    if not (student / "model.onnx").exists():
        console.print("[dim]no student yet — building ONNX[/dim]")
        build_student(run)
    name = name or run.name
    dst = skills_dir() / name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(student / "model.onnx", dst / "model.onnx")
    meta = read_json(student / "skill.json")
    meta["name"] = name
    write_json(dst / "skill.json", meta)
    copy_runtime(dst)
    copied = []
    for f in ["report.md", "report.json", "confusion.png", "disagreements.png",
              "stress.md", "stress.json", "stress.png"]:
        if (run / f).exists():
            shutil.copy(run / f, dst / f)
            copied.append(f)
    return {"skill_dir": str(dst), "files": sorted(p.name for p in dst.iterdir()),
            "size_mb": round((dst / "model.onnx").stat().st_size / 1e6, 2),
            "metrics": meta.get("metrics", {}), "reports_copied": copied}


def dummy_skill(name: str, classes: list[str]) -> dict:
    """Random-head skill with the real interface/architecture, for demo integration."""
    import torch

    from .backbones import DEFAULT_BACKBONE, create_backbone

    torch.manual_seed(0)
    backbone, cfg = create_backbone(DEFAULT_BACKBONE, pretrained=False)
    head = make_head(cfg["dim"], len(classes))
    dst = skills_dir() / name
    diff = export_onnx(make_student(backbone, head), cfg["input_size"], dst / "model.onnx")
    spec = {
        "name": name,
        "query": f"which of: {', '.join(classes)}?",
        "task_summary": "Classify the image into one of the given classes.",
        "visual_cues": [],
        "question": {
            "type": "choice", "name": name,
            "instructions": "Which option best describes the image?",
            "choices": [{"value": c, "description": c} for c in classes],
        },
        "edge_cases": [], "min_conf": 0.6, "propagate_back": 0, "status": "ready",
    }
    write_skill(dst, name=name, classes=classes, cfg=cfg, backbone=DEFAULT_BACKBONE, spec=spec,
                extra={"dummy": True})
    copy_runtime(dst)
    return {"skill_dir": str(dst), "classes": classes, "dummy": True, "onnx_max_abs_diff": diff,
            "size_mb": round((dst / "model.onnx").stat().st_size / 1e6, 2)}
