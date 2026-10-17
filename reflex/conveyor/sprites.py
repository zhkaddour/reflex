"""Object sprites for the conveyor sim: generated once with an OpenAI image model (transparent
PNG, top-down), cached in data/sprites/<kind>/<i>.png. `procedural=True` draws simple stand-ins
instead (no API key, for offline testing)."""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from ..utils import ReflexError, console

DEFAULT_IMAGE_MODEL = "gpt-image-1.5"  # gpt-image-2 does not support transparent backgrounds
SPRITE_LONG_SIDE = 256

# kind -> (physical long side in mm, variant descriptions). The first kind is the default target.
KINDS: dict[str, tuple[int, list[str]]] = {
    "plastic_bottle": (240, ["empty clear plastic water bottle with a blue cap",
                             "green tinted plastic soda bottle with a label",
                             "crushed clear plastic bottle with a white cap",
                             "brown plastic cola bottle with a red label",
                             "small clear plastic juice bottle with an orange cap"]),
    "glass_bottle": (250, ["empty green glass wine bottle",
                           "brown glass beer bottle with a paper label",
                           "clear glass bottle with a metal cap",
                           "cobalt blue glass bottle",
                           "small clear glass sauce bottle"]),
    "aluminium_can": (130, ["red aluminium soda can",
                            "crushed silver aluminium can",
                            "blue energy drink can",
                            "green aluminium beer can",
                            "dented yellow aluminium can"]),
    "cardboard": (220, ["flattened piece of brown corrugated cardboard",
                        "small closed brown cardboard box",
                        "grey cardboard egg carton",
                        "brown cardboard tube",
                        "torn piece of a pizza box"]),
    "paper": (150, ["crumpled ball of white paper",
                    "folded newspaper",
                    "crumpled brown paper bag",
                    "loose sheet of printed office paper, slightly wrinkled",
                    "crumpled yellow sticky notes"]),
    "carton": (190, ["empty white milk carton",
                     "orange juice tetra pak carton",
                     "small chocolate milk carton",
                     "flattened tetra pak drink carton",
                     "yoghurt pot"]),
    "plastic_bag": (230, ["crumpled clear plastic bag",
                          "crumpled white plastic shopping bag",
                          "crumpled blue plastic bag",
                          "plastic film wrapper",
                          "crumpled black plastic bag"]),
}

PROMPT = ("A single {desc}, photographed from directly above (top-down view), lying on a flat surface. "
          "Isolated object on a transparent background, no shadow, no text overlay, photorealistic "
          "product photo, the whole object visible and filling most of the frame.")


def sprites_dir() -> Path:
    return Path(os.environ.get("REFLEX_SPRITES_DIR", "data/sprites"))


def _clean(rgba: np.ndarray) -> np.ndarray:
    """Hard-ish alpha (image models leave a soft halo), crop to the object, resize."""
    a = rgba[..., 3].astype(np.float32)
    a = np.clip((a - 60) / 140, 0, 1)
    rgba = rgba.copy()
    rgba[..., 3] = (a * 255).astype(np.uint8)
    ys, xs = np.where(a > 0.05)
    if len(xs) == 0:
        raise ValueError("empty sprite")
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = rgba.shape[:2]
    s = SPRITE_LONG_SIDE / max(h, w)
    return cv2.resize(rgba, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _generate_one(client, model: str, desc: str, quality: str) -> np.ndarray:
    r = client.images.generate(model=model, prompt=PROMPT.format(desc=desc), size="1024x1024",
                               quality=quality, background="transparent", output_format="png")
    buf = np.frombuffer(base64.b64decode(r.data[0].b64_json), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("image model returned no alpha channel")
    return _clean(img)


def generate(out: Path | None = None, kinds: list[str] | None = None, per_kind: int = 5,
             quality: str = "low", model: str | None = None, workers: int = 6, force: bool = False) -> dict:
    """Generate missing sprites with the image model (cached: existing files are kept)."""
    from openai import OpenAI

    out = Path(out or sprites_dir())
    model = model or os.environ.get("REFLEX_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    kinds = kinds or list(KINDS)
    if unknown := [k for k in kinds if k not in KINDS]:
        raise ReflexError(f"unknown kinds {unknown}; known: {list(KINDS)}")
    jobs = [(k, i, KINDS[k][1][i % len(KINDS[k][1])]) for k in kinds for i in range(per_kind)
            if force or not (out / k / f"{i}.png").exists()]
    client = OpenAI()
    ok, failed = 0, []
    console.print(f"generating {len(jobs)} sprites with {model} ({quality}) → {out}")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_generate_one, client, model, d, quality): (k, i, d) for k, i, d in jobs}
        for f in as_completed(futs):
            k, i, d = futs[f]
            try:
                (out / k).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / k / f"{i}.png"), f.result())
                ok += 1
                console.print(f"  [green]✓[/green] {k}/{i}  {d}")
            except Exception as e:  # noqa: BLE001 — keep the rest
                failed.append(f"{k}/{i}: {e}")
                console.print(f"  [red]✗[/red] {k}/{i}  {e}")
    manifest = {k: {"long_side_mm": KINDS[k][0], "variants": KINDS[k][1]} for k in KINDS}
    (out / "manifest.json").write_text(json.dumps({"model": model, "kinds": manifest}, indent=2))
    return {"dir": str(out), "generated": ok, "failed": failed, "model": model,
            "total": sum(len(list((out / k).glob("*.png"))) for k in kinds if (out / k).exists())}


# ---------------------------------------------------------------- procedural stand-ins

_PROC_COLORS = {"plastic_bottle": (225, 230, 235), "glass_bottle": (60, 120, 50), "aluminium_can": (40, 40, 200),
                "cardboard": (70, 120, 170), "paper": (240, 240, 240), "carton": (230, 200, 120),
                "plastic_bag": (200, 160, 120)}


def _procedural(kind: str, rng: np.random.Generator) -> np.ndarray:
    h, w = (90, 256) if kind in ("plastic_bottle", "glass_bottle") else (150, 200)
    img = np.zeros((h, w, 4), np.uint8)
    color = tuple(int(np.clip(c + rng.integers(-30, 30), 0, 255)) for c in _PROC_COLORS[kind]) + (255,)
    if kind.endswith("bottle"):
        cv2.rectangle(img, (40, 10), (w - 10, h - 10), color, -1)
        cv2.rectangle(img, (0, h // 2 - 14), (45, h // 2 + 14), (200, 120, 30, 255) if kind == "plastic_bottle"
                      else (150, 150, 150, 255), -1)
    elif kind in ("aluminium_can", "paper"):
        cv2.circle(img, (w // 2, h // 2), h // 2 - 4, color, -1)
    else:
        cv2.rectangle(img, (8, 8), (w - 8, h - 8), color, -1)
    return img


def load(root: Path | None = None, procedural: bool = False, seed: int = 0) -> dict[str, list[np.ndarray]]:
    """kind -> list of RGBA sprites (long side ~256 px)."""
    root = Path(root or sprites_dir())
    if procedural:
        rng = np.random.default_rng(seed)
        return {k: [_procedural(k, rng) for _ in range(3)] for k in KINDS}
    out = {}
    for k in KINDS:
        files = sorted((root / k).glob("*.png")) if (root / k).exists() else []
        imgs = [im for im in (cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in files)
                if im is not None and im.ndim == 3 and im.shape[2] == 4]
        if imgs:
            out[k] = imgs
    if not out:
        raise ReflexError(f"no sprites in {root} — run `reflex conveyor sprites` (or use --procedural)")
    return out
