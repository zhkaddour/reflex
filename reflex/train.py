"""Stage 4: frozen pretrained backbone + small head trained on the teacher's soft labels.

Split is by time/episode, never random by frame (neighbouring frames are near-duplicates)."""

from __future__ import annotations

import time
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .backbones import create_backbone
from .runtime import preprocess, spec_classes
from .utils import ReflexError, console, device, load_frames, load_spec, read_jsonl, write_json

TEST_FRAC = 0.2
GAP_SEC = 1.0


# ---------------------------------------------------------------- data

def load_dataset(run: Path) -> tuple[list[dict], list[str]]:
    """Labelled frames with soft-label vectors in class order."""
    s = load_spec(run)
    classes = spec_classes(s)
    labels = {r["frame_id"]: r for r in read_jsonl(run / "labels_final.jsonl")}
    if not labels:
        raise ReflexError("no labels — run `reflex label` first")
    data = []
    for f in load_frames(run):
        r = labels.get(f["frame_id"])
        if r is None:
            continue
        y = np.array([r["probs"].get(c, 0.0) for c in classes], np.float32)
        data.append({**f, "y": y / max(y.sum(), 1e-6), "top": classes.index(r["top"]),
                     "weight": float(r.get("weight", r["conf"])), "source": r.get("source", "teacher")})
    return data, classes


def time_split(data: list[dict], test_frac: float = TEST_FRAC, gap_sec: float = GAP_SEC) -> dict:
    """>=3 episodes: hold out whole episodes. Otherwise: last 20% of each episode is test,
    with a 1 s gap before it dropped from training."""
    episodes = sorted({d["episode"] for d in data})
    train, test, gap = [], [], []
    if len(episodes) >= 3:
        n_test = max(1, round(test_frac * len(episodes)))
        test_eps = {episodes[int((k + 0.5) * len(episodes) / n_test)] for k in range(n_test)}
        for d in data:
            (test if d["episode"] in test_eps else train).append(d["frame_id"])
        method = "episode"
    else:
        test_eps = set()
        for ep in episodes:
            ts = np.array([d["t_sec"] for d in data if d["episode"] == ep])
            cut = np.quantile(ts, 1 - test_frac)
            is_video_time = len(ts) > 1 and (ts.max() - ts.min()) > 0
            for d in data:
                if d["episode"] != ep:
                    continue
                if d["t_sec"] >= cut:
                    test.append(d["frame_id"])
                elif is_video_time and d["t_sec"] >= cut - gap_sec:
                    gap.append(d["frame_id"])
                else:
                    train.append(d["frame_id"])
        method = "time"
    assert not (set(train) & set(test)) and not (set(gap) & (set(train) | set(test))), "split overlap"
    if not train or not test:
        raise ReflexError(f"split failed (train={len(train)}, test={len(test)}) — need more frames/episodes")
    return {"method": method, "test_episodes": sorted(test_eps), "train": train, "test": test, "gap": gap}


def augment(img: np.ndarray, rng: np.random.Generator, hflip_ok: bool) -> np.ndarray:
    h, w = img.shape[:2]
    if hflip_ok and rng.random() < 0.5:
        img = img[:, ::-1]
    s = rng.uniform(0.8, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    img = cv2.resize(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), (w, h))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.75, 1.25)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    img = (img - img.mean()) * rng.uniform(0.8, 1.2) + img.mean()
    return np.clip(img, 0, 255).astype(np.uint8)


def embed(model, imgs: list[np.ndarray], cfg: dict, dev: str, bs: int = 64) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = np.concatenate([preprocess(im, cfg["input_size"], cfg["mean"], cfg["std"]) for im in imgs[i:i + bs]])
            out.append(model(torch.from_numpy(x).to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, cfg["dim"]), np.float32)


def cached_embeddings(run: Path, key: str, ids: list[str], make_imgs, model, cfg, dev) -> np.ndarray:
    """Embeddings for ids, cached in run/emb/<key>.npz; only missing ids are computed."""
    path = run / "emb" / f"{key}.npz"
    cache = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["feats"]))
    missing = [i for i in ids if i not in cache]
    if missing:
        t0 = time.time()
        feats = embed(model, make_imgs(missing), cfg, dev)
        cache.update(zip(missing, feats))
        path.parent.mkdir(exist_ok=True)
        np.savez(path, ids=np.array(list(cache)), feats=np.stack(list(cache.values())))
        console.print(f"[dim]embedded {len(missing)} images ({key}) in {time.time() - t0:.1f}s[/dim]")
    return np.stack([cache[i] for i in ids])


# ---------------------------------------------------------------- training

def _soft_ce(logits, y, w):
    import torch

    return (-(y * torch.log_softmax(logits, 1)).sum(1) * w).sum() / w.sum()


def train_head(X, Y, W, hidden: int, epochs: int, seed: int = 0):
    import torch

    from .export import make_head

    torch.manual_seed(seed)
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    head = make_head(X.shape[1], Y.shape[1], hidden, mu, sigma)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt, Wt = map(lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)), (X, Y, W))
    n, bs = len(Xt), 256
    gen = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = _soft_ce(head(Xt[b]), Yt[b], Wt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    return head, float(loss)


def finetune_backbone(run, backbone, head, data, train_ids, cfg, dev, hflip_ok, epochs: int, unfreeze_frac: float = 0.3):
    """Unfreeze the last ~30% of backbone parameters and train end to end on images."""
    import torch

    from .export import make_student

    model = make_student(backbone, head).to(dev)
    params = list(backbone.parameters())
    for i, p in enumerate(params):
        p.requires_grad = i >= int(len(params) * (1 - unfreeze_frac))
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": head.parameters(), "lr": 5e-4}], weight_decay=1e-2)
    by_id = {d["frame_id"]: d for d in data}
    rng = np.random.default_rng(0)
    model.eval()  # keep BatchNorm statistics frozen; gradients still flow
    for ep in range(epochs):
        order = rng.permutation(train_ids)
        tot = 0.0
        for i in range(0, len(order), 32):
            batch = [by_id[f] for f in order[i:i + 32]]
            imgs = [augment(cv2.imread(str(run / d["path"])), rng, hflip_ok) for d in batch]
            x = torch.from_numpy(np.concatenate([preprocess(im, cfg["input_size"], cfg["mean"], cfg["std"])
                                                 for im in imgs])).to(dev)
            y = torch.from_numpy(np.stack([d["y"] for d in batch])).to(dev)
            w = torch.tensor([d["weight"] for d in batch], dtype=torch.float32, device=dev)
            loss = _soft_ce(model(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(batch)
        console.print(f"[dim]finetune epoch {ep + 1}/{epochs}: loss {tot / len(order):.4f}[/dim]")
    for p in backbone.parameters():
        p.requires_grad = True
    return backbone.cpu(), head.cpu()


def train(run: Path, backbone: str = "mobilenetv3_small_100", epochs: int = 300, hidden: int = 0, aug: int = 2,
          finetune: bool = False, ft_epochs: int = 3) -> dict:
    import torch

    from .export import build_student

    t_start = time.time()
    s = load_spec(run)
    data, classes = load_dataset(run)
    sp = time_split(data)
    write_json(run / "split.json", sp)
    by_id = {d["frame_id"]: d for d in data}
    dev = device()
    console.print(f"backbone [bold]{backbone}[/bold] on {dev}; split by {sp['method']}: "
                  f"{len(sp['train'])} train / {len(sp['test'])} test / {len(sp['gap'])} gap")
    model, cfg = create_backbone(backbone)
    model = model.to(dev)

    def clean_imgs(ids):
        return [cv2.imread(str(run / by_id[i]["path"])) for i in ids]

    all_ids = sp["train"] + sp["test"]
    E = dict(zip(all_ids, cached_embeddings(run, backbone, all_ids, clean_imgs, model, cfg, dev)))
    Xs, Ys, Ws = [np.stack([E[i] for i in sp["train"]])], [], []
    Ys.append(np.stack([by_id[i]["y"] for i in sp["train"]]))
    Ws.append(np.array([by_id[i]["weight"] for i in sp["train"]], np.float32))
    for k in range(aug):
        def aug_imgs(ids, k=k):
            out = []
            for i in ids:
                rng = np.random.default_rng(zlib.crc32(f"{i}/{k}".encode()))
                out.append(augment(cv2.imread(str(run / by_id[i]["path"])), rng, s.get("hflip_ok", True)))
            return out
        Xs.append(cached_embeddings(run, f"{backbone}_aug{k}", sp["train"], aug_imgs, model, cfg, dev))
        Ys.append(Ys[0])
        Ws.append(Ws[0])
    X, Y, W = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)

    # Class balancing (sqrt inverse frequency of the teacher's top label).
    tops = np.concatenate([[by_id[i]["top"] for i in sp["train"]]] * (aug + 1))
    counts = np.bincount(tops, minlength=len(classes)).astype(np.float32)
    cw = np.where(counts > 0, np.sqrt(counts.sum() / (len(classes) * np.maximum(counts, 1))), 0.0)
    W = W * cw[tops]

    t0 = time.time()
    head, loss = train_head(X, Y, W, hidden, epochs)
    head_s = time.time() - t0

    def agreement(ids):
        with torch.no_grad():
            p = head(torch.from_numpy(np.stack([E[i] for i in ids]))).argmax(1).numpy()
        return float((p == np.array([by_id[i]["top"] for i in ids])).mean())

    if (run / "backbone_ft.pt").exists():
        (run / "backbone_ft.pt").unlink()
    if finetune:
        model, head = finetune_backbone(run, model, head, data, sp["train"], cfg, dev, s.get("hflip_ok", True), ft_epochs)
        torch.save(model.state_dict(), run / "backbone_ft.pt")
    torch.save({"state_dict": head.state_dict(), "hidden": hidden}, run / "head.pt")

    info = {"backbone": backbone, "classes": classes, "input_size": cfg["input_size"], "mean": cfg["mean"],
            "std": cfg["std"], "feat_dim": cfg["dim"], "hidden": hidden, "epochs": epochs, "aug": aug,
            "finetuned": finetune, "device": dev, "split_method": sp["method"],
            "n_train": len(sp["train"]), "n_test": len(sp["test"]), "n_gap": len(sp["gap"]),
            "train_class_counts": {c: int(counts[i] // (aug + 1)) for i, c in enumerate(classes)},
            "test_class_counts": dict(Counter(classes[by_id[i]["top"]] for i in sp["test"])),
            "final_loss": round(loss, 4), "head_train_s": round(head_s, 2)}
    if not finetune:  # quick numbers from cached embeddings (eval gives the real ONNX numbers)
        info["train_agreement"] = round(agreement(sp["train"]), 4)
        info["test_agreement_quick"] = round(agreement(sp["test"]), 4)
    write_json(run / "train.json", info)
    info.update(build_student(run))
    info["total_s"] = round(time.time() - t_start, 1)
    write_json(run / "train.json", info)
    console.print(f"[green]trained[/green] in {info['total_s']}s — "
                  f"test agreement (quick) {info.get('test_agreement_quick', 'n/a')}, ONNX {info['size_mb']} MB")
    return info

