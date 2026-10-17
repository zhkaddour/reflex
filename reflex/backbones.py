"""Backbone menu (timm, pretrained weights from Hugging Face)."""

from __future__ import annotations

from .utils import ReflexError

BACKBONES = {
    "mobilenetv3_small_100": {
        "params": "1.5M", "use": "Speed matters most (RPS, conveyor)", "kwargs": {}},
    "efficientnet_b0": {
        "params": "4.0M", "use": "Balanced", "kwargs": {}},
    "vit_small_patch14_dinov2.lvd142m": {
        "params": "~22M", "use": "Hardest / most specific tasks", "kwargs": {"img_size": 224}},
}
DEFAULT_BACKBONE = "mobilenetv3_small_100"


def create_backbone(name: str, pretrained: bool = True):
    """-> (model returning pooled features, cfg {input_size:(H,W), mean, std, dim})."""
    import timm
    import torch

    if name not in BACKBONES:
        raise ReflexError(f"unknown backbone '{name}'. Choose from: {', '.join(BACKBONES)}")
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, **BACKBONES[name]["kwargs"])
    model.eval()
    dc = timm.data.resolve_model_data_config(model)
    _, h, w = dc["input_size"]
    with torch.no_grad():
        dim = model(torch.zeros(1, 3, h, w)).shape[-1]
    return model, {"input_size": [h, w], "mean": list(dc["mean"]), "std": list(dc["std"]), "dim": int(dim)}
