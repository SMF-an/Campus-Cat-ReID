from __future__ import annotations

from io import BytesIO
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

try:
    import timm
except ImportError:  # timm is optional in some environments
    timm = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fixed DINOv3 model used by the project.
DINOV3_MODEL_NAME = "vit_base_patch16_dinov3"


def _build_preprocess(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _BackboneWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)

        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if isinstance(feats, dict):
            for key in ("image_features", "x", "features", "last_hidden_state"):
                if key in feats:
                    feats = feats[key]
                    break

        if torch.is_tensor(feats) and feats.ndim == 3:
            feats = feats[:, 0, :] if feats.shape[1] > 1 else feats.mean(dim=1)

        if not torch.is_tensor(feats):
            raise TypeError(f"Unexpected backbone output: {type(feats)!r}")
        return feats


def build_backbone(
    device: str = "cpu",
    pretrained: bool = True,
    image_size: int = 224,
) -> Tuple[nn.Module, int, Callable[[Image.Image], torch.Tensor]]:
    if timm is None:
        raise RuntimeError("timm is required for DINOv3 backbones. Install with: pip install timm")

    try:
        model = timm.create_model(DINOV3_MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="token")
    except Exception as exc:
        raise RuntimeError(f"Failed to create DINOv3 model from timm: {DINOV3_MODEL_NAME}") from exc

    feature_dim = int(getattr(model, "num_features", 0) or getattr(model, "embed_dim", 0) or 0)
    if feature_dim <= 0:
        raise RuntimeError(f"Could not determine feature dimension for model {DINOV3_MODEL_NAME!r}")

    return _BackboneWrapper(model).to(device), feature_dim, _build_preprocess(image_size)


class VisionFeatureExtractor:
    def __init__(
        self,
        device: str = "cpu",
        checkpoint: Optional[str] = None,
        pretrained: bool = True,
        image_size: int = 224,
    ):
        self.device = device
        self.emb_head: Optional[nn.Module] = None

        self.model, self.dim, self.preprocess = build_backbone(device=device, pretrained=pretrained, image_size=image_size)

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def extract(self, image: Image.Image) -> np.ndarray:
        if self.model is None or self.preprocess is None:
            vec = np.random.rand(getattr(self, "dim", 512)).astype(np.float32)
            return vec / np.linalg.norm(vec)

        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
            if self.emb_head is not None:
                feats = self.emb_head(feats)

        if not torch.is_tensor(feats):
            raise TypeError(f"Embedding is not a tensor: {type(feats)!r}")

        vec = feats.detach().cpu().numpy()[0].astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        backbone_state = {}
        for key, value in ckpt.items():
            if key.startswith("backbone."):
                backbone_state[key[len("backbone.") :]] = value

        if backbone_state:
            try:
                self.model.load_state_dict(backbone_state, strict=False)
            except Exception:
                pass

        if "emb_head.proj.weight" in ckpt:
            weight = ckpt["emb_head.proj.weight"]
            out_dim, in_dim = weight.shape
            proj = nn.Linear(in_dim, out_dim, bias=("emb_head.proj.bias" in ckpt))
            state = {"weight": weight}
            if "emb_head.proj.bias" in ckpt:
                state["bias"] = ckpt["emb_head.proj.bias"]
            proj.load_state_dict(state)
            self.emb_head = proj.to(self.device)
            self.dim = out_dim


def load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")
