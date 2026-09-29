from __future__ import annotations

import math
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

CLASS_NAMES = ["real", "synthetic", "tampered"]


def _resolve_weights(backbone_name: str, use_pretrained: bool):
    if not use_pretrained:
        return {"weights": None}
    try:
        if backbone_name == "efficientnet_b0":
            return {"weights": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1}
        if backbone_name == "convnext_tiny":
            return {"weights": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1}
        if backbone_name == "vit_b_16":
            return {"weights": tvm.ViT_B_16_Weights.IMAGENET1K_V1}
    except AttributeError:
        return {"pretrained": use_pretrained}
    return {"weights": None}


class EfficientNetBackbone(nn.Module):
    out_channels = 1280

    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        kwargs = _resolve_weights("efficientnet_b0", use_pretrained)
        model = tvm.efficientnet_b0(**kwargs)
        self.features = model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ConvNeXtBackbone(nn.Module):
    out_channels = 768

    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        kwargs = _resolve_weights("convnext_tiny", use_pretrained)
        model = tvm.convnext_tiny(**kwargs)
        self.features = model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ViTBackbone(nn.Module):
    out_channels = 768

    def __init__(self, image_size: int = 224, use_pretrained: bool = True):
        super().__init__()
        kwargs = _resolve_weights("vit_b_16", use_pretrained)
        model = tvm.vit_b_16(image_size=image_size, **kwargs)
        self.model = model
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = self.model._process_input(x)
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x[:, 1:, :]
        token_count = x.shape[1]
        side = int(math.sqrt(token_count))
        x = x.transpose(1, 2).reshape(n, self.out_channels, side, side)
        return x


def build_backbone(backbone_name: str, image_size: int, use_pretrained: bool) -> nn.Module:
    if backbone_name == "efficientnet_b0":
        return EfficientNetBackbone(use_pretrained=use_pretrained)
    if backbone_name == "convnext_tiny":
        return ConvNeXtBackbone(use_pretrained=use_pretrained)
    if backbone_name == "vit_b_16":
        return ViTBackbone(image_size=image_size, use_pretrained=use_pretrained)
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def partial_load(module: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    current = module.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    current.update(filtered)
    module.load_state_dict(current, strict=False)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(dim)
        self.noise_norm = nn.LayerNorm(dim)
        self.rgb_to_noise = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.noise_to_rgb = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, rgb_feat: torch.Tensor, noise_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)
        noise_tokens = noise_feat.flatten(2).transpose(1, 2)
        rgb_q = self.rgb_norm(rgb_tokens)
        noise_q = self.noise_norm(noise_tokens)
        rgb_attended, _ = self.rgb_to_noise(rgb_q, noise_q, noise_q, need_weights=False)
        noise_attended, _ = self.noise_to_rgb(noise_q, rgb_q, rgb_q, need_weights=False)
        fused = self.mix(torch.cat([rgb_attended, noise_attended], dim=-1))
        fused = fused.transpose(1, 2).reshape(b, c, h, w)
        return fused


class DualStreamPixelAgent(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        image_size: int = 224,
        fusion_dim: int = 256,
        num_classes: int = 3,
        use_pretrained: bool = True,
        rgb_init_checkpoint: str = "",
        noise_init_checkpoint: str = "",
        proposal_threshold: float = 0.45,
        proposal_top_k: int = 5,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.proposal_threshold = proposal_threshold
        self.proposal_top_k = proposal_top_k

        self.rgb_backbone = build_backbone(backbone_name, image_size, use_pretrained)
        self.noise_backbone = build_backbone(backbone_name, image_size, use_pretrained)
        if rgb_init_checkpoint:
            partial_load(self.rgb_backbone, rgb_init_checkpoint)
        if noise_init_checkpoint:
            partial_load(self.noise_backbone, noise_init_checkpoint)

        in_dim = int(getattr(self.rgb_backbone, "out_channels"))
        self.rgb_proj = nn.Conv2d(in_dim, fusion_dim, kernel_size=1)
        self.noise_proj = nn.Conv2d(in_dim, fusion_dim, kernel_size=1)
        self.fusion = CrossAttentionFusion(dim=fusion_dim, num_heads=4)
        self.post_fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, num_classes),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_dim // 2, 1, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb_feat = self.rgb_proj(self.rgb_backbone(rgb))
        noise_feat = self.noise_proj(self.noise_backbone(noise))

        if rgb_feat.shape[-2:] != noise_feat.shape[-2:]:
            noise_feat = F.interpolate(
                noise_feat, size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False
            )

        fused = self.fusion(rgb_feat, noise_feat)
        fused = self.post_fusion(fused)

        logits = self.classifier(fused)
        heatmap_logits = self.heatmap_head(fused)
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=(rgb.shape[-2], rgb.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "logits": logits,
            "heatmap_logits": heatmap_logits,
        }

    @torch.no_grad()
    def predict_structured(self, rgb: torch.Tensor, noise: torch.Tensor) -> List[Dict[str, Any]]:
        outputs = self.forward(rgb, noise)
        probs = torch.softmax(outputs["logits"], dim=-1)
        heatmap = torch.sigmoid(outputs["heatmap_logits"])

        results: List[Dict[str, Any]] = []
        for idx in range(rgb.shape[0]):
            heat = heatmap[idx, 0].detach().cpu().numpy().astype(np.float32)
            cls_probs = probs[idx].detach().cpu().numpy().astype(np.float32)
            label_idx = int(cls_probs.argmax())
            proposals = self.extract_proposals(heat)
            results.append(
                {
                    "label": CLASS_NAMES[label_idx],
                    "confidence": float(cls_probs[label_idx]),
                    "class_probs": {
                        CLASS_NAMES[i]: float(cls_probs[i]) for i in range(len(CLASS_NAMES))
                    },
                    "low_level_confidence": float(cls_probs[2]),
                    "global_synthetic_score": float(cls_probs[1]),
                    "anomaly_heatmap": heat,
                    "suspicious_regions": proposals,
                }
            )
        return results

    def extract_proposals(self, heatmap: np.ndarray) -> List[Dict[str, Any]]:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        binary = (heatmap >= self.proposal_threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        proposals: List[Dict[str, Any]] = []
        height, width = heatmap.shape
        for comp_idx in range(1, num_labels):
            x, y, w, h, area = stats[comp_idx]
            if area <= 4:
                continue
            component_mask = labels == comp_idx
            score = float(heatmap[component_mask].mean())
            proposals.append(
                {
                    "box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                    "box_permille": [
                        int(round((x / max(1, width)) * 1000)),
                        int(round((y / max(1, height)) * 1000)),
                        int(round(((x + w) / max(1, width)) * 1000)),
                        int(round(((y + h) / max(1, height)) * 1000)),
                    ],
                    "score": score,
                    "area": int(area),
                    "area_ratio": float(area / max(1, height * width)),
                }
            )
        proposals.sort(key=lambda row: row["score"], reverse=True)
        return proposals[: self.proposal_top_k]
