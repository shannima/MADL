from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def compute_losses(
    logits: torch.Tensor,
    heatmap_logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    mask_valid: torch.Tensor,
    cls_loss_weight: float = 1.0,
    bce_loss_weight: float = 1.0,
    dice_loss_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(logits, labels)

    valid = mask_valid.view(-1, 1, 1, 1)
    if float(valid.sum()) > 0.0:
        bce_map = F.binary_cross_entropy_with_logits(heatmap_logits, masks, reduction="none")
        valid_pixels = valid.expand_as(bce_map)
        bce_loss = (bce_map * valid_pixels).sum() / valid_pixels.sum().clamp_min(1.0)
        valid_indices = mask_valid > 0.5
        d_loss = dice_loss(heatmap_logits[valid_indices], masks[valid_indices])
    else:
        zero = logits.new_zeros(())
        bce_loss = zero
        d_loss = zero

    total = (
        (cls_loss_weight * cls_loss) + (bce_loss_weight * bce_loss) + (dice_loss_weight * d_loss)
    )
    return {
        "total": total,
        "cls_loss": cls_loss,
        "bce_loss": bce_loss,
        "dice_loss": d_loss,
    }
