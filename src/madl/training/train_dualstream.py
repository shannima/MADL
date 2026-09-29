from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from madl.backbones.dualstream.dataset import PixelForgeryDataset
from madl.backbones.dualstream.losses import compute_losses
from madl.backbones.dualstream.model import CLASS_NAMES, DualStreamPixelAgent


@dataclass
class EpochMetrics:
    loss: float
    cls_loss: float
    bce_loss: float
    dice_loss: float
    accuracy: float
    tampered_iou: float


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dual-stream pixel forensics agent.")
    parser.add_argument("--train_real_dir", required=True)
    parser.add_argument("--train_tampered_dir", required=True)
    parser.add_argument("--train_mask_dir", required=True)
    parser.add_argument("--train_synthetic_dirs", nargs="+", required=True)
    parser.add_argument("--val_real_dir", required=True)
    parser.add_argument("--val_tampered_dir", required=True)
    parser.add_argument("--val_mask_dir", required=True)
    parser.add_argument("--val_synthetic_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--backbone",
        default="convnext_tiny",
        choices=["efficientnet_b0", "convnext_tiny", "vit_b_16"],
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples_per_class", type=int, default=0)
    parser.add_argument("--max_val_samples_per_class", type=int, default=0)
    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--bce_loss_weight", type=float, default=1.0)
    parser.add_argument("--dice_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="")
    parser.add_argument("--rgb_init_checkpoint", default="")
    parser.add_argument("--noise_init_checkpoint", default="")
    parser.add_argument("--no_pretrained", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    targets = (targets >= 0.5).float()
    pred = pred.flatten(1)
    targets = targets.flatten(1)
    intersection = (pred * targets).sum(dim=1)
    union = pred.sum(dim=1) + targets.sum(dim=1) - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1e-6), torch.ones_like(union))


def build_loader(
    dataset: PixelForgeryDataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model: DualStreamPixelAgent,
    loader: DataLoader,
    optimizer,
    device: str,
    cls_loss_weight: float,
    bce_loss_weight: float,
    dice_loss_weight: float,
    training: bool,
) -> EpochMetrics:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_bce = 0.0
    total_dice = 0.0
    correct = 0
    total = 0
    tampered_ious = []

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch["rgb"], batch["noise"])
            losses = compute_losses(
                logits=outputs["logits"],
                heatmap_logits=outputs["heatmap_logits"],
                labels=batch["label"],
                masks=batch["mask"],
                mask_valid=batch["mask_valid"],
                cls_loss_weight=cls_loss_weight,
                bce_loss_weight=bce_loss_weight,
                dice_loss_weight=dice_loss_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

        batch_size = int(batch["label"].shape[0])
        total_loss += float(losses["total"].detach().item()) * batch_size
        total_cls += float(losses["cls_loss"].detach().item()) * batch_size
        total_bce += float(losses["bce_loss"].detach().item()) * batch_size
        total_dice += float(losses["dice_loss"].detach().item()) * batch_size

        pred = outputs["logits"].argmax(dim=-1)
        correct += int((pred == batch["label"]).sum().item())
        total += batch_size

        tampered_mask = batch["label"] == 2
        if bool(tampered_mask.any()):
            ious = compute_iou_from_logits(
                outputs["heatmap_logits"][tampered_mask], batch["mask"][tampered_mask]
            )
            tampered_ious.extend(float(v) for v in ious.detach().cpu().tolist())

        iterator.set_postfix(
            loss=f"{(total_loss / max(1, total)):.4f}",
            acc=f"{(correct / max(1, total)):.4f}",
        )

    return EpochMetrics(
        loss=total_loss / max(1, total),
        cls_loss=total_cls / max(1, total),
        bce_loss=total_bce / max(1, total),
        dice_loss=total_dice / max(1, total),
        accuracy=correct / max(1, total),
        tampered_iou=float(np.mean(tampered_ious)) if tampered_ious else 0.0,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_dataset = PixelForgeryDataset(
        real_dir=args.train_real_dir,
        tampered_dir=args.train_tampered_dir,
        mask_dir=args.train_mask_dir,
        synthetic_dirs=args.train_synthetic_dirs,
        image_size=args.image_size,
        split="train",
        max_samples_per_class=args.max_train_samples_per_class,
        seed=args.seed,
    )
    val_dataset = PixelForgeryDataset(
        real_dir=args.val_real_dir,
        tampered_dir=args.val_tampered_dir,
        mask_dir=args.val_mask_dir,
        synthetic_dirs=args.val_synthetic_dirs,
        image_size=args.image_size,
        split="val",
        max_samples_per_class=args.max_val_samples_per_class,
        seed=args.seed + 100,
    )
    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    model = DualStreamPixelAgent(
        backbone_name=args.backbone,
        image_size=args.image_size,
        use_pretrained=not args.no_pretrained,
        rgb_init_checkpoint=args.rgb_init_checkpoint,
        noise_init_checkpoint=args.noise_init_checkpoint,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_score = -1.0
    history = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", -1.0))

    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.cls_loss_weight,
            args.bce_loss_weight,
            args.dice_loss_weight,
            training=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=args.device,
            cls_loss_weight=args.cls_loss_weight,
            bce_loss_weight=args.bce_loss_weight,
            dice_loss_weight=args.dice_loss_weight,
            training=False,
        )
        epoch_row = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(epoch_row)
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        score = float(val_metrics.accuracy + val_metrics.tampered_iou)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": max(best_score, score),
            "args": vars(args),
            "class_names": CLASS_NAMES,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        if score > best_score:
            best_score = score
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))

        print(
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_tampered_iou={val_metrics.tampered_iou:.4f}"
        )


if __name__ == "__main__":
    main()
