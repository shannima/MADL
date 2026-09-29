from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterator, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm

from madl.localization.visual_ranker import (
    ConvNeXtCandidateRanker,
    VisualCandidateDataset,
    visual_candidate_ranking_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Tiny visual Agent-C candidate ranker."
    )
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_candidates_per_image", type=int, default=0)
    parser.add_argument("--no_pretrained", action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--freeze_visual_branch", action="store_true", default=False)
    parser.add_argument("--augment_train", action="store_true", default=False)
    parser.add_argument("--use_geometry_features", action="store_true", default=False)
    parser.add_argument("--use_source_embedding", action="store_true", default=False)
    parser.add_argument("--source_vocab_size", type=int, default=128)
    parser.add_argument("--source_embedding_dim", type=int, default=8)
    parser.add_argument("--pointwise_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_weight", type=float, default=0.35)
    parser.add_argument("--listwise_weight", type=float, default=0.15)
    parser.add_argument("--hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--ranking_margin", type=float, default=0.20)
    parser.add_argument("--target_gap", type=float, default=0.20)
    parser.add_argument("--grouped_batches", action="store_true", default=False)
    parser.add_argument(
        "--trust_replay_mask_paths",
        action="store_true",
        default=False,
        help="Skip raw-mask existence filtering during dataset initialization; use only with frozen replay manifests.",
    )
    parser.add_argument(
        "--trust_replay_image_paths",
        action="store_true",
        default=False,
        help="Construct image paths directly from replay stems instead of probing every extension/zero-padding variant.",
    )
    parser.add_argument(
        "--trusted_image_ext",
        default=".png",
        help="Image extension used with --trust_replay_image_paths.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Warm-start from an existing visual candidate ranker checkpoint.",
    )
    parser.add_argument(
        "--focus_pairs_json",
        default="",
        help="Targeted error report containing focus_pairs for oversampling.",
    )
    parser.add_argument(
        "--focus_repeats",
        type=int,
        default=1,
        help="Repeat focus-image grouped batches this many times.",
    )
    parser.add_argument(
        "--focus_hard_only",
        action="store_true",
        default=False,
        help="Use only high-SAM low-IoU focus pairs.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grouped_candidate_batches(
    dataset: VisualCandidateDataset,
    *,
    max_batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    focus_stems: Set[str] | None = None,
    focus_repeats: int = 1,
) -> List[List[int]]:
    focus_stems = set(focus_stems or set())
    focus_repeats = max(1, int(focus_repeats))
    groups: Dict[str, List[int]] = {}
    for index, sample in enumerate(dataset.samples):
        groups.setdefault(str(sample.stem), []).append(index)
    batches: List[List[int]] = []
    for stem, indices in groups.items():
        repeat_count = focus_repeats if stem in focus_stems else 1
        for _ in range(repeat_count):
            for start in range(0, len(indices), int(max_batch_size)):
                batches.append(indices[start : start + int(max_batch_size)])
    if shuffle:
        random.Random(int(seed)).shuffle(batches)
    return batches


class StaticBatchSampler(Sampler[List[int]]):
    def __init__(self, batches: List[List[int]]) -> None:
        self.batches = [list(batch) for batch in batches if batch]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def load_focus_stems(focus_pairs_json: str, *, hard_only: bool = False) -> Set[str]:
    if not focus_pairs_json:
        return set()
    with open(focus_pairs_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    pairs = data if isinstance(data, list) else data.get("focus_pairs", [])
    stems: Set[str] = set()
    for row in pairs:
        if not isinstance(row, dict):
            continue
        if hard_only and not bool(row.get("selected_is_high_sam_low_iou", False)):
            continue
        stem = str(row.get("stem", ""))
        if stem:
            stems.add(stem)
    return stems


def load_initial_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(compatible_state, strict=False)
    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = "..." if len(skipped) > 8 else ""
        print(
            f"[warm-start] skipped {len(skipped)} incompatible checkpoint tensors: {preview}{suffix}"
        )


def make_loader(
    dataset: VisualCandidateDataset,
    batch_size: int,
    num_workers: int,
    *,
    grouped_batches: bool = False,
    seed: int = 42,
    focus_stems: Set[str] | None = None,
    focus_repeats: int = 1,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    worker_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = bool(persistent_workers)
        worker_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    if grouped_batches:
        return DataLoader(
            dataset,
            batch_sampler=StaticBatchSampler(
                build_grouped_candidate_batches(
                    dataset,
                    max_batch_size=batch_size,
                    shuffle=True,
                    seed=seed,
                    focus_stems=focus_stems,
                    focus_repeats=focus_repeats,
                )
            ),
            **worker_kwargs,
        )
    weights = []
    focus_stems = set(focus_stems or set())
    for sample in dataset.samples:
        target = float(sample.target)
        focus_weight = 3.0 if str(sample.stem) in focus_stems else 0.0
        weights.append(1.0 + focus_weight + 7.0 * float(target >= 0.5) + 2.0 * target)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **worker_kwargs,
    )


def freeze_backbone(model: ConvNeXtCandidateRanker) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def freeze_visual_branch(model: ConvNeXtCandidateRanker) -> None:
    for module in (model.input_adapter, model.features, model.head):
        for parameter in module.parameters():
            parameter.requires_grad = False


def train_epoch(
    model: ConvNeXtCandidateRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_pointwise = 0.0
    total_pairwise = 0.0
    total_listwise = 0.0
    total = 0
    amp_enabled = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for batch in tqdm(loader, leave=False):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).clamp(0.0, 1.0)
        sam_scores = batch["sam_score"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        source_ids = batch["source_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(inputs, geometry=geometry, source_ids=source_ids)
            losses = visual_candidate_ranking_loss(
                logits,
                targets,
                batch["stem"],
                sam_scores=sam_scores,
                pointwise_weight=args.pointwise_weight,
                pairwise_weight=args.pairwise_weight,
                listwise_weight=args.listwise_weight,
                hard_negative_weight=args.hard_negative_weight,
                margin=args.ranking_margin,
                target_gap=args.target_gap,
            )
            loss = losses["total_loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(inputs.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_pointwise += float(losses["pointwise_loss"].detach().item()) * batch_size
        total_pairwise += float(losses["pairwise_loss"].detach().item()) * batch_size
        total_listwise += float(losses["listwise_loss"].detach().item()) * batch_size
        total += batch_size
    return {
        "loss": total_loss / max(1, total),
        "pointwise_loss": total_pointwise / max(1, total),
        "pairwise_loss": total_pairwise / max(1, total),
        "listwise_loss": total_listwise / max(1, total),
    }


@torch.no_grad()
def evaluate_candidate_selection(
    model: ConvNeXtCandidateRanker,
    input_json: str,
    image_dir: str,
    image_size: int,
    device: str,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    verify_raw_masks: bool = True,
    trust_image_paths: bool = False,
    trusted_image_ext: str = ".png",
) -> Dict[str, Any]:
    with open(input_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    rows = list(data.get("per_image") or [])
    model.eval()
    row_by_stem = {str(row.get("stem", "")): row for row in rows}
    best_by_stem: Dict[str, tuple[float, float]] = {}
    try:
        val_dataset = VisualCandidateDataset(
            input_json=input_json,
            image_dir=image_dir,
            image_size=image_size,
            augment=False,
            source_vocab_size=int(getattr(model, "source_vocab_size", 128)),
            verify_raw_masks=verify_raw_masks,
            trust_image_paths=trust_image_paths,
            trusted_image_ext=trusted_image_ext,
        )
    except ValueError:
        val_dataset = None
    if val_dataset is not None:
        worker_kwargs = {"num_workers": max(0, int(num_workers)), "pin_memory": bool(pin_memory)}
        if int(num_workers) > 0:
            worker_kwargs["persistent_workers"] = bool(persistent_workers)
            worker_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            **worker_kwargs,
        )
        amp_enabled = device.startswith("cuda")
        for batch in tqdm(val_loader, leave=False):
            inputs = batch["input"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            source_ids = batch["source_id"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                scores = (
                    torch.sigmoid(model(inputs, geometry=geometry, source_ids=source_ids))
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            targets = batch["target"].detach().cpu().numpy().tolist()
            stems = list(batch["stem"])
            for stem, score, target in zip(stems, scores, targets):
                score = float(score)
                target = float(target)
                previous = best_by_stem.get(str(stem))
                if previous is None or score > previous[0]:
                    best_by_stem[str(stem)] = (score, target)
    selected_ious: List[float] = []
    oracle_ious: List[float] = []
    current_ious: List[float] = []
    for stem, row in row_by_stem.items():
        selected_ious.append(float(best_by_stem.get(stem, (0.0, 0.0))[1]))
        oracle_ious.append(float(row.get("oracle_best_iou", 0.0)))
        current_ious.append(float(row.get("selected_iou", 0.0)))
    return {
        "count": len(rows),
        "visual_ranker_mean_iou": float(np.mean(selected_ious)) if selected_ious else 0.0,
        "visual_ranker_median_iou": float(np.median(selected_ious)) if selected_ious else 0.0,
        "visual_ranker_success_at_0_5": float(np.mean([value >= 0.5 for value in selected_ious]))
        if selected_ious
        else 0.0,
        "current_agent_c_mean_iou": float(np.mean(current_ious)) if current_ious else 0.0,
        "oracle_mean_iou": float(np.mean(oracle_ious)) if oracle_ious else 0.0,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    train_dataset = VisualCandidateDataset(
        input_json=args.train_json,
        image_dir=args.train_image_dir,
        image_size=args.image_size,
        max_candidates_per_image=args.max_candidates_per_image,
        augment=args.augment_train,
        source_vocab_size=int(args.source_vocab_size),
        verify_raw_masks=not bool(args.trust_replay_mask_paths),
        trust_image_paths=bool(args.trust_replay_image_paths),
        trusted_image_ext=str(args.trusted_image_ext),
    )
    focus_stems = load_focus_stems(args.focus_pairs_json, hard_only=bool(args.focus_hard_only))
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        grouped_batches=bool(args.grouped_batches),
        seed=int(args.seed),
        focus_stems=focus_stems,
        focus_repeats=int(args.focus_repeats),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),
    )
    model = ConvNeXtCandidateRanker(
        image_size=args.image_size,
        use_pretrained=not args.no_pretrained,
        use_geometry_features=bool(args.use_geometry_features),
        use_source_embedding=bool(args.use_source_embedding),
        source_vocab_size=int(args.source_vocab_size),
        source_embedding_dim=int(args.source_embedding_dim),
    ).to(args.device)
    if args.init_checkpoint:
        load_initial_checkpoint(model, args.init_checkpoint)
    if args.freeze_backbone:
        freeze_backbone(model)
    if args.freeze_visual_branch:
        freeze_visual_branch(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: List[Dict[str, Any]] = []
    best_score = -1.0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args)
        val_metrics = evaluate_candidate_selection(
            model,
            args.val_json,
            args.val_image_dir,
            args.image_size,
            args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
            verify_raw_masks=not bool(args.trust_replay_mask_paths),
            trust_image_paths=bool(args.trust_replay_image_paths),
            trusted_image_ext=str(args.trusted_image_ext),
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "val": val_metrics,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        score = float(val_metrics["visual_ranker_mean_iou"])
        if score > best_score:
            best_score = score
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
