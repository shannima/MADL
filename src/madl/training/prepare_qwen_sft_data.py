from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_SYSTEM_PROMPT = (
    "You are an image-forensics proposal-grounding agent. Judge the image category and provide verifiable evidence. "
    "Output strict JSON only with keys label, is_tampered, is_synthetic, chain_of_thought, and evidence_regions. "
    "The label must be one of real, tampered, or synthetic. "
    "For real images, set is_tampered=false, is_synthetic=false, and evidence_regions=[]. "
    "For tampered images, set is_tampered=true, is_synthetic=false, and provide local evidence_regions. "
    "For synthetic images, set is_tampered=false, is_synthetic=true, and provide a global evidence region. "
    "Each evidence region must contain box_2d, confidence, evidence_scope, evidence_type, and description. "
    "Use box_2d as [x_min, y_min, x_max, y_max] in 0-1000 coordinates. "
    "Natural object edges, shadows, perspective changes, and compression artifacts are not sufficient evidence. "
    "Allowed evidence_scope values are none, local, and global. "
    "Allowed evidence_type values are natural_capture, localized_edit, and whole_image_generation."
)
DEFAULT_USER_PROMPT = (
    "<image>\n"
    "Perform an image-forensics proposal-grounding task and output strict JSON only. "
    "The JSON must contain label, is_tampered, is_synthetic, chain_of_thought, and evidence_regions. "
    "Use label in {real, tampered, synthetic}. "
    "For localized tampering, provide evidence_regions with box_2d in 0-1000 coordinates, "
    "confidence, evidence_scope, evidence_type, and description. "
    "When localized evidence is present, include both a tight candidate and a context candidate when useful."
)


def iter_images(root: str) -> List[str]:
    paths: List[str] = []
    root_path = Path(root)
    for suffix in IMAGE_SUFFIXES:
        paths.extend(str(path) for path in root_path.glob(f"*{suffix}"))
        paths.extend(str(path) for path in root_path.glob(f"*{suffix.upper()}"))
    return sorted(set(paths))


def find_mask_for_image(mask_dir: str, image_path: str) -> str:
    stem = Path(image_path).stem
    candidates = []
    for suffix in IMAGE_SUFFIXES:
        candidates.append(Path(mask_dir) / f"{stem}_mask{suffix}")
        candidates.append(Path(mask_dir) / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _box_to_permille(
    x: int, y: int, width: int, height: int, image_width: int, image_height: int
) -> List[int]:
    return [
        max(0, min(1000, int(round((x / max(1, image_width)) * 1000.0)))),
        max(0, min(1000, int(round((y / max(1, image_height)) * 1000.0)))),
        max(0, min(1000, int(round(((x + width) / max(1, image_width)) * 1000.0)))),
        max(0, min(1000, int(round(((y + height) / max(1, image_height)) * 1000.0)))),
    ]


def coerce_box_permille(box: List[int]) -> List[int]:
    x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    x0 = max(0, min(1000, x0))
    y0 = max(0, min(1000, y0))
    x1 = max(x0 + 1, min(1000, x1))
    y1 = max(y0 + 1, min(1000, y1))
    return [x0, y0, x1, y1]


def box_area_permille(box: List[int]) -> float:
    x0, y0, x1, y1 = coerce_box_permille(box)
    return float(max(0, x1 - x0) * max(0, y1 - y0) / 1_000_000.0)


def box_iou_permille(a: List[int], b: List[int]) -> float:
    ax0, ay0, ax1, ay1 = coerce_box_permille(a)
    bx0, by0, bx1, by1 = coerce_box_permille(b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def expand_box_permille(box: List[int], scale: float) -> List[int]:
    x0, y0, x1, y1 = [float(value) for value in coerce_box_permille(box)]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    width = max(1.0, (x1 - x0) * float(scale))
    height = max(1.0, (y1 - y0) * float(scale))
    return coerce_box_permille(
        [
            int(round(cx - width / 2.0)),
            int(round(cy - height / 2.0)),
            int(round(cx + width / 2.0)),
            int(round(cy + height / 2.0)),
        ]
    )


def dedupe_boxes(boxes: List[List[int]], iou_threshold: float = 0.90) -> List[List[int]]:
    deduped: List[List[int]] = []
    for box in boxes:
        normalized = coerce_box_permille(box)
        if any(box_iou_permille(normalized, kept) >= iou_threshold for kept in deduped):
            continue
        deduped.append(normalized)
    return deduped


def proposal_boxes_with_context(boxes: List[List[int]], max_regions: int = 5) -> List[dict]:
    base_boxes = dedupe_boxes([coerce_box_permille(box) for box in boxes])
    proposal_boxes: List[tuple[List[int], float, str]] = []
    for idx, box in enumerate(base_boxes):
        description = (
            "primary localized edit candidate" if idx == 0 else "secondary localized edit candidate"
        )
        confidence = 0.94 if idx == 0 else 0.88
        proposal_boxes.append((box, confidence, description))

    for box in base_boxes[:2]:
        area = box_area_permille(box)
        scale = 2.20 if area < 0.01 else 1.65 if area < 0.04 else 1.25
        context_box = expand_box_permille(box, scale)
        proposal_boxes.append((context_box, 0.76, "context window around localized edit candidate"))

    regions: List[dict] = []
    seen: List[List[int]] = []
    for box, confidence, description in proposal_boxes:
        if any(box_iou_permille(box, kept) >= 0.90 for kept in seen):
            continue
        seen.append(box)
        regions.append(
            {
                "description": description,
                "box_2d": box,
                "confidence": confidence,
                "evidence_scope": "local",
                "evidence_type": "localized_edit",
            }
        )
        if max_regions and len(regions) >= max_regions:
            break
    return regions


def extract_component_boxes_permille(
    mask_path: str,
    max_boxes: int = 3,
    min_area_ratio: float = 0.0005,
) -> List[List[int]]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    binary = (mask > 127).astype(np.uint8)
    image_height, image_width = binary.shape[:2]
    image_area = max(1, image_height * image_width)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: List[List[int]] = []
    for label_id in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label_id]]
        if area / image_area < min_area_ratio:
            continue
        box = _box_to_permille(x, y, width, height, image_width, image_height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append(box)
        if max_boxes and len(boxes) >= max_boxes:
            break
    return boxes


def extract_union_box_permille(mask_path: str) -> List[int]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    binary = (mask > 127).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return []
    image_height, image_width = binary.shape[:2]
    x, y, width, height = cv2.boundingRect(coords)
    return _box_to_permille(x, y, width, height, image_width, image_height)


def _record(image_path: str, response: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": DEFAULT_USER_PROMPT},
            {
                "role": "assistant",
                "content": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": [image_path],
    }


def make_real_record(image_path: str) -> dict:
    return _record(
        image_path,
        {
            "label": "real",
            "is_tampered": False,
            "is_synthetic": False,
            "chain_of_thought": "No supported localized manipulation or whole-image generation cues are present; natural image artifacts are treated as non-evidence.",
            "evidence_regions": [],
        },
    )


def make_tampered_record(image_path: str, boxes: List[List[int]]) -> dict:
    evidence_regions = proposal_boxes_with_context(boxes)
    return _record(
        image_path,
        {
            "label": "tampered",
            "is_tampered": True,
            "is_synthetic": False,
            "chain_of_thought": "Localized forensic evidence is concentrated in the proposed region; downstream pixel agents should verify the boundary and segmentation.",
            "evidence_regions": evidence_regions,
        },
    )


def make_synthetic_record(image_path: str) -> dict:
    return _record(
        image_path,
        {
            "label": "synthetic",
            "is_tampered": False,
            "is_synthetic": True,
            "chain_of_thought": "Whole-image generation cues are treated as global evidence rather than a localized edit.",
            "evidence_regions": [
                {
                    "description": "whole-image generation region",
                    "box_2d": [0, 0, 1000, 1000],
                    "confidence": 0.95,
                    "evidence_scope": "global",
                    "evidence_type": "whole_image_generation",
                }
            ],
        },
    )


def sample_items(items: List[str], limit: int, rng: random.Random) -> List[str]:
    if limit <= 0 or limit >= len(items):
        selected = list(items)
        rng.shuffle(selected)
        return selected
    return rng.sample(items, limit)


def build_dataset(
    real_dir: str,
    tampered_dir: str,
    mask_dir: str,
    synthetic_dirs: Iterable[str],
    output_path: str,
    n_per_class: int = 10000,
    seed: int = 42,
    max_boxes: int = 3,
    include_union_box: bool = True,
    min_area_ratio: float = 0.0005,
) -> dict:
    rng = random.Random(seed)
    records: List[dict] = []
    stats = {
        "real": 0,
        "tampered": 0,
        "synthetic": 0,
        "tampered_skipped_no_mask": 0,
        "tampered_skipped_empty_mask": 0,
    }

    for image_path in tqdm(sample_items(iter_images(real_dir), n_per_class, rng), desc="real"):
        records.append(make_real_record(image_path))
        stats["real"] += 1

    tampered_images = sample_items(iter_images(tampered_dir), 0, rng)
    for image_path in tqdm(tampered_images, desc="tampered"):
        if stats["tampered"] >= n_per_class > 0:
            break
        mask_path = find_mask_for_image(mask_dir, image_path)
        if not mask_path:
            stats["tampered_skipped_no_mask"] += 1
            continue
        boxes = extract_component_boxes_permille(
            mask_path,
            max_boxes=max_boxes,
            min_area_ratio=min_area_ratio,
        )
        if include_union_box:
            union_box = extract_union_box_permille(mask_path)
            if union_box and union_box not in boxes:
                boxes.insert(0, union_box)
                boxes = boxes[:max_boxes]
        if not boxes:
            stats["tampered_skipped_empty_mask"] += 1
            continue
        records.append(make_tampered_record(image_path, boxes))
        stats["tampered"] += 1

    synthetic_images: List[str] = []
    for synthetic_dir in synthetic_dirs:
        synthetic_images.extend(iter_images(str(synthetic_dir)))
    for image_path in tqdm(sample_items(synthetic_images, n_per_class, rng), desc="synthetic"):
        records.append(make_synthetic_record(image_path))
        stats["synthetic"] += 1

    rng.shuffle(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["total"] = len(records)
    stats["output_path"] = str(output)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen proposal-grounding SFT data from SID masks."
    )
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--tampered_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument(
        "--synthetic_dirs",
        nargs="+",
        required=True,
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_per_class", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_boxes", type=int, default=3)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--no_union_box", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        real_dir=args.real_dir,
        tampered_dir=args.tampered_dir,
        mask_dir=args.mask_dir,
        synthetic_dirs=args.synthetic_dirs,
        output_path=args.output_path,
        n_per_class=args.n_per_class,
        seed=args.seed,
        max_boxes=args.max_boxes,
        include_union_box=not args.no_union_box,
        min_area_ratio=args.min_area_ratio,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
