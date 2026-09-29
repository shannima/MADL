from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from madl.localization.internal_schema import CandidateRegion
from madl.localization.report_normalization import normalize_evidence_regions


class CachedProposalBank:
    """Loads cached Agent-A proposal dossiers and exposes a stable proposal bank."""

    def __init__(self, cache_dist_dir: str) -> None:
        self.cache_dist_dir = str(cache_dist_dir)

    def dossier_path(self, stem: str) -> str:
        return os.path.join(self.cache_dist_dir, f"{stem}_dossier.json")

    def available_stems(self, limit: int = 0) -> List[str]:
        paths = sorted(glob.glob(os.path.join(self.cache_dist_dir, "*_dossier.json")))
        stems = [Path(path).name.replace("_dossier.json", "") for path in paths]
        if limit and limit > 0:
            return stems[:limit]
        return stems

    def load_report(self, stem: str) -> Dict[str, Any]:
        path = self.dossier_path(stem)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cached proposal dossier not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def load_regions(self, stem: str, max_regions: int = 6) -> List[CandidateRegion]:
        report = self.load_report(stem)
        return regions_from_report(report, source=f"cached:{stem}", max_regions=max_regions)


def _coerce_box(box: Any) -> Optional[List[int]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    x0 = max(0, min(1000, x0))
    y0 = max(0, min(1000, y0))
    x1 = max(x0 + 1, min(1000, x1))
    y1 = max(y0 + 1, min(1000, y1))
    return [x0, y0, x1, y1]


def box_iou_permille(a: List[int], b: List[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _remap_crop_box_to_full_permille(
    box: List[int],
    crop_bounds: Any,
    image_size: tuple[int, int] | None,
) -> List[int]:
    if image_size is None:
        return box
    if not isinstance(crop_bounds, (list, tuple)) or len(crop_bounds) != 4:
        return box
    try:
        crop_x0, crop_y0, crop_x1, crop_y1 = [float(value) for value in crop_bounds]
        image_width, image_height = [float(value) for value in image_size]
    except (TypeError, ValueError):
        return box
    if image_width <= 0 or image_height <= 0 or crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return box

    x0, y0, x1, y1 = [float(value) for value in box]
    crop_width = crop_x1 - crop_x0
    crop_height = crop_y1 - crop_y0
    full_x0 = crop_x0 + (x0 / 1000.0) * crop_width
    full_y0 = crop_y0 + (y0 / 1000.0) * crop_height
    full_x1 = crop_x0 + (x1 / 1000.0) * crop_width
    full_y1 = crop_y0 + (y1 / 1000.0) * crop_height
    remapped = [
        int(round((full_x0 / image_width) * 1000.0)),
        int(round((full_y0 / image_height) * 1000.0)),
        int(round((full_x1 / image_width) * 1000.0)),
        int(round((full_y1 / image_height) * 1000.0)),
    ]
    return _coerce_box(remapped) or box


def regions_from_report(
    report: Dict[str, Any],
    source: str,
    max_regions: int = 6,
    image_size: tuple[int, int] | None = None,
) -> List[CandidateRegion]:
    raw_regions = normalize_evidence_regions(report.get("evidence_regions", []))
    if not raw_regions:
        for key, value in report.items():
            if ("evidence" in key or "region" in key) and isinstance(value, list):
                if any(isinstance(item, dict) and "box_2d" in item for item in value):
                    raw_regions = normalize_evidence_regions(value)
                    break

    regions: List[CandidateRegion] = []
    for idx, item in enumerate(raw_regions):
        if not isinstance(item, dict):
            continue
        box = _coerce_box(item.get("box_2d"))
        if box is None:
            continue
        box = _remap_crop_box_to_full_permille(
            box, report.get("_multiscale_crop_bounds"), image_size
        )
        confidence = item.get("confidence", item.get("score", 0.75))
        try:
            confidence_f = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_f = 0.75
        regions.append(
            CandidateRegion(
                region_id=f"p{idx + 1}",
                box_2d=box,
                confidence=confidence_f,
                description=str(item.get("description", "suspicious_region")),
                evidence_type=str(item.get("evidence_type", "localized_edit")),
                evidence_scope=str(item.get("evidence_scope", "local")),
                source=str(item.get("source", source)),
            )
        )
        if max_regions and len(regions) >= max_regions:
            break
    return regions


def merge_candidate_regions(
    regions: Iterable[CandidateRegion],
    max_regions: int = 6,
    iou_threshold: float = 0.65,
) -> List[CandidateRegion]:
    """Keep a compact top-k proposal set while removing near-duplicate boxes."""
    ranked = sorted(
        list(regions),
        key=lambda region: (float(region.confidence), -_region_area(region.box_2d)),
        reverse=True,
    )
    merged: List[CandidateRegion] = []
    for region in ranked:
        if any(box_iou_permille(region.box_2d, kept.box_2d) >= iou_threshold for kept in merged):
            continue
        merged.append(
            CandidateRegion(
                region_id=f"p{len(merged) + 1}",
                box_2d=region.box_2d,
                confidence=region.confidence,
                description=region.description,
                evidence_type=region.evidence_type,
                evidence_scope=region.evidence_scope,
                source=region.source,
            )
        )
        if max_regions and len(merged) >= max_regions:
            break
    return merged


def build_uncertainty_regions(
    regions: Iterable[CandidateRegion],
    scales: Iterable[float] = (1.5, 2.0, 3.0),
    include_union: bool = True,
) -> List[CandidateRegion]:
    """Augment live LMM boxes with scale and union proposals without using GT."""
    base_regions = list(regions)
    refined: List[CandidateRegion] = list(base_regions)
    for region in base_regions:
        for scale in scales:
            box = _scale_box_permille(region.box_2d, float(scale))
            refined.append(
                CandidateRegion(
                    region_id=f"{region.region_id}_s{float(scale):.2f}",
                    box_2d=box,
                    confidence=max(
                        0.05, float(region.confidence) * _scale_confidence(float(scale))
                    ),
                    description=f"{region.description} uncertainty scale {float(scale):.2f}",
                    evidence_type=region.evidence_type,
                    evidence_scope=region.evidence_scope,
                    source=f"uncertainty:scale_{float(scale):.2f}:{region.source}",
                )
            )

    if include_union and len(base_regions) >= 2:
        union_box = _union_boxes_permille([region.box_2d for region in base_regions])
        if union_box is not None:
            refined.append(
                CandidateRegion(
                    region_id="p_union",
                    box_2d=union_box,
                    confidence=max(
                        (float(region.confidence) for region in base_regions), default=0.75
                    )
                    * 0.86,
                    description="union of live LMM proposals",
                    evidence_type="localized_edit",
                    evidence_scope="local",
                    source="uncertainty:union",
                )
            )

    return refined


def build_spatial_sweep_regions() -> List[CandidateRegion]:
    """Add GT-free edge sweep proposals for small border-adjacent edits."""
    boxes = [
        ("bottom_edge", [0, 930, 300, 1000]),
        ("bottom_edge", [250, 930, 550, 1000]),
        ("bottom_edge", [500, 930, 800, 1000]),
        ("bottom_edge", [700, 930, 1000, 1000]),
        ("bottom_edge", [550, 900, 850, 1000]),
        ("left_bottom", [0, 500, 320, 980]),
        ("left_bottom", [0, 600, 320, 1000]),
        ("left_bottom", [0, 250, 300, 1000]),
    ]
    regions: List[CandidateRegion] = []
    for idx, (sweep_name, box) in enumerate(boxes, start=1):
        regions.append(
            CandidateRegion(
                region_id=f"edge_{idx}",
                box_2d=box,
                confidence=0.68 if sweep_name == "bottom_edge" else 0.67,
                description=f"{sweep_name.replace('_', '-')} forensic sweep",
                evidence_type="localized_edit",
                evidence_scope="local",
                source=f"spatial_sweep:{sweep_name}:{idx}",
            )
        )
    return regions


def _scale_box_permille(box: List[int], scale: float) -> List[int]:
    x0, y0, x1, y1 = [float(value) for value in box]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    width = max(1.0, (x1 - x0) * scale)
    height = max(1.0, (y1 - y0) * scale)
    scaled = [
        int(round(max(0.0, cx - width / 2.0))),
        int(round(max(0.0, cy - height / 2.0))),
        int(round(min(1000.0, cx + width / 2.0))),
        int(round(min(1000.0, cy + height / 2.0))),
    ]
    return _coerce_box(scaled) or box


def _scale_confidence(scale: float) -> float:
    if scale <= 1.5:
        return 0.96
    if scale <= 2.0:
        return 0.92
    return 0.86


def _union_boxes_permille(boxes: Iterable[List[int]]) -> Optional[List[int]]:
    valid_boxes = [box for box in boxes if _coerce_box(box) is not None]
    if not valid_boxes:
        return None
    union_box = [
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    ]
    return _coerce_box(union_box)


def _region_area(box: List[int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def stem_from_image_path(image_path: str) -> str:
    return Path(image_path).stem
