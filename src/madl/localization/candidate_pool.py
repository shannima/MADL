from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from madl.localization.internal_schema import CandidateRegion
from madl.localization.proposals import (
    build_spatial_sweep_regions,
    merge_candidate_regions,
    regions_from_report,
)

_SUPPORTED_IMAGE_DTYPES = frozenset({np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)})
_V4_HEATMAP_FAMILIES = (
    "multiscale_grid",
    "component",
    "grid",
    "peak",
    "micro",
    "edge",
)
_V4_FILL_ORDER = (
    "layout",
    "multiscale_grid",
    "component",
    "grid",
    "peak",
    "micro",
    "edge",
    "sweep",
)
_V4_EXTRAS_ORDER = (
    "multiscale_grid",
    "component",
    "grid",
    "peak",
    "micro",
    "edge",
    "layout",
    "sweep",
    "weak",
)


def _validated_request(
    image_path: str,
    max_regions: int,
    cached_report: Dict[str, Any] | None,
    include_grid: bool,
    include_layout: bool,
    include_spatial_sweep: bool,
) -> np.ndarray:
    if type(max_regions) is not int or max_regions <= 0:
        raise ValueError("max_regions must be a positive non-bool integer")
    for name, value in (
        ("include_grid", include_grid),
        ("include_layout", include_layout),
        ("include_spatial_sweep", include_spatial_sweep),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")
    if cached_report is not None and not isinstance(cached_report, Mapping):
        raise TypeError("cached_report must be a mapping or None")
    if not isinstance(image_path, (str, bytes, os.PathLike)) or not os.fspath(image_path):
        raise ValueError("image_path must be a non-empty path")
    image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype not in _SUPPORTED_IMAGE_DTYPES
    ):
        raise ValueError("decoded image must be a non-empty numeric BGR array")
    try:
        finite = bool(np.isfinite(image).all())
    except TypeError as error:
        raise ValueError("decoded image must contain finite numeric values") from error
    if not finite:
        raise ValueError("decoded image must contain finite numeric values")
    return image


def _validated_heatmap(heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(heatmap, np.ndarray)
        or heatmap.ndim != 2
        or heatmap.shape != image.shape[:2]
        or heatmap.dtype != np.float32
        or heatmap.size == 0
        or not bool(np.isfinite(heatmap).all())
        or float(np.min(heatmap)) < 0.0
        or float(np.max(heatmap)) > 1.0
    ):
        raise ValueError("anomaly heatmap must be finite float32 in [0, 1] and match image shape")
    return heatmap


def _validated_regions(regions: List[CandidateRegion], max_regions: int) -> List[CandidateRegion]:
    if not isinstance(regions, list) or len(regions) > max_regions:
        raise ValueError("candidate region count exceeds max_regions")
    boxes: set[tuple[int, int, int, int]] = set()
    for index, region in enumerate(regions, start=1):
        if not isinstance(region, CandidateRegion) or region.region_id != f"p{index}":
            raise ValueError("candidate regions must be contiguous p1..pN packets")
        if not isinstance(region.source, str) or not region.source:
            raise ValueError("candidate region source must be non-empty")
        if isinstance(region.confidence, bool):
            raise ValueError("candidate confidence must be finite in [0, 1]")
        try:
            confidence = float(region.confidence)
        except (TypeError, ValueError) as error:
            raise ValueError("candidate confidence must be finite in [0, 1]") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("candidate confidence must be finite in [0, 1]")
        box = region.box_2d
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(value) is not int for value in box)
        ):
            raise ValueError("candidate box must contain four integer permille values")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("candidate box must be bounded and nondegenerate")
        box_key = (x0, y0, x1, y1)
        if box_key in boxes:
            raise ValueError("candidate boxes must be deduplicated")
        boxes.add(box_key)
    return regions


def _validated_result(
    regions: List[CandidateRegion],
    heatmap: np.ndarray,
    image: np.ndarray,
    max_regions: int,
) -> Tuple[List[CandidateRegion], np.ndarray]:
    return _validated_regions(regions, max_regions), _validated_heatmap(heatmap, image)


def build_pixel_candidate_regions(
    image_path: str,
    max_regions: int = 16,
    cached_report: Dict[str, Any] | None = None,
    include_grid: bool = True,
    include_layout: bool = True,
    include_spatial_sweep: bool = True,
) -> Tuple[List[CandidateRegion], np.ndarray]:
    """Build a GT-free, high-recall candidate pool from local forensic cues."""
    image = _validated_request(
        image_path,
        max_regions,
        cached_report,
        include_grid,
        include_layout,
        include_spatial_sweep,
    )

    heatmap = build_local_anomaly_heatmap(image)
    component_regions = _component_regions(heatmap)
    grid_regions: List[CandidateRegion] = []
    if include_grid:
        grid_regions = _grid_regions(heatmap, max_regions=max(4, max_regions))
    layout_regions = _layout_anchor_regions() if include_layout else []
    sweep_regions: List[CandidateRegion] = []
    if include_spatial_sweep:
        sweep_regions = _with_source_prefix(build_spatial_sweep_regions(), "pixel:")
    weak_regions: List[CandidateRegion] = []
    if cached_report:
        weak_regions = regions_from_report(
            cached_report, source="weak_lmm_report", max_regions=0, image_size=_image_size(image)
        )

    merged = _balanced_candidate_regions(
        component_regions=component_regions,
        grid_regions=grid_regions,
        layout_regions=layout_regions,
        sweep_regions=sweep_regions,
        weak_regions=weak_regions,
        max_regions=max_regions,
    )
    return _validated_result(merged, heatmap, image, max_regions)


def build_pixel_candidate_regions_v3(
    image_path: str,
    max_regions: int = 40,
    cached_report: Dict[str, Any] | None = None,
    include_grid: bool = True,
    include_layout: bool = True,
    include_spatial_sweep: bool = True,
) -> Tuple[List[CandidateRegion], np.ndarray]:
    """Build a candidate pool with more budget for tiny and top/edge tampering."""
    image = _validated_request(
        image_path,
        max_regions,
        cached_report,
        include_grid,
        include_layout,
        include_spatial_sweep,
    )

    heatmap = build_local_anomaly_heatmap(image)
    component_regions = _component_regions(heatmap)
    micro_regions = _micro_component_regions(heatmap)
    peak_regions = _peak_regions(heatmap, max_regions=max(12, max_regions // 2))
    edge_micro_regions = _edge_micro_anchor_regions(heatmap)
    grid_regions: List[CandidateRegion] = []
    if include_grid:
        grid_regions = _grid_regions(heatmap, max_regions=max(8, max_regions // 2))
    layout_regions = _layout_anchor_regions() if include_layout else []
    sweep_regions: List[CandidateRegion] = []
    if include_spatial_sweep:
        sweep_regions = _with_source_prefix(build_spatial_sweep_regions(), "pixel:")
    weak_regions: List[CandidateRegion] = []
    if cached_report:
        weak_regions = regions_from_report(
            cached_report, source="weak_lmm_report", max_regions=0, image_size=_image_size(image)
        )

    merged = _balanced_candidate_regions_v3(
        weak_regions=weak_regions,
        peak_regions=peak_regions,
        micro_regions=micro_regions,
        edge_micro_regions=edge_micro_regions,
        component_regions=component_regions,
        grid_regions=grid_regions,
        layout_regions=layout_regions,
        sweep_regions=sweep_regions,
        max_regions=max_regions,
    )
    return _validated_result(merged, heatmap, image, max_regions)


def build_pixel_candidate_regions_v4(
    image_path: str,
    max_regions: int = 40,
    cached_report: Dict[str, Any] | None = None,
    include_grid: bool = True,
    include_layout: bool = True,
    include_spatial_sweep: bool = True,
) -> Tuple[List[CandidateRegion], np.ndarray]:
    """Hybrid pool: preserve layout coverage while adding heatmap-ranked small grids."""
    image = _validated_request(
        image_path,
        max_regions,
        cached_report,
        include_grid,
        include_layout,
        include_spatial_sweep,
    )

    heatmap = build_local_anomaly_heatmap(image)
    component_regions = _component_regions(heatmap)
    micro_regions = _micro_component_regions(heatmap)
    peak_regions = _peak_regions(heatmap, max_regions=max(8, max_regions // 4))
    multiscale_grid_regions: List[CandidateRegion] = []
    if include_grid:
        multiscale_grid_regions = _multiscale_grid_regions(
            heatmap, max_regions=max(12, max_regions // 2)
        )
    edge_micro_regions = _edge_micro_anchor_regions(heatmap)
    grid_regions: List[CandidateRegion] = []
    if include_grid:
        grid_regions = _grid_regions(heatmap, max_regions=max(8, max_regions // 4))
    layout_regions = _layout_anchor_regions() if include_layout else []
    sweep_regions: List[CandidateRegion] = []
    if include_spatial_sweep:
        sweep_regions = _with_source_prefix(build_spatial_sweep_regions(), "pixel:")
    weak_regions: List[CandidateRegion] = []
    if cached_report:
        weak_regions = regions_from_report(
            cached_report, source="weak_lmm_report", max_regions=0, image_size=_image_size(image)
        )

    merged = _balanced_candidate_regions_v4(
        weak_regions=weak_regions,
        layout_regions=layout_regions,
        multiscale_grid_regions=multiscale_grid_regions,
        component_regions=component_regions,
        grid_regions=grid_regions,
        peak_regions=peak_regions,
        micro_regions=micro_regions,
        edge_micro_regions=edge_micro_regions,
        sweep_regions=sweep_regions,
        max_regions=max_regions,
    )
    return _validated_result(merged, heatmap, image, max_regions)


def build_regions_from_model_heatmap(
    image_path: str,
    heatmap: np.ndarray,
    *,
    max_regions: int = 40,
    semantic_report: Dict[str, Any] | None = None,
) -> Tuple[List[CandidateRegion], np.ndarray]:
    """Build the production candidate pool from Agent B's dual-stream heatmap.

    This entry point mirrors the v4 candidate quotas but uses the learned
    dual-stream response supplied by the model instead of recomputing a
    handcrafted anomaly map.
    """
    image = _validated_request(
        image_path,
        max_regions,
        semantic_report,
        True,
        True,
        True,
    )
    learned_heatmap = _validated_heatmap(np.asarray(heatmap, dtype=np.float32), image)
    component_regions = _component_regions(learned_heatmap)
    micro_regions = _micro_component_regions(learned_heatmap)
    peak_regions = _peak_regions(learned_heatmap, max_regions=max(8, max_regions // 4))
    multiscale_grid_regions = _multiscale_grid_regions(
        learned_heatmap,
        max_regions=max(12, max_regions // 2),
    )
    edge_micro_regions = _edge_micro_anchor_regions(learned_heatmap)
    grid_regions = _grid_regions(learned_heatmap, max_regions=max(8, max_regions // 4))
    layout_regions = _layout_anchor_regions()
    sweep_regions = _with_source_prefix(build_spatial_sweep_regions(), "pixel:")
    weak_regions: List[CandidateRegion] = []
    if semantic_report:
        weak_regions = regions_from_report(
            semantic_report,
            source="agent_a_semantic_prior",
            max_regions=0,
            image_size=_image_size(image),
        )
    merged = _balanced_candidate_regions_v4(
        weak_regions=weak_regions,
        layout_regions=layout_regions,
        multiscale_grid_regions=multiscale_grid_regions,
        component_regions=component_regions,
        grid_regions=grid_regions,
        peak_regions=peak_regions,
        micro_regions=micro_regions,
        edge_micro_regions=edge_micro_regions,
        sweep_regions=sweep_regions,
        max_regions=max_regions,
    )
    return _validated_result(merged, learned_heatmap, image, max_regions)


def build_local_anomaly_heatmap(image_bgr: np.ndarray) -> np.ndarray:
    """Combine residual, edge, and color inconsistency cues into a [0, 1] heatmap."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    image_float = image_bgr.astype(np.float32) / 255.0

    blur_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    residual = np.abs(gray - blur_gray)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    blur_color = cv2.GaussianBlur(image_float, (0, 0), sigmaX=5.0)
    color_residual = np.mean(np.abs(image_float - blur_color), axis=2)

    heatmap = (
        0.38 * _robust_normalize(residual)
        + 0.24 * _robust_normalize(laplacian)
        + 0.24 * _robust_normalize(edge)
        + 0.14 * _robust_normalize(color_residual)
    )
    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (5, 5), sigmaX=0)
    return _robust_normalize(heatmap)


def _component_regions(heatmap: np.ndarray) -> List[CandidateRegion]:
    height, width = heatmap.shape
    regions: List[CandidateRegion] = []
    if float(np.max(heatmap)) <= 1e-6:
        return regions

    thresholds = sorted(
        {
            max(0.18, float(np.percentile(heatmap, 88))),
            max(0.24, float(np.percentile(heatmap, 93))),
            max(0.30, float(np.percentile(heatmap, 97))),
        }
    )
    min_area = max(12, int(round(width * height * 0.0003)))
    max_area = int(round(width * height * 0.70))
    kernel = np.ones((5, 5), dtype=np.uint8)
    region_idx = 0

    for threshold in thresholds:
        binary = (heatmap >= threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_idx in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[label_idx]]
            if area < min_area or area > max_area:
                continue
            box = _pad_pixel_box([x, y, x + w, y + h], width, height, pad_ratio=0.035)
            px0, py0, px1, py1 = box
            score_crop = heatmap[py0:py1, px0:px1]
            confidence = _score_to_confidence(float(np.mean(score_crop)), float(np.max(score_crop)))
            region_idx += 1
            regions.append(
                CandidateRegion(
                    region_id=f"component_{region_idx}",
                    box_2d=_pixel_box_to_permille(box, width, height),
                    confidence=confidence,
                    description=f"local high-frequency anomaly component at threshold {threshold:.3f}",
                    evidence_type="localized_edit",
                    evidence_scope="local",
                    source=f"pixel:component:{threshold:.3f}",
                )
            )
    return regions


def _micro_component_regions(heatmap: np.ndarray) -> List[CandidateRegion]:
    height, width = heatmap.shape
    regions: List[CandidateRegion] = []
    if float(np.max(heatmap)) <= 1e-6:
        return regions
    thresholds = sorted(
        {
            max(0.22, float(np.percentile(heatmap, 94))),
            max(0.30, float(np.percentile(heatmap, 97))),
            max(0.38, float(np.percentile(heatmap, 99))),
        }
    )
    min_area = max(3, int(round(width * height * 0.00004)))
    max_area = int(round(width * height * 0.08))
    kernel = np.ones((3, 3), dtype=np.uint8)
    idx = 0
    for threshold in thresholds:
        binary = (heatmap >= threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_idx in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[label_idx]]
            if area < min_area or area > max_area:
                continue
            base = [x, y, x + w, y + h]
            short_side = max(1, min(width, height))
            min_side = max(8, int(round(short_side * 0.055)))
            cx = (base[0] + base[2]) / 2.0
            cy = (base[1] + base[3]) / 2.0
            half_w = max(min_side / 2.0, (base[2] - base[0]) * 0.85)
            half_h = max(min_side / 2.0, (base[3] - base[1]) * 0.85)
            box = [
                int(round(cx - half_w)),
                int(round(cy - half_h)),
                int(round(cx + half_w)),
                int(round(cy + half_h)),
            ]
            box = _pad_pixel_box(box, width, height, pad_ratio=0.02)
            crop = heatmap[box[1] : box[3], box[0] : box[2]]
            confidence = _score_to_confidence(
                float(np.mean(crop)) if crop.size else 0.0,
                float(np.max(crop)) if crop.size else 0.0,
            )
            idx += 1
            regions.append(
                CandidateRegion(
                    region_id=f"micro_{idx}",
                    box_2d=_pixel_box_to_permille(box, width, height),
                    confidence=min(0.88, confidence + 0.04),
                    description=f"tiny high-frequency anomaly component at threshold {threshold:.3f}",
                    evidence_type="localized_edit",
                    evidence_scope="local",
                    source=f"pixel:micro_component:{threshold:.3f}",
                )
            )
    return sorted(regions, key=lambda region: region.confidence, reverse=True)


def _peak_regions(heatmap: np.ndarray, max_regions: int) -> List[CandidateRegion]:
    height, width = heatmap.shape
    if float(np.max(heatmap)) <= 1e-6:
        return []
    dilated = cv2.dilate(heatmap, np.ones((9, 9), dtype=np.uint8))
    peak_mask = (heatmap >= dilated - 1e-6) & (
        heatmap >= max(0.20, float(np.percentile(heatmap, 96)))
    )
    ys, xs = np.where(peak_mask)
    candidates = sorted(
        ((float(heatmap[y, x]), int(x), int(y)) for y, x in zip(ys, xs)), reverse=True
    )
    regions: List[CandidateRegion] = []
    kept_centers: List[tuple[int, int]] = []
    min_distance = max(6, int(round(min(width, height) * 0.06)))
    size_specs = [
        ("tiny", 0.090, 0.090),
        ("small", 0.140, 0.140),
        ("wide", 0.220, 0.120),
        ("tall", 0.120, 0.220),
    ]
    for score, x, y in candidates:
        if any((x - px) ** 2 + (y - py) ** 2 < min_distance**2 for px, py in kept_centers):
            continue
        kept_centers.append((x, y))
        for label, rel_w, rel_h in size_specs:
            box_w = max(8, int(round(width * rel_w)))
            box_h = max(8, int(round(height * rel_h)))
            box = [x - box_w // 2, y - box_h // 2, x + box_w // 2, y + box_h // 2]
            box = _pad_pixel_box(box, width, height, pad_ratio=0.01)
            regions.append(
                CandidateRegion(
                    region_id=f"peak_{len(regions) + 1}",
                    box_2d=_pixel_box_to_permille(box, width, height),
                    confidence=max(0.40, min(0.86, 0.38 + score * 0.46)),
                    description=f"heatmap peak {label} candidate",
                    evidence_type="localized_edit",
                    evidence_scope="local",
                    source=f"pixel:peak:{label}",
                )
            )
            if len(regions) >= max_regions:
                return _dedupe_preserve_order(regions, max_regions=max_regions, iou_threshold=0.88)
    return _dedupe_preserve_order(regions, max_regions=max_regions, iou_threshold=0.88)


def _multiscale_grid_regions(heatmap: np.ndarray, max_regions: int) -> List[CandidateRegion]:
    height, width = heatmap.shape
    if float(np.max(heatmap)) <= 1e-6:
        return []
    windows: List[tuple[float, List[int], str]] = []
    for grid_name, rows, cols in (("grid6", 6, 6), ("grid8", 8, 8), ("grid10", 10, 10)):
        cell_w = max(4, int(round(width / cols)))
        cell_h = max(4, int(round(height / rows)))
        stride_w = max(2, cell_w // 2)
        stride_h = max(2, cell_h // 2)
        for y0 in range(0, max(1, height - cell_h + 1), stride_h):
            for x0 in range(0, max(1, width - cell_w + 1), stride_w):
                x1 = min(width, x0 + cell_w)
                y1 = min(height, y0 + cell_h)
                crop = heatmap[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                high_tail = (
                    float(np.mean(crop[crop >= np.percentile(crop, 85)]))
                    if np.max(crop) > 0
                    else 0.0
                )
                score = 0.50 * float(np.max(crop)) + 0.35 * high_tail + 0.15 * float(np.mean(crop))
                windows.append((score, [x0, y0, x1, y1], grid_name))

    windows.sort(key=lambda item: item[0], reverse=True)
    regions: List[CandidateRegion] = []
    for idx, (score, box, grid_name) in enumerate(windows, start=1):
        regions.append(
            CandidateRegion(
                region_id=f"multiscale_grid_{idx}",
                box_2d=_pixel_box_to_permille(
                    _pad_pixel_box(box, width, height, pad_ratio=0.01), width, height
                ),
                confidence=max(0.08, min(0.78, 0.20 + float(score) * 0.58)),
                description=f"{grid_name} heatmap-ranked local window",
                evidence_type="localized_edit",
                evidence_scope="local",
                source=f"pixel:multiscale_grid:{grid_name}",
            )
        )
        if len(regions) >= max_regions * 3:
            break
    return _dedupe_preserve_order(regions, max_regions=max_regions, iou_threshold=0.75)


def _edge_micro_anchor_regions(heatmap: np.ndarray) -> List[CandidateRegion]:
    height, width = heatmap.shape
    specs = [
        ("top_left_tiny", [0, 0, 180, 180]),
        ("top_center_tiny", [410, 0, 590, 180]),
        ("top_right_tiny", [820, 0, 1000, 180]),
        ("top_left_small", [0, 0, 280, 260]),
        ("top_center_small", [350, 0, 650, 260]),
        ("top_right_small", [720, 0, 1000, 260]),
        ("left_top_small", [0, 80, 220, 320]),
        ("right_top_small", [780, 80, 1000, 320]),
        ("left_edge_tiny", [0, 410, 180, 590]),
        ("right_edge_tiny", [820, 410, 1000, 590]),
        ("bottom_left_tiny", [0, 820, 180, 1000]),
        ("bottom_right_tiny", [820, 820, 1000, 1000]),
    ]
    regions: List[CandidateRegion] = []
    for idx, (name, permille_box) in enumerate(specs, start=1):
        px_box = [
            int(round(permille_box[0] / 1000.0 * width)),
            int(round(permille_box[1] / 1000.0 * height)),
            int(round(permille_box[2] / 1000.0 * width)),
            int(round(permille_box[3] / 1000.0 * height)),
        ]
        crop = heatmap[px_box[1] : px_box[3], px_box[0] : px_box[2]]
        score = float(np.max(crop)) if crop.size else 0.0
        regions.append(
            CandidateRegion(
                region_id=f"edge_micro_{idx}",
                box_2d=permille_box,
                confidence=max(0.34, min(0.76, 0.34 + score * 0.34)),
                description=f"edge-aware tiny anchor {name}",
                evidence_type="localized_edit",
                evidence_scope="local",
                source=f"pixel:edge_micro:{name}",
            )
        )
    return sorted(regions, key=lambda region: region.confidence, reverse=True)


def _grid_regions(heatmap: np.ndarray, max_regions: int) -> List[CandidateRegion]:
    height, width = heatmap.shape
    windows: List[tuple[float, List[int], str]] = []
    for grid_name, rows, cols in (("grid4", 4, 4), ("grid5", 5, 5)):
        cell_w = max(1, int(round(width / cols)))
        cell_h = max(1, int(round(height / rows)))
        stride_w = max(1, cell_w // 2)
        stride_h = max(1, cell_h // 2)
        for y0 in range(0, max(1, height - cell_h + 1), stride_h):
            for x0 in range(0, max(1, width - cell_w + 1), stride_w):
                x1 = min(width, x0 + cell_w)
                y1 = min(height, y0 + cell_h)
                crop = heatmap[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                high_tail = (
                    float(np.mean(crop[crop >= np.percentile(crop, 80)]))
                    if np.max(crop) > 0
                    else 0.0
                )
                score = 0.65 * high_tail + 0.35 * float(np.max(crop))
                windows.append((score, [x0, y0, x1, y1], grid_name))

    windows.sort(key=lambda item: item[0], reverse=True)
    regions: List[CandidateRegion] = []
    for idx, (score, box, grid_name) in enumerate(windows[: max(1, max_regions)], start=1):
        regions.append(
            CandidateRegion(
                region_id=f"grid_{idx}",
                box_2d=_pixel_box_to_permille(
                    _pad_pixel_box(box, width, height, pad_ratio=0.02), width, height
                ),
                confidence=max(0.05, min(0.72, 0.18 + float(score) * 0.54)),
                description=f"{grid_name} local anomaly window",
                evidence_type="localized_edit",
                evidence_scope="local",
                source=f"pixel:grid:{grid_name}",
            )
        )
    return regions


def _layout_anchor_regions() -> List[CandidateRegion]:
    boxes = [
        ("v2_lower_wide", [100, 200, 900, 1000]),
        ("v2_center_large", [200, 300, 700, 800]),
        ("v2_bottom_right_large", [400, 400, 1000, 1000]),
        ("v2_left_lower_tall", [100, 300, 500, 1000]),
        ("v2_center_mid", [300, 400, 700, 800]),
        ("v2_upper_right_tall", [400, 0, 800, 700]),
        ("v2_bottom_wide", [200, 700, 800, 1000]),
        ("v2_center_small", [336, 252, 669, 585]),
        ("v2_left_large", [0, 200, 400, 900]),
        ("full_image", [0, 0, 1000, 1000]),
        ("v2_right_mid", [588, 336, 921, 669]),
        ("v2_horizontal_center_band", [0, 150, 1000, 850]),
        ("v2_right_lower_tall", [500, 300, 800, 900]),
        ("v2_left_center_mid", [168, 252, 501, 585]),
        ("v2_vertical_center_band", [400, 0, 650, 1000]),
        ("v2_bottom_right_quadrant", [600, 600, 1000, 1000]),
        ("v2_upper_right_wide", [500, 100, 900, 500]),
        ("v2_upper_left_wide", [200, 0, 600, 400]),
        ("v2_lower_center_mid", [300, 600, 550, 850]),
        ("v2_center_lower_small", [300, 450, 550, 700]),
        ("v2_left_small", [150, 200, 350, 400]),
        ("v2_bottom_right_small", [600, 675, 850, 925]),
        ("v2_right_tall", [600, 200, 900, 800]),
        ("v2_left_lower_mid", [168, 504, 501, 837]),
        ("v2_lower_large", [200, 500, 900, 900]),
        ("v2_bottom_left_quadrant", [0, 500, 333, 1000]),
        ("v2_center_tiny", [450, 350, 600, 500]),
        ("v2_bottom_left_small", [150, 600, 350, 800]),
        ("v2_bottom_left_mid", [225, 750, 475, 1000]),
        ("v2_mid_horizontal_strip", [0, 300, 1000, 550]),
        ("v2_bottom_center_small", [450, 600, 700, 850]),
        ("v2_left_center_small", [150, 350, 350, 550]),
        ("v2_upper_center_small", [400, 100, 600, 300]),
        ("v2_right_edge_small", [800, 600, 950, 750]),
        ("v2_center_right_small", [500, 350, 700, 550]),
        ("v2_upper_center_mid", [375, 225, 625, 475]),
        ("v2_lower_right_small", [550, 650, 750, 850]),
        ("v2_top_left_wide", [0, 0, 600, 300]),
        ("v2_bottom_center_tall", [450, 750, 700, 1000]),
        ("v2_center_vertical_strip", [375, 0, 525, 1000]),
        ("v2_right_lower_mid", [588, 504, 921, 837]),
    ]
    regions: List[CandidateRegion] = []
    for idx, (name, box) in enumerate(boxes, start=1):
        regions.append(
            CandidateRegion(
                region_id=f"layout_{idx}",
                box_2d=box,
                confidence=max(0.45, 0.74 - idx * 0.006),
                description=f"multi-scale layout anchor {name}",
                evidence_type="localized_edit",
                evidence_scope="local",
                source=f"pixel:layout:{name}",
            )
        )
    return regions


def _balanced_candidate_regions(
    component_regions: List[CandidateRegion],
    grid_regions: List[CandidateRegion],
    layout_regions: List[CandidateRegion],
    sweep_regions: List[CandidateRegion],
    weak_regions: List[CandidateRegion],
    max_regions: int,
) -> List[CandidateRegion]:
    if max_regions <= 0:
        return merge_candidate_regions(
            [*weak_regions, *layout_regions, *component_regions, *grid_regions, *sweep_regions],
            max_regions=0,
            iou_threshold=0.60,
        )

    weak_quota = min(len(weak_regions), 3, max_regions)
    remaining = max_regions - weak_quota
    layout_quota = min(len(layout_regions), max(1, int(round(remaining * 0.80))))
    remaining -= layout_quota
    component_quota = min(len(component_regions), max(0, remaining))
    remaining -= component_quota
    grid_quota = min(len(grid_regions), max(0, remaining))
    remaining -= grid_quota
    sweep_quota = min(len(sweep_regions), max(0, remaining))

    selected: List[CandidateRegion] = []
    selected.extend(
        merge_candidate_regions(weak_regions, max_regions=weak_quota, iou_threshold=0.60)
    )
    selected.extend(layout_regions[:layout_quota])
    selected.extend(
        merge_candidate_regions(component_regions, max_regions=component_quota, iou_threshold=0.60)
    )
    selected.extend(
        merge_candidate_regions(grid_regions, max_regions=grid_quota, iou_threshold=0.60)
    )
    selected.extend(
        merge_candidate_regions(sweep_regions, max_regions=sweep_quota, iou_threshold=0.60)
    )
    return _renumber_regions(
        _dedupe_preserve_order(selected, max_regions=max_regions, iou_threshold=0.90)
    )


def _balanced_candidate_regions_v3(
    *,
    weak_regions: List[CandidateRegion],
    peak_regions: List[CandidateRegion],
    micro_regions: List[CandidateRegion],
    edge_micro_regions: List[CandidateRegion],
    component_regions: List[CandidateRegion],
    grid_regions: List[CandidateRegion],
    layout_regions: List[CandidateRegion],
    sweep_regions: List[CandidateRegion],
    max_regions: int,
) -> List[CandidateRegion]:
    if max_regions <= 0:
        return merge_candidate_regions(
            [
                *weak_regions,
                *peak_regions,
                *micro_regions,
                *edge_micro_regions,
                *component_regions,
                *grid_regions,
                *layout_regions,
                *sweep_regions,
            ],
            max_regions=0,
            iou_threshold=0.60,
        )

    quotas = {
        "weak": min(len(weak_regions), 2, max_regions),
        "peak": min(len(peak_regions), max(4, int(round(max_regions * 0.22)))),
        "micro": min(len(micro_regions), max(3, int(round(max_regions * 0.15)))),
        "edge": min(len(edge_micro_regions), max(4, int(round(max_regions * 0.15)))),
        "component": min(len(component_regions), max(3, int(round(max_regions * 0.12)))),
        "grid": min(len(grid_regions), max(4, int(round(max_regions * 0.12)))),
        "layout": min(len(layout_regions), max(6, int(round(max_regions * 0.20)))),
    }
    used = sum(quotas.values())
    quotas["sweep"] = min(len(sweep_regions), max(0, max_regions - used))

    selected: List[CandidateRegion] = []
    selected.extend(
        merge_candidate_regions(weak_regions, max_regions=quotas["weak"], iou_threshold=0.60)
    )
    selected.extend(
        merge_candidate_regions(peak_regions, max_regions=quotas["peak"], iou_threshold=0.82)
    )
    selected.extend(
        merge_candidate_regions(micro_regions, max_regions=quotas["micro"], iou_threshold=0.72)
    )
    selected.extend(
        merge_candidate_regions(edge_micro_regions, max_regions=quotas["edge"], iou_threshold=0.72)
    )
    selected.extend(
        merge_candidate_regions(
            component_regions, max_regions=quotas["component"], iou_threshold=0.60
        )
    )
    selected.extend(
        merge_candidate_regions(grid_regions, max_regions=quotas["grid"], iou_threshold=0.60)
    )
    selected.extend(layout_regions[: quotas["layout"]])
    selected.extend(
        merge_candidate_regions(sweep_regions, max_regions=quotas["sweep"], iou_threshold=0.60)
    )

    if len(selected) < max_regions:
        extras = [
            *peak_regions,
            *micro_regions,
            *edge_micro_regions,
            *grid_regions,
            *layout_regions,
            *component_regions,
            *sweep_regions,
        ]
        selected.extend(extras)
    return _renumber_regions(
        _dedupe_preserve_order(selected, max_regions=max_regions, iou_threshold=0.90)
    )


def _v4_quota_plan(
    counts: Mapping[str, int],
    *,
    max_regions: int,
) -> Dict[str, int]:
    """Allocate a bounded, label-free hybrid V4 evidence budget."""

    if type(max_regions) is not int or max_regions <= 0:
        raise ValueError("max_regions must be a positive non-bool integer")
    names = ("weak", *_V4_FILL_ORDER[:-1], "sweep")
    if set(counts) != set(names):
        raise ValueError("V4 quota counts must name every evidence family exactly")
    available: Dict[str, int] = {}
    for name in names:
        value = counts[name]
        if type(value) is not int or value < 0:
            raise ValueError("V4 quota counts must be non-negative integers")
        available[name] = value
    quotas = {name: 0 for name in names}

    if max_regions == 1:
        priority = ("layout", *_V4_HEATMAP_FAMILIES, "weak", "sweep")
        chosen = next((name for name in priority if available[name]), None)
        if chosen is not None:
            quotas[chosen] = 1
        return quotas

    remaining = max_regions
    heatmap_choice = next((name for name in _V4_HEATMAP_FAMILIES if available[name]), None)
    if available["layout"] and heatmap_choice is not None:
        layout_target = min(
            available["layout"],
            max(1, int(math.ceil(max_regions * 0.40))),
            max_regions - 1,
        )
        quotas["layout"] = layout_target
        quotas[heatmap_choice] = 1
        remaining -= layout_target + 1
    elif available["layout"]:
        layout_target = min(available["layout"], max_regions)
        quotas["layout"] = layout_target
        remaining -= layout_target
    elif heatmap_choice is not None:
        quotas[heatmap_choice] = 1
        remaining -= 1

    weak_quota = min(available["weak"], 2, remaining)
    quotas["weak"] = weak_quota
    remaining -= weak_quota

    while remaining > 0:
        progressed = False
        for name in _V4_FILL_ORDER:
            if quotas[name] >= available[name]:
                continue
            quotas[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _interleave_region_groups(
    groups: Mapping[str, List[CandidateRegion]],
    order: tuple[str, ...],
) -> List[CandidateRegion]:
    interleaved: List[CandidateRegion] = []
    depth = max((len(groups[name]) for name in order), default=0)
    for index in range(depth):
        for name in order:
            if index < len(groups[name]):
                interleaved.append(groups[name][index])
    return interleaved


def _merge_region_quota(
    regions: List[CandidateRegion],
    quota: int,
    *,
    iou_threshold: float,
) -> List[CandidateRegion]:
    if quota <= 0:
        return []
    return merge_candidate_regions(regions, max_regions=quota, iou_threshold=iou_threshold)


def _balanced_candidate_regions_v4(
    *,
    weak_regions: List[CandidateRegion],
    layout_regions: List[CandidateRegion],
    multiscale_grid_regions: List[CandidateRegion],
    component_regions: List[CandidateRegion],
    grid_regions: List[CandidateRegion],
    peak_regions: List[CandidateRegion],
    micro_regions: List[CandidateRegion],
    edge_micro_regions: List[CandidateRegion],
    sweep_regions: List[CandidateRegion],
    max_regions: int,
) -> List[CandidateRegion]:
    if max_regions <= 0:
        return merge_candidate_regions(
            [
                *weak_regions,
                *layout_regions,
                *multiscale_grid_regions,
                *component_regions,
                *grid_regions,
                *peak_regions,
                *micro_regions,
                *edge_micro_regions,
                *sweep_regions,
            ],
            max_regions=0,
            iou_threshold=0.60,
        )

    groups = {
        "weak": weak_regions,
        "layout": layout_regions,
        "multiscale_grid": multiscale_grid_regions,
        "component": component_regions,
        "grid": grid_regions,
        "peak": peak_regions,
        "micro": micro_regions,
        "edge": edge_micro_regions,
        "sweep": sweep_regions,
    }
    quotas = _v4_quota_plan(
        {name: len(regions) for name, regions in groups.items()},
        max_regions=max_regions,
    )

    selected: List[CandidateRegion] = []
    selected.extend(layout_regions[: quotas["layout"]])
    selected.extend(
        _merge_region_quota(multiscale_grid_regions, quotas["multiscale_grid"], iou_threshold=0.74)
    )
    selected.extend(_merge_region_quota(component_regions, quotas["component"], iou_threshold=0.60))
    selected.extend(_merge_region_quota(grid_regions, quotas["grid"], iou_threshold=0.60))
    selected.extend(_merge_region_quota(peak_regions, quotas["peak"], iou_threshold=0.82))
    selected.extend(_merge_region_quota(micro_regions, quotas["micro"], iou_threshold=0.72))
    selected.extend(_merge_region_quota(edge_micro_regions, quotas["edge"], iou_threshold=0.72))
    selected.extend(_merge_region_quota(weak_regions, quotas["weak"], iou_threshold=0.60))
    selected.extend(_merge_region_quota(sweep_regions, quotas["sweep"], iou_threshold=0.60))

    deduped = _dedupe_preserve_order(selected, max_regions=max_regions, iou_threshold=0.90)
    if len(deduped) < max_regions:
        extras = _interleave_region_groups(groups, _V4_EXTRAS_ORDER)
        deduped = _dedupe_preserve_order(
            [*deduped, *extras],
            max_regions=max_regions,
            iou_threshold=0.90,
        )
    return _renumber_regions(deduped)


def _dedupe_preserve_order(
    regions: List[CandidateRegion], max_regions: int, iou_threshold: float
) -> List[CandidateRegion]:
    kept: List[CandidateRegion] = []
    for region in regions:
        if any(_boxes_overlap(region.box_2d, existing.box_2d, iou_threshold) for existing in kept):
            continue
        kept.append(region)
        if len(kept) >= max_regions:
            break
    return kept


def _boxes_overlap(a: List[int], b: List[int], iou_threshold: float) -> bool:
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
    return union > 0 and (inter / union) >= iou_threshold


def _renumber_regions(regions: List[CandidateRegion]) -> List[CandidateRegion]:
    renumbered: List[CandidateRegion] = []
    for idx, region in enumerate(regions, start=1):
        renumbered.append(
            CandidateRegion(
                region_id=f"p{idx}",
                box_2d=region.box_2d,
                confidence=region.confidence,
                description=region.description,
                evidence_type=region.evidence_type,
                evidence_scope=region.evidence_scope,
                source=region.source,
            )
        )
    return renumbered


def _with_source_prefix(regions: List[CandidateRegion], prefix: str) -> List[CandidateRegion]:
    return [
        CandidateRegion(
            region_id=region.region_id,
            box_2d=region.box_2d,
            confidence=min(0.64, float(region.confidence)),
            description=region.description,
            evidence_type=region.evidence_type,
            evidence_scope=region.evidence_scope,
            source=f"{prefix}{region.source}",
        )
        for region in regions
    ]


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _score_to_confidence(mean_score: float, max_score: float) -> float:
    return max(0.05, min(0.95, 0.30 + 0.35 * mean_score + 0.30 * max_score))


def _pad_pixel_box(box: List[int], width: int, height: int, pad_ratio: float) -> List[int]:
    x0, y0, x1, y1 = box
    pad_x = int(round(max(1, (x1 - x0) * pad_ratio)))
    pad_y = int(round(max(1, (y1 - y0) * pad_ratio)))
    return [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    ]


def _pixel_box_to_permille(box: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box
    return [
        int(round((x0 / max(1, width)) * 1000.0)),
        int(round((y0 / max(1, height)) * 1000.0)),
        int(round((x1 / max(1, width)) * 1000.0)),
        int(round((y1 / max(1, height)) * 1000.0)),
    ]


def _image_size(image_bgr: np.ndarray) -> tuple[int, int]:
    height, width = image_bgr.shape[:2]
    return width, height
