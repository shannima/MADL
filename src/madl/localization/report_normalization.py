from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "tampered", "suspicious"}
    return bool(value)


def _coerce_region(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        if "box_2d" in item:
            return dict(item)
        if "box" in item:
            region = dict(item)
            region["box_2d"] = region["box"]
            return region
        return None
    if isinstance(item, (list, tuple)) and len(item) == 4:
        return {"box_2d": list(item)}
    return None


def normalize_evidence_regions(raw_regions: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_regions, dict):
        iterable: Iterable[Any] = [raw_regions]
    elif isinstance(raw_regions, list):
        iterable = raw_regions
    else:
        iterable = []

    regions: List[Dict[str, Any]] = []
    for item in iterable:
        region = _coerce_region(item)
        if region is not None:
            regions.append(region)
    return regions


def normalize_forensic_report(
    report: Dict[str, Any] | None,
    fallback_regions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Convert heterogeneous LMM outputs into the MADL report protocol."""
    record = deepcopy(report or {})
    regions = normalize_evidence_regions(record.get("evidence_regions", []))
    if not regions and fallback_regions:
        regions = normalize_evidence_regions(fallback_regions)
    record["evidence_regions"] = regions

    if _truthy(record.get("is_tampered", False)) or _truthy(record.get("suspicious", False)):
        record["is_tampered"] = True
    else:
        record["is_tampered"] = False

    if _truthy(record.get("is_synthetic", False)):
        record["is_synthetic"] = True
    else:
        raw_label = str(record.get("label", record.get("recommended_label", ""))).strip().lower()
        record["is_synthetic"] = raw_label in {"synthetic", "full_synthetic", "full synthetic"}
    return record
