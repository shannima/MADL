"""Agent B: pixel-forensic evidence extraction, candidate masks, and learned ranking."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from madl.schemas import AgentBEvidence


def _to_permille(box: Sequence[float]) -> list[int] | None:
    if len(box) != 4:
        return None
    values = [float(value) for value in box]
    scale = 1000.0 if max(abs(value) for value in values) > 1.0 else 1.0
    normalized = [min(1.0, max(0.0, value / scale)) for value in values]
    x0, y0, x1, y1 = normalized
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(round(value * 1000.0)) for value in normalized]


def _merged_regions(
    spatial_priors: Sequence[Sequence[float]], report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index, box in enumerate(spatial_priors):
        converted = _to_permille(box)
        if converted is not None and tuple(converted) not in seen:
            seen.add(tuple(converted))
            regions.append({"box_2d": converted, "source": f"agent_a:{index}"})
    for index, region in enumerate(report.get("suspicious_regions", ()) or ()):
        if not isinstance(region, Mapping):
            continue
        box = region.get("box_permille", region.get("box_2d", region.get("box")))
        if not isinstance(box, (list, tuple)):
            continue
        converted = _to_permille(box)
        if converted is not None and tuple(converted) not in seen:
            seen.add(tuple(converted))
            regions.append({"box_2d": converted, "source": f"dualstream:{index}"})
    return regions


class PixelForensicsAgent:
    """Coordinate the dual-stream model, SAM candidate generation, and visual ranker."""

    def __init__(
        self, pixel_runtime: Any, segmenter: Any, ranker: Any, candidate_builder: Any | None = None
    ) -> None:
        self.pixel_runtime = pixel_runtime
        self.segmenter = segmenter
        self.ranker = ranker
        self.candidate_builder = candidate_builder

    def analyze(self, image: Any, spatial_priors: tuple = ()) -> AgentBEvidence:
        report = dict(self.pixel_runtime.predict(image) or {})
        heatmap = report.get("anomaly_heatmap")
        if self.candidate_builder is None:
            regions = _merged_regions(spatial_priors, report)
        else:
            regions = list(
                self.candidate_builder.build(
                    image,
                    heatmap,
                    spatial_priors=spatial_priors,
                    pixel_report=report,
                )
            )
        candidates = list(self.segmenter.segment(image, regions, heatmap=heatmap) or ())
        selected = self.ranker.select(image, heatmap, candidates) if candidates else None
        class_probs = report.get("class_probs", {})
        tampered_probability = float(
            class_probs.get("tampered", report.get("low_level_confidence", 0.0))
        )
        if selected is None:
            return AgentBEvidence(
                tampered_probability=tampered_probability,
                selected_mask=None,
                mask_score=0.0,
                candidate_count=len(candidates),
                heatmap=heatmap,
                candidate_scores=tuple(float(item.get("score", 0.0)) for item in candidates),
                metadata={"pixel_report": report, "regions": regions},
            )
        mask = selected.get("mask", selected.get("selected_mask"))
        score = float(selected.get("visual_ranker_score", selected.get("score", 0.0)))
        return AgentBEvidence(
            tampered_probability=tampered_probability,
            selected_mask=mask,
            mask_score=score,
            candidate_count=len(candidates),
            heatmap=heatmap,
            candidate_scores=tuple(float(item.get("score", 0.0)) for item in candidates),
            metadata={"pixel_report": report, "regions": regions, "selected": selected},
        )
