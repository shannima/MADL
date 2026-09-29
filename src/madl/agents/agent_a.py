"""Agent A: Qwen-based semantic verification and calibrated classification evidence."""

from __future__ import annotations

from typing import Any, Mapping

from madl.schemas import AgentAEvidence, ClassLabel


def _coerce_label(report: Mapping[str, Any]) -> ClassLabel:
    raw = str(report.get("label", report.get("recommended_label", ""))).strip().lower()
    aliases = {
        "authentic": ClassLabel.REAL,
        "real": ClassLabel.REAL,
        "synthetic": ClassLabel.SYNTHETIC,
        "fully_synthetic": ClassLabel.SYNTHETIC,
        "full_synthetic": ClassLabel.SYNTHETIC,
        "tampered": ClassLabel.TAMPERED,
        "manipulated": ClassLabel.TAMPERED,
        "locally_tampered": ClassLabel.TAMPERED,
    }
    if raw in aliases:
        return aliases[raw]
    if bool(report.get("is_tampered", False)):
        return ClassLabel.TAMPERED
    if bool(report.get("is_synthetic", False)):
        return ClassLabel.SYNTHETIC
    return ClassLabel.REAL


def _coerce_scores(report: Mapping[str, Any], label: ClassLabel) -> dict[ClassLabel, float]:
    raw = report.get("class_probs", report.get("scores", {}))
    if isinstance(raw, Mapping) and raw:
        return {member: float(raw.get(member.value, raw.get(member, 0.0))) for member in ClassLabel}
    confidence = float(report.get("confidence", report.get("label_confidence", 0.85)))
    confidence = min(1.0, max(1.0 / 3.0, confidence))
    remainder = (1.0 - confidence) / 2.0
    return {member: confidence if member is label else remainder for member in ClassLabel}


def _coerce_priors(report: Mapping[str, Any]) -> tuple[tuple[float, float, float, float], ...]:
    priors = []
    for region in report.get("evidence_regions", ()) or ():
        if not isinstance(region, Mapping):
            continue
        box = region.get("box_2d", region.get("box"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        values = tuple(float(value) for value in box)
        scale = 1000.0 if max(abs(value) for value in values) > 1.0 else 1.0
        normalized = tuple(min(1.0, max(0.0, value / scale)) for value in values)
        x0, y0, x1, y1 = normalized
        if x1 > x0 and y1 > y0:
            priors.append(normalized)
    return tuple(priors)


class SemanticVerificationAgent:
    """Normalize a Qwen-style backend into the public Agent A evidence contract."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def _run_backend(self, image: Any) -> Mapping[str, Any]:
        if hasattr(self.backend, "analyze"):
            return self.backend.analyze(image)
        if hasattr(self.backend, "analyze_image"):
            return self.backend.analyze_image(str(image))
        raise TypeError("Agent A backend must implement analyze() or analyze_image()")

    def analyze(self, image: Any) -> AgentAEvidence:
        report = dict(self._run_backend(image) or {})
        label = _coerce_label(report)
        return AgentAEvidence(
            label=label,
            scores=_coerce_scores(report, label),
            semantic_summary=str(
                report.get(
                    "explanation", report.get("chain_of_thought", report.get("reasoning", ""))
                )
            ),
            spatial_priors=_coerce_priors(report),
            metadata={"raw_report": report},
        )

    def review(
        self,
        image: Any,
        initial: AgentAEvidence,
        pixel_evidence: Any,
    ) -> AgentAEvidence:
        reviewer = getattr(self.backend, "review_candidates", None)
        if not callable(reviewer):
            return initial
        report = reviewer(image, initial.metadata.get("raw_report", {}), pixel_evidence)
        if not report:
            return initial
        label = _coerce_label(report)
        return AgentAEvidence(
            label=label,
            scores=_coerce_scores(report, label),
            semantic_summary=str(
                report.get("explanation", report.get("reasoning", initial.semantic_summary))
            ),
            spatial_priors=initial.spatial_priors,
            metadata={"initial": initial.metadata, "review_report": dict(report)},
        )
