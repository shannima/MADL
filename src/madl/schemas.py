"""Public evidence contracts exchanged by the three MADL agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence


class ClassLabel(str, Enum):
    """The unified three-way forensic label space."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    TAMPERED = "tampered"


class DecisionState(str, Enum):
    """Explicit states used by Agent C to reconcile heterogeneous evidence."""

    AGREEMENT = "agreement"
    SUPPRESSION = "suppression"
    OVERRIDE = "override"
    CONFLICT = "conflict"


def _label(value: ClassLabel | str) -> ClassLabel:
    return value if isinstance(value, ClassLabel) else ClassLabel(str(value).lower())


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return value


@dataclass(frozen=True)
class AgentAEvidence:
    """Semantic, classification, and weak spatial-prior evidence from Agent A."""

    label: ClassLabel | str
    scores: Mapping[ClassLabel | str, float]
    semantic_summary: str = ""
    spatial_priors: tuple[tuple[float, float, float, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_label = _label(self.label)
        raw = {_label(key): float(value) for key, value in self.scores.items()}
        complete = {label: max(0.0, raw.get(label, 0.0)) for label in ClassLabel}
        total = sum(complete.values())
        if not isfinite(total) or total <= 0.0:
            raise ValueError("Agent A scores must contain positive finite mass")
        normalized = {label: value / total for label, value in complete.items()}
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "scores", normalized)
        object.__setattr__(self, "spatial_priors", tuple(tuple(box) for box in self.spatial_priors))

    @property
    def confidence(self) -> float:
        return float(self.scores[self.label])


@dataclass(frozen=True)
class AgentBEvidence:
    """Pixel-level manipulation evidence and the selected localization mask."""

    tampered_probability: float
    selected_mask: Any | None
    mask_score: float
    candidate_count: int
    heatmap: Any | None = None
    candidate_scores: Sequence[float] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probability = _probability(self.tampered_probability, "tampered_probability")
        score = _probability(self.mask_score, "mask_score")
        count = int(self.candidate_count)
        if count < 0:
            raise ValueError("candidate_count cannot be negative")
        if probability >= 0.5 and score > 0.0 and self.selected_mask is None:
            raise ValueError("positive pixel evidence must include its selected mask")
        object.__setattr__(self, "tampered_probability", probability)
        object.__setattr__(self, "mask_score", score)
        object.__setattr__(self, "candidate_count", count)
        object.__setattr__(self, "candidate_scores", tuple(float(v) for v in self.candidate_scores))


@dataclass(frozen=True)
class MADLResult:
    """Final three-way label, conditional mask, and auditable decision trace."""

    label: ClassLabel | str
    confidence: float
    mask: Any | None
    decision_state: DecisionState | str
    trace: tuple[str, ...]
    agent_a: AgentAEvidence | None = None
    agent_b: AgentBEvidence | None = None
    explanation: str = ""

    def __post_init__(self) -> None:
        label = _label(self.label)
        state = self.decision_state
        if not isinstance(state, DecisionState):
            state = DecisionState(str(state).lower())
        confidence = _probability(self.confidence, "confidence")
        if label is ClassLabel.TAMPERED and self.mask is None:
            raise ValueError("a tampered MADL result must retain a localization mask")
        if label is not ClassLabel.TAMPERED and self.mask is not None:
            raise ValueError("non-tampered MADL results cannot expose a localization mask")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "decision_state", state)
        object.__setattr__(self, "trace", tuple(self.trace))
