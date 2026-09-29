"""Conflict-aware adjudication for semantic and pixel-level evidence."""

from __future__ import annotations

from dataclasses import dataclass

from madl.schemas import (
    AgentAEvidence,
    AgentBEvidence,
    ClassLabel,
    DecisionState,
    MADLResult,
)


@dataclass(frozen=True)
class AdjudicationConfig:
    """Frozen thresholds used by Agent C."""

    local_evidence_threshold: float = 0.60
    strong_local_evidence_threshold: float = 0.80
    semantic_override_threshold: float = 0.85

    def __post_init__(self) -> None:
        values = (
            self.local_evidence_threshold,
            self.strong_local_evidence_threshold,
            self.semantic_override_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("all Agent C thresholds must lie in [0, 1]")
        if self.strong_local_evidence_threshold < self.local_evidence_threshold:
            raise ValueError("strong local threshold must not be below the local threshold")


class ConflictAwareAdjudicator:
    """Agent C: produce a traceable decision from Agent A and Agent B evidence."""

    def __init__(self, config: AdjudicationConfig | None = None) -> None:
        self.config = config or AdjudicationConfig()

    def decide(self, semantic: AgentAEvidence, pixel: AgentBEvidence) -> MADLResult:
        local_score = min(pixel.tampered_probability, pixel.mask_score)
        valid_mask = (
            pixel.selected_mask is not None and local_score >= self.config.local_evidence_threshold
        )
        strong_mask = valid_mask and local_score >= self.config.strong_local_evidence_threshold

        if semantic.label is ClassLabel.TAMPERED and strong_mask:
            state = DecisionState.AGREEMENT
            label = ClassLabel.TAMPERED
            mask = pixel.selected_mask
            confidence = min(semantic.confidence, local_score)
        elif (
            semantic.label is ClassLabel.TAMPERED
            and valid_mask
            and semantic.confidence >= self.config.semantic_override_threshold
        ):
            state = DecisionState.OVERRIDE
            label = ClassLabel.TAMPERED
            mask = pixel.selected_mask
            confidence = min(
                semantic.confidence, max(local_score, self.config.local_evidence_threshold)
            )
        elif semantic.label is not ClassLabel.TAMPERED and strong_mask:
            state = DecisionState.CONFLICT
            label = semantic.label
            mask = None
            confidence = semantic.confidence
        elif semantic.label is not ClassLabel.TAMPERED:
            state = (
                DecisionState.SUPPRESSION
                if pixel.tampered_probability > 0.0
                else DecisionState.AGREEMENT
            )
            label = semantic.label
            mask = None
            confidence = semantic.confidence
        else:
            # A predicts tampering but B cannot provide the spatial support required
            # for a local-tampering result. Preserve the best non-local semantic class.
            state = DecisionState.CONFLICT
            alternatives = {
                ClassLabel.REAL: semantic.scores[ClassLabel.REAL],
                ClassLabel.SYNTHETIC: semantic.scores[ClassLabel.SYNTHETIC],
            }
            label = max(alternatives, key=alternatives.get)
            mask = None
            confidence = alternatives[label]

        trace = (
            f"agent_a:{semantic.label.value}:{semantic.confidence:.4f}",
            f"agent_b:tampered:{pixel.tampered_probability:.4f}:mask:{pixel.mask_score:.4f}",
            f"agent_c:{state.value}:{label.value}",
        )
        return MADLResult(
            label=label,
            confidence=confidence,
            mask=mask,
            decision_state=state,
            trace=trace,
            agent_a=semantic,
            agent_b=pixel,
            explanation=semantic.semantic_summary,
        )
