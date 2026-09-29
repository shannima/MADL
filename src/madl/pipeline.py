"""Unified Agent A -> Agent B -> Agent A review -> Agent C pipeline."""

from __future__ import annotations

from typing import Any, Protocol

from madl.schemas import AgentAEvidence, AgentBEvidence, MADLResult


class AgentAProtocol(Protocol):
    def analyze(self, image: Any) -> AgentAEvidence: ...


class AgentBProtocol(Protocol):
    def analyze(self, image: Any, spatial_priors: tuple = ()) -> AgentBEvidence: ...


class AgentCProtocol(Protocol):
    def decide(
        self, semantic_evidence: AgentAEvidence, pixel_evidence: AgentBEvidence
    ) -> MADLResult: ...


class MADLPipeline:
    """Model-agnostic orchestration with injectable, independently testable agents."""

    def __init__(
        self, agent_a: AgentAProtocol, agent_b: AgentBProtocol, agent_c: AgentCProtocol
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c

    def predict(self, image: Any) -> MADLResult:
        semantic = self.agent_a.analyze(image)
        pixel = self.agent_b.analyze(image, spatial_priors=semantic.spatial_priors)
        reviewer = getattr(self.agent_a, "review", None)
        if callable(reviewer):
            semantic = reviewer(image, semantic, pixel)
        return self.agent_c.decide(semantic, pixel)
