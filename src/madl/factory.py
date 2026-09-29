"""Construct the model-backed MADL pipeline from portable release weights."""

from __future__ import annotations

from madl.agents import (
    ConflictAwareAdjudicator,
    PixelForensicsAgent,
    SemanticVerificationAgent,
)
from madl.config import MADLConfig
from madl.pipeline import MADLPipeline
from madl.runtime import (
    DualStreamCandidateBuilder,
    DualStreamRuntime,
    QwenForensicsRuntime,
    SAMCandidateSegmenter,
    VisualRankerAdapter,
)


def build_pipeline(config: MADLConfig | None = None, *, device: str | None = None) -> MADLPipeline:
    config = config or MADLConfig()
    config.validate_portable()
    agent_a = SemanticVerificationAgent(QwenForensicsRuntime(config.resolve_weight("agent_a")))
    agent_b = PixelForensicsAgent(
        pixel_runtime=DualStreamRuntime(config.resolve_weight("agent_b"), device=device),
        candidate_builder=DualStreamCandidateBuilder(max_regions=40),
        segmenter=SAMCandidateSegmenter(config.resolve_weight("sam"), device=device),
        ranker=VisualRankerAdapter(config.resolve_weight("visual_ranker"), device=device),
    )
    return MADLPipeline(
        agent_a=agent_a,
        agent_b=agent_b,
        agent_c=ConflictAwareAdjudicator(),
    )
