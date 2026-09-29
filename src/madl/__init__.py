"""MADL: multi-agent image forgery detection and localization."""

from .config import MADLConfig
from .pipeline import MADLPipeline
from .schemas import (
    AgentAEvidence,
    AgentBEvidence,
    ClassLabel,
    DecisionState,
    MADLResult,
)

__all__ = [
    "AgentAEvidence",
    "AgentBEvidence",
    "ClassLabel",
    "DecisionState",
    "MADLConfig",
    "MADLPipeline",
    "MADLResult",
]

__version__ = "1.0.0"
