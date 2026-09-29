import unittest

from madl.agents.agent_c import AdjudicationConfig, ConflictAwareAdjudicator
from madl.schemas import AgentAEvidence, AgentBEvidence, ClassLabel, DecisionState

MASK = [[0, 1], [0, 1]]


def a_evidence(label, confidence):
    remainder = (1.0 - confidence) / 2.0
    return AgentAEvidence(
        label=label,
        scores={
            ClassLabel.REAL: remainder,
            ClassLabel.SYNTHETIC: remainder,
            ClassLabel.TAMPERED: remainder,
            label: confidence,
        },
    )


class AgentCTests(unittest.TestCase):
    def setUp(self):
        self.agent = ConflictAwareAdjudicator(
            AdjudicationConfig(
                local_evidence_threshold=0.60,
                strong_local_evidence_threshold=0.80,
                semantic_override_threshold=0.85,
            )
        )

    def test_agreement_retains_tampered_mask(self):
        result = self.agent.decide(
            a_evidence(ClassLabel.TAMPERED, 0.90),
            AgentBEvidence(0.92, MASK, 0.87, 3),
        )
        self.assertEqual(result.label, ClassLabel.TAMPERED)
        self.assertEqual(result.decision_state, DecisionState.AGREEMENT)
        self.assertEqual(result.mask, MASK)

    def test_suppression_rejects_weak_spurious_mask(self):
        result = self.agent.decide(
            a_evidence(ClassLabel.REAL, 0.92),
            AgentBEvidence(0.40, None, 0.0, 0),
        )
        self.assertEqual(result.label, ClassLabel.REAL)
        self.assertEqual(result.decision_state, DecisionState.SUPPRESSION)
        self.assertIsNone(result.mask)

    def test_override_uses_strong_semantic_tampered_evidence(self):
        result = self.agent.decide(
            a_evidence(ClassLabel.TAMPERED, 0.93),
            AgentBEvidence(0.68, MASK, 0.61, 1),
        )
        self.assertEqual(result.label, ClassLabel.TAMPERED)
        self.assertEqual(result.decision_state, DecisionState.OVERRIDE)

    def test_conflict_does_not_expose_a_disputed_mask(self):
        result = self.agent.decide(
            a_evidence(ClassLabel.SYNTHETIC, 0.90),
            AgentBEvidence(0.91, MASK, 0.89, 4),
        )
        self.assertEqual(result.label, ClassLabel.SYNTHETIC)
        self.assertEqual(result.decision_state, DecisionState.CONFLICT)
        self.assertIsNone(result.mask)


if __name__ == "__main__":
    unittest.main()
