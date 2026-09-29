import unittest

from madl.schemas import (
    AgentAEvidence,
    AgentBEvidence,
    ClassLabel,
    DecisionState,
    MADLResult,
)


class SchemaTests(unittest.TestCase):
    def test_tampered_result_requires_a_mask(self):
        with self.assertRaises(ValueError):
            MADLResult(
                label=ClassLabel.TAMPERED,
                confidence=0.9,
                mask=None,
                decision_state=DecisionState.AGREEMENT,
                trace=("agent_a:tampered", "agent_b:tampered"),
            )

    def test_evidence_scores_are_normalized_and_complete(self):
        evidence = AgentAEvidence(
            label=ClassLabel.SYNTHETIC,
            scores={"real": 1.0, "synthetic": 3.0, "tampered": 0.0},
            semantic_summary="global generation artifacts",
        )
        self.assertAlmostEqual(sum(evidence.scores.values()), 1.0)
        self.assertEqual(evidence.confidence, 0.75)

    def test_agent_b_rejects_inconsistent_mask_evidence(self):
        with self.assertRaises(ValueError):
            AgentBEvidence(
                tampered_probability=0.9,
                selected_mask=None,
                mask_score=0.8,
                candidate_count=2,
            )


if __name__ == "__main__":
    unittest.main()
