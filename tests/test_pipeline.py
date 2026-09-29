import unittest

from madl.pipeline import MADLPipeline
from madl.schemas import AgentAEvidence, AgentBEvidence, ClassLabel, DecisionState


class FakeAgentA:
    def __init__(self):
        self.reviewed = False

    def analyze(self, image):
        return AgentAEvidence(
            label=ClassLabel.TAMPERED,
            scores={"real": 0.05, "synthetic": 0.05, "tampered": 0.90},
            spatial_priors=((0.1, 0.2, 0.8, 0.9),),
        )

    def review(self, image, initial, pixel_evidence):
        self.reviewed = True
        return initial


class FakeAgentB:
    def __init__(self):
        self.received_priors = None

    def analyze(self, image, spatial_priors=()):
        self.received_priors = spatial_priors
        return AgentBEvidence(0.91, [[1]], 0.88, 2)


class FakeAgentC:
    def decide(self, semantic_evidence, pixel_evidence):
        from madl.schemas import MADLResult

        return MADLResult(
            label=ClassLabel.TAMPERED,
            confidence=0.90,
            mask=pixel_evidence.selected_mask,
            decision_state=DecisionState.AGREEMENT,
            trace=("fake",),
        )


class PipelineTests(unittest.TestCase):
    def test_pipeline_passes_priors_and_runs_candidate_review(self):
        agent_a = FakeAgentA()
        agent_b = FakeAgentB()
        pipeline = MADLPipeline(agent_a=agent_a, agent_b=agent_b, agent_c=FakeAgentC())
        result = pipeline.predict("image.jpg")
        self.assertEqual(agent_b.received_priors, ((0.1, 0.2, 0.8, 0.9),))
        self.assertTrue(agent_a.reviewed)
        self.assertEqual(result.label, ClassLabel.TAMPERED)


if __name__ == "__main__":
    unittest.main()
