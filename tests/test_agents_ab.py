import unittest

from madl.agents.agent_a import SemanticVerificationAgent
from madl.agents.agent_b import PixelForensicsAgent
from madl.schemas import ClassLabel


class FakeQwenBackend:
    def analyze(self, image):
        return {
            "label": "tampered",
            "class_probs": {"real": 0.05, "synthetic": 0.10, "tampered": 0.85},
            "chain_of_thought": "localized boundary inconsistency",
            "evidence_regions": [{"box_2d": [100, 200, 500, 800]}],
        }


class FakePixelRuntime:
    def predict(self, image):
        return {
            "class_probs": {"real": 0.05, "synthetic": 0.10, "tampered": 0.85},
            "anomaly_heatmap": [[0.1, 0.9], [0.2, 0.8]],
            "suspicious_regions": [{"box_permille": [120, 220, 480, 780]}],
        }


class FakeSegmenter:
    def segment(self, image, regions, heatmap=None):
        self.regions = regions
        return [
            {"mask": [[0, 1], [0, 1]], "score": 0.82, "proposal_source": "sam:0"},
            {"mask": [[1, 0], [1, 0]], "score": 0.65, "proposal_source": "sam:1"},
        ]


class FakeRanker:
    def select(self, image, heatmap, candidates):
        return candidates[0]


class AgentABTests(unittest.TestCase):
    def test_agent_a_normalizes_report_and_permille_regions(self):
        evidence = SemanticVerificationAgent(FakeQwenBackend()).analyze("image.png")
        self.assertEqual(evidence.label, ClassLabel.TAMPERED)
        self.assertEqual(evidence.spatial_priors, ((0.1, 0.2, 0.5, 0.8),))
        self.assertAlmostEqual(evidence.confidence, 0.85)

    def test_agent_b_combines_semantic_and_bottom_up_regions(self):
        segmenter = FakeSegmenter()
        evidence = PixelForensicsAgent(
            pixel_runtime=FakePixelRuntime(),
            segmenter=segmenter,
            ranker=FakeRanker(),
        ).analyze("image.png", spatial_priors=((0.1, 0.2, 0.5, 0.8),))
        self.assertEqual(evidence.selected_mask, [[0, 1], [0, 1]])
        self.assertEqual(evidence.candidate_count, 2)
        self.assertGreaterEqual(len(segmenter.regions), 2)
        self.assertAlmostEqual(evidence.tampered_probability, 0.85)


if __name__ == "__main__":
    unittest.main()
