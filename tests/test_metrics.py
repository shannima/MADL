import unittest

from madl.evaluation.metrics import classification_metrics, localization_metrics


class MetricTests(unittest.TestCase):
    def test_classification_metrics_report_macro_and_class_recall(self):
        report = classification_metrics(
            ["real", "synthetic", "tampered", "tampered"],
            ["real", "synthetic", "real", "tampered"],
        )
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["per_class"]["tampered"]["recall"], 0.5)
        self.assertIn("macro_f1", report)

    def test_localization_metrics_match_binary_definition(self):
        report = localization_metrics(
            prediction=[[1, 1], [0, 0]],
            target=[[1, 0], [1, 0]],
        )
        self.assertAlmostEqual(report["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(report["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
