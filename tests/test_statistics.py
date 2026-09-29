import unittest

from madl.evaluation.statistics import exact_mcnemar_p


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_is_symmetric(self):
        self.assertEqual(exact_mcnemar_p(13, 2), exact_mcnemar_p(2, 13))
        self.assertAlmostEqual(exact_mcnemar_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
