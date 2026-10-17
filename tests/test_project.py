"""Tests target leakage prevention, probability math and routing boundaries."""
import unittest
import numpy as np
import torch

from data import clean_training, normalize_text
from encoder import mean_pool
from metrics import (evaluate, expected_calibration_error, fit_temperature, paired_bootstrap,
                     probabilities, routing_metrics, select_threshold, wilson_interval)
from model import IntentHead


class ProtocolTests(unittest.TestCase):
    def test_normalization_equivalence(self):
        self.assertEqual(normalize_text("  MY Card!!! "), normalize_text("my card"))
        self.assertEqual(normalize_text("ＣＡＲＤ"), "card")

    def test_test_overlap_removed_without_test_labels(self):
        rows = [{"text": "My card?", "label": 0}, {"text": "safe query", "label": 1}]
        kept, counts = clean_training(rows, ["MY CARD!"])
        self.assertEqual(kept, [rows[1]])
        self.assertEqual(counts["test_overlap_train_rows_removed"], 1)

    def test_conflicting_duplicates_excluded(self):
        rows = [{"text": "a", "label": 0}, {"text": "A!", "label": 1},
                {"text": "b", "label": 1}, {"text": "B?", "label": 1}]
        kept, counts = clean_training(rows, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["text"], "b")
        self.assertEqual(counts["conflicting_train_rows_removed"], 2)
        self.assertEqual(counts["duplicate_train_rows_removed"], 1)


class MathTests(unittest.TestCase):
    def test_padding_does_not_change_pooling(self):
        hidden = torch.tensor([[[1., 0.], [0., 1.], [999., 999.]]])
        actual = mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
        torch.testing.assert_close(actual, expected)

    def test_temperature_preserves_class_order(self):
        logits = np.array([[1000., 1002., -1000.], [1., 3., 2.]])
        np.testing.assert_array_equal(probabilities(logits).argmax(1), probabilities(logits, 2.5).argmax(1))
        np.testing.assert_allclose(probabilities(logits).sum(1), 1)

    def test_temperature_fitting_does_not_worsen_calibration_nll(self):
        logits = np.array([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
        labels = np.array([0, 1, 1, 1])
        temperature = fit_temperature(logits, labels)
        self.assertLessEqual(evaluate(probabilities(logits, temperature), labels)["nll"], evaluate(probabilities(logits), labels)["nll"])

    def test_confidence_one_is_included_in_ece(self):
        value = expected_calibration_error(np.array([[1., 0.], [0., 1.]]), np.array([1, 1]))
        self.assertAlmostEqual(value, 0.5)

    def test_negative_temperature_rejected(self):
        with self.assertRaises(ValueError):
            probabilities(np.array([[1., 0.]]), -1)

    def test_bootstrap_is_paired(self):
        first = np.array([1, 0, 1, 1])
        same = paired_bootstrap(first, first, repetitions=50)
        self.assertEqual(same["accuracy_difference"], 0)
        self.assertEqual(same["percentile_95"], [0, 0])

    def test_zero_accepted_has_no_invented_accuracy(self):
        result = routing_metrics(np.array([[0.6, 0.4]]), [0], 1.000001)
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["accepted_accuracy"])
        self.assertEqual(result["accepted_accuracy_wilson_95"], [None, None])

    def test_unmet_target_routes_everything_to_a_person(self):
        threshold = select_threshold(np.array([[0.9, 0.1]]), [1], [0., 0.5, 0.9], min_accepted=1)
        self.assertGreater(threshold, 1)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(95, 100)
        self.assertLess(low, 0.95)
        self.assertGreater(high, 0.95)

    def test_both_heads_support_backpropagation(self):
        for hidden_dim in [0, 128]:
            model = IntentHead(384, 77, hidden_dim)
            logits = model(torch.randn(4, 384))
            self.assertEqual(tuple(logits.shape), (4, 77))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5, 20, 76]))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
