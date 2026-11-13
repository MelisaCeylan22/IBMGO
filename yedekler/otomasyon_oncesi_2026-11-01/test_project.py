"""Telekom test verisi ve algoritmalar için küçük otomatik doğrulamalar.

Çalıştırma: python -B -m unittest test_project.py -v
"""

import unittest
from unittest.mock import patch

import numpy as np

import bmgo
import baselines
from data_generator import generate_telecom_data
from fitness import get_suite_metrics


class TelecomGeneratorTests(unittest.TestCase):
    def test_standard_profile_is_backward_compatible(self):
        expected = {50: 46, 100: 68, 250: 149, 500: 170, 1000: 178}
        for size, requirement_count in expected.items():
            with self.subTest(size=size):
                coverage, _, _, features, _, _ = generate_telecom_data(
                    size, seed=42, include_metadata=True, scenario_profile="standard"
                )
                self.assertEqual(coverage.shape, (size, requirement_count))
                self.assertEqual(len(features), requirement_count)

    def test_expanded_profile_has_realistic_additional_context(self):
        counts = []
        for size in (50, 100, 250, 500, 1000):
            with self.subTest(size=size):
                coverage, durations, names, features, deps, meta = generate_telecom_data(
                    size, seed=42, include_metadata=True, scenario_profile="expanded"
                )
                counts.append(len(features))
                self.assertEqual(coverage.shape, (size, len(features)))
                self.assertTrue(np.all(coverage.sum(axis=0) > 0))
                self.assertTrue(np.all(durations > 0))
                self.assertTrue(np.all(np.tril(deps) == 0))
                self.assertEqual(len(names), size)
                self.assertEqual(len({c["id"] for c in meta["test_cases"]}), size)
        self.assertEqual(counts[0], 46)
        self.assertTrue(all(b > a for a, b in zip(counts, counts[1:])))
        self.assertGreaterEqual(counts[-1], 300)

    def test_expanded_examples_and_reproducibility(self):
        first = generate_telecom_data(500, seed=42, include_metadata=True, scenario_profile="expanded")
        second = generate_telecom_data(500, seed=42, include_metadata=True, scenario_profile="expanded")
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[4], second[4]))
        self.assertEqual(first[5]["test_cases"], second[5]["test_cases"])
        families = {case["family"] for case in first[5]["test_cases"]}
        for required_family in ("auth_expired", "portability_pending", "duplicate_billing",
                                "prorated_upgrade", "quota_reset", "blacklisted"):
            self.assertIn(required_family, families)

    def test_extra_offers_and_customer_eligibility(self):
        _, _, _, _, _, meta = generate_telecom_data(
            1000, seed=42, include_metadata=True, scenario_profile="expanded"
        )
        self.assertEqual(
            {o["code"] for o in meta["offers"]},
            {f"KMP{i:02d}" for i in range(1, 10)},
        )
        cases = meta["test_cases"]
        expectations = {
            "second_line_ineligible": ("KMP07", "parent_line_active", False),
            "age_ineligible": ("KMP08", "customer_age", 35),
            "roaming_disabled": ("KMP09", "roaming_enabled", False),
        }
        for family, (offer_code, field, value) in expectations.items():
            matches = [case for case in cases if case["family"] == family]
            self.assertTrue(matches, msg=f"{family} uretilmemis")
            self.assertTrue(
                all(case["offer_code"] == offer_code and case[field] == value
                    for case in matches),
            )
        self.assertTrue({"G30", "G31", "G32"}.issubset(
            {r["id"] for r in meta["requirements"]}
        ))


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coverage, cls.durations, _, _, cls.deps, cls.metadata = generate_telecom_data(
            50, seed=42, include_metadata=True, scenario_profile="expanded"
        )

    def assert_feasible(self, solution):
        metrics = get_suite_metrics(
            solution, self.coverage, self.durations,
            self.deps, self.metadata["critical_feature_indices"],
        )
        self.assertEqual(metrics["critical_missing"], 0)
        self.assertEqual(metrics["broken_dependencies"], 0)
        self.assertGreaterEqual(metrics["coverage_ratio"], 0.9 - 1e-9)

    def test_all_methods_feasible(self):
        args = dict(
            coverage_matrix=self.coverage,
            durations=self.durations,
            dependency_matrix=self.deps,
            critical_feature_indices=self.metadata["critical_feature_indices"],
            min_coverage=0.9,
        )
        raw, _, _ = bmgo.binary_mountain_gazelle_optimizer(
            **args, pop_size=4, max_iter=2, random_seed=42,
        )
        improved, _, _ = bmgo.binary_mountain_gazelle_optimizer(
            **args, pop_size=4, max_iter=2, random_seed=42,
            adaptive_binary=True, diverse_initialization=True, local_search_trials=4,
        )
        ga, _, _ = baselines.genetic_algorithm_optimizer(
            **args, pop_size=4, max_generations=10, random_seed=42,
        )
        greedy, _ = baselines.greedy_regression_selector(**args)
        for sol in (raw, improved, ga, greedy):
            self.assert_feasible(sol)

    def test_equal_fitness_evaluation_budget(self):
        args = dict(
            coverage_matrix=self.coverage,
            durations=self.durations,
            dependency_matrix=self.deps,
            critical_feature_indices=self.metadata["critical_feature_indices"],
            min_coverage=0.9,
        )
        raw_count = 0
        improved_count = 0
        ga_count = 0
        orig_bmgo = bmgo.calculate_fitness
        orig_ga = baselines.calculate_fitness

        def count_raw(*a, **kw):
            nonlocal raw_count
            raw_count += 1
            return orig_bmgo(*a, **kw)

        def count_improved(*a, **kw):
            nonlocal improved_count
            improved_count += 1
            return orig_bmgo(*a, **kw)

        def count_ga(*a, **kw):
            nonlocal ga_count
            ga_count += 1
            return orig_ga(*a, **kw)

        with patch.object(bmgo, "calculate_fitness", side_effect=count_raw):
            bmgo.binary_mountain_gazelle_optimizer(
                **args, pop_size=4, max_iter=5, random_seed=42,
            )
        with patch.object(bmgo, "calculate_fitness", side_effect=count_improved):
            bmgo.binary_mountain_gazelle_optimizer(
                **args, pop_size=4, max_iter=4, random_seed=42,
                adaptive_binary=True, diverse_initialization=True,
                local_search_trials=4,
            )
        with patch.object(baselines, "calculate_fitness", side_effect=count_ga):
            baselines.genetic_algorithm_optimizer(
                **args, pop_size=4, max_generations=20, random_seed=42,
            )
        self.assertEqual((raw_count, improved_count, ga_count), (84, 84, 84))

    def test_reference_chaotic_spiral_variant_is_reproducible(self):
        common = dict(
            coverage_matrix=self.coverage,
            durations=self.durations,
            dependency_matrix=self.deps,
            critical_feature_indices=self.metadata["critical_feature_indices"],
            min_coverage=0.90,
            pop_size=4,
            max_iter=4,
            adaptive_binary=True,
            diverse_initialization=True,
            local_search_trials=2,
            spiral_search_trials=2,
            reference_mgo_formulas=True,
            chaotic_initialization=True,
            nonlinear_control=True,
            random_seed=48,
        )
        first = bmgo.binary_mountain_gazelle_optimizer(**common)
        second = bmgo.binary_mountain_gazelle_optimizer(**common)
        self.assert_feasible(first[0])
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertEqual(first[1], second[1])
        self.assertTrue(np.array_equal(first[2], second[2]))

        with patch.object(bmgo, "calculate_fitness", wraps=bmgo.calculate_fitness) as counter:
            bmgo.binary_mountain_gazelle_optimizer(**common)
            # 4 baslangic + 4 iterasyon * (4*4 hareket + 2 swap + 2 spiral)
            self.assertEqual(counter.call_count, 84)

    def test_s_shaped_binary_transfer_respects_constraints(self):
        solution, score, _ = bmgo.binary_mountain_gazelle_optimizer(
            self.coverage,
            self.durations,
            self.deps,
            pop_size=4,
            max_iter=3,
            random_seed=42,
            critical_feature_indices=self.metadata["critical_feature_indices"],
            min_coverage=0.90,
            reference_mgo_formulas=True,
            chaotic_initialization=True,
            nonlinear_control=True,
            binary_transfer="s_assign",
        )
        self.assertTrue(np.isfinite(score))
        self.assert_feasible(solution)

    def test_mixed_mgo_movements_are_reproducible_and_feasible(self):
        common = dict(
            coverage_matrix=self.coverage,
            durations=self.durations,
            dependency_matrix=self.deps,
            critical_feature_indices=self.metadata["critical_feature_indices"],
            min_coverage=0.9,
            pop_size=4, max_iter=3, random_seed=99,
            adaptive_binary=True, diverse_initialization=True,
            local_search_trials=4,
            reference_mgo_formulas=True,
            reference_mgo_probability=0.5,
            chaotic_initialization=True, nonlinear_control=True,
        )
        first = bmgo.binary_mountain_gazelle_optimizer(**common)
        second = bmgo.binary_mountain_gazelle_optimizer(**common)
        self.assert_feasible(first[0])
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertEqual(first[1], second[1])

        with self.assertRaises(ValueError):
            bmgo.binary_mountain_gazelle_optimizer(
                **{**common, "reference_mgo_probability": -0.1}
            )

    def test_large_neighborhood_preserves_hard_constraints(self):
        common = dict(
            coverage_matrix=self.coverage,
            durations=self.durations,
            dependency_matrix=self.deps,
            critical_feature_indices=self.metadata["critical_feature_indices"],
            min_coverage=0.90,
            pop_size=4, max_iter=4, random_seed=72,
            adaptive_binary=True, diverse_initialization=True,
            reference_mgo_formulas=True, chaotic_initialization=True,
            nonlinear_control=True, local_search_trials=4,
            local_search_strategy="large",
        )
        with patch.object(bmgo, "calculate_fitness", wraps=bmgo.calculate_fitness) as counter:
            solution, _, _ = bmgo.binary_mountain_gazelle_optimizer(**common)
        self.assertEqual(counter.call_count, 84)
        self.assert_feasible(solution)

    def test_dimension_adaptive_policy_is_fixed_before_holdout(self):
        small = bmgo.dimension_adaptive_settings(250)
        medium = bmgo.dimension_adaptive_settings(500)
        large = bmgo.dimension_adaptive_settings(1000)
        self.assertFalse(small.get("reference_mgo_formulas", False))
        self.assertEqual(medium["local_search_strategy"], "large")
        self.assertEqual(large["reference_mgo_probability"], 0.5)
        for settings in (small, medium, large):
            self.assertEqual(settings["local_search_trials"], 12)


if __name__ == "__main__":
    unittest.main()
