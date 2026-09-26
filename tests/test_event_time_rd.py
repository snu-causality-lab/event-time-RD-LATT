"""Synthetic checks; the RBC passthrough check is not a validation of RBC theory."""

import unittest

import numpy as np
import pandas as pd
from rdrobust import rdrobust

from event_time_rd import covariate_balance, estimate_rd


class EventTimeRDTests(unittest.TestCase):
    def setUp(self):
        self.x = np.r_[np.linspace(-2, -0.01, 160), np.linspace(0.01, 2, 160)]
        rng = np.random.default_rng(812)
        self.z = rng.normal(size=len(self.x))
        self.y = 1 + 0.3 * self.x + 0.2 * self.x ** 2 + 1.5 * (self.x > 0)
        self.y += rng.normal(scale=0.3, size=len(self.x))

    def test_conventional_jump_against_independent_weighted_least_squares(self):
        widths = (1.1, 1.4)
        intercepts = []
        for side, width in enumerate(widths):
            keep = ((self.x < 0) if side == 0 else (self.x > 0)) & (np.abs(self.x) < width)
            x, y = self.x[keep], self.y[keep]
            root_weight = np.sqrt(1 - np.abs(x) / width)
            design = np.column_stack((np.ones(len(x)), x))
            coefficient = np.linalg.lstsq(design * root_weight[:, None], y * root_weight, rcond=None)[0]
            intercepts.append(coefficient[0])
        result = estimate_rd(self.y, self.x, bandwidth=widths)
        self.assertAlmostEqual(result["estimate"], intercepts[1] - intercepts[0], places=10)
        self.assertEqual(result["bandwidth_left"], widths[0])
        self.assertEqual(result["bias_bandwidth_right"], widths[1])
        self.assertEqual(result["n_effective_left"], np.sum((self.x < 0) & (self.x > -widths[0])))

    def test_rbc_extraction_matches_library_output_only(self):
        result = estimate_rd(self.y, self.x, self.z, bandwidth=1.2)
        direct = rdrobust(self.y, self.x, covs=self.z[:, None], covs_drop=False,
                          c=0, p=1, q=2, kernel="triangular", h=1.2, b=1.2,
                          bwselect="mserd", vce="nn", level=95)
        for key, expected in {
            "estimate_bias_corrected": direct.coef.loc["Bias-Corrected"].iloc[0],
            "se_robust": direct.se.loc["Robust"].iloc[0],
            "ci_robust_low": direct.ci.loc["Robust"].iloc[0],
            "ci_robust_high": direct.ci.loc["Robust"].iloc[1],
            "p_value_robust": direct.pv.loc["Robust"].iloc[0],
        }.items():
            self.assertAlmostEqual(result[key], expected, places=12)

    def test_complete_case_mask_zero_exclusion_and_fractional_time(self):
        y, x, z = self.y.copy(), self.x.copy(), self.z.copy()
        y[0], x[1], z[2], x[3] = np.nan, np.nan, np.nan, 0
        result = estimate_rd(y, x, z, bandwidth=1.2)
        clean = estimate_rd(y[4:], x[4:], z[4:], bandwidth=1.2)
        self.assertEqual((result["n_input"], result["n_used"], result["n_excluded"]), (320, 316, 4))
        self.assertAlmostEqual(result["estimate"], clean["estimate"], places=12)
        # All |time| < 1 observations would collapse to zero under calendar-day truncation.
        self.assertGreater(result["n_effective_left"], 50)

    def test_input_support_and_rank_guards(self):
        invalid = [
            dict(y=self.y[:-1], event_time=self.x),
            dict(y=self.y, event_time=self.x[:, None]),
            dict(y=self.y, event_time=np.abs(self.x)),
            dict(y=self.y, event_time=np.sign(self.x)),
            dict(y=self.y, event_time=self.x, covariates=np.ones(len(self.x))),
            dict(y=self.y, event_time=self.x, covariates=np.column_stack((self.z, self.z))),
            dict(y=self.y, event_time=self.x, covariates=self.z[:-1]),
            dict(y=self.y[:18], event_time=self.x[:18]),
            dict(y=self.y, event_time=self.x, bandwidth=0.02),
            dict(y=self.y, event_time=self.x, bandwidth=(1, -1)),
            dict(y=self.y, event_time=self.x, bandwidth=np.nan),
            dict(y=self.y, event_time=np.arange(len(self.x)).astype("datetime64[D]")),
            dict(y=pd.Series(self.y), event_time=pd.Series(self.x, index=np.arange(len(self.x))[::-1])),
        ]
        for args in invalid:
            with self.subTest(args=list(args)):
                with self.assertRaises(ValueError):
                    estimate_rd(**args)
        for field in ("y", "event_time", "covariates"):
            args = dict(y=self.y.copy(), event_time=self.x.copy(), covariates=self.z.copy())
            args[field][0] = np.inf
            # Infinite values must be rejected even when another field is missing.
            args["event_time" if field == "y" else "y"][0] = np.nan
            with self.subTest(infinite=field), self.assertRaises(ValueError):
                estimate_rd(**args)

    def test_covariate_rank_is_pooled_not_required_on_each_side(self):
        z = np.where(self.x < 0, 0, self.z)
        result = estimate_rd(self.y, self.x, z, bandwidth=1.2)
        self.assertTrue(np.isfinite(result["estimate"]))

    def test_degenerate_inference_is_rejected(self):
        outcomes = {
            "constant_nonzero": np.full(len(self.x), 10.0),
            "constant_zero": np.zeros(len(self.x)),
            "constant_with_roundoff": np.full(len(self.x), 0.1),
            "constant_only_inside_bandwidth": np.where(np.abs(self.x) < 1.2, 10.0, self.y),
            "local_constant_with_roundoff": np.where(np.abs(self.x) < 1.2, 0.1, self.y),
        }
        for label, y in outcomes.items():
            with self.subTest(outcome=label), np.errstate(divide="ignore", invalid="ignore"):
                with self.assertRaisesRegex(ValueError, "[Uu]ndefined|[Dd]egenerate"):
                    estimate_rd(y, self.x, bandwidth=1.2)

    def test_zero_pvalues_are_valid_with_positive_uncertainty(self):
        result = estimate_rd(self.y + 100 * (self.x > 0), self.x, bandwidth=1.2)
        self.assertEqual(result["p_value"], 0)
        self.assertEqual(result["p_value_robust"], 0)
        self.assertGreater(result["se"], 0)
        self.assertGreater(result["se_robust"], 0)

    def test_balance_distributions_counts_and_undefined_cases(self):
        x = np.r_[-np.ones(40), np.ones(40), 0, -2, 2]
        v = np.linspace(-1, 1, 40)
        columns = {
            "same": np.r_[v, v, 99, 99, 99],
            "different_variance": np.r_[v, 3 * v, 99, 99, 99],
            "constant": np.ones(83),
            "missing": np.r_[np.full(40, np.nan), v, 99, 99, 99],
            "category": ["a"] * 30 + ["b"] * 10 + ["a"] * 10 + ["b"] * 30 + ["outside"] * 3,
            "sparse": ["a"] * 79 + ["b"] + ["outside"] * 3,
        }
        result = covariate_balance(x, columns, bandwidth=2, categorical=("category", "sparse")).set_index("covariate")
        self.assertEqual(result.loc["same", "statistic"], 0)
        self.assertEqual(result.loc["same", "p_value"], 1)
        self.assertAlmostEqual(v.mean(), (3 * v).mean())
        points = np.sort(np.r_[v, 3 * v])
        ks_distance = max(abs(np.mean(v <= point) - np.mean(3 * v <= point)) for point in points)
        self.assertAlmostEqual(result.loc["different_variance", "statistic"], ks_distance)
        self.assertGreater(ks_distance, 0)
        self.assertAlmostEqual(result.loc["category", "statistic"], 18.05)
        self.assertEqual((result.loc["same", "n_left"], result.loc["same", "n_right"]), (40, 40))
        for name, status in (("constant", "constant"), ("missing", "missing_side"), ("sparse", "sparse_table")):
            self.assertEqual(result.loc[name, "status"], status)
            self.assertTrue(np.isnan(result.loc[name, "p_value"]))
        self.assertEqual(result.loc["missing", "n_left"], 0)

    def test_balance_original_ks_calibration(self):
        left = np.linspace(-1, 1, 40)
        right = 3 * left
        result = covariate_balance(np.r_[-np.ones(40), np.ones(40)],
                                   {"value": np.r_[left, right]}, bandwidth=2).iloc[0]
        points = np.sort(np.r_[left, right])
        statistic = max(abs(np.mean(left <= point) - np.mean(right <= point)) for point in points)
        n_eff = len(left) * len(right) / (len(left) + len(right))
        terms = np.arange(1, 101)
        expected_p = 2 * np.sum((-1.0) ** (terms - 1) * np.exp(-2 * terms ** 2 * statistic ** 2 * n_eff))
        # Algorithm-fidelity check via the limiting survival series, not a proof of calibration.
        self.assertAlmostEqual(result["p_value"], expected_p, places=12)

    def test_balance_default_uses_all_supplied_nonzero_times(self):
        x = np.r_[self.x, 0, np.nan]
        values = {"value": np.r_[self.z, 99, 99]}
        whole_window = covariate_balance(x, values).iloc[0]
        selected_window = covariate_balance(x, values, bandwidth=1).iloc[0]
        self.assertEqual((whole_window["n_left"], whole_window["n_right"]), (160, 160))
        self.assertEqual((selected_window["n_left"], selected_window["n_right"]), (80, 80))

    def test_balance_alignment_and_column_guards(self):
        with self.assertRaises(ValueError):
            covariate_balance(pd.Series(self.x), pd.DataFrame({"z": self.z}, index=np.arange(len(self.x))[::-1]), bandwidth=1)
        with self.assertRaises(ValueError):
            covariate_balance(self.x, {"z": self.z}, bandwidth=1, categorical=("unknown",))
        with self.assertRaises(ValueError):
            covariate_balance(self.x, {"z": self.z[:-1]}, bandwidth=1)

    def test_categorical_requires_column_names_not_a_bare_string(self):
        columns = {name: np.tile([0, 1], len(self.x) // 2) for name in ("a", "b", "ab")}
        with self.assertRaisesRegex(ValueError, "categorical.*collection"):
            covariate_balance(self.x, columns, categorical="ab")


if __name__ == "__main__":
    unittest.main()
