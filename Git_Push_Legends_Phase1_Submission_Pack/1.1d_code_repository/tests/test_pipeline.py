"""Fast contracts for data integrity, leakage control and submission format."""

from __future__ import annotations

import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_and_engineer_data
from src.hcp import analyse_linkages, forward_pressure_summary, regional_projection
from src.models import (
    forecast_sarima,
    make_direct_samples,
    metric_set,
    stationarity_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]


class DataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_and_engineer_data(ROOT)

    def test_all_source_coverage_and_target(self):
        self.assertEqual(len(self.bundle.monthly), 288)
        self.assertEqual(self.bundle.audit["target_non_null"], 276)
        self.assertEqual(self.bundle.audit["shipping_exact_duplicates_removed"], 3)
        self.assertEqual(self.bundle.monthly.index.max(), pd.Timestamp("2023-12-01"))

    def test_rich_shipping_features_exist(self):
        expected = {
            "shipping_std_now",
            "shipping_month_return_now",
            "shipping_extreme_up_days_now",
            "shipping_late_minus_early_lag3",
        }
        self.assertTrue(expected.issubset(self.bundle.origin_features.columns))

    def test_supervised_training_cutoff_prevents_future_labels(self):
        samples = make_direct_samples(self.bundle.origin_features, self.bundle.target)
        cutoff = pd.Timestamp("2020-12-01")
        train = samples.through(cutoff)
        self.assertTrue((train.label_ends <= cutoff).all())
        self.assertGreater(len(train.x), 100)


class ModelContracts(unittest.TestCase):
    def test_sarima_output_stationarity_and_metrics(self):
        bundle = load_and_engineer_data(ROOT)
        prediction, fit = forecast_sarima(
            bundle.target,
            pd.Timestamp("2020-12-01"),
            (1, 1, 1),
            (1, 0, 1, 12),
        )
        actual = bundle.target.loc["2021-01-01":"2021-12-01"].to_numpy(dtype=float)
        self.assertEqual(prediction.shape, (12,))
        self.assertTrue(fit["converged"])
        metrics = metric_set(actual, prediction)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        diagnostics = stationarity_diagnostics(bundle.target)
        self.assertEqual(
            set(diagnostics),
            {"level", "first_difference", "seasonal_difference_12"},
        )
        self.assertLess(diagnostics["first_difference"]["adf_p_value"], 0.05)

    def test_submission_uses_sarima_not_ridge(self):
        for relative in ["README.md", "src/pipeline.py", "src/reporting.py"]:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("SARIMA", source)
            self.assertNotIn("Direct Ridge", source)
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dependency in [
            "numpy==2.4.6",
            "pandas==2.3.3",
            "Pillow==11.3.0",
            "pypdf==6.14.2",
            "PyMuPDF==1.28.0",
            "statsmodels==0.14.5",
        ]:
            self.assertIn(dependency, requirements)

    def test_hcp_two_indicators_and_forward_pressure(self):
        bundle = load_and_engineer_data(ROOT)
        linkage = analyse_linkages(bundle.monthly)
        self.assertEqual(
            set(linkage["indicator_tests"]),
            {"regional_food_inflation", "regional_general_inflation"},
        )
        for result in linkage["indicator_tests"].values():
            self.assertGreater(result["regression_n"], 250)
            self.assertTrue(np.isfinite(result["lag1_coefficient"]))
            self.assertGreaterEqual(result["lag1_p_value"], 0.0)
            self.assertLessEqual(result["lag1_p_value"], 1.0)
            self.assertGreaterEqual(result["granger"]["p_value"], 0.0)
            self.assertLessEqual(result["granger"]["p_value"], 1.0)

        forecast = np.linspace(12.0, 4.0, 12)
        projection = regional_projection(bundle.monthly, forecast)
        pressure = forward_pressure_summary(projection)
        self.assertEqual(pressure["months_at_or_above_10_percent"], 3)
        self.assertEqual(pressure["peak_month"], "2024-01")
        self.assertGreater(
            pressure["first_half_average_yoy"],
            pressure["second_half_average_yoy"],
        )

    def test_predictions_contract_if_generated(self):
        path = ROOT / "outputs" / "predictions.csv"
        if not path.exists():
            self.skipTest("Pipeline output not generated yet")
        frame = pd.read_csv(path)
        self.assertEqual(list(frame.columns), ["year_month", "forecast"])
        self.assertEqual(len(frame), 12)
        self.assertEqual(frame.iloc[0]["year_month"], "2024-01")
        self.assertEqual(frame.iloc[-1]["year_month"], "2024-12")
        self.assertTrue(np.isfinite(frame["forecast"]).all())

    def test_submission_pack_contract_if_generated(self):
        path = ROOT / "output" / "Git_Push_Legends_Phase1_Submission_Pack.zip"
        if not path.exists():
            self.skipTest("Submission pack not generated yet")
        expected = {
            "1.1a_predictions.csv",
            "1.1b_feature_engineering_report.pdf",
            "1.1c_model_comparison_report.pdf",
            "1.1d_code_repository.zip",
            "1.2a_hcp_linkage_memo.pdf",
            "1.2b_hcp_visualisations.pdf",
            "submission_manifest.json",
        }
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()), expected)
            self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
