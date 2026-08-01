import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.fetch_and_compute_irr import (
    DataProvenanceError,
    annualize_monthly_irr,
    calculate_cohort,
    descriptive_summary,
    load_query_config,
    periodic_irr,
)


class IrrMathTests(unittest.TestCase):
    def test_monthly_irr_is_annualized_compoundingly(self) -> None:
        self.assertAlmostEqual(annualize_monthly_irr(0.01), 1.01**12 - 1)

    def test_periodic_irr_recovers_known_one_period_return(self) -> None:
        self.assertAlmostEqual(periodic_irr([-100.0, 110.0]), 0.10, places=10)

    def test_multiple_sign_changes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            periodic_irr([-100.0, 150.0, -40.0, 10.0])


class ProvenanceTests(unittest.TestCase):
    def test_unverified_estat_config_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.yaml"
            path.write_text(
                yaml.safe_dump({"verified": False, "queries": [{"industry": "x"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(DataProvenanceError):
                load_query_config(path)


class CohortTests(unittest.TestCase):
    def test_fixed_horizon_is_recorded(self) -> None:
        index = pd.date_range("2010-01-31", periods=12, freq="ME")
        returns = pd.Series(np.full(12, 0.01), index=index)
        result = calculate_cohort(
            "industry",
            {2010: 100.0},
            {2010: 120.0},
            returns,
            2010,
            12,
        )
        self.assertEqual(result.horizon_months, 12)
        self.assertEqual(result.first_month, "2010-01")
        self.assertEqual(result.last_month, "2010-12")
        self.assertTrue(np.isfinite(result.annualized_irr))

    def test_incomplete_horizon_is_rejected(self) -> None:
        index = pd.date_range("2010-01-31", periods=6, freq="ME")
        returns = pd.Series(np.full(6, 0.01), index=index)
        with self.assertRaises(DataProvenanceError):
            calculate_cohort(
                "industry",
                {2010: 100.0},
                {2010: 120.0},
                returns,
                2010,
                12,
            )

    def test_summary_does_not_emit_p_values(self) -> None:
        index = pd.date_range("2010-01-31", periods=24, freq="ME")
        returns = pd.Series(np.full(24, 0.01), index=index)
        results = [
            calculate_cohort("industry", {2010: 100.0}, {2010: 120.0}, returns, 2010, 12),
            calculate_cohort(
                "industry",
                {2011: 100.0},
                {2011: 120.0},
                returns,
                2011,
                12,
            ),
        ]
        summary = descriptive_summary(results)[0]
        self.assertEqual(summary["inference_status"], "not_tested")
        self.assertNotIn("p_value", summary)


if __name__ == "__main__":
    unittest.main()
