import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "ml"))

import isolation_forest_period_auto_huaye as isolation


def make_isolation_input() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    base_timestamp = pd.Timestamp("2018-07-02 00:00:00")

    for period_index, period in enumerate(isolation.PERIOD_ORDER):
        period_rows = 24 + period_index * 3
        for row_index in range(period_rows):
            values = rng.normal(
                loc=period_index,
                scale=0.1,
                size=len(isolation.MODEL_FEATURES),
            )
            row = {
                "timestamp": base_timestamp + pd.Timedelta(days=period_index, minutes=row_index),
                "floor": 6,
                "zone": [1, 2, 4, 5][row_index % 4],
                "hour": row_index % 24,
                "operating_period": period,
                "baseline_anomaly_label": 0,
                "baseline_anomaly_count": 0,
                "lof_anomaly_label": 0,
                "lof_anomaly_score": float(row_index) / period_rows,
            }

            if row_index == period_rows - 1 and period_index % 2 == 0:
                values += 8.0
                row["baseline_anomaly_label"] = 1
                row["baseline_anomaly_count"] = 2
            if row_index >= period_rows - 2 and period_index in {1, 4}:
                values -= 7.0
                row["lof_anomaly_label"] = 1

            for column, value in zip(isolation.MODEL_FEATURES, values):
                row[column] = float(value)
            for column, value in zip(isolation.RAW_SENSOR_COLUMNS, values):
                row[column] = float(value)
            rows.append(row)

    return pd.DataFrame(rows)


class IsolationForestPeriodAutoTest(unittest.TestCase):
    def test_period_models_summaries_and_validation_on_synthetic_data(self) -> None:
        source = make_isolation_input()

        results, models = isolation.fit_period_models(source)
        summaries = isolation.build_summaries(results)
        validations = isolation.validate_results(
            source,
            results,
            summaries,
            models,
            expected_row_count=len(source),
        )

        self.assertEqual(set(models), set(isolation.PERIOD_ORDER))
        self.assertEqual(len(results), len(source))
        self.assertGreater(int(results["period_if_anomaly_label"].sum()), 0)
        self.assertTrue(
            np.array_equal(
                results["period_if_anomaly_label"].to_numpy(dtype=bool),
                results["period_if_decision_function"].lt(0).to_numpy(),
            )
        )

        period_summary = summaries["period_summary"]
        expected_rates = 100.0 * period_summary["anomaly_count"] / period_summary["rows"]
        pd.testing.assert_series_equal(
            period_summary["anomaly_rate_pct"],
            expected_rates,
            check_names=False,
        )
        self.assertTrue((period_summary["anomaly_rate_pct"] >= 0).all())
        self.assertTrue((period_summary["anomaly_rate_pct"] <= 100).all())
        self.assertEqual(period_summary["anomaly_count"].dtype, np.dtype("int64"))
        self.assertTrue(all(validation["passed"] for validation in validations))

    def test_save_outputs_writes_reloadable_files(self) -> None:
        source = make_isolation_input()
        results, models = isolation.fit_period_models(source)
        summaries = isolation.build_summaries(results)
        validations = isolation.validate_results(
            source,
            results,
            summaries,
            models,
            expected_row_count=len(source),
        )

        with tempfile.TemporaryDirectory(dir=isolation.PROJECT_DIR) as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(isolation, "OUTPUT_DIR", output_dir):
                summary_payload = isolation.save_outputs(results, summaries, validations)

            full_results = output_dir / isolation.OUTPUT_FILES["full_results"]
            anomalies = output_dir / isolation.OUTPUT_FILES["anomalies"]
            modelling_summary = output_dir / isolation.OUTPUT_FILES["modelling_summary"]

            self.assertTrue(full_results.is_file())
            self.assertTrue(anomalies.is_file())
            self.assertTrue(modelling_summary.is_file())
            self.assertEqual(len(pd.read_csv(full_results)), len(results))
            self.assertEqual(len(pd.read_csv(anomalies)), int(results["period_if_anomaly_label"].sum()))
            self.assertEqual(summary_payload["row_count"], len(results))
            self.assertEqual(json.loads(modelling_summary.read_text(encoding="utf-8"))["row_count"], len(results))


if __name__ == "__main__":
    unittest.main()
