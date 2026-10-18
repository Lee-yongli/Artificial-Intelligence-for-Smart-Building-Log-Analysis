import json
import math
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
                "lof_reference_missing": 0,
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
            for column, value in zip(isolation.ROLLING_COLUMNS, values):
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
                results["period_if_auto_anomaly_label"].to_numpy(dtype=bool),
                results["period_if_decision_function"].lt(0).to_numpy(),
            )
        )

        for period, group in results.groupby("operating_period", observed=True):
            with self.subTest(period=period):
                selected = group["period_if_anomaly_label"].eq(1)
                expected_count = max(1, math.ceil(len(group) * isolation.IF_CALIBRATED_ANOMALY_RATE))
                self.assertEqual(int(selected.sum()), expected_count)
                self.assertGreaterEqual(
                    group.loc[selected, "period_if_anomaly_score"].min(),
                    group.loc[~selected, "period_if_anomaly_score"].max(),
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
        for name in ["period_summary", "zone_summary", "hour_summary"]:
            table = summaries[name]
            expected = 100.0 * table["anomaly_count"].astype(np.int64) / table["rows"]
            np.testing.assert_allclose(table["anomaly_rate_pct"], expected)
            self.assertTrue(table["anomaly_rate_pct"].between(0, 100).all())
        expected_auto = 100.0 * period_summary["auto_anomaly_count"].astype(np.int64) / period_summary["rows"]
        np.testing.assert_allclose(period_summary["auto_anomaly_rate_pct"], expected_auto)

    def test_event_overlap_percentages_do_not_overflow_for_small_events(self) -> None:
        # A fully supported 35-minute event must have 100% overlap, even when
        # source labels and grouped counts fit into an int8.
        event_rows = make_isolation_input().iloc[:35].copy()
        event_rows["timestamp"] = pd.date_range("2018-07-02", periods=len(event_rows), freq="min")
        event_rows["zone"] = 1
        event_rows["operating_period"] = "night_off"
        for column in ["period_if_anomaly_label", "if_lof_overlap", "if_baseline_overlap"]:
            event_rows[column] = np.ones(len(event_rows), dtype=np.int8)
        event_rows["period_if_anomaly_score"] = 0.7
        event_rows["period_if_decision_function"] = -0.2
        events = isolation.build_if_events(event_rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(int(events.iloc[0]["anomaly_points"]), 35)
        self.assertEqual(float(events.iloc[0]["baseline_overlap_pct"]), 100.0)

    def test_validation_rejects_wrong_size_and_wrong_calibration(self) -> None:
        source = make_isolation_input()
        results, models = isolation.fit_period_models(source)
        summaries = isolation.build_summaries(results)

        # Keep the strict production-size default while allowing explicit fixtures.
        with self.assertRaisesRegex(AssertionError, "expected_row_count"):
            isolation.validate_results(source, results, summaries, models)
        with self.assertRaisesRegex(AssertionError, "expected_row_count"):
            isolation.validate_results(
                source, results, summaries, models, expected_row_count=len(source) + 1
            )

        corrupted = results.copy()
        first_normal = corrupted.index[corrupted["period_if_anomaly_label"].eq(0)][0]
        corrupted.loc[first_normal, "period_if_anomaly_label"] = 1
        bad_summaries = isolation.build_summaries(corrupted)
        with self.assertRaisesRegex(AssertionError, "calibrated_period_rate"):
            isolation.validate_results(
                source, corrupted, bad_summaries, models, expected_row_count=len(source)
            )

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
            self.assertTrue((output_dir / isolation.OUTPUT_FILES["events"]).is_file())
            self.assertEqual(len(pd.read_csv(full_results)), len(results))
            self.assertEqual(len(pd.read_csv(anomalies)), int(results["period_if_anomaly_label"].sum()))
            self.assertEqual(summary_payload["row_count"], len(results))
            self.assertEqual(summary_payload["auto_anomaly_count"], int(results["period_if_auto_anomaly_label"].sum()))
            self.assertEqual(summary_payload["event_count"], len(summaries["events"]))
            self.assertEqual(json.loads(modelling_summary.read_text(encoding="utf-8"))["row_count"], len(results))


if __name__ == "__main__":
    unittest.main()
