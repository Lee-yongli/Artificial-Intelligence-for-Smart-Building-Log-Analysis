import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from ml_data_preparation import (  # noqa: E402
    BASELINE_LABEL_COLUMN,
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURE_COLUMNS,
    RAW_MEASUREMENT_COLUMNS,
    ROLLING_MEASUREMENT_COLUMNS,
    run_pipeline,
)


class MLDataPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "baseline_ready.csv.gz"
        self.scores_path = self.root / "v3_scores.csv.gz"
        self.output_dir = self.root / "output"
        self.source = self._make_source_data()
        self.scores = self._make_score_data(self.source)
        self.source.to_csv(self.input_path, index=False)
        self.scores.to_csv(self.scores_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _make_source_data() -> pd.DataFrame:
        timestamps = pd.to_datetime(
            [
                "2018-07-02 00:04:00",
                "2018-07-02 07:04:00",
                "2018-07-02 08:04:00",
                "2018-07-02 12:04:00",
                "2018-07-02 17:04:00",
                "2018-07-02 19:04:00",
                "2018-07-07 10:04:00",
                "2018-07-08 22:04:00",
            ]
        )
        hours = timestamps.hour
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "year": timestamps.year,
                "floor": 6,
                "zone": [1, 2, 4, 5, 1, 2, 4, 5],
                "month": timestamps.month,
                "date": timestamps.date.astype(str),
                "hour": hours,
                "minute": timestamps.minute,
                "day_of_week": timestamps.dayofweek,
                "day_name": timestamps.day_name(),
                "is_weekend": timestamps.dayofweek >= 5,
                "day_type": np.where(timestamps.dayofweek >= 5, "Weekend", "Weekday"),
                "is_work_hour": (timestamps.dayofweek < 5) & (hours >= 8) & (hours <= 17),
                "minute_of_day": hours * 60 + timestamps.minute,
            }
        )
        for index, column in enumerate(RAW_MEASUREMENT_COLUMNS, start=1):
            frame[column] = np.linspace(index, index + 3.5, len(frame))
            frame[f"{column}_was_missing"] = False
        for index, column in enumerate(ROLLING_MEASUREMENT_COLUMNS, start=1):
            frame[column] = np.linspace(index * 2, index * 2 + 7, len(frame))
        frame.loc[3, ROLLING_MEASUREMENT_COLUMNS[0]] = np.nan
        return frame

    @staticmethod
    def _make_score_data(source: pd.DataFrame) -> pd.DataFrame:
        variables = [
            "total_AC_power_kw",
            "light_power_kw",
            "plug_power_kw",
            "temperature_c",
            "humidity_pct",
            "lux",
        ]
        rows = []
        for source_index, row in source.iterrows():
            for variable_index, variable in enumerate(variables):
                rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "floor": row["floor"],
                        "zone": row["zone"],
                        "variable": variable,
                        "is_anomaly": source_index == 2 and variable_index in {0, 1},
                    }
                )
        return pd.DataFrame(rows)

    def test_pipeline_outputs_sklearn_ready_dataset(self) -> None:
        summary = run_pipeline(
            input_path=self.input_path,
            scores_path=self.scores_path,
            output_dir=self.output_dir,
            chunksize=5,
            preview_rows=3,
        )

        output_path = self.output_dir / "floor6_zone1245_ml_ready_2018_jul01_sep01.csv.gz"
        feature_list_path = self.output_dir / "feature_list.json"
        output = pd.read_csv(output_path)
        feature_list = json.loads(feature_list_path.read_text(encoding="utf-8"))
        model_features = feature_list["model_feature_columns"]

        self.assertEqual(len(output), len(self.source))
        self.assertEqual(model_features, [column for column in output.columns if column in model_features])
        self.assertNotIn(BASELINE_LABEL_COLUMN, model_features)
        self.assertFalse(set(IDENTIFIER_COLUMNS).intersection(model_features))
        self.assertEqual(int(output[BASELINE_LABEL_COLUMN].sum()), 1)
        self.assertEqual(int(output.loc[2, "baseline_anomaly_count"]), 2)
        self.assertFalse(output[model_features].isna().any().any())
        self.assertTrue(np.isfinite(output[model_features].to_numpy()).all())
        self.assertTrue(any(name.startswith("categorical__zone_") for name in model_features))
        self.assertTrue(any(name.startswith("categorical__hour_") for name in model_features))
        self.assertTrue(any(name.startswith("categorical__operating_period_") for name in model_features))

        scaled_features = feature_list["scaled_numerical_features"]
        np.testing.assert_allclose(output[scaled_features].mean().to_numpy(), 0, atol=1e-6)
        self.assertEqual(len(scaled_features), len(NUMERICAL_FEATURE_COLUMNS))
        self.assertEqual(summary["feature_nan_count"], 0)
        self.assertEqual(summary["feature_infinite_count"], 0)
        self.assertEqual(summary["ml_ready_nan_count"], 0)
        self.assertEqual(summary["ml_ready_infinite_count"], 0)
        self.assertEqual(summary["duplicate_rows"], 0)
        self.assertTrue(summary["model_features_are_numeric"])
        self.assertTrue(summary["output_reload_check"])

    def test_duplicate_source_identifiers_are_rejected(self) -> None:
        duplicate_source = pd.concat([self.source, self.source.iloc[[0]]], ignore_index=True)
        duplicate_source.to_csv(self.input_path, index=False)

        with self.assertRaisesRegex(ValueError, "duplicate identifier"):
            run_pipeline(
                input_path=self.input_path,
                scores_path=self.scores_path,
                output_dir=self.output_dir,
            )


if __name__ == "__main__":
    unittest.main()
