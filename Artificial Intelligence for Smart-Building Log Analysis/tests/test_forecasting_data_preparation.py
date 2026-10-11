import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "ml"))

from forecasting_data_preparation import (  # noqa: E402
    DEFAULT_INPUT,
    MISSING_FLAG_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
    assign_splits,
    build_causal_features,
    feature_columns,
    fit_and_apply_training_imputation,
    validate_prediction_ready,
)


class ForecastingDataPreparationTest(unittest.TestCase):
    @staticmethod
    def _frame(start: str, periods: int = 35) -> pd.DataFrame:
        timestamps = pd.date_range(start, periods=periods, freq="min")
        rows = []
        for zone, offset in [(1, 0.0), (2, 1000.0)]:
            for index, timestamp in enumerate(timestamps):
                row = {"timestamp": timestamp, "floor": 6, "zone": zone}
                for sensor_index, sensor in enumerate(SENSOR_COLUMNS):
                    row[sensor] = offset + index + sensor_index / 10
                    row[f"{sensor}_was_missing"] = False
                rows.append(row)
        return pd.DataFrame(rows).sort_values(["zone", "timestamp"]).reset_index(drop=True)

    def test_default_input_is_eda_notebook_final_dataset(self) -> None:
        self.assertEqual(
            DEFAULT_INPUT.name,
            "floor6_zone1245_cleaned_features_2018_jul01_sep01.csv.gz",
        )
        self.assertEqual(
            DEFAULT_INPUT.parent.name,
            "floor6_zone1245_2018_jul01_sep01_total_ac_eda",
        )

    def test_target_and_lags_never_cross_zone_boundaries(self) -> None:
        featured, audit = build_causal_features(self._frame("2018-07-01"))
        first_by_zone = featured.groupby("zone").head(1)
        self.assertTrue(first_by_zone["total_AC_power_kw_lag_1"].isna().all())
        zone_one = featured.loc[featured["zone"].eq(1)].reset_index(drop=True)
        zone_two = featured.loc[featured["zone"].eq(2)].reset_index(drop=True)
        self.assertEqual(zone_one.loc[0, TARGET_COLUMN], 15.0)
        self.assertEqual(zone_two.loc[0, TARGET_COLUMN], 1015.0)
        self.assertFalse(audit["target_is_interval_average"])

    def test_originally_missing_target_is_not_ground_truth(self) -> None:
        frame = self._frame("2018-07-01")
        target_row = frame.index[
            frame["zone"].eq(1)
            & frame["timestamp"].eq(pd.Timestamp("2018-07-01 00:15"))
        ][0]
        frame.loc[target_row, "total_AC_power_kw_was_missing"] = True
        frame.loc[target_row, "total_AC_power_kw"] = 9999.0
        featured, _ = build_causal_features(frame)
        prediction = featured.loc[
            featured["zone"].eq(1)
            & featured["timestamp"].eq(pd.Timestamp("2018-07-01 00:00"))
        ].iloc[0]
        self.assertTrue(np.isnan(prediction[TARGET_COLUMN]))

    def test_split_targets_remain_inside_their_period(self) -> None:
        periods = 65 * 24 * 60
        featured, _ = build_causal_features(self._frame("2018-07-01", periods))
        splits, _ = assign_splits(featured)
        self.assertLess(splits["train"]["target_timestamp"].max(), pd.Timestamp("2018-08-14"))
        self.assertGreaterEqual(
            splits["validation"]["target_timestamp"].min(), pd.Timestamp("2018-08-14")
        )
        self.assertLess(
            splits["validation"]["target_timestamp"].max(), pd.Timestamp("2018-08-24")
        )
        self.assertGreaterEqual(
            splits["test"]["target_timestamp"].min(), pd.Timestamp("2018-08-24")
        )

    def test_imputation_statistics_come_from_training_only(self) -> None:
        featured, _ = build_causal_features(self._frame("2018-07-01", 65 * 24 * 60))
        splits, _ = assign_splits(featured)
        validation_index = splits["validation"].index[0]
        splits["validation"].loc[validation_index, "temperature_c"] = np.nan
        prepared, medians, audit = fit_and_apply_training_imputation(splits)
        train_median = splits["train"]["temperature_c"].median()
        saved_median = medians.set_index("feature").loc["temperature_c", "training_median"]
        self.assertEqual(saved_median, train_median)
        self.assertEqual(prepared["validation"].loc[validation_index, "temperature_c"], train_median)
        self.assertEqual(audit["imputer_fit_scope"], "training split only")

    def test_prediction_ready_features_are_complete_and_safe(self) -> None:
        featured, _ = build_causal_features(self._frame("2018-07-01", 65 * 24 * 60))
        splits, _ = assign_splits(featured)
        prepared, _, _ = fit_and_apply_training_imputation(splits)
        audit = validate_prediction_ready(prepared)
        numeric, categorical = feature_columns()
        self.assertEqual(audit["feature_count"], len(numeric) + len(categorical))
        self.assertEqual(audit["missing_output_feature_values"], 0)
        self.assertFalse(audit["forbidden_feature_columns_present"])

    def test_missing_flag_columns_match_sensor_columns(self) -> None:
        self.assertEqual(
            MISSING_FLAG_COLUMNS,
            [f"{column}_was_missing" for column in SENSOR_COLUMNS],
        )


if __name__ == "__main__":
    unittest.main()
