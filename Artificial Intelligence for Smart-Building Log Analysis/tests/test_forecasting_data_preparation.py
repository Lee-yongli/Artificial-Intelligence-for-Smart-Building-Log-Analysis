import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "ml"))

from forecasting_data_preparation import (  # noqa: E402
    SENSOR_COLUMNS,
    TARGET_COLUMN,
    TARGET_TIMESTAMP_COLUMN,
    build_causal_features,
    fit_train_only_imputation,
    split_chronologically,
)


class ForecastingDataPreparationTest(unittest.TestCase):
    @staticmethod
    def _source(start: str, periods: int, zones=(1, 2)) -> pd.DataFrame:
        timestamps = pd.date_range(start, periods=periods, freq="min")
        frames = []
        for zone in zones:
            frame = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "floor": 6,
                    "zone": zone,
                }
            )
            for sensor_index, sensor in enumerate(SENSOR_COLUMNS):
                frame[sensor] = zone * 1000 + np.arange(periods) + sensor_index / 10
                frame[f"{sensor}_was_missing"] = False
            frames.append(frame)
        return pd.concat(frames, ignore_index=True).sort_values(
            ["zone", "timestamp"]
        ).reset_index(drop=True)

    def test_target_and_lags_are_zone_specific(self) -> None:
        featured, audit = build_causal_features(self._source("2018-07-01", 40))
        first_rows = featured.groupby("zone").head(1)
        self.assertTrue(first_rows["total_AC_power_kw_lag_1"].isna().all())
        self.assertEqual(audit["cross_zone_lag_rows"], 0)

        zone_one = featured.loc[featured["zone"].eq(1)].reset_index(drop=True)
        zone_two = featured.loc[featured["zone"].eq(2)].reset_index(drop=True)
        self.assertEqual(zone_one.loc[0, TARGET_COLUMN], 1015.0)
        self.assertEqual(zone_two.loc[0, TARGET_COLUMN], 2015.0)
        self.assertEqual(
            zone_one.loc[0, TARGET_TIMESTAMP_COLUMN],
            zone_one.loc[0, "timestamp"] + pd.Timedelta(minutes=15),
        )

    def test_originally_missing_future_ac_is_not_ground_truth(self) -> None:
        source = self._source("2018-07-01", 40, zones=(1,))
        source.loc[15, "total_AC_power_kw"] = 9999.0
        source.loc[15, "total_AC_power_kw_was_missing"] = True
        featured, _ = build_causal_features(source)
        self.assertTrue(pd.isna(featured.loc[0, TARGET_COLUMN]))

    def test_split_boundaries_use_prediction_and_target_time(self) -> None:
        source = self._source("2018-07-01", 63 * 24 * 60, zones=(1,))
        featured, _ = build_causal_features(source)
        splits, _ = split_chronologically(featured)
        self.assertLess(
            splits["train"][TARGET_TIMESTAMP_COLUMN].max(), pd.Timestamp("2018-08-14")
        )
        self.assertGreaterEqual(
            splits["validation"]["timestamp"].min(), pd.Timestamp("2018-08-14")
        )
        self.assertLess(
            splits["validation"][TARGET_TIMESTAMP_COLUMN].max(),
            pd.Timestamp("2018-08-24"),
        )
        self.assertGreaterEqual(
            splits["test"]["timestamp"].min(), pd.Timestamp("2018-08-24")
        )
        self.assertLess(
            splits["test"][TARGET_TIMESTAMP_COLUMN].max(), pd.Timestamp("2018-09-02")
        )

    def test_imputation_statistics_are_fitted_on_train_only(self) -> None:
        source = self._source("2018-07-01", 1500, zones=(1,))
        featured, _ = build_causal_features(source)
        train = featured.iloc[:1460].copy()
        validation = featured.iloc[1460:1480].copy()
        test = featured.iloc[1480:1500].copy()
        train.loc[train.index[:2], "temperature_c"] = np.nan
        validation["temperature_c"] = np.nan
        test["temperature_c"] = np.nan
        splits = {"train": train, "validation": validation, "test": test}

        prepared, _, _, medians = fit_train_only_imputation(splits)
        expected = float(train["temperature_c"].median())
        self.assertEqual(medians["temperature_c"], expected)
        self.assertTrue(prepared["validation"]["temperature_c"].eq(expected).all())
        self.assertTrue(prepared["test"]["temperature_c"].eq(expected).all())


if __name__ == "__main__":
    unittest.main()
