"""Prepare leakage-safe Week 7 forecasting datasets for Floor 6.

The workflow uses the final cleaned dataset produced by
``Data_pre/Floor6_Zone1245_Data_Preparation_EDA_executed.ipynb``. It creates
strictly causal features, defines a same-zone future 15-minute average target,
performs purged chronological splits, fits median imputation on training data
only, and exports prediction-ready datasets plus a leakage audit.

No forecasting model is trained by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_DIR
    / "outputs/floor6_zone1245_2018_jul01_sep01_total_ac_eda"
    / "floor6_zone1245_cleaned_features_2018_jul01_sep01.csv.gz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/week7_prediction_ready_dataset"

KEY_COLUMNS = ["timestamp", "floor", "zone"]
SENSOR_COLUMNS = [
    "total_AC_power_kw",
    "light_power_kw",
    "plug_power_kw",
    "temperature_c",
    "humidity_pct",
    "lux",
]
MISSING_FLAG_COLUMNS = [f"{column}_was_missing" for column in SENSOR_COLUMNS]
LAG_MINUTES = [1, 5, 15, 30, 60, 1440]
ROLLING_MEAN_MINUTES = [15, 30, 60]
ROLLING_STD_MINUTES = [15, 60]
FORECAST_HORIZON_MINUTES = 15
CAUSAL_FORWARD_FILL_LIMIT = 15
TARGET_COLUMN = "target_total_AC_power_kw_next_15min_mean"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"
FORBIDDEN_FEATURE_COLUMNS = {
    "baseline_anomaly_label",
    "baseline_anomaly_count",
    "ac_state",
    "transition_type",
    "ac_off_threshold_kw",
}


@dataclass(frozen=True)
class SplitDefinition:
    name: str
    prediction_start: str
    target_end_exclusive: str


SPLITS = (
    SplitDefinition("train", "2018-07-01", "2018-08-14"),
    SplitDefinition("validation", "2018-08-14", "2018-08-24"),
    SplitDefinition("test", "2018-08-24", "2018-09-02"),
)


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    """Convert common boolean encodings while rejecting ambiguous values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        invalid = ~series.fillna(0).isin([0, 1])
        if invalid.any():
            raise ValueError(f"{column} contains values other than 0/1")
        return series.fillna(0).astype(bool)
    normalized = series.fillna("false").astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise ValueError(f"{column} contains invalid boolean values")
    return normalized.map(mapping).astype(bool)


def load_source(path: Path = DEFAULT_INPUT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate, sort, and audit the EDA notebook's final dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Forecasting source dataset not found: {path}")

    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + SENSOR_COLUMNS + MISSING_FLAG_COLUMNS)
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Source dataset is missing required columns: {missing_columns}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_timestamps = int(frame["timestamp"].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Source contains {invalid_timestamps} invalid timestamps")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError("Source identifier columns contain missing values")

    duplicate_keys = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_keys:
        raise ValueError(f"Source contains {duplicate_keys} duplicate timestamp/floor/zone keys")

    for column in MISSING_FLAG_COLUMNS:
        frame[column] = _as_bool(frame[column], column)

    frame = frame.sort_values(["zone", "timestamp"], kind="stable").reset_index(drop=True)
    deltas = frame.groupby("zone", sort=False)["timestamp"].diff().dropna()
    non_minute_intervals = int((deltas != pd.Timedelta(minutes=1)).sum())
    zones = sorted(frame["zone"].unique().tolist())

    resolved_path = path.resolve()
    try:
        report_path = resolved_path.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        report_path = str(resolved_path)

    quality: dict[str, Any] = {
        "source_path": report_path,
        "source_file": path.name,
        "source_notebook": "Data_pre/Floor6_Zone1245_Data_Preparation_EDA_executed.ipynb",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "floor_values": ",".join(map(str, sorted(frame["floor"].unique().tolist()))),
        "zone_values": ",".join(map(str, zones)),
        "zone_count": len(zones),
        "min_timestamp": frame["timestamp"].min().isoformat(sep=" "),
        "max_timestamp": frame["timestamp"].max().isoformat(sep=" "),
        "unique_timestamps": int(frame["timestamp"].nunique()),
        "duplicate_key_rows": duplicate_keys,
        "non_one_minute_intervals": non_minute_intervals,
        "sampling_frequency": "1 minute" if non_minute_intervals == 0 else "irregular",
        "source_sensor_nan_values_before_restoring_flags": int(
            frame[SENSOR_COLUMNS].isna().sum().sum()
        ),
    }
    for column, flag in zip(SENSOR_COLUMNS, MISSING_FLAG_COLUMNS):
        quality[f"originally_missing_{column}"] = int(frame[flag].sum())
    return frame, quality


def feature_columns() -> tuple[list[str], list[str]]:
    """Return logical numeric and categorical feature columns."""
    numeric = (
        SENSOR_COLUMNS
        + MISSING_FLAG_COLUMNS
        + [f"total_AC_power_kw_lag_{minutes}" for minutes in LAG_MINUTES]
        + [f"total_AC_power_kw_rolling_mean_{minutes}" for minutes in ROLLING_MEAN_MINUTES]
        + [f"total_AC_power_kw_rolling_std_{minutes}" for minutes in ROLLING_STD_MINUTES]
        + ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
    )
    return numeric, ["zone_category"]


def build_causal_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create target and predictors using only information available at time t."""
    data = frame.copy()

    # The EDA source used bidirectional interpolation and whole-period medians.
    # Restore every flagged measurement to missing before target/feature creation.
    for column, flag in zip(SENSOR_COLUMNS, MISSING_FLAG_COLUMNS):
        data.loc[data[flag], column] = np.nan

    zone_ac = data.groupby("zone", sort=False)["total_AC_power_kw"]
    future_values = [zone_ac.shift(-offset) for offset in range(1, FORECAST_HORIZON_MINUTES + 1)]
    future_window = pd.concat(future_values, axis=1)
    data[TARGET_COLUMN] = future_window.mean(axis=1, skipna=False)
    data[TARGET_TIMESTAMP_COLUMN] = data.groupby("zone", sort=False)["timestamp"].shift(
        -FORECAST_HORIZON_MINUTES
    )
    expected_target_timestamp = data["timestamp"] + pd.Timedelta(
        minutes=FORECAST_HORIZON_MINUTES
    )
    exact_horizon = data[TARGET_TIMESTAMP_COLUMN].eq(expected_target_timestamp)
    data.loc[~exact_horizon, TARGET_COLUMN] = np.nan

    # Forward-fill only short gaps and never across a zone boundary. Any
    # remaining feature gaps are filled later with training-only medians.
    for column in SENSOR_COLUMNS:
        data[column] = data.groupby("zone", sort=False)[column].ffill(
            limit=CAUSAL_FORWARD_FILL_LIMIT
        )

    zone_ac = data.groupby("zone", sort=False)["total_AC_power_kw"]
    for minutes in LAG_MINUTES:
        data[f"total_AC_power_kw_lag_{minutes}"] = zone_ac.shift(minutes)
    for minutes in ROLLING_MEAN_MINUTES:
        data[f"total_AC_power_kw_rolling_mean_{minutes}"] = (
            zone_ac.rolling(window=minutes, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    for minutes in ROLLING_STD_MINUTES:
        data[f"total_AC_power_kw_rolling_std_{minutes}"] = (
            zone_ac.rolling(window=minutes, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

    data["hour"] = data["timestamp"].dt.hour.astype("int8")
    data["day_of_week"] = data["timestamp"].dt.dayofweek.astype("int8")
    data["is_weekend"] = (data["day_of_week"] >= 5).astype("int8")
    minute_of_day = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    radians = 2 * np.pi * minute_of_day / (24 * 60)
    data["hour_sin"] = np.sin(radians)
    data["hour_cos"] = np.cos(radians)
    data["zone_category"] = data["zone"].astype(str)

    first_rows = data.groupby("zone", sort=False).head(1)
    if not first_rows["total_AC_power_kw_lag_1"].isna().all():
        raise AssertionError("lag_1 crossed a zone boundary")

    audit = {
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "target_definition": (
            "same-zone mean total_AC_power_kw from t+1 to t+15 after prediction timestamp"
        ),
        "target_is_interval_average": True,
        "target_window_start_offset_minutes": 1,
        "target_window_end_offset_minutes": FORECAST_HORIZON_MINUTES,
        "causal_forward_fill_limit_minutes": CAUSAL_FORWARD_FILL_LIMIT,
        "invalid_or_unavailable_target_rows": int(data[TARGET_COLUMN].isna().sum()),
        "non_exact_horizon_rows": int((~exact_horizon).sum()),
        "source_rolling_columns_used": False,
        "source_operating_state_columns_used": False,
        "anomaly_reference_columns_used": False,
        "lag_features_zone_specific": True,
        "rolling_windows_centered": False,
    }
    return data, audit


def assign_splits(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Create chronological splits based on both prediction and target time."""
    supervised = frame.loc[
        frame[TARGET_COLUMN].notna() & frame[TARGET_TIMESTAMP_COLUMN].notna()
    ].copy()
    result: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {}

    for definition in SPLITS:
        start = pd.Timestamp(definition.prediction_start)
        end = pd.Timestamp(definition.target_end_exclusive)
        mask = (
            supervised["timestamp"].ge(start)
            & supervised[TARGET_TIMESTAMP_COLUMN].ge(start)
            & supervised[TARGET_TIMESTAMP_COLUMN].lt(end)
        )
        split = supervised.loc[mask].copy()
        if split.empty:
            raise ValueError(f"Chronological split {definition.name!r} is empty")
        result[definition.name] = split
        summary[definition.name] = {
            "rows": int(len(split)),
            "prediction_min": split["timestamp"].min().isoformat(sep=" "),
            "prediction_max": split["timestamp"].max().isoformat(sep=" "),
            "target_min": split[TARGET_TIMESTAMP_COLUMN].min().isoformat(sep=" "),
            "target_max": split[TARGET_TIMESTAMP_COLUMN].max().isoformat(sep=" "),
        }

    if result["train"][TARGET_TIMESTAMP_COLUMN].max() >= result["validation"]["timestamp"].min():
        raise AssertionError("Training targets overlap the validation prediction period")
    if result["validation"][TARGET_TIMESTAMP_COLUMN].max() >= result["test"]["timestamp"].min():
        raise AssertionError("Validation targets overlap the test prediction period")
    return result, summary


def fit_and_apply_training_imputation(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Fit feature medians on training only and apply them to every split."""
    numeric, categorical = feature_columns()
    train_medians = splits["train"][numeric].median(numeric_only=True)
    unavailable = sorted(set(numeric).difference(train_medians.index))
    unavailable += sorted(train_medians.index[train_medians.isna()].tolist())
    if unavailable:
        raise ValueError(f"Training medians unavailable for features: {sorted(set(unavailable))}")

    prepared: dict[str, pd.DataFrame] = {}
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    for name, split in splits.items():
        output = split.copy()
        before[name] = int(output[numeric + categorical].isna().sum().sum())
        output[numeric] = output[numeric].fillna(train_medians)
        after[name] = int(output[numeric + categorical].isna().sum().sum())
        if after[name]:
            raise AssertionError(f"{name} retains missing model feature values")
        prepared[name] = output

    median_table = train_medians.rename("training_median").rename_axis("feature").reset_index()
    audit = {
        "imputer_fit_scope": "training split only",
        "imputation_strategy": "per-feature training median after limited causal forward-fill",
        "feature_missing_before_imputation": before,
        "feature_missing_after_imputation": after,
        "scaling_applied": False,
    }
    return prepared, median_table, audit


def validate_prediction_ready(
    splits: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Run runtime assertions for alignment, ordering, leakage, and completeness."""
    numeric, categorical = feature_columns()
    features = numeric + categorical
    forbidden_present = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(features))
    anomaly_present = sorted(column for column in features if column.startswith("baseline_anomaly"))
    if forbidden_present or anomaly_present:
        raise AssertionError(f"Forbidden forecasting features found: {forbidden_present + anomaly_present}")

    checks: dict[str, Any] = {
        "feature_count": len(features),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "forbidden_feature_columns_present": False,
        "anomaly_reference_columns_present": False,
        "all_target_window_end_timestamps_exactly_15_minutes_ahead": True,
        "all_splits_sorted_within_zone": True,
        "duplicate_output_keys": 0,
        "missing_output_feature_values": 0,
    }

    for name, split in splits.items():
        horizon = split[TARGET_TIMESTAMP_COLUMN] - split["timestamp"]
        if not horizon.eq(pd.Timedelta(minutes=FORECAST_HORIZON_MINUTES)).all():
            raise AssertionError(f"{name} contains a target window end that is not exactly 15 minutes ahead")
        if not all(group["timestamp"].is_monotonic_increasing for _, group in split.groupby("zone")):
            raise AssertionError(f"{name} is not chronological within each zone")
        duplicates = int(split.duplicated(KEY_COLUMNS).sum())
        missing_features = int(split[features].isna().sum().sum())
        checks["duplicate_output_keys"] += duplicates
        checks["missing_output_feature_values"] += missing_features

    if checks["duplicate_output_keys"] or checks["missing_output_feature_values"]:
        raise AssertionError(f"Prediction-ready validation failed: {checks}")
    return checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_prediction_ready(
    splits: dict[str, pd.DataFrame], data_dir: Path
) -> pd.DataFrame:
    """Write compressed prediction-ready splits and return a file manifest."""
    numeric, categorical = feature_columns()
    columns = (
        ["timestamp", TARGET_TIMESTAMP_COLUMN, "floor", "zone"]
        + numeric
        + categorical
        + [TARGET_COLUMN]
    )
    rows: list[dict[str, Any]] = []
    for name, split in splits.items():
        path = data_dir / f"prediction_ready_{name}.csv.gz"
        split[columns].to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        rows.append(
            {
                "split": name,
                "file": path.name,
                "rows": int(len(split)),
                "columns": len(columns),
                "feature_count": len(numeric) + len(categorical),
                "prediction_min": split["timestamp"].min().isoformat(sep=" "),
                "prediction_max": split["timestamp"].max().isoformat(sep=" "),
                "target_min": split[TARGET_TIMESTAMP_COLUMN].min().isoformat(sep=" "),
                "target_max": split[TARGET_TIMESTAMP_COLUMN].max().isoformat(sep=" "),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _audit_table(*sections: tuple[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section, values in sections:
        for key, value in values.items():
            rows.append(
                {
                    "section": section,
                    "metric": key,
                    "value": json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value,
                }
            )
    return pd.DataFrame(rows)


def _write_leakage_report(
    path: Path,
    source_quality: dict[str, Any],
    feature_audit: dict[str, Any],
    split_summary: dict[str, Any],
    imputation_audit: dict[str, Any],
    validation_audit: dict[str, Any],
) -> None:
    split_lines = "\n".join(
        f"- **{name}**: {values['rows']:,} rows; prediction {values['prediction_min']} to "
        f"{values['prediction_max']}; targets {values['target_min']} to {values['target_max']}"
        for name, values in split_summary.items()
    )
    text = f"""# Week 7 prediction-ready dataset leakage audit

## Source and forecast definition

- Source: `{source_quality['source_file']}`, the final main dataset written by `{source_quality['source_notebook']}`.
- Source rows: {source_quality['rows']:,}; zones: {source_quality['zone_values']}; coverage: {source_quality['min_timestamp']} to {source_quality['max_timestamp']}.
- Target: same-zone mean `total_AC_power_kw` from t+1 to t+{FORECAST_HORIZON_MINUTES} after the prediction timestamp.
- The target is an interval-average forecast over the next 15 minutes, not a single future-minute point value.
- Logical feature count: {validation_audit['feature_count']}.

## Chronological splits

{split_lines}

Both prediction timestamps and target-window end timestamps must be inside the same split. The final 15 prediction minutes before each split boundary are therefore purged.

## Leakage controls

| Check | Result |
| --- | --- |
| Target and lags calculated within zone | PASS |
| Every retained target-window end exactly 15 minutes ahead | PASS |
| Trailing, non-centred rolling windows | PASS |
| Source bidirectional/global imputations masked using `_was_missing` flags | PASS |
| Forward-fill causal, within zone, limited to {CAUSAL_FORWARD_FILL_LIMIT} minutes | PASS |
| Remaining medians fitted on training only | PASS |
| Source rolling and operating-state columns excluded | PASS |
| Anomaly reference columns excluded | PASS |
| Split target overlap prevented | PASS |
| Output feature values complete after train-only imputation | PASS |
| Duplicate output keys | {validation_audit['duplicate_output_keys']} |

Feature missing values before train-only median imputation: `{json.dumps(imputation_audit['feature_missing_before_imputation'], sort_keys=True)}`. After imputation: `{json.dumps(imputation_audit['feature_missing_after_imputation'], sort_keys=True)}`.

## Exclusions and limitations

- Unavailable target rows: {feature_audit['invalid_or_unavailable_target_rows']:,}.
- No scaler was fitted during data preparation; any later model-specific scaler must be fitted on the training split only.
- Leakage safety after source cleaning depends on the correctness of the EDA notebook's preserved `_was_missing` flags. All flagged imputed sensor values were masked before target and feature construction.
- The data covers one floor and approximately two months, so it cannot establish seasonal or cross-floor generalisation.
- No forecasting model was trained or evaluated in this stage.
"""
    path.write_text(text, encoding="utf-8")


def run_preparation(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Run preparation, validation, and artifact export without model training."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source, source_quality = load_source(Path(input_path))
    featured, feature_audit = build_causal_features(source)
    raw_splits, split_summary = assign_splits(featured)
    prepared_splits, median_table, imputation_audit = fit_and_apply_training_imputation(
        raw_splits
    )
    validation_audit = validate_prediction_ready(prepared_splits)
    manifest = save_prediction_ready(prepared_splits, data_dir)

    feature_audit["final_usable_rows_across_splits"] = int(
        sum(len(split) for split in prepared_splits.values())
    )
    feature_audit["excluded_rows_total"] = int(
        len(featured) - feature_audit["final_usable_rows_across_splits"]
    )

    median_table.to_csv(reports_dir / "training_feature_medians.csv", index=False)
    manifest.to_csv(reports_dir / "prediction_ready_manifest.csv", index=False)
    _audit_table(
        ("source", source_quality),
        ("feature_engineering", feature_audit),
        ("imputation", imputation_audit),
        ("validation", validation_audit),
        *[(f"split_{name}", values) for name, values in split_summary.items()],
    ).to_csv(reports_dir / "leakage_audit.csv", index=False)

    numeric, categorical = feature_columns()
    config = {
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "target_definition": "same-zone mean total_AC_power_kw from t+1 to t+15 after prediction timestamp",
        "target_is_interval_average": True,
        "target_window_start_offset_minutes": 1,
        "target_window_end_offset_minutes": FORECAST_HORIZON_MINUTES,
        "causal_forward_fill_limit": CAUSAL_FORWARD_FILL_LIMIT,
        "lag_minutes": LAG_MINUTES,
        "rolling_mean_minutes": ROLLING_MEAN_MINUTES,
        "rolling_std_minutes": ROLLING_STD_MINUTES,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": TARGET_COLUMN,
        "splits": [asdict(item) for item in SPLITS],
        "model_training_performed": False,
    }
    (reports_dir / "preparation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    _write_leakage_report(
        reports_dir / "leakage_audit.md",
        source_quality,
        feature_audit,
        split_summary,
        imputation_audit,
        validation_audit,
    )

    return {
        "source_quality": source_quality,
        "feature_audit": feature_audit,
        "split_summary": split_summary,
        "imputation_audit": imputation_audit,
        "validation_audit": validation_audit,
        "manifest": manifest.to_dict(orient="records"),
        "output_dir": str(output_dir.resolve()),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_preparation(args.input, args.output_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
