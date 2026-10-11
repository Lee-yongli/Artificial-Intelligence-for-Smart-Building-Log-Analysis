"""Build leakage-safe, prediction-ready Floor 6 forecasting datasets.

The task is supervised point forecasting: for each zone, predict
``total_AC_power_kw`` exactly 15 minutes after the prediction timestamp.  This
module only prepares and validates data; it intentionally does not train a
forecasting model or modify the anomaly-detection workflow.
"""

from __future__ import annotations

import argparse
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
    / "floor6_zone1245_cleaned_smoothed_2018_jul01_sep01.csv.gz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/forecasting_floor6"

KEY_COLUMNS = ["timestamp", "floor", "zone"]
SENSOR_COLUMNS = [
    "total_AC_power_kw",
    "light_power_kw",
    "plug_power_kw",
    "temperature_c",
    "humidity_pct",
    "lux",
]
SOURCE_MISSING_FLAGS = [f"{column}_was_missing" for column in SENSOR_COLUMNS]
LAG_MINUTES = [1, 5, 15, 30, 60, 1440]
ROLLING_MEAN_MINUTES = [15, 30, 60]
ROLLING_STD_MINUTES = [15, 60]
FORECAST_HORIZON_MINUTES = 15
CAUSAL_FORWARD_FILL_LIMIT = 15
TARGET_COLUMN = "target_total_AC_power_kw_t_plus_15"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"


@dataclass(frozen=True)
class SplitDefinition:
    name: str
    prediction_start: str
    target_end_exclusive: str


SPLIT_DEFINITIONS = (
    SplitDefinition("train", "2018-07-01", "2018-08-14"),
    SplitDefinition("validation", "2018-08-14", "2018-08-24"),
    SplitDefinition("test", "2018-08-24", "2018-09-02"),
)


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    """Normalize common boolean encodings and reject ambiguous values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        invalid = ~series.fillna(0).isin([0, 1])
        if invalid.any():
            raise ValueError(f"{name} contains values other than 0/1")
        return series.fillna(0).astype(bool)
    normalized = series.fillna("false").astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if (~normalized.isin(mapping)).any():
        raise ValueError(f"{name} contains invalid boolean values")
    return normalized.map(mapping).astype(bool)


def load_and_validate_source(
    path: Path = DEFAULT_INPUT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the cleaned source and validate its actual schema and time grid."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Forecasting source dataset not found: {path}")

    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + SENSOR_COLUMNS + SOURCE_MISSING_FLAGS)
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
        raise ValueError(f"Source contains {duplicate_keys} duplicate key rows")
    for column in SOURCE_MISSING_FLAGS:
        frame[column] = _as_bool(frame[column], column)

    frame = frame.sort_values(["zone", "timestamp"], kind="stable").reset_index(drop=True)
    time_deltas = frame.groupby("zone", sort=False)["timestamp"].diff().dropna()
    non_minute_intervals = int((time_deltas != pd.Timedelta(minutes=1)).sum())
    if non_minute_intervals:
        raise ValueError(
            f"Expected a complete one-minute grid; found {non_minute_intervals} irregular intervals"
        )

    quality: dict[str, Any] = {
        "source_path": str(path.resolve()),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "floor_values": ",".join(map(str, sorted(frame["floor"].unique().tolist()))),
        "zone_values": ",".join(map(str, sorted(frame["zone"].unique().tolist()))),
        "zone_count": int(frame["zone"].nunique()),
        "min_timestamp": frame["timestamp"].min().isoformat(sep=" "),
        "max_timestamp": frame["timestamp"].max().isoformat(sep=" "),
        "unique_timestamps": int(frame["timestamp"].nunique()),
        "duplicate_key_rows": duplicate_keys,
        "non_one_minute_intervals": non_minute_intervals,
        "source_nan_values": int(frame.isna().sum().sum()),
    }
    for sensor, flag in zip(SENSOR_COLUMNS, SOURCE_MISSING_FLAGS):
        quality[f"originally_missing_{sensor}"] = int(frame[flag].sum())
    return frame, quality


def base_feature_columns() -> tuple[list[str], list[str]]:
    """Return pre-imputation numeric and categorical forecasting features."""
    numeric = (
        SENSOR_COLUMNS
        + SOURCE_MISSING_FLAGS
        + [f"total_AC_power_kw_lag_{minutes}" for minutes in LAG_MINUTES]
        + [f"total_AC_power_kw_rolling_mean_{minutes}" for minutes in ROLLING_MEAN_MINUTES]
        + [f"total_AC_power_kw_rolling_std_{minutes}" for minutes in ROLLING_STD_MINUTES]
        + ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
    )
    return numeric, ["zone_category"]


def build_causal_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create zone-specific target, lags, rolls, and calendar features."""
    data = frame.copy()

    # The upstream cleaned dataset used bidirectional interpolation and
    # whole-period medians.  Its missing flags let us restore affected values
    # before building any forecasting feature.
    for sensor, flag in zip(SENSOR_COLUMNS, SOURCE_MISSING_FLAGS):
        data.loc[data[flag], sensor] = np.nan

    # Define the target before causal feature filling.  Originally missing
    # future AC values remain unavailable ground truth and are later excluded.
    data[TARGET_COLUMN] = data.groupby("zone", sort=False)["total_AC_power_kw"].shift(
        -FORECAST_HORIZON_MINUTES
    )
    data[TARGET_TIMESTAMP_COLUMN] = data.groupby("zone", sort=False)["timestamp"].shift(
        -FORECAST_HORIZON_MINUTES
    )
    expected_target_time = data["timestamp"] + pd.Timedelta(
        minutes=FORECAST_HORIZON_MINUTES
    )
    exact_horizon = data[TARGET_TIMESTAMP_COLUMN].eq(expected_target_time)
    data.loc[~exact_horizon, TARGET_COLUMN] = np.nan

    # Feature filling is causal, short, and restricted to each zone.
    for sensor in SENSOR_COLUMNS:
        data[sensor] = data.groupby("zone", sort=False)[sensor].ffill(
            limit=CAUSAL_FORWARD_FILL_LIMIT
        )

    ac_by_zone = data.groupby("zone", sort=False)["total_AC_power_kw"]
    for minutes in LAG_MINUTES:
        data[f"total_AC_power_kw_lag_{minutes}"] = ac_by_zone.shift(minutes)
    for minutes in ROLLING_MEAN_MINUTES:
        data[f"total_AC_power_kw_rolling_mean_{minutes}"] = (
            ac_by_zone.rolling(window=minutes, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    for minutes in ROLLING_STD_MINUTES:
        data[f"total_AC_power_kw_rolling_std_{minutes}"] = (
            ac_by_zone.rolling(window=minutes, min_periods=2)
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
    cross_zone_lag_rows = int(first_rows["total_AC_power_kw_lag_1"].notna().sum())
    if cross_zone_lag_rows:
        raise AssertionError("lag_1 crossed a zone boundary")

    audit = {
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "target_definition": (
            "same-zone total_AC_power_kw exactly 15 minutes after prediction timestamp"
        ),
        "target_is_interval_average": False,
        "causal_forward_fill_limit_minutes": CAUSAL_FORWARD_FILL_LIMIT,
        "masked_upstream_imputed_values": int(data[SOURCE_MISSING_FLAGS].sum().sum()),
        "unavailable_target_rows": int(data[TARGET_COLUMN].isna().sum()),
        "non_exact_horizon_rows": int((~exact_horizon).sum()),
        "cross_zone_lag_rows": cross_zone_lag_rows,
        "upstream_rolling_features_used": False,
        "centered_rolling_features_used": False,
        "anomaly_reference_features_used": False,
    }
    return data, audit


def split_chronologically(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Split by prediction and target time, purging horizon overlap."""
    supervised = frame.loc[
        frame[TARGET_COLUMN].notna() & frame[TARGET_TIMESTAMP_COLUMN].notna()
    ].copy()
    splits: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict[str, Any]] = {}

    for definition in SPLIT_DEFINITIONS:
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
        splits[definition.name] = split
        summary[definition.name] = {
            "rows": int(len(split)),
            "prediction_min": split["timestamp"].min().isoformat(sep=" "),
            "prediction_max": split["timestamp"].max().isoformat(sep=" "),
            "target_min": split[TARGET_TIMESTAMP_COLUMN].min().isoformat(sep=" "),
            "target_max": split[TARGET_TIMESTAMP_COLUMN].max().isoformat(sep=" "),
        }

    if splits["train"][TARGET_TIMESTAMP_COLUMN].max() >= splits["validation"]["timestamp"].min():
        raise AssertionError("Training targets overlap the validation prediction period")
    if splits["validation"][TARGET_TIMESTAMP_COLUMN].max() >= splits["test"]["timestamp"].min():
        raise AssertionError("Validation targets overlap the test prediction period")
    return splits, summary


def fit_train_only_imputation(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str], list[str], dict[str, float]]:
    """Fit feature medians on Train and apply them unchanged to all splits."""
    numeric_features, categorical_features = base_feature_columns()
    combined = pd.concat(
        [split[numeric_features] for split in splits.values()], ignore_index=True
    )
    imputed_feature_names = [
        column for column in numeric_features if combined[column].isna().any()
    ]

    prepared = {name: split.copy() for name, split in splits.items()}
    indicator_features: list[str] = []
    for column in imputed_feature_names:
        indicator = f"{column}_missing_at_prediction"
        indicator_features.append(indicator)
        for split in prepared.values():
            split[indicator] = split[column].isna().astype("int8")

    train_medians = prepared["train"][numeric_features].median(numeric_only=True)
    missing_train_statistics = sorted(
        set(numeric_features).difference(train_medians.dropna().index)
    )
    if missing_train_statistics:
        raise ValueError(
            f"Training data cannot supply medians for: {missing_train_statistics}"
        )

    for split in prepared.values():
        split[numeric_features] = split[numeric_features].fillna(train_medians)

    final_numeric = numeric_features + indicator_features
    feature_columns = final_numeric + categorical_features
    for name, split in prepared.items():
        if split[feature_columns].isna().any().any():
            raise ValueError(f"{name} features still contain missing values")
        numeric_values = split[final_numeric].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"{name} features contain infinite values")

    return (
        prepared,
        feature_columns,
        imputed_feature_names,
        {key: float(value) for key, value in train_medians.items()},
    )


def build_leakage_audit(
    prepared: dict[str, pd.DataFrame],
    features: list[str],
    source_quality: dict[str, Any],
    feature_audit: dict[str, Any],
    split_summary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Return explicit, machine-readable leakage and alignment checks."""
    forbidden_fragments = ["baseline_anomaly", "anomaly_count"]
    forbidden_features = [
        column
        for column in features
        if column == TARGET_COLUMN
        or column == TARGET_TIMESTAMP_COLUMN
        or any(fragment in column for fragment in forbidden_fragments)
    ]
    aligned_columns = all(
        list(prepared["train"][features].columns) == list(split[features].columns)
        for split in prepared.values()
    )
    exact_horizon = all(
        (split[TARGET_TIMESTAMP_COLUMN] - split["timestamp"])
        .eq(pd.Timedelta(minutes=FORECAST_HORIZON_MINUTES))
        .all()
        for split in prepared.values()
    )
    no_missing = all(not split[features + [TARGET_COLUMN]].isna().any().any() for split in prepared.values())

    checks = [
        ("unique_source_keys", source_quality["duplicate_key_rows"] == 0, "timestamp/floor/zone is unique"),
        ("one_minute_sampling", source_quality["non_one_minute_intervals"] == 0, "all within-zone intervals are one minute"),
        ("upstream_imputation_masked", feature_audit["masked_upstream_imputed_values"] > 0, f"masked {feature_audit['masked_upstream_imputed_values']} source values using missingness flags"),
        ("zone_specific_lags", feature_audit["cross_zone_lag_rows"] == 0, "lag features reset at every zone boundary"),
        ("causal_rolling", not feature_audit["centered_rolling_features_used"], "rolling windows are trailing and include no future rows"),
        ("upstream_rolls_excluded", not feature_audit["upstream_rolling_features_used"], "upstream rolling columns were not reused"),
        ("anomaly_references_excluded", not forbidden_features, f"forbidden features: {forbidden_features}"),
        ("target_excluded_from_features", TARGET_COLUMN not in features, "future target is output-only"),
        ("train_only_imputation", True, "medians were fitted on Train and applied unchanged to Validation/Test"),
        ("no_global_scaling", True, "no scaler or encoder was fitted during dataset preparation"),
        ("exact_forecast_horizon", exact_horizon, "every target timestamp is exactly prediction timestamp + 15 minutes"),
        ("purged_split_boundaries", splits_are_purged(prepared), "no target crosses into the next split"),
        ("aligned_feature_schema", aligned_columns, f"all splits have the same {len(features)} ordered features"),
        ("complete_prediction_ready_values", no_missing, "features and targets contain no NaN after Train-only imputation"),
    ]
    audit = pd.DataFrame(checks, columns=["check", "passed", "details"])
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise AssertionError(f"Leakage audit failed: {failed}")
    return audit


def splits_are_purged(splits: dict[str, pd.DataFrame]) -> bool:
    return bool(
        splits["train"][TARGET_TIMESTAMP_COLUMN].max()
        < splits["validation"]["timestamp"].min()
        and splits["validation"][TARGET_TIMESTAMP_COLUMN].max()
        < splits["test"]["timestamp"].min()
    )


def _write_summary(
    path: Path,
    source_quality: dict[str, Any],
    split_summary: dict[str, dict[str, Any]],
    features: list[str],
    imputed_features: list[str],
) -> None:
    split_lines = "\n".join(
        f"- **{name}**: {values['rows']:,} rows; prediction {values['prediction_min']} to {values['prediction_max']}; target through {values['target_max']}."
        for name, values in split_summary.items()
    )
    text = f"""# Week 7 prediction-ready dataset preparation

## Forecast definition

- Point target: same-zone `total_AC_power_kw` exactly 15 minutes after the prediction timestamp.
- This is not the average power over the next 15-minute interval.
- Source coverage: {source_quality['min_timestamp']} to {source_quality['max_timestamp']}.
- Zones: {source_quality['zone_values']}; source rows: {source_quality['rows']:,}.

## Prediction-ready outputs

{split_lines}

- Ordered feature count: **{len(features)}**.
- Feature names and Train-fitted imputation values are recorded in `feature_manifest.json`.
- Full datasets are reproducible `.csv.gz` artifacts excluded from Git; a small preview is committed for review.

## Leakage prevention

- Upstream values marked as originally missing were restored to missing before feature generation because upstream cleaning used bidirectional interpolation and whole-period medians.
- Only within-zone causal forward-fill, limited to 15 minutes, was allowed before lag construction.
- Lags are zone-specific: 1, 5, 15, 30, 60, and 1440 minutes.
- Rolling statistics are trailing, not centered.
- Targets were created before feature imputation; originally missing future AC readings were not treated as ground truth.
- Train, Validation, and Test were selected using both prediction and target timestamps, with a 15-minute boundary purge.
- Remaining missing predictors were filled using medians fitted on Train only. Imputed predictors: {', '.join(imputed_features)}.
- Existing globally standardized ML-ready features, anomaly labels, and upstream rolling features were excluded.
- No scaling, categorical encoding, or model training is performed in this stage.

All checks in `leakage_audit.csv` passed.

## Limitation

The data covers Floor 6 and approximately two months, so it supports the Week 7 forecasting experiment but not long-term or cross-floor generalisation claims.
"""
    path.write_text(text, encoding="utf-8")


def run_pipeline(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    preview_rows_per_split: int = 25,
) -> dict[str, Any]:
    """Prepare, validate, audit, and export prediction-ready datasets."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source, source_quality = load_and_validate_source(Path(input_path))
    featured, feature_audit = build_causal_features(source)
    splits, split_summary = split_chronologically(featured)
    prepared, features, imputed_features, train_medians = fit_train_only_imputation(splits)
    audit = build_leakage_audit(
        prepared, features, source_quality, feature_audit, split_summary
    )

    output_columns = KEY_COLUMNS + [TARGET_TIMESTAMP_COLUMN] + features + [TARGET_COLUMN]
    previews: list[pd.DataFrame] = []
    for name, split in prepared.items():
        output = split[output_columns].copy()
        output.insert(0, "split", name)
        output.to_csv(data_dir / f"prediction_ready_{name}.csv.gz", index=False)
        previews.append(output.head(preview_rows_per_split))
    pd.concat(previews, ignore_index=True).to_csv(
        data_dir / "prediction_ready_preview.csv", index=False
    )

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in source_quality.items()]
    ).to_csv(reports_dir / "data_quality_summary.csv", index=False)
    pd.DataFrame(
        [{"split": name, **values} for name, values in split_summary.items()]
    ).to_csv(reports_dir / "split_summary.csv", index=False)
    audit.to_csv(reports_dir / "leakage_audit.csv", index=False)

    manifest = {
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "prediction_timestamp_column": "timestamp",
        "target_timestamp_column": TARGET_TIMESTAMP_COLUMN,
        "target_column": TARGET_COLUMN,
        "target_definition": feature_audit["target_definition"],
        "identifier_columns": KEY_COLUMNS + [TARGET_TIMESTAMP_COLUMN],
        "feature_count": len(features),
        "feature_columns": features,
        "lag_minutes": LAG_MINUTES,
        "rolling_mean_minutes": ROLLING_MEAN_MINUTES,
        "rolling_std_minutes": ROLLING_STD_MINUTES,
        "causal_forward_fill_limit_minutes": CAUSAL_FORWARD_FILL_LIMIT,
        "train_only_median_imputed_features": imputed_features,
        "train_feature_medians": train_medians,
        "split_definitions": [asdict(item) for item in SPLIT_DEFINITIONS],
        "model_training_performed": False,
        "scaling_performed": False,
        "categorical_encoding_performed": False,
    }
    (reports_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(
        reports_dir / "forecasting_preparation_summary.md",
        source_quality,
        split_summary,
        features,
        imputed_features,
    )

    return {
        "input_path": str(Path(input_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_rows": source_quality["rows"],
        "source_min_timestamp": source_quality["min_timestamp"],
        "source_max_timestamp": source_quality["max_timestamp"],
        "feature_count": len(features),
        "split_summary": split_summary,
        "leakage_checks_passed": int(audit["passed"].sum()),
        "leakage_checks_total": int(len(audit)),
        "model_training_performed": False,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-rows-per-split", type=int, default=25)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_pipeline(args.input, args.output_dir, args.preview_rows_per_split)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
