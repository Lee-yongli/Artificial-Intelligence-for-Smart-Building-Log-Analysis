"""Build the shared ML-ready dataset for Week 6 anomaly detection work."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DATASET = (
    PROJECT_DIR
    / "outputs/floor6_zone1245_2018_jul01_sep01_total_ac_eda"
    / "floor6_zone1245_baseline_ready_2018_jul01_sep01.csv.gz"
)
DEFAULT_BASELINE_SCORES = (
    PROJECT_DIR
    / "outputs/floor6_zone1245_2018_jul01_sep01_statistical_baseline_v3_15min_operation_aware"
    / "baseline_tables/09_baseline_anomaly_scores_long.csv.gz"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs/floor6_zone1245_ml_ready"

KEY_COLUMNS = ["timestamp", "floor", "zone"]
IDENTIFIER_COLUMNS = ["timestamp", "floor", "zone"]
RAW_MEASUREMENT_COLUMNS = [
    "total_AC_power_kw",
    "light_power_kw",
    "plug_power_kw",
    "temperature_c",
    "humidity_pct",
    "lux",
]
MISSING_FLAG_COLUMNS = [f"{column}_was_missing" for column in RAW_MEASUREMENT_COLUMNS]
ROLLING_MEASUREMENT_COLUMNS = [
    "total_AC_power_kw_rolling_mean_15min",
    "light_power_kw_rolling_mean_15min",
    "plug_power_kw_rolling_mean_15min",
    "temperature_c_rolling_mean_15min",
    "humidity_pct_rolling_mean_15min",
    "lux_rolling_mean_15min",
]
ENGINEERED_NUMERICAL_COLUMNS = ["minute_of_day_sin", "minute_of_day_cos"]
NUMERICAL_FEATURE_COLUMNS = ROLLING_MEASUREMENT_COLUMNS + ENGINEERED_NUMERICAL_COLUMNS
CATEGORICAL_FEATURE_COLUMNS = ["zone", "hour", "operating_period"]
REFERENCE_COLUMNS = [
    "date",
    "day_type",
    "hour",
    "minute",
    "minute_of_day",
    "is_weekend",
    "is_work_hour",
    "operating_period",
] + RAW_MEASUREMENT_COLUMNS + MISSING_FLAG_COLUMNS
BASELINE_LABEL_COLUMN = "baseline_anomaly_label"
BASELINE_COUNT_COLUMN = "baseline_anomaly_count"
RANDOM_STATE = 42


def assign_operating_period(frame: pd.DataFrame) -> pd.Series:
    """Apply the same operating-period definition used by the v3 baseline."""
    conditions = [
        frame["is_weekend"].astype(bool),
        frame["hour"].between(0, 6),
        frame["hour"].eq(7),
        frame["hour"].between(8, 16),
        frame["hour"].between(17, 18),
        frame["hour"].between(19, 23),
    ]
    labels = [
        "weekend",
        "night_off",
        "ramp_up",
        "core_work",
        "ramp_down",
        "evening_off",
    ]
    return pd.Series(np.select(conditions, labels, default="unknown"), index=frame.index)


def build_preprocessor() -> ColumnTransformer:
    """Create the reusable sklearn preprocessing pipeline."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )


def _required_input_columns() -> List[str]:
    return sorted(
        set(
            KEY_COLUMNS
            + [
                "date",
                "day_type",
                "hour",
                "minute",
                "minute_of_day",
                "is_weekend",
                "is_work_hour",
            ]
            + RAW_MEASUREMENT_COLUMNS
            + MISSING_FLAG_COLUMNS
            + ROLLING_MEASUREMENT_COLUMNS
        )
    )


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def load_feature_source(input_path: Path) -> pd.DataFrame:
    """Load baseline-ready data and add deterministic time features."""
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    _validate_columns(frame, _required_input_columns(), input_path.name)

    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError("Identifier columns contain missing values")
    duplicate_keys = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_keys:
        raise ValueError(f"Input dataset contains {duplicate_keys} duplicate identifier rows")

    frame["operating_period"] = assign_operating_period(frame)
    radians = 2 * np.pi * frame["minute_of_day"].astype(float) / (24 * 60)
    frame["minute_of_day_sin"] = np.sin(radians)
    frame["minute_of_day_cos"] = np.cos(radians)
    return frame


def _coerce_anomaly_label(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        invalid = ~values.isin([0, 1])
        if invalid.any():
            raise ValueError("Baseline is_anomaly contains values other than 0 and 1")
        return values.astype(bool)

    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].unique().tolist())[:5]
        raise ValueError(f"Baseline is_anomaly contains invalid values: {examples}")
    return normalized.map(mapping).astype(bool)


def load_baseline_labels(scores_path: Path, chunksize: int = 250_000) -> pd.DataFrame:
    """Aggregate long v3 anomaly points to one reference label per source row."""
    usecols = KEY_COLUMNS + ["variable", "is_anomaly"]
    partials: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        scores_path,
        usecols=usecols,
        parse_dates=["timestamp"],
        chunksize=chunksize,
    ):
        chunk["is_anomaly"] = _coerce_anomaly_label(chunk["is_anomaly"])
        partials.append(
            chunk.groupby(KEY_COLUMNS, as_index=False).agg(
                **{
                    BASELINE_LABEL_COLUMN: ("is_anomaly", "max"),
                    BASELINE_COUNT_COLUMN: ("is_anomaly", "sum"),
                }
            )
        )

    if not partials:
        raise ValueError(f"Baseline score dataset is empty: {scores_path}")

    labels = pd.concat(partials, ignore_index=True)
    labels = labels.groupby(KEY_COLUMNS, as_index=False).agg(
        **{
            BASELINE_LABEL_COLUMN: (BASELINE_LABEL_COLUMN, "max"),
            BASELINE_COUNT_COLUMN: (BASELINE_COUNT_COLUMN, "sum"),
        }
    )
    labels[BASELINE_LABEL_COLUMN] = labels[BASELINE_LABEL_COLUMN].astype("int8")
    labels[BASELINE_COUNT_COLUMN] = labels[BASELINE_COUNT_COLUMN].astype("int8")
    return labels


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_groups(feature_names: Sequence[str]) -> Dict[str, List[str]]:
    names = list(feature_names)
    return {
        "scaled_numerical_features": [name for name in names if name.startswith("numerical__")],
        "encoded_categorical_features": [name for name in names if name.startswith("categorical__")],
    }


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_DATASET,
    scores_path: Path = DEFAULT_BASELINE_SCORES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunksize: int = 250_000,
    preview_rows: int = 1_000,
) -> Dict[str, Any]:
    """Generate the ML-ready dataset and its reproducibility metadata."""
    input_path = Path(input_path)
    scores_path = Path(scores_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_feature_source(input_path)
    original_rows = len(source)
    source_missing_values = int(source.isna().sum().sum())
    labels = load_baseline_labels(scores_path, chunksize=chunksize)

    merged = source.merge(labels, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if merged[[BASELINE_LABEL_COLUMN, BASELINE_COUNT_COLUMN]].isna().any().any():
        missing_labels = int(merged[BASELINE_LABEL_COLUMN].isna().sum())
        raise ValueError(f"Baseline labels are missing for {missing_labels} source rows")

    preprocessor = build_preprocessor()
    feature_matrix = preprocessor.fit_transform(merged)
    feature_names = preprocessor.get_feature_names_out().tolist()
    feature_frame = pd.DataFrame(feature_matrix, columns=feature_names, index=merged.index).astype(np.float32)
    groups = _feature_groups(feature_names)

    if BASELINE_LABEL_COLUMN in feature_names or BASELINE_COUNT_COLUMN in feature_names:
        raise AssertionError("Baseline reference label leaked into the model feature matrix")
    if set(IDENTIFIER_COLUMNS).intersection(feature_names):
        raise AssertionError("Raw identifier columns leaked into the model feature matrix")
    if feature_frame.isna().any().any():
        raise ValueError("ML feature matrix contains NaN values after preprocessing")
    if not np.isfinite(feature_frame.to_numpy()).all():
        raise ValueError("ML feature matrix contains infinite values")

    output_columns = IDENTIFIER_COLUMNS + REFERENCE_COLUMNS
    ml_ready = pd.concat(
        [
            merged[output_columns].reset_index(drop=True),
            feature_frame.reset_index(drop=True),
            merged[[BASELINE_LABEL_COLUMN, BASELINE_COUNT_COLUMN]].astype("int8").reset_index(drop=True),
        ],
        axis=1,
    )
    ml_ready_numeric = ml_ready.select_dtypes(include=[np.number])
    ml_ready_nan_count = int(ml_ready.isna().sum().sum())
    ml_ready_infinite_count = int(np.isinf(ml_ready_numeric.to_numpy()).sum())
    duplicate_rows = int(ml_ready.duplicated().sum())

    output_dataset = output_dir / "floor6_zone1245_ml_ready_2018_jul01_sep01.csv.gz"
    temporary_dataset = output_dir / ".floor6_zone1245_ml_ready_2018_jul01_sep01.tmp.csv.gz"
    ml_ready.to_csv(temporary_dataset, index=False)
    temporary_dataset.replace(output_dataset)
    ml_ready.head(preview_rows).to_csv(output_dir / "ml_ready_preview.csv", index=False)

    pipeline_path = output_dir / "preprocessing_pipeline.pkl"
    with pipeline_path.open("wb") as pipeline_file:
        pickle.dump(preprocessor, pipeline_file)

    scaled_columns = groups["scaled_numerical_features"]
    scaled_means = feature_frame[scaled_columns].mean()
    scaled_stds = feature_frame[scaled_columns].std(ddof=0)
    label_counts = ml_ready[BASELINE_LABEL_COLUMN].value_counts().sort_index()

    feature_manifest: Dict[str, Any] = {
        "identifier_columns": IDENTIFIER_COLUMNS,
        "raw_reference_columns": REFERENCE_COLUMNS,
        "numerical_source_features": NUMERICAL_FEATURE_COLUMNS,
        "categorical_source_features": CATEGORICAL_FEATURE_COLUMNS,
        **groups,
        "model_feature_columns": feature_names,
        "baseline_label_column": BASELINE_LABEL_COLUMN,
        "baseline_count_column": BASELINE_COUNT_COLUMN,
        "baseline_label_is_model_feature": False,
        "identifier_columns_are_model_features": False,
    }
    _write_json(output_dir / "feature_list.json", feature_manifest)

    config: Dict[str, Any] = {
        "input_dataset": _relative_path(input_path),
        "baseline_scores_dataset": _relative_path(scores_path),
        "output_dataset": _relative_path(output_dataset),
        "pipeline_file": _relative_path(pipeline_path),
        "random_state": RANDOM_STATE,
        "numerical_pipeline": ["median imputation", "StandardScaler"],
        "categorical_pipeline": ["most-frequent imputation", "OneHotEncoder(handle_unknown='ignore')"],
        "operating_period_definition": {
            "weekend": "all weekend rows",
            "night_off": "weekday hours 00:00-06:59",
            "ramp_up": "weekday hour 07:00-07:59",
            "core_work": "weekday hours 08:00-16:59",
            "ramp_down": "weekday hours 17:00-18:59",
            "evening_off": "weekday hours 19:00-23:59",
        },
        "baseline_label_definition": "1 when any v3 variable-level is_anomaly flag is true for the timestamp/floor/zone row; otherwise 0",
        "data_leakage_control": "Baseline labels, baseline counts, raw identifiers, and reference columns are excluded from model_feature_columns.",
    }
    _write_json(output_dir / "preprocessing_config.json", config)

    reloaded_columns = pd.read_csv(output_dataset, nrows=10).columns.tolist()
    if reloaded_columns != ml_ready.columns.tolist():
        raise ValueError("Saved ML-ready dataset columns changed when reloaded")

    summary: Dict[str, Any] = {
        "rows_original": original_rows,
        "rows_ml_ready": len(ml_ready),
        "columns_ml_ready": len(ml_ready.columns),
        "source_missing_values": source_missing_values,
        "ml_ready_nan_count": ml_ready_nan_count,
        "ml_ready_infinite_count": ml_ready_infinite_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_identifier_rows": int(ml_ready.duplicated(IDENTIFIER_COLUMNS).sum()),
        "numerical_source_feature_count": len(NUMERICAL_FEATURE_COLUMNS),
        "categorical_source_feature_count": len(CATEGORICAL_FEATURE_COLUMNS),
        "encoded_categorical_feature_count": len(groups["encoded_categorical_features"]),
        "final_model_feature_count": len(feature_names),
        "feature_nan_count": int(feature_frame.isna().sum().sum()),
        "feature_infinite_count": int(np.isinf(feature_frame.to_numpy()).sum()),
        "scaled_feature_max_abs_mean": float(scaled_means.abs().max()),
        "scaled_feature_min_std": float(scaled_stds.min()),
        "scaled_feature_max_std": float(scaled_stds.max()),
        "baseline_label_column": BASELINE_LABEL_COLUMN,
        "baseline_normal_rows": int(label_counts.get(0, 0)),
        "baseline_anomaly_rows": int(label_counts.get(1, 0)),
        "baseline_label_in_model_features": BASELINE_LABEL_COLUMN in feature_names,
        "identifier_in_model_features": bool(set(IDENTIFIER_COLUMNS).intersection(feature_names)),
        "model_features_are_numeric": all(
            pd.api.types.is_numeric_dtype(feature_frame[column]) for column in feature_names
        ),
        "output_reload_check": True,
    }
    _write_json(output_dir / "preprocessing_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DATASET)
    parser.add_argument("--baseline-scores", type=Path, default=DEFAULT_BASELINE_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--preview-rows", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        input_path=args.input,
        scores_path=args.baseline_scores,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        preview_rows=args.preview_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
