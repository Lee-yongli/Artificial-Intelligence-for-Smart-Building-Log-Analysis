"""
定义同一区域未来 15 分钟平均 AC 功率作为预测目标，
按时间顺序划分训练集、验证集和测试集，只使用训练集统计量进行中位数填补，
并输出 prediction-ready 数据集

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


# 项目根目录，用于构建默认输入和输出路径。
PROJECT_DIR = Path(__file__).resolve().parents[1]

# 输入数据：来自前面 EDA / cleaning 阶段生成的清洗后 Floor 6 数据。
DEFAULT_INPUT = (
    PROJECT_DIR
    / "outputs/floor6_zone1245_2018_jul01_sep01_total_ac_eda"
    / "floor6_zone1245_cleaned_features_2018_jul01_sep01.csv.gz"
)

# 输出目录：保存 Week 7 预测任务的数据集、配置文件和检查报告。
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/week7_prediction_ready_dataset"

# 每一行数据的身份字段。
KEY_COLUMNS = ["timestamp", "floor", "zone"]

# 主要能耗和环境变量。
SENSOR_COLUMNS = [
    "total_AC_power_kw",
    "light_power_kw",
    "plug_power_kw",
    "temperature_c",
    "humidity_pct",
    "lux",
]

# 缺失值标记列，用来识别哪些传感器值在清洗阶段曾经被填补过。
MISSING_FLAG_COLUMNS = [f"{column}_was_missing" for column in SENSOR_COLUMNS]

# lag features：使用过去不同时间点的 AC 功率。
LAG_MINUTES = [1, 5, 15, 30, 60, 1440]

# rolling features：描述过去一段时间的平均水平和波动情况。
ROLLING_MEAN_MINUTES = [15, 30, 60]
ROLLING_STD_MINUTES = [15, 60]

# 预测未来 15 分钟的平均 AC 功率。
FORECAST_HORIZON_MINUTES = 15
CAUSAL_FORWARD_FILL_LIMIT = 15
TARGET_COLUMN = "target_total_AC_power_kw_next_15min_mean"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"

# 这些列不允许进入预测特征，避免把异常检测结果或后验运行状态作为输入。
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


# 按时间顺序定义训练集、验证集和测试集，而不是随机划分。
SPLITS = (
    SplitDefinition("train", "2018-07-01", "2018-08-14"),
    SplitDefinition("validation", "2018-08-14", "2018-08-24"),
    SplitDefinition("test", "2018-08-24", "2018-09-02"),
)


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    """将常见布尔编码统一转换为 True/False，并拒绝含义不明确的值。"""
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
    """读取源数据，并进行基础质量检查和审计记录。

    这里检查必须字段、时间戳、重复 key、缺失值标记和采样间隔。
    """
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
    """定义预测模型最终使用的数值特征和类别特征。"""
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
    """构建预测目标和因果特征。

    因果特征表示模型输入只使用当前时间点和过去的信息，不使用未来信息。
    """
    data = frame.copy()

    # EDA 源数据曾经使用双向插值和全时段中位数。
    # 为了避免未来信息进入特征，先把所有被标记为原本缺失的传感器值恢复为 NaN。
    for column, flag in zip(SENSOR_COLUMNS, MISSING_FLAG_COLUMNS):
        data.loc[data[flag], column] = np.nan

    # 构建预测目标：
    # 对同一个 zone，取当前时间之后 1 到 15 分钟的 total_AC_power_kw，
    # 计算平均值作为未来 15 分钟平均 AC 功率。
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

    # 只对短时间缺失做 forward fill，并且绝不跨越 zone 边界。
    # 剩余缺失值后面只用训练集的中位数填补。
    for column in SENSOR_COLUMNS:
        data[column] = data.groupby("zone", sort=False)[column].ffill(
            limit=CAUSAL_FORWARD_FILL_LIMIT
        )

    # 构建 lag features，表示过去不同时间点的 AC 功率。
    zone_ac = data.groupby("zone", sort=False)["total_AC_power_kw"]
    for minutes in LAG_MINUTES:
        data[f"total_AC_power_kw_lag_{minutes}"] = zone_ac.shift(minutes)

    # 构建 rolling mean，表示最近一段时间的平均运行水平。
    for minutes in ROLLING_MEAN_MINUTES:
        data[f"total_AC_power_kw_rolling_mean_{minutes}"] = (
            zone_ac.rolling(window=minutes, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # 构建 rolling standard deviation，表示最近一段时间 AC 功率是否稳定。
    for minutes in ROLLING_STD_MINUTES:
        data[f"total_AC_power_kw_rolling_std_{minutes}"] = (
            zone_ac.rolling(window=minutes, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

    # 构建时间特征和周期编码，让模型理解一天内的循环规律。
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
    """按时间顺序划分训练集、验证集和测试集。

    prediction timestamp 和 target window 的结束时间都必须落在同一个 split 内，
    这样可以避免训练集目标跨入验证集或测试集。
    """
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
    """只在训练集上计算特征中位数，并应用到所有 split。

    这样可以避免验证集或测试集的信息泄露到训练阶段。
    """
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
    """对最终 prediction-ready 数据集进行运行时检查。

    检查内容包括时间对齐、排序、重复 key、缺失值和禁止特征。
    """
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
    """保存压缩后的 train / validation / test 数据集，并返回文件清单。"""
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
    """运行完整数据准备流程，不进行模型训练。

    流程包括读取源数据、构建目标和特征、时间顺序划分、训练集填补、
    数据验证，以及导出数据集和审计报告。
    """
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
