"""Week 5 contextual baseline and MAD anomaly detection for Floor 6.

The workflow consumes the cleaned Week 4 one-minute table for Floor 6 Zones
1, 2, 4 and 5. Detection uses the existing 15-record rolling-mean AC feature;
the underlying observations remain one minute apart.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart_building_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
WEEK4_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "floor6_zone1245_2018_jul01_sep01_total_ac_eda"
)
INPUT_FILE = (
    WEEK4_OUTPUT_DIR
    / "floor6_zone1245_cleaned_smoothed_2018_jul01_sep01.csv.gz"
)
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "floor6_zone1245_2018_jul01_sep01_baseline_mad"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_FILE = OUTPUT_DIR / "floor6_zone1245_baseline_mad_results.csv.gz"

TARGET_COLUMN = "total_AC_power_kw_rolling_mean_15min"
RAW_TARGET_COLUMN = "total_AC_power_kw"
CONTEXT_COLUMNS = ["zone", "day_type", "hour"]
EXPECTED_FLOOR = 6
EXPECTED_ZONES = [1, 2, 4, 5]
ANALYSIS_START = pd.Timestamp("2018-07-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2018-09-01 23:59:00")
ROBUST_Z_THRESHOLD = 3.5
MAD_SCALE_CONSTANT = 0.6745
MIN_CONTEXT_OBSERVATIONS = 1_000
TOP_RESULT_COUNT = 50


def load_cleaned_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load the Week 4 Floor 6 modelling table without changing it."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Week 4 cleaned/smoothed input not found: {input_file}"
        )

    required = {
        "timestamp",
        "year",
        "floor",
        "zone",
        RAW_TARGET_COLUMN,
        TARGET_COLUMN,
        "day_type",
        "hour",
        "minute",
        "is_weekend",
    }
    available = set(pd.read_csv(input_file, nrows=0).columns)
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"Week 4 input is missing required columns: {missing}")

    data = pd.read_csv(input_file, parse_dates=["timestamp"])
    data[RAW_TARGET_COLUMN] = pd.to_numeric(data[RAW_TARGET_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Week 4 input contains invalid timestamps.")
    return data.sort_values(["zone", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def make_context_sample_size_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    """Compare hour contexts with a finer 15-minute alternative."""
    valid = data.loc[data[TARGET_COLUMN].notna()].copy()
    valid["time_slot_15min"] = (
        valid["timestamp"].dt.floor("15min").dt.strftime("%H:%M")
    )
    designs = [
        ("zone + day_type + hour", ["zone", "day_type", "hour"]),
        (
            "zone + day_type + 15-minute time slot",
            ["zone", "day_type", "time_slot_15min"],
        ),
    ]
    rows = []
    for name, columns in designs:
        sizes = valid.groupby(columns, observed=True)[TARGET_COLUMN].size()
        rows.append(
            {
                "context_design": name,
                "contexts": len(sizes),
                "minimum_valid_observations": int(sizes.min()),
                "percentile_05_valid_observations": float(sizes.quantile(0.05)),
                "median_valid_observations": float(sizes.median()),
                "maximum_valid_observations": int(sizes.max()),
                "selected_for_v1": name == "zone + day_type + hour",
            }
        )
    return pd.DataFrame(rows)


def estimate_empirical_resolution(data: pd.DataFrame) -> tuple[float, str]:
    """Estimate the observed spacing of the smoothed Floor 6 AC signal."""
    values = data[TARGET_COLUMN].dropna().to_numpy(dtype=float)
    if values.size < 2:
        raise ValueError("Not enough valid rolling AC values to estimate resolution.")
    unique_values = np.unique(np.round(values, 6))
    positive_steps = np.diff(unique_values)
    positive_steps = positive_steps[positive_steps > 1e-6]
    if not positive_steps.size:
        raise ValueError("Could not estimate positive empirical AC resolution.")
    return float(np.round(np.quantile(positive_steps, 0.01), 6)), TARGET_COLUMN


def build_contextual_baseline(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build zone-day-type-hour mean and median baselines from valid targets."""
    valid = data[TARGET_COLUMN].notna()
    contexts = (
        data.loc[valid]
        .groupby(CONTEXT_COLUMNS, observed=True)[TARGET_COLUMN]
        .agg(
            baseline_mean="mean",
            baseline_median="median",
            baseline_count="count",
        )
        .reset_index()
    )
    contexts["mean_median_difference"] = (
        contexts["baseline_mean"] - contexts["baseline_median"]
    )
    contexts["mean_median_abs_difference"] = contexts[
        "mean_median_difference"
    ].abs()
    merged = data.merge(
        contexts,
        on=CONTEXT_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    return merged, contexts


def apply_contextual_mad(
    data: pd.DataFrame,
    baseline_contexts: pd.DataFrame,
    fallback_epsilon_kw: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate residuals, contextual MAD scores and statistical labels."""
    scored = data.copy()
    valid = scored[TARGET_COLUMN].notna()
    scored["residual"] = scored[TARGET_COLUMN] - scored["baseline_median"]
    scored["absolute_residual"] = scored["residual"].abs()

    residual_median = (
        scored.loc[valid]
        .groupby(CONTEXT_COLUMNS, observed=True)["residual"]
        .median()
        .rename("residual_median")
        .reset_index()
    )
    scored = scored.merge(
        residual_median,
        on=CONTEXT_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    scored["absolute_residual_deviation_from_median"] = (
        scored["residual"] - scored["residual_median"]
    ).abs()
    mad = (
        scored.loc[valid]
        .groupby(CONTEXT_COLUMNS, observed=True)[
            "absolute_residual_deviation_from_median"
        ]
        .median()
        .rename("contextual_mad")
        .reset_index()
    )
    scored = scored.merge(
        mad,
        on=CONTEXT_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    contexts = (
        baseline_contexts.merge(
            residual_median,
            on=CONTEXT_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        .merge(mad, on=CONTEXT_COLUMNS, how="left", validate="one_to_one")
    )
    contexts["mad_zero_flag"] = contexts["contextual_mad"].eq(0)

    positive_mad = valid & scored["contextual_mad"].gt(0)
    zero_mad = valid & scored["contextual_mad"].eq(0)
    scored["robust_z_score"] = np.nan
    scored.loc[positive_mad, "robust_z_score"] = (
        MAD_SCALE_CONSTANT
        * (
            scored.loc[positive_mad, "residual"]
            - scored.loc[positive_mad, "residual_median"]
        )
        / scored.loc[positive_mad, "contextual_mad"]
    )
    scored["absolute_robust_z_score"] = scored["robust_z_score"].abs()
    scored["fallback_epsilon_kw"] = fallback_epsilon_kw

    flags = pd.Series(pd.NA, index=scored.index, dtype="Int64")
    flags.loc[positive_mad] = (
        scored.loc[positive_mad, "absolute_robust_z_score"]
        > ROBUST_Z_THRESHOLD
    ).astype(int)
    flags.loc[zero_mad] = (
        scored.loc[zero_mad, "absolute_residual_deviation_from_median"]
        > fallback_epsilon_kw
    ).astype(int)
    scored["mad_anomaly_flag"] = flags
    scored["mad_anomaly_label"] = pd.Series(
        np.select(
            [flags.eq(1).fillna(False), flags.eq(0).fillna(False)],
            ["Anomaly", "Normal"],
            default=None,
        ),
        index=scored.index,
        dtype="string",
    )
    scored["anomaly_detection_method"] = "not_evaluated_missing_target"
    scored.loc[positive_mad, "anomaly_detection_method"] = (
        "robust_z_contextual_mad"
    )
    scored.loc[zero_mad, "anomaly_detection_method"] = (
        "mad_zero_empirical_resolution_fallback"
    )
    return scored, contexts


def anomaly_summary(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    valid = data.loc[data[TARGET_COLUMN].notna()]
    summary = (
        valid.groupby(group_columns, observed=True)
        .agg(
            observations=(TARGET_COLUMN, "count"),
            anomalies=("mad_anomaly_flag", "sum"),
        )
        .reset_index()
    )
    summary["anomalies"] = summary["anomalies"].astype("int64")
    summary["anomaly_rate_pct"] = (
        100 * summary["anomalies"] / summary["observations"]
    )
    return summary.sort_values(group_columns, kind="stable").reset_index(drop=True)


def make_mean_median_summaries(contexts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    abs_difference = contexts["mean_median_abs_difference"]
    weights = contexts["baseline_count"]
    summary = pd.DataFrame(
        [
            {
                "contexts": len(contexts),
                "mean_signed_difference_kw": contexts[
                    "mean_median_difference"
                ].mean(),
                "mean_absolute_difference_kw": abs_difference.mean(),
                "median_absolute_difference_kw": abs_difference.median(),
                "percentile_95_absolute_difference_kw": abs_difference.quantile(0.95),
                "weighted_mean_absolute_difference_kw": np.average(
                    abs_difference, weights=weights
                ),
                "maximum_absolute_difference_kw": abs_difference.max(),
                "contexts_mean_above_median": int(
                    contexts["mean_median_difference"].gt(0).sum()
                ),
                "contexts_mean_above_median_pct": 100
                * contexts["mean_median_difference"].gt(0).mean(),
            }
        ]
    )
    largest = contexts.sort_values(
        ["mean_median_abs_difference", *CONTEXT_COLUMNS],
        ascending=[False, True, True, True],
        kind="stable",
    ).head(30)
    return {
        "mean_median_summary": summary,
        "mean_median_largest_differences": largest.reset_index(drop=True),
    }


def make_mad_diagnostics(contexts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mad = contexts["contextual_mad"].dropna()
    distribution = pd.DataFrame(
        [
            {
                "count": int(mad.count()),
                "zero_count": int(mad.eq(0).sum()),
                "minimum": mad.min(),
                "percentile_01": mad.quantile(0.01),
                "percentile_05": mad.quantile(0.05),
                "median": mad.median(),
                "mean": mad.mean(),
                "percentile_95": mad.quantile(0.95),
                "percentile_99": mad.quantile(0.99),
                "maximum": mad.max(),
            }
        ]
    )
    smallest = (
        contexts.loc[
            contexts["contextual_mad"].gt(0),
            CONTEXT_COLUMNS
            + ["baseline_median", "contextual_mad", "baseline_count"],
        ]
        .sort_values(
            ["contextual_mad", *CONTEXT_COLUMNS], kind="stable"
        )
        .head(30)
        .reset_index(drop=True)
    )
    return {
        "contextual_mad_distribution": distribution,
        "smallest_nonzero_mad_contexts": smallest,
    }


def assign_anomaly_runs(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Assign exact one-minute anomaly runs independently within each zone."""
    runs = anomalies.sort_values(["zone", "timestamp"], kind="stable").copy()
    if runs.empty:
        runs["anomaly_run_id"] = pd.Series(dtype="string")
        runs["anomaly_run_length"] = pd.Series(dtype="int64")
        return runs
    gap = runs.groupby("zone", sort=False)["timestamp"].diff()
    runs["_new_run"] = (
        gap.isna() | gap.ne(pd.Timedelta(minutes=1))
    ).astype("int64")
    runs["anomaly_run_number"] = (
        runs.groupby("zone", sort=False)["_new_run"].cumsum().astype("int64")
    )
    runs["anomaly_run_id"] = (
        "Z"
        + runs["zone"].astype("int64").astype(str)
        + "-R"
        + runs["anomaly_run_number"].astype(str).str.zfill(4)
    )
    runs["anomaly_run_length"] = (
        runs.groupby(["zone", "anomaly_run_number"], sort=False)["timestamp"]
        .transform("size")
        .astype("int64")
    )
    return runs.drop(columns="_new_run")


def make_anomaly_run_summary(data: pd.DataFrame) -> pd.DataFrame:
    anomalies = assign_anomaly_runs(data.loc[data["mad_anomaly_flag"].eq(1)])
    if anomalies.empty:
        return pd.DataFrame()
    summary = (
        anomalies.groupby(
            ["zone", "anomaly_run_number", "anomaly_run_id"], sort=False
        )
        .agg(
            run_start=("timestamp", "min"),
            run_end=("timestamp", "max"),
            anomaly_run_length=("timestamp", "size"),
            maximum_absolute_robust_z=("absolute_robust_z_score", "max"),
            maximum_absolute_residual_kw=("absolute_residual", "max"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["anomaly_run_length", "maximum_absolute_robust_z", "zone", "run_start"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_representative_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    anomalies = assign_anomaly_runs(data.loc[data["mad_anomaly_flag"].eq(1)])
    if anomalies.empty:
        return anomalies
    anomalies["detection_rank"] = anomalies[
        "absolute_robust_z_score"
    ].fillna(
        anomalies["absolute_residual_deviation_from_median"]
        / anomalies["fallback_epsilon_kw"]
    )
    selected: list[pd.Series] = []
    selected_keys: set[tuple[pd.Timestamp, int]] = set()

    def add_row(row: pd.Series, reason: str) -> None:
        key = (row["timestamp"], int(row["zone"]))
        if key in selected_keys:
            return
        item = row.copy()
        item["selection_reason"] = reason
        direction = "above" if item["residual"] > 0 else "below"
        if pd.notna(item["robust_z_score"]):
            score_text = f"absolute robust z={item['absolute_robust_z_score']:.2f}"
        else:
            score_text = (
                "deviation exceeds the estimated empirical-resolution fallback"
            )
        item["why_flagged"] = (
            f"Smoothed AC power was {abs(item['residual']):.4f} kW {direction} "
            f"its contextual median; {score_text}. This is a statistical anomaly, "
            "not a confirmed fault."
        )
        selected.append(item)
        selected_keys.add(key)

    robust = anomalies.dropna(subset=["absolute_robust_z_score"])
    if not robust.empty:
        add_row(
            robust.nlargest(1, "absolute_robust_z_score").iloc[0],
            "Highest robust MAD score",
        )
    add_row(
        anomalies.nlargest(1, "residual").iloc[0],
        "Largest positive contextual residual",
    )
    add_row(
        anomalies.nsmallest(1, "residual").iloc[0],
        "Largest negative contextual residual",
    )
    longest = int(anomalies["anomaly_run_length"].max())
    sustained = anomalies.loc[anomalies["anomaly_run_length"].eq(longest)]
    add_row(
        sustained.nlargest(1, "detection_rank").iloc[0],
        f"Representative point from a {longest}-minute anomaly run",
    )
    for zone in EXPECTED_ZONES:
        zone_rows = anomalies.loc[anomalies["zone"].eq(zone)]
        if not zone_rows.empty:
            add_row(
                zone_rows.nlargest(1, "detection_rank").iloc[0],
                f"Highest-ranked example in Zone {zone}",
            )

    result = pd.DataFrame(selected)
    preferred = [
        "selection_reason",
        "timestamp",
        "floor",
        "zone",
        "day_type",
        "hour",
        RAW_TARGET_COLUMN,
        TARGET_COLUMN,
        "baseline_median",
        "residual",
        "contextual_mad",
        "robust_z_score",
        "absolute_robust_z_score",
        "anomaly_run_id",
        "anomaly_run_length",
        "anomaly_detection_method",
        "why_flagged",
    ]
    return result[[column for column in preferred if column in result]].reset_index(
        drop=True
    )


def make_manual_spot_checks(data: pd.DataFrame, count: int = 8) -> pd.DataFrame:
    candidates = data.loc[
        data[TARGET_COLUMN].notna() & data["contextual_mad"].gt(0)
    ]
    if candidates.empty:
        return pd.DataFrame()
    positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    checks = candidates.iloc[positions].copy()
    checks["manual_residual"] = checks[TARGET_COLUMN] - checks["baseline_median"]
    checks["manual_robust_z_score"] = (
        MAD_SCALE_CONSTANT
        * (checks["manual_residual"] - checks["residual_median"])
        / checks["contextual_mad"]
    )
    checks["residual_check_abs_error"] = (
        checks["residual"] - checks["manual_residual"]
    ).abs()
    checks["robust_z_check_abs_error"] = (
        checks["robust_z_score"] - checks["manual_robust_z_score"]
    ).abs()
    columns = [
        "timestamp",
        "floor",
        "zone",
        "day_type",
        "hour",
        TARGET_COLUMN,
        "baseline_median",
        "residual",
        "manual_residual",
        "residual_median",
        "contextual_mad",
        "robust_z_score",
        "manual_robust_z_score",
        "residual_check_abs_error",
        "robust_z_check_abs_error",
    ]
    return checks[columns].reset_index(drop=True)


def validate_interleaved_zone_runs() -> tuple[bool, str]:
    synthetic = pd.DataFrame(
        {
            "zone": [1, 2, 1, 2, 1],
            "timestamp": pd.to_datetime(
                [
                    "2018-07-01 00:00",
                    "2018-07-01 00:00",
                    "2018-07-01 00:01",
                    "2018-07-01 00:01",
                    "2018-07-01 00:02",
                ]
            ),
        }
    )
    assigned = assign_anomaly_runs(synthetic)
    observed = assigned.groupby("zone", sort=True).agg(
        run_ids=("anomaly_run_id", "nunique"),
        minimum_length=("anomaly_run_length", "min"),
        maximum_length=("anomaly_run_length", "max"),
    )
    passed = (
        tuple(observed.loc[1]) == (1, 3, 3)
        and tuple(observed.loc[2]) == (1, 2, 2)
    )
    detail = (
        f"Z1 runs={int(observed.loc[1, 'run_ids'])}, "
        f"length={int(observed.loc[1, 'maximum_length'])}; "
        f"Z2 runs={int(observed.loc[2, 'run_ids'])}, "
        f"length={int(observed.loc[2, 'maximum_length'])}"
    )
    return passed, detail


def validate_results(
    data: pd.DataFrame,
    contexts: pd.DataFrame,
    spot_checks: pd.DataFrame,
) -> pd.DataFrame:
    valid = data[TARGET_COLUMN].notna()
    positive_mad = valid & data["contextual_mad"].gt(0)
    zero_mad = valid & data["contextual_mad"].eq(0)
    expected_dates = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="1min")
    zone_diffs = (
        data.sort_values(["zone", "timestamp"], kind="stable")
        .groupby("zone", sort=False)["timestamp"]
        .diff()
        .dropna()
    )

    valid_residuals = data.loc[valid, CONTEXT_COLUMNS + ["residual"]].copy()
    valid_residuals["manual_residual_median"] = valid_residuals.groupby(
        CONTEXT_COLUMNS, observed=True
    )["residual"].transform("median")
    valid_residuals["manual_mad"] = (
        valid_residuals["residual"]
        - valid_residuals["manual_residual_median"]
    ).abs()
    valid_residuals["manual_mad"] = valid_residuals.groupby(
        CONTEXT_COLUMNS, observed=True
    )["manual_mad"].transform("median")
    run_passed, run_detail = validate_interleaved_zone_runs()

    expected_robust_flags = data.loc[
        positive_mad, "absolute_robust_z_score"
    ].gt(ROBUST_Z_THRESHOLD)
    expected_fallback_flags = data.loc[
        zero_mad, "absolute_residual_deviation_from_median"
    ].gt(data.loc[zero_mad, "fallback_epsilon_kw"])

    checks = [
        (
            "floor_is_exactly_6",
            set(data["floor"].unique()) == {EXPECTED_FLOOR},
            f"floors={sorted(data['floor'].unique().tolist())}",
        ),
        (
            "zones_are_exactly_1_2_4_5",
            set(data["zone"].unique()) == set(EXPECTED_ZONES),
            f"zones={sorted(data['zone'].unique().tolist())}",
        ),
        (
            "zone_3_absent",
            not data["zone"].eq(3).any(),
            "Zone 3 rows=0",
        ),
        (
            "wrong_floors_absent",
            not data["floor"].isin([3, 4, 5, 7]).any(),
            "Floors 3/4/5/7 rows=0",
        ),
        (
            "analysis_period_matches_week4",
            data["timestamp"].min() == ANALYSIS_START
            and data["timestamp"].max() == ANALYSIS_END,
            f"{data['timestamp'].min()} through {data['timestamp'].max()}",
        ),
        (
            "one_minute_frequency_and_complete_zone_grids",
            zone_diffs.eq(pd.Timedelta(minutes=1)).all()
            and data.groupby("zone")["timestamp"].nunique().eq(len(expected_dates)).all(),
            f"expected timestamps per zone={len(expected_dates):,}",
        ),
        (
            "timestamp_floor_zone_key_unique",
            not data.duplicated(["timestamp", "floor", "zone"]).any(),
            f"duplicate keys={int(data.duplicated(['timestamp', 'floor', 'zone']).sum())}",
        ),
        (
            "rolling_ac_target_exists",
            TARGET_COLUMN in data.columns,
            TARGET_COLUMN,
        ),
        (
            "raw_ac_field_exists",
            RAW_TARGET_COLUMN in data.columns,
            RAW_TARGET_COLUMN,
        ),
        (
            "baseline_counts_use_valid_targets_only",
            int(contexts["baseline_count"].sum()) == int(valid.sum()),
            f"baseline count sum={int(contexts['baseline_count'].sum()):,}; valid={int(valid.sum()):,}",
        ),
        (
            "context_keys_are_unique",
            not contexts.duplicated(CONTEXT_COLUMNS).any(),
            f"contexts={len(contexts):,}",
        ),
        (
            "no_cross_zone_baseline_mixing",
            data.loc[valid, "baseline_count"].notna().all()
            and data.loc[valid].groupby(CONTEXT_COLUMNS)["baseline_median"].nunique().le(1).all(),
            "Every valid row matched one zone-day_type-hour baseline",
        ),
        (
            "residual_calculation_correct",
            np.allclose(
                data.loc[valid, "residual"],
                data.loc[valid, TARGET_COLUMN]
                - data.loc[valid, "baseline_median"],
                equal_nan=True,
            ),
            "residual = rolling AC target - contextual median",
        ),
        (
            "contextual_mad_calculation_correct",
            np.allclose(
                data.loc[valid, "contextual_mad"],
                valid_residuals["manual_mad"],
                equal_nan=True,
            ),
            "MAD recomputed independently inside zone-day_type-hour contexts",
        ),
        (
            "mad_zero_has_no_division_by_zero",
            data.loc[zero_mad, "robust_z_score"].isna().all(),
            f"MAD=0 observations={int(zero_mad.sum()):,}",
        ),
        (
            "no_infinite_robust_z_scores",
            not np.isinf(
                data["robust_z_score"].to_numpy(dtype=float, na_value=np.nan)
            ).any(),
            "No positive or negative infinity",
        ),
        (
            "threshold_and_fallback_decisions_correct",
            np.array_equal(
                data.loc[positive_mad, "mad_anomaly_flag"].to_numpy(dtype=int),
                expected_robust_flags.to_numpy(dtype=int),
            )
            and np.array_equal(
                data.loc[zero_mad, "mad_anomaly_flag"].to_numpy(dtype=int),
                expected_fallback_flags.to_numpy(dtype=int),
            ),
            f"abs(robust z)>{ROBUST_Z_THRESHOLD}; MAD=0 uses empirical resolution",
        ),
        (
            "missing_smoothed_targets_excluded",
            data.loc[~valid, "mad_anomaly_flag"].isna().all(),
            f"excluded rows={int((~valid).sum()):,}",
        ),
        (
            "interleaved_zone_run_segmentation",
            run_passed,
            run_detail,
        ),
        (
            "manual_formula_spot_checks",
            not spot_checks.empty
            and spot_checks["residual_check_abs_error"].max() < 1e-12
            and spot_checks["robust_z_check_abs_error"].max() < 1e-12,
            f"spot checks={len(spot_checks)}",
        ),
    ]
    validation = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise AssertionError(f"Floor 6 Week 5 validation failed: {failed}")
    return validation


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _baseline_window(
    data: pd.DataFrame, contexts: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    zone_range = contexts.groupby("zone")["baseline_median"].agg(
        lambda values: values.max() - values.min()
    )
    zone = int(zone_range.idxmax())
    start = pd.Timestamp("2018-07-02 00:00:00")
    end = start + pd.Timedelta(days=2)
    window = data.loc[
        data["zone"].eq(zone) & data["timestamp"].between(start, end, inclusive="left")
    ]
    return window, zone


def _focus_anomaly(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    anomalies = data.loc[data["mad_anomaly_flag"].eq(1)].copy()
    anomalies["detection_rank"] = anomalies[
        "absolute_robust_z_score"
    ].fillna(
        anomalies["absolute_residual_deviation_from_median"]
        / anomalies["fallback_epsilon_kw"]
    )
    focus = anomalies.nlargest(1, "detection_rank").iloc[0]
    window = data.loc[
        data["zone"].eq(focus["zone"])
        & data["timestamp"].between(
            focus["timestamp"] - pd.Timedelta(hours=4),
            focus["timestamp"] + pd.Timedelta(hours=4),
        )
    ].copy()
    return window, focus


def make_figures(
    data: pd.DataFrame,
    contexts: pd.DataFrame,
    by_zone: pd.DataFrame,
) -> pd.DataFrame:
    """Create only Week 5 model-result figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[tuple[int, str, str]] = []

    baseline_window, baseline_zone = _baseline_window(data, contexts)
    path = FIGURE_DIR / "week5_01_smoothed_ac_vs_contextual_baseline.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        baseline_window["timestamp"],
        baseline_window[TARGET_COLUMN],
        label="15-minute rolling-mean AC",
        color="#3264A8",
        linewidth=1.3,
    )
    ax.plot(
        baseline_window["timestamp"],
        baseline_window["baseline_median"],
        label="Contextual median baseline",
        color="#D27A35",
        linewidth=1.8,
    )
    ax.set(
        title=f"Smoothed AC and contextual median baseline: Floor 6, Zone {baseline_zone}",
        xlabel="Timestamp",
        ylabel="AC power (kW)",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save_figure(fig, path)
    records.append((1, path.name, "Smoothed AC compared with its contextual median baseline"))

    focus_window, focus = _focus_anomaly(data)
    focus_anomalies = focus_window.loc[focus_window["mad_anomaly_flag"].eq(1)]
    path = FIGURE_DIR / "week5_02_smoothed_ac_baseline_with_mad_anomalies.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(focus_window["timestamp"], focus_window[TARGET_COLUMN], label="15-minute rolling-mean AC", color="#3264A8", linewidth=1.3)
    ax.plot(focus_window["timestamp"], focus_window["baseline_median"], label="Contextual median baseline", color="#D27A35", linewidth=1.8)
    ax.scatter(focus_anomalies["timestamp"], focus_anomalies[TARGET_COLUMN], label="MAD statistical anomaly", color="#B94D4D", s=24, zorder=3)
    ax.set(title=f"MAD anomalies around a high-scoring case: Floor 6, Zone {int(focus['zone'])}", xlabel="Timestamp", ylabel="AC power (kW)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save_figure(fig, path)
    records.append((2, path.name, "Smoothed AC, baseline and MAD anomaly points"))

    path = FIGURE_DIR / "week5_03_contextual_residual_over_time.png"
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(focus_window["timestamp"], focus_window["residual"], color="#3264A8", linewidth=1.2, label="Residual")
    ax.scatter(focus_anomalies["timestamp"], focus_anomalies["residual"], color="#B94D4D", s=24, label="MAD statistical anomaly", zorder=3)
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set(title=f"Contextual residuals: Floor 6, Zone {int(focus['zone'])}", xlabel="Timestamp", ylabel="Rolling AC − baseline (kW)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save_figure(fig, path)
    records.append((3, path.name, "Residuals for the same high-scoring anomaly window"))

    path = FIGURE_DIR / "week5_04_mad_anomaly_rate_by_zone.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(by_zone["zone"].astype(str), by_zone["anomaly_rate_pct"], color="#3264A8")
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set(title="MAD statistical anomaly rate by Floor 6 zone", xlabel="Zone", ylabel="Anomaly rate (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, path)
    records.append((4, path.name, "Statistical anomaly rates for Zones 1, 2, 4 and 5"))

    path = FIGURE_DIR / "week5_05_mean_vs_median_contextual_baseline.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(contexts["baseline_median"], contexts["baseline_mean"], c=contexts["mean_median_abs_difference"], cmap="viridis", alpha=0.75, s=28)
    low = min(contexts["baseline_median"].min(), contexts["baseline_mean"].min())
    high = max(contexts["baseline_median"].max(), contexts["baseline_mean"].max())
    ax.plot([low, high], [low, high], linestyle="--", color="#555555", linewidth=1, label="Mean = median")
    ax.set(title="Mean and median contextual AC baselines", xlabel="Contextual median (kW)", ylabel="Contextual mean (kW)")
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Absolute mean-median difference (kW)")
    ax.grid(alpha=0.15)
    _save_figure(fig, path)
    records.append((5, path.name, "Mean and median comparison across 192 Floor 6 contexts"))

    path = FIGURE_DIR / "week5_06_mad_explanation_case_study.png"
    fig, (top_ax, residual_ax) = plt.subplots(2, 1, figsize=(15.5, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    fig.suptitle("How the contextual baseline and MAD identify unusual AC behaviour", x=0.07, y=0.97, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.07, 0.925, f"Case study: Floor 6, Zone {int(focus['zone'])} on {focus['timestamp']:%d %b %Y at %H:%M}", fontsize=11.5, color="#4A4A4A")
    top_ax.plot(focus_window["timestamp"], focus_window[TARGET_COLUMN], color="#3264A8", linewidth=1.4, label="15-minute rolling-mean AC")
    top_ax.plot(focus_window["timestamp"], focus_window["baseline_median"], color="#D27A35", linewidth=1.9, label="Contextual median baseline")
    top_ax.scatter(focus_anomalies["timestamp"], focus_anomalies[TARGET_COLUMN], color="#B94D4D", edgecolor="white", linewidth=0.35, s=27, zorder=4, label="MAD statistical anomaly")
    top_ax.scatter([focus["timestamp"]], [focus[TARGET_COLUMN]], marker="D", color="#F2C14E", edgecolor="#333333", s=78, zorder=5)
    top_ax.annotate(f"Selected point\n|robust z| = {focus['absolute_robust_z_score']:.1f}" if pd.notna(focus["absolute_robust_z_score"]) else "Selected fallback anomaly", xy=(focus["timestamp"], focus[TARGET_COLUMN]), xytext=(18, 48), textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=10.5, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.95})
    top_ax.set(title="Smoothed AC compared with expected behaviour", ylabel="AC power (kW)")
    top_ax.legend(frameon=False, ncol=3, fontsize=10)
    top_ax.grid(alpha=0.18)
    residual_ax.plot(focus_window["timestamp"], focus_window["residual"], color="#3264A8", linewidth=1.3, label="Residual")
    residual_ax.scatter(focus_anomalies["timestamp"], focus_anomalies["residual"], color="#B94D4D", edgecolor="white", linewidth=0.35, s=27, zorder=4, label="Flagged residual")
    residual_ax.axhline(0, color="#444444", linewidth=0.9)
    residual_ax.set(title="Residuals measure deviation from context, not simply high AC power", ylabel="Rolling AC − baseline (kW)", xlabel="Timestamp")
    residual_ax.legend(frameon=False, ncol=2, fontsize=10)
    residual_ax.grid(alpha=0.18)
    residual_ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    residual_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    explanation = (
        "HOW TO READ THIS FIGURE\n\n"
        "• Baseline: typical rolling AC power for the same\n  zone, day type and hour.\n\n"
        "• Residual: smoothed AC power minus the baseline.\n\n"
        "• MAD rule: flag when |robust z| > 3.5.\n\n"
        "• Flags are statistical anomalies, not confirmed faults."
    )
    fig.text(0.745, 0.69, explanation, va="top", fontsize=10.5, linespacing=1.35, bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F7F9FC", "edgecolor": "#D5DCE5"})
    point_text = (
        "SELECTED POINT\n\n"
        f"Raw AC             {focus[RAW_TARGET_COLUMN]:.3f} kW\n"
        f"Rolling AC         {focus[TARGET_COLUMN]:.3f} kW\n"
        f"Context baseline   {focus['baseline_median']:.3f} kW\n"
        f"Residual           {focus['residual']:+.3f} kW\n"
        f"Context MAD        {focus['contextual_mad']:.3f} kW"
    )
    fig.text(0.745, 0.31, point_text, va="top", fontsize=10.5, family="monospace", linespacing=1.4, bbox={"boxstyle": "round,pad=0.7", "facecolor": "#FFF9E8", "edgecolor": "#E5D39A"})
    fig.subplots_adjust(left=0.07, right=0.70, top=0.87, bottom=0.09, hspace=0.22)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    records.append((6, path.name, "Two-panel explanation of the Floor 6 contextual baseline and MAD rule"))

    return pd.DataFrame(records, columns=["figure_number", "file_name", "description"])


def write_outputs(
    data: pd.DataFrame,
    contexts: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(RESULT_FILE, index=False, compression="gzip")
    paths = {"results": RESULT_FILE}
    context_path = OUTPUT_DIR / "context_baselines.csv"
    contexts.to_csv(context_path, index=False)
    paths["context_baselines"] = context_path
    for name, table in tables.items():
        path = OUTPUT_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def run_week5_analysis(input_file: Path = INPUT_FILE) -> dict[str, object]:
    """Run the complete Floor 6 Week 5 workflow."""
    raw = load_cleaned_data(input_file)
    context_sample_sizes = make_context_sample_size_diagnostic(raw)
    empirical_resolution_kw, resolution_source = estimate_empirical_resolution(raw)
    with_baseline, baseline = build_contextual_baseline(raw)
    scored, contexts = apply_contextual_mad(
        with_baseline,
        baseline,
        fallback_epsilon_kw=empirical_resolution_kw,
    )

    valid = scored[TARGET_COLUMN].notna()
    anomaly = scored["mad_anomaly_flag"].eq(1)
    mad_zero_contexts = int(contexts["mad_zero_flag"].sum())
    mad_zero_observations = int(
        scored.loc[valid & scored["contextual_mad"].eq(0)].shape[0]
    )
    fallback_anomalies = int(
        scored["anomaly_detection_method"]
        .eq("mad_zero_empirical_resolution_fallback")
        .mul(scored["mad_anomaly_flag"].eq(1))
        .sum()
    )
    overall_summary = pd.DataFrame(
        [
            {
                "input_file": input_file.name,
                "total_rows": len(scored),
                "valid_smoothed_observations_analysed": int(valid.sum()),
                "excluded_smoothed_target_rows": int((~valid).sum()),
                "total_anomalies": int(anomaly.sum()),
                "anomaly_rate_pct": 100 * anomaly.sum() / valid.sum(),
                "context_grouping": "zone + day_type + hour",
                "number_of_contexts": len(contexts),
                "mad_zero_contexts": mad_zero_contexts,
                "mad_zero_context_observations": mad_zero_observations,
                "fallback_anomalies": fallback_anomalies,
                "robust_z_threshold": ROBUST_Z_THRESHOLD,
                "estimated_empirical_resolution_kw": empirical_resolution_kw,
                "empirical_resolution_source": resolution_source,
                "analysis_start": scored["timestamp"].min(),
                "analysis_end": scored["timestamp"].max(),
                "floor": EXPECTED_FLOOR,
                "zones": "1,2,4,5",
                "raw_interpretation_field": RAW_TARGET_COLUMN,
                "detection_target": TARGET_COLUMN,
            }
        ]
    )

    scored["analysis_date"] = scored["timestamp"].dt.strftime("%Y-%m-%d")
    by_zone = anomaly_summary(scored, ["zone"])
    by_day_type = anomaly_summary(scored, ["day_type"])
    by_hour = anomaly_summary(scored, ["hour"])
    by_date = anomaly_summary(scored, ["analysis_date"])
    context_rates = anomaly_summary(scored, CONTEXT_COLUMNS)
    highest_context_rates = (
        context_rates.loc[
            context_rates["observations"].ge(MIN_CONTEXT_OBSERVATIONS)
        ]
        .sort_values(
            ["anomaly_rate_pct", "anomalies", *CONTEXT_COLUMNS],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
        .head(30)
        .reset_index(drop=True)
    )

    scored["detection_rank"] = scored["absolute_robust_z_score"].fillna(
        scored["absolute_residual_deviation_from_median"]
        / scored["fallback_epsilon_kw"]
    )
    top_columns = [
        "timestamp",
        "floor",
        "zone",
        "day_type",
        "hour",
        RAW_TARGET_COLUMN,
        TARGET_COLUMN,
        "baseline_median",
        "residual",
        "residual_median",
        "contextual_mad",
        "robust_z_score",
        "absolute_robust_z_score",
        "fallback_epsilon_kw",
        "anomaly_detection_method",
        "detection_rank",
    ]
    highest_scoring = (
        scored.loc[anomaly]
        .nlargest(TOP_RESULT_COUNT, "detection_rank")[top_columns]
        .reset_index(drop=True)
    )
    representative = select_representative_anomalies(scored)
    run_summary = make_anomaly_run_summary(scored)
    spot_checks = make_manual_spot_checks(scored)
    validation = validate_results(scored, contexts, spot_checks)
    mean_median_tables = make_mean_median_summaries(contexts)
    mad_tables = make_mad_diagnostics(contexts)
    figure_index = make_figures(scored, contexts, by_zone)

    output_columns = [
        "timestamp",
        "year",
        "floor",
        "zone",
        "source_file",
        RAW_TARGET_COLUMN,
        TARGET_COLUMN,
        "day_type",
        "hour",
        "minute",
        "is_weekend",
        "ac_meter_count",
        "has_ac_meter",
        "baseline_mean",
        "baseline_median",
        "baseline_count",
        "mean_median_difference",
        "mean_median_abs_difference",
        "residual",
        "absolute_residual",
        "residual_median",
        "contextual_mad",
        "robust_z_score",
        "absolute_robust_z_score",
        "absolute_residual_deviation_from_median",
        "fallback_epsilon_kw",
        "mad_anomaly_flag",
        "mad_anomaly_label",
        "anomaly_detection_method",
    ]
    scored_output = scored[output_columns]
    tables = {
        "overall_summary": overall_summary,
        "by_zone": by_zone,
        "by_day_type": by_day_type,
        "by_hour": by_hour,
        "by_date": by_date,
        "context_sample_size_diagnostic": context_sample_sizes,
        "highest_anomaly_rate_contexts": highest_context_rates,
        "highest_scoring_anomalies": highest_scoring,
        "representative_anomalies": representative,
        "anomaly_run_summary": run_summary,
        "manual_spot_checks": spot_checks,
        "validation": validation,
        "figure_index": figure_index,
        **mean_median_tables,
        **mad_tables,
    }
    output_paths = write_outputs(scored_output, contexts, tables)

    print(f"Input rows: {len(scored):,}")
    print(f"Valid smoothed observations: {int(valid.sum()):,}")
    print(f"Excluded smoothed targets: {int((~valid).sum()):,}")
    print(
        f"Statistical anomalies: {int(anomaly.sum()):,} "
        f"({100 * anomaly.sum() / valid.sum():.4f}%)"
    )
    print(f"Contexts: {len(contexts):,}; MAD=0 contexts: {mad_zero_contexts:,}")
    print(f"Fallback anomalies: {fallback_anomalies:,}")
    print(
        "Estimated empirical rolling-AC resolution: "
        f"{empirical_resolution_kw:g} kW ({resolution_source})"
    )

    return {
        "overall_summary": overall_summary,
        "by_zone": by_zone,
        "by_day_type": by_day_type,
        "by_hour": by_hour,
        "by_date": by_date,
        "context_sample_size_diagnostic": context_sample_sizes,
        "highest_anomaly_rate_contexts": highest_context_rates,
        "highest_scoring_anomalies": highest_scoring,
        "representative_anomalies": representative,
        "anomaly_run_summary": run_summary,
        "manual_spot_checks": spot_checks,
        "validation": validation,
        "contexts": contexts,
        "mean_median_summary": mean_median_tables["mean_median_summary"],
        "mean_median_largest_differences": mean_median_tables[
            "mean_median_largest_differences"
        ],
        "contextual_mad_distribution": mad_tables[
            "contextual_mad_distribution"
        ],
        "smallest_nonzero_mad_contexts": mad_tables[
            "smallest_nonzero_mad_contexts"
        ],
        "figure_index": figure_index,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    run_week5_analysis()
