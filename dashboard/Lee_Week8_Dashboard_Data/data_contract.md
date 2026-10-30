# Dashboard data contract

## Scope

This delivery exposes Gavin's schema 3.0 results through stable browser-facing names. Prediction and anomaly detection remain separate datasets because their timestamps, coverage and preprocessing differ.

Do not join prediction and anomaly rows by array index. Do not label prediction timestamps outside the anomaly coverage as normal.

## Prediction series

`data/prediction_series.csv` and `dashboard_data.json → prediction.series` contain hourly values for charts.

| Field | Type | Meaning |
|---|---|---|
| `split` | string | `validation`, `test` or `sep6_2018_09_06` |
| `zone` | integer | Zone 1, 2, 4 or 5 |
| `period` | string | Operating period for the target-window end time |
| `time` | string | Building-local naive hourly timestamp |
| `date` | string | Date derived from `time` |
| `actual` | number (kW) | Mean observed AC power for the target interval |
| `predicted` | number (kW) | Random Forest prediction |
| `persistence` | number (kW) | Persistence baseline prediction |
| `sample_count` | integer | Minute-level rows represented by the hourly mean |

Hourly values are for chart display. Calculate MAE and RMSE from `prediction.metrics`, not from the hourly means.

## Prediction metrics

`data/prediction_metrics.csv` and `dashboard_data.json → prediction.metrics` contain daily filter buckets and sufficient statistics.

Aggregate filtered metrics as follows:

- `MAE = sum(sum_abs_error) / sum(n)`
- `RMSE = sqrt(sum(sum_squared_error) / sum(n))`
- `Bias = sum(sum_error) / sum(n)`

Do not average group-level MAE or RMSE values.

## Anomaly summary

`data/anomaly_summary.csv` and `dashboard_data.json → anomaly.series` contain daily filter buckets.

| Field | Type | Meaning |
|---|---|---|
| `date` | string | Observation date |
| `zone` | integer | Zone 1, 2, 4 or 5 |
| `period` | string | Observation operating period |
| `model` | string | Default is `if_calibrated`; other values are references or comparison methods |
| `observations` | integer | Records in the bucket |
| `anomaly_count` | integer | Records carrying the selected anomaly label |
| `anomaly_rate_pct` | number | `100 × anomaly_count / observations` |

Gavin's `data/anomaly_results.csv.gz` remains the authoritative source. This delivery provides a normalized copy below, but intentionally does not load all 362,880 point rows into the default browser JSON.

## Point-level files

The normalized detail files are compressed so they do not slow down the default dashboard page:

- `data/anomaly_points.csv.gz`: 362,880 records with `time`, `zone`, `period`, `anomaly`, `anomaly_score`, comparison labels and physical readings.
- `data/prediction_points.csv.gz`: 113,622 records with `time`, `zone`, `period`, `actual`, `predicted`, persistence and errors.

`anomaly` is the formal period-calibrated Isolation Forest label. It is an integer flag (`0` or `1`), not a confirmed equipment-fault label. `anomaly_score` is comparable only within the same operating-period model.

## Time and units

- Power, MAE, RMSE and bias use kW.
- Rates ending in `_pct` are percentages on a 0–100 scale.
- Timestamps are source building-local naive times. The browser must not apply timezone conversion.
- Anomaly candidates are not confirmed equipment faults.
