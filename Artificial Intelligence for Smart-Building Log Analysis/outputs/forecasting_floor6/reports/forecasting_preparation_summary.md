# Week 7 prediction-ready dataset preparation

## Forecast definition

- Point target: same-zone `total_AC_power_kw` exactly 15 minutes after the prediction timestamp.
- This is not the average power over the next 15-minute interval.
- Source coverage: 2018-07-01 00:00:00 to 2018-09-01 23:59:00.
- Zones: 1,2,4,5; source rows: 362,880.

## Prediction-ready outputs

- **train**: 253,380 rows; prediction 2018-07-01 00:00:00 to 2018-08-13 23:44:00; target through 2018-08-13 23:59:00.
- **validation**: 56,273 rows; prediction 2018-08-14 00:00:00 to 2018-08-23 23:44:00; target through 2018-08-23 23:59:00.
- **test**: 51,780 rows; prediction 2018-08-24 00:00:00 to 2018-09-01 23:44:00; target through 2018-09-01 23:59:00.

- Ordered feature count: **46**.
- Feature names and Train-fitted imputation values are recorded in `feature_manifest.json`.
- Full datasets are reproducible `.csv.gz` artifacts excluded from Git; a small preview is committed for review.

## Leakage prevention

- Upstream values marked as originally missing were restored to missing before feature generation because upstream cleaning used bidirectional interpolation and whole-period medians.
- Only within-zone causal forward-fill, limited to 15 minutes, was allowed before lag construction.
- Lags are zone-specific: 1, 5, 15, 30, 60, and 1440 minutes.
- Rolling statistics are trailing, not centered.
- Targets were created before feature imputation; originally missing future AC readings were not treated as ground truth.
- Train, Validation, and Test were selected using both prediction and target timestamps, with a 15-minute boundary purge.
- Remaining missing predictors were filled using medians fitted on Train only. Imputed predictors: total_AC_power_kw, light_power_kw, plug_power_kw, temperature_c, humidity_pct, lux, total_AC_power_kw_lag_1, total_AC_power_kw_lag_5, total_AC_power_kw_lag_15, total_AC_power_kw_lag_30, total_AC_power_kw_lag_60, total_AC_power_kw_lag_1440, total_AC_power_kw_rolling_mean_15, total_AC_power_kw_rolling_mean_30, total_AC_power_kw_rolling_mean_60, total_AC_power_kw_rolling_std_15, total_AC_power_kw_rolling_std_60.
- Existing globally standardized ML-ready features, anomaly labels, and upstream rolling features were excluded.
- No scaling, categorical encoding, or model training is performed in this stage.

All checks in `leakage_audit.csv` passed.

## Limitation

The data covers Floor 6 and approximately two months, so it supports the Week 7 forecasting experiment but not long-term or cross-floor generalisation claims.
