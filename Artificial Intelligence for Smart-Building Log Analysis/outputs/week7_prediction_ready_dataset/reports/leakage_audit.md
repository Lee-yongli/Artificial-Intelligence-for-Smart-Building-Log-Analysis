# Week 7 prediction-ready dataset leakage audit

## Source and forecast definition

- Source: `floor6_zone1245_cleaned_features_2018_jul01_sep01.csv.gz`, the final main dataset written by `Data_pre/Floor6_Zone1245_Data_Preparation_EDA_executed.ipynb`.
- Source rows: 362,880; zones: 1,2,4,5; coverage: 2018-07-01 00:00:00 to 2018-09-01 23:59:00.
- Target: same-zone `total_AC_power_kw` exactly 15 minutes after the prediction timestamp.
- The target is a point forecast, not the mean over the next 15-minute interval.
- Logical feature count: 29.

## Chronological splits

- **train**: 253,380 rows; prediction 2018-07-01 00:00:00 to 2018-08-13 23:44:00; targets 2018-07-01 00:15:00 to 2018-08-13 23:59:00
- **validation**: 56,273 rows; prediction 2018-08-14 00:00:00 to 2018-08-23 23:44:00; targets 2018-08-14 00:15:00 to 2018-08-23 23:59:00
- **test**: 51,780 rows; prediction 2018-08-24 00:00:00 to 2018-09-01 23:44:00; targets 2018-08-24 00:15:00 to 2018-09-01 23:59:00

Both prediction and target timestamps must be inside the same split. The final 15 prediction minutes before each split boundary are therefore purged.

## Leakage controls

| Check | Result |
| --- | --- |
| Target and lags calculated within zone | PASS |
| Every retained target exactly 15 minutes ahead | PASS |
| Trailing, non-centred rolling windows | PASS |
| Source bidirectional/global imputations masked using `_was_missing` flags | PASS |
| Forward-fill causal, within zone, limited to 15 minutes | PASS |
| Remaining medians fitted on training only | PASS |
| Source rolling and operating-state columns excluded | PASS |
| Anomaly reference columns excluded | PASS |
| Split target overlap prevented | PASS |
| Output feature values complete after train-only imputation | PASS |
| Duplicate output keys | 0 |

Feature missing values before train-only median imputation: `{"test": 1881, "train": 8657, "validation": 4703}`. After imputation: `{"test": 0, "train": 0, "validation": 0}`.

## Exclusions and limitations

- Unavailable target rows: 1,327.
- No scaler was fitted during data preparation; any later model-specific scaler must be fitted on the training split only.
- Leakage safety after source cleaning depends on the correctness of the EDA notebook's preserved `_was_missing` flags. All flagged imputed sensor values were masked before target and feature construction.
- The data covers one floor and approximately two months, so it cannot establish seasonal or cross-floor generalisation.
- No forecasting model was trained or evaluated in this stage.
