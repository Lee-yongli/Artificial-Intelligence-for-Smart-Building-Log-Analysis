# Week 6 ML Data Preparation

`ml_data_preparation.py` converts the Floor 6 baseline-ready dataset into the shared input for downstream Isolation Forest and LOF experiments. It does not train anomaly detection models.

## Inputs

- Baseline-ready wide data: `outputs/floor6_zone1245_2018_jul01_sep01_total_ac_eda/floor6_zone1245_baseline_ready_2018_jul01_sep01.csv.gz`
- v3 long baseline scores: `outputs/floor6_zone1245_2018_jul01_sep01_statistical_baseline_v3_15min_operation_aware/baseline_tables/09_baseline_anomaly_scores_long.csv.gz`

The v3 `is_anomaly` values are aggregated by `timestamp`, `floor`, and `zone`. `baseline_anomaly_label` is 1 when any variable is anomalous for that row. `baseline_anomaly_count` records how many of the six variables are anomalous.

## Model Features

Numerical source features:

- Six 15-minute rolling measurement features
- `minute_of_day_sin` and `minute_of_day_cos`

Numerical values use median imputation followed by `StandardScaler`.

Categorical source features:

- `zone`
- `hour`
- `operating_period`

Categorical values use most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")`.

The baseline label, baseline anomaly count, identifiers, raw measurements, and reference columns are not model features. Downstream code must select only `model_feature_columns` from `feature_list.json` to avoid data leakage.

## Run

From the repository root:

```bash
python3 "Artificial Intelligence for Smart-Building Log Analysis/ml_data_preparation.py"
```

Optional paths can be supplied with `--input`, `--baseline-scores`, and `--output-dir`.

## Outputs

The default output directory is `outputs/floor6_zone1245_ml_ready/`:

- `floor6_zone1245_ml_ready_2018_jul01_sep01.csv.gz`: full shared dataset; generated locally and ignored by Git
- `feature_list.json`: exact model features and column roles
- `preprocessing_config.json`: transformations, paths, label definition, and leakage controls
- `preprocessing_summary.json`: row counts, feature counts, labels, and quality checks
- `preprocessing_pipeline.pkl`: fitted sklearn transformer; generated locally and ignored by Git
- `ml_ready_preview.csv`: reviewable 1,000-row preview

For a new train/test evaluation, fit the preprocessing pipeline on the training period only. The checked output is fitted on the complete shared Week 6 dataset so all team members receive one consistent feature matrix.

## Tests

```bash
python3 -m unittest discover -s "Artificial Intelligence for Smart-Building Log Analysis/tests" -v
```
