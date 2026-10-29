# Artificial Intelligence for Smart-Building Log Analysis

This repository contains the analysis project folder for the Industry Research Project on AI-assisted smart-building log analysis.

## Main Project Folder

```text
Artificial Intelligence for Smart-Building Log Analysis/
```

This folder contains:

- `Artificial_Intelligence_for_Smart_Building_Log_Analysis.ipynb`: main Jupyter notebook.
- `merge_2018_2019_floors_3_4_5_7.ipynb`: focused notebook that only merges 2018-2019 data for floors 3, 4, 5, and 7.
- `outputs/`: small reviewable summary files, EDA charts, and preview tables.
- `Raw data/2018Floor6.csv`: the original Floor 6 data used by the current analysis and the September 6 evaluation.

## Data Policy

The repository includes the original CU-BEMS Floor 6 file used by the current analysis:

```text
Artificial Intelligence for Smart-Building Log Analysis/Raw data/2018Floor6.csv
```

It also includes the prediction-ready train, validation and test splits, plus both CSV and compressed CSV versions of the September 6 actual table, under:

```text
Artificial Intelligence for Smart-Building Log Analysis/outputs/week7_prediction_ready_dataset/data/
```

The September 6 table is `prediction_ready_2018-09-06_from_raw.csv` (or `.csv.gz`). It contains 5,760 zone-minute rows with actual measurements, engineered features and the same future-15-minute average AC-power target used for training. Of these, 5,625 have a valid future target; the remaining 135 targets are missing and are excluded from error metrics. Prediction outputs remain under `outputs/week7_forecasting_gavin_rf/data/`.

The other 13 raw floor files used by the earlier multi-floor notebooks are not included. Running those historical notebooks still requires their original inputs; the supplied Floor 6 file supports the current four-zone analysis. Additional raw or generated files remain ignored unless explicitly included.

Generated outputs are selectively included; other large intermediate files remain excluded from Git. The Week 6 ML-ready dataset is committed so the LOF and Isolation Forest experiments can run immediately after cloning.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

On Windows, enable long path support before cloning or pulling this repository because some generated analysis outputs live in deeply nested folders:

```bash
git config --global core.longpaths true
```

Open the notebook:

```bash
jupyter notebook "Artificial Intelligence for Smart-Building Log Analysis/Artificial_Intelligence_for_Smart_Building_Log_Analysis.ipynb"
```

## Current Status

The project currently includes:

- Merging 14 floor-level CSV files.
- Creating a zone-minute feature table.
- Cleaning and resampling 2018-2019 data for floors 3, 4, 5, and 7 to 15-minute zone-level records.
- Running exploratory data analysis.
- Saving EDA summaries and plots.
- Checking NaN values in the cleaned 15-minute dataset.

Next steps are EDA on the cleaned selected-floor dataset, feature engineering, rolling-window features, train/test splitting, and anomaly detection experiments.

## Week 6 ML Data Preparation

The reusable Floor 6 ML preprocessing pipeline is implemented in:

```text
Artificial Intelligence for Smart-Building Log Analysis/ml/ML_Data_Preparation_Feature_Engineering.ipynb
```

Open it from the repository root:

```bash
jupyter notebook "Artificial Intelligence for Smart-Building Log Analysis/ml/ML_Data_Preparation_Feature_Engineering.ipynb"
```

The generated feature list separates model inputs from identifiers, raw reference values, and the v3 statistical baseline label.

The current repository includes the generated Week 6 ML-ready dataset:

```text
Artificial Intelligence for Smart-Building Log Analysis/outputs/floor6_zone1245_ml_ready/floor6_zone1245_ml_ready_2018_jul01_sep01.csv.gz
```

After installing dependencies, the Isolation Forest experiment can be run from the repository root:

```bash
python "Artificial Intelligence for Smart-Building Log Analysis/ml/isolation_forest_period_auto_huaye.py"
```

Run the test suite with:

```bash
python -m unittest discover "Artificial Intelligence for Smart-Building Log Analysis/tests"
```

## Week 7 Forecasting — Gavin

Lee's Week 7 preparation now builds prediction-ready train/validation/test datasets for `target_total_AC_power_kw_next_15min_mean`: the same-zone average `total_AC_power_kw` from t+1 through t+15. The lightweight configuration, manifest, training medians, and leakage audit are saved under `outputs/week7_prediction_ready_dataset/reports/`; the reproducible split datasets are committed under `outputs/week7_prediction_ready_dataset/data/` so the forecasting inputs are available after cloning.

Open [ML_Forecasting_Gavin.ipynb](<Artificial Intelligence for Smart-Building Log Analysis/ml/ML_Forecasting_Gavin.ipynb>) and run the cells in order. The notebook evaluates a persistence baseline, trains RandomForestRegressor candidates, selects by validation MAE, and reports MAE/RMSE/Bias for validation, test, and the 2018-09-06 actual table.

Forecasting outputs are saved under `outputs/week7_forecasting_gavin_rf/`. The main comparison files are `data/validation_predictions.csv.gz`, `data/test_predictions.csv.gz`, and `data/sep6_2018_09_06_predictions.csv.gz`; each contains `actual_next15_mean_kW`, `persistence_kW`, `rf_kW`, and error columns. Reports include overall metrics plus error breakdowns by zone, hour, calendar operating period, and zone-period.

The 2018-09-06 actual table is generated from `Raw data/2018Floor6.csv` with the same 34 columns as Lee's prediction-ready splits. It is stored under `outputs/week7_prediction_ready_dataset/data/` and uses the same future-15-minute average target.
