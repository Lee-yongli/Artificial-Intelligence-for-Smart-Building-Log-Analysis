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
- `Raw Data/`: local-only raw CSV folder. This folder is ignored by Git and is not uploaded.

## Data Policy

Raw CU-BEMS CSV files are not included in this GitHub repository because they are large and should be managed outside Git.

To run the notebook locally, place the 14 raw CSV files here:

```text
Artificial Intelligence for Smart-Building Log Analysis/Raw Data/
```

Expected files:

```text
2018Floor1.csv ... 2018Floor7.csv
2019Floor1.csv ... 2019Floor7.csv
```

Most large generated `.csv.gz` outputs are excluded from Git. The Week 6 ML-ready dataset is committed so the LOF and Isolation Forest experiments can run immediately after cloning.

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
