# Artificial Intelligence for Smart-Building Log Analysis

This repository contains the working files for the Industry Research Project on AI-assisted smart-building log analysis.

## Project Overview

The project studies smart-building operational data from the CU-BEMS dataset. The current workflow focuses on data integration, exploratory data analysis, data cleaning, 15-minute resampling, missing-value inspection, and preparation for later feature engineering and anomaly detection.

## Repository Structure

- `Artificial Intelligence for Smart-Building Log Analysis/Artificial_Intelligence_for_Smart_Building_Log_Analysis.ipynb`
  Main Jupyter notebook for merging the 14 CSV files, creating the first zone-level feature table, running EDA, and checking missing values.

- `Artificial Intelligence for Smart-Building Log Analysis/outputs/`
  Small summary outputs and preview files that are suitable for GitHub review.

- `2 Proposal Report/`
  Proposal documents, presentation material, speaker notes, and supporting project documentation.

- `1 Progress Report/`
  Progress report template/material.

## Data Policy

The raw CSV files and large generated datasets are intentionally excluded from Git because they are large and duplicated.

To reproduce the workflow locally, place the 14 raw CSV files here:

```text
Artificial Intelligence for Smart-Building Log Analysis/Raw Data/
```

Expected raw file names:

```text
2018Floor1.csv ... 2018Floor7.csv
2019Floor1.csv ... 2019Floor7.csv
```

Large generated outputs such as `combined_wide_2018_2019.csv.gz`, `features_zone_minute_2018_2019.csv.gz`, and `cleaned_zone_15min_2018_2019.csv.gz` are also excluded. They should be regenerated locally from the notebook when needed.

## Environment

Install the main Python dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook in Jupyter:

```bash
jupyter notebook "Artificial Intelligence for Smart-Building Log Analysis/Artificial_Intelligence_for_Smart_Building_Log_Analysis.ipynb"
```

## Current Analysis Stage

The project has completed:

- Merging 14 floor-level CSV files.
- Creating a modelling-friendly zone-minute feature table.
- Running exploratory data analysis.
- Producing small EDA summaries and charts.
- Checking NaN counts in the cleaned 15-minute dataset.

Next recommended stage:

- Complete feature engineering on the cleaned 15-minute zone-level table.
- Add rolling-window features by floor, zone, and measurement.
- Prepare train/test splits for anomaly detection experiments.
