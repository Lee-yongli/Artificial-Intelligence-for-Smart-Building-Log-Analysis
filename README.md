# Artificial Intelligence for Smart-Building Log Analysis

This repository contains the analysis project folder for the Industry Research Project on AI-assisted smart-building log analysis.

## Main Project Folder

```text
Artificial Intelligence for Smart-Building Log Analysis/
```

This folder contains:

- `Artificial_Intelligence_for_Smart_Building_Log_Analysis.ipynb`: main Jupyter notebook.
- `merge_2018_2019_floors_3_4_5_7.ipynb`: focused notebook that only merges 2018-2019 data for floors 3, 4, 5, and 7.
- `Floor6_Zone1245_Data_Preparation_EDA_executed.ipynb`: Floor 6 Week 4 data preparation and EDA source of truth.
- `baseline_mad_floor6_zone1245_2018_jul01_sep01.ipynb`: Week 5 contextual median baseline and MAD statistical anomaly detection.
- `week5_floor6_contextual_mad.py`: reusable implementation used by the Week 5 notebook.
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

Large generated `.csv.gz` outputs are also excluded from Git. They can be regenerated from the notebook.
This includes the complete Week 5 row-level anomaly result; compact Week 5 summaries and figures remain reviewable in `outputs/`.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook "Artificial Intelligence for Smart-Building Log Analysis/Artificial_Intelligence_for_Smart_Building_Log_Analysis.ipynb"
```

## Current Status

The project currently includes:

- Data integration for 2018-2019 Floors 3, 4, 5, and 7: completed.
- Data cleaning and 15-minute floor-zone resampling: completed.
- Exploratory data analysis and validation: completed.
- Floor 6 Zone 1/2/4/5 data preparation and EDA for 2018-07-01 through 2018-09-01: completed.
- Week 5 contextual median baseline by zone, weekday/weekend, and hour: completed using the one-minute Floor 6 observations and the 15-minute rolling-mean AC feature.
- MAD-based Floor 6 statistical anomaly detection and Week 5 result summaries: completed.

The next modelling step is an independent unsupervised Isolation Forest comparison. MAD flags are statistical pseudo-labels and should not be used as ground-truth training labels.
