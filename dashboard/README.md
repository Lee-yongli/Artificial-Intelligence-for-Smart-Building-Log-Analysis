# Week 8 Smart Building Dashboard Delivery

## Main file

Open `dashboard.html` to view the integrated dashboard.

For the most consistent browser behaviour, open a terminal in this folder and run:

```bash
python3 -m http.server 8765
```

Then visit `http://127.0.0.1:8765/dashboard.html`.

## Included work

- Huaye's dashboard layout.
- Lee's normalized dashboard JSON/CSV data and adapters.
- Gavin's validated schema 3.0 model results embedded in the delivery data.
- Working anomaly filters for zone, period and model.
- Plain-language anomaly percentage explanations that update with the selected model, zone and operating period.
- Event-level priority review for all 134 sustained anomaly events.
- Transparent triage tiers based on duration and cross-method support.
- Representative IF-only, LOF-only and agreement cases.
- Test-set prediction view with zone and operating-period filters.
- Actual-versus-predicted chart, MAE, RMSE and record counts.
- Plain-language explanation of the prediction chart and error metrics.
- IF/LOF/statistical-baseline comparison values and chart.
- Explicit public-data disclosure and client-data integration roadmap.
- Automated validation report.

## Important interpretation

- Anomalies are candidates, not confirmed equipment faults.
- CU-BEMS is a public proxy dataset because client operational logs were not available during development.
- Client deployment requires schema mapping, local baseline reconstruction and operator validation.
- The formal Isolation Forest output uses a 5% per-period screening rule.
- Review tiers are workflow heuristics, not fault probabilities.
- Prediction and anomaly results cover different ranges and must not be joined by row position.
- Prediction error metrics use minute-level records; hourly values are used only for chart display.

## Key files

- `dashboard.html`: main dashboard page.
- `dashboard_app.js`: page rendering, filters and chart integration.
- `Lee_Week8_Dashboard_Data/dashboard_data.json`: filter-ready dashboard data.
- `Lee_Week8_Dashboard_Data/data/`: CSV and compressed point-level outputs.
- `Lee_Week8_Dashboard_Data/validation_report.json`: completed validation checks.
