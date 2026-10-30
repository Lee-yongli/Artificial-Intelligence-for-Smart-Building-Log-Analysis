# Lee Week 8 — Dashboard data integration

This folder is the browser-facing data layer between Gavin's final model delivery and Huaye/Penelope's dashboard work. It standardises field names, exposes filter-ready tables and keeps anomaly and prediction coverage separate.

## Use in the existing HTML

For a page that may be opened directly with `file://`, load the non-module files before the dashboard's own script:

```html
<script src="../Lee_Week8_Dashboard_Data/dashboard_data.js"></script>
<script src="../Lee_Week8_Dashboard_Data/dashboard_display_values.js"></script>
<script src="../Lee_Week8_Dashboard_Data/dashboard_adapter.js"></script>
```

The page can then read:

```javascript
const data = window.LeeDashboardData;
const display = window.LeeDashboardDisplay;

const anomaly = window.LeeDashboard.anomalyMetrics(data, {
  model: "if_calibrated",
  zone: "Zone 1",
  period: "core_work"
});

const prediction = window.LeeDashboard.predictionChart(data, {
  split: "test",
  zone: "Zone 1"
});
```

When the dashboard is served over HTTP, `dashboard_adapter.mjs` can be imported as an ES module and `dashboard_data.json` can be fetched.

## Main outputs

| File | Purpose |
|---|---|
| `dashboard_data.json` | Full filter-ready data contract for the dashboard |
| `dashboard_data.js` | Same data as a global variable; works when opening HTML directly |
| `dashboard_display_values.json/js` | Default KPI and chart values for Test / Zone 1 |
| `dashboard_adapter.js` | Traditional-script filtering and metric functions |
| `dashboard_adapter.mjs` | ES-module version of the adapter |
| `data/prediction_series.csv` | `zone`, `period`, `time`, `actual`, `predicted` and persistence chart data |
| `data/anomaly_summary.csv` | Daily anomaly counts and rates by zone, period and model |
| `data/prediction_metrics.csv` | Filterable MAE/RMSE sufficient statistics |
| `data/comparison_summary.csv` | Filterable IF/LOF/baseline agreement counts |
| `data/anomaly_points.csv.gz` | Standardized point rows with `time`, `zone`, `period`, `anomaly` and score |
| `data/prediction_points.csv.gz` | Standardized point rows with `time`, `zone`, `period`, `actual` and `predicted` |
| `data_contract.md` | Field meanings, units, aggregation and join rules |
| `validation_report.json` | Automated row, field and headline-value checks |

## Default dashboard values

- Records: 362,880
- Zones: 1, 2, 4 and 5
- Primary anomaly candidates: 18,144 (5% calibrated per operating period)
- Default prediction view: Test / Zone 1
- Default prediction n: 12,945
- Default MAE: 0.27809719297647817 kW
- Default RMSE: 0.9474701409855618 kW

## Rebuild and verify

Run from this folder with Node.js 18 or later:

```bash
node build_dashboard_data.mjs
node verify_dashboard_data.mjs
```

The build reads `../Gavin_Week8_Dashboard_Delivery/dashboard_data.json`. It does not train models or alter Gavin's delivery.
