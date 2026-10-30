import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { once } from "node:events";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { createGunzip, createGzip } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const sourceDir = join(here, "..", "Gavin_Week8_Dashboard_Delivery");
const outputDataDir = join(here, "data");
const sourcePath = join(sourceDir, "dashboard_data.json");

const sourceText = await readFile(sourcePath, "utf8");
const source = JSON.parse(sourceText);
const sourceHash = createHash("sha256").update(sourceText).digest("hex");

const numberSort = (a, b) => Number(a) - Number(b);
const textSort = (a, b) => String(a).localeCompare(String(b));
const unique = (values, sort = textSort) => [...new Set(values)].sort(sort);
const sum = (rows, field) => rows.reduce((total, row) => total + Number(row[field] ?? 0), 0);

const predictionSeries = source.prediction_hourly.map(row => ({
  split: row.split,
  zone: Number(row.zone),
  period: row.period,
  time: row.hour,
  date: row.hour.slice(0, 10),
  actual: row.actual,
  predicted: row.predicted,
  persistence: row.persistence,
  sample_count: row.n,
}));

const anomalySummary = source.anomaly_buckets.map(row => ({
  date: row.date,
  zone: Number(row.zone),
  period: row.period,
  model: row.model,
  observations: row.observations,
  anomaly_count: row.anomaly_count,
  anomaly_rate_pct: row.anomaly_rate_pct,
}));

const predictionMetrics = source.prediction_buckets.map(row => ({
  split: row.split,
  date: row.date,
  zone: Number(row.zone),
  period: row.period,
  model: row.model,
  n: row.n,
  sum_abs_error: row.sum_abs_error,
  sum_squared_error: row.sum_squared_error,
  sum_error: row.sum_error,
  mae_kw: row.mae_kw,
  rmse_kw: row.rmse_kw,
  bias_kw: row.bias_kw,
}));

const comparisonSummary = source.comparison_buckets.map(row => ({
  date: row.date,
  zone: Number(row.zone),
  period: row.period,
  mode: row.mode,
  left_model: row.left_model,
  right_model: row.right_model,
  observations: row.observations,
  left_count: row.left_count,
  right_count: row.right_count,
  intersection: row.intersection,
  union: row.union,
  left_only: row.left_only,
  right_only: row.right_only,
  normal_both: row.normal_both,
  jaccard: row.jaccard,
}));

function aggregatePrediction(filters) {
  const rows = predictionMetrics.filter(row => Object.entries(filters).every(([key, value]) => row[key] === value));
  const n = sum(rows, "n");
  return {
    ...filters,
    n,
    mae_kw: n ? sum(rows, "sum_abs_error") / n : null,
    rmse_kw: n ? Math.sqrt(sum(rows, "sum_squared_error") / n) : null,
    bias_kw: n ? sum(rows, "sum_error") / n : null,
  };
}

function aggregateAnomaly(groupField) {
  const rows = anomalySummary.filter(row => row.model === "if_calibrated");
  const groups = new Map();
  for (const row of rows) {
    const key = row[groupField];
    const value = groups.get(key) ?? { observations: 0, anomaly_count: 0 };
    value.observations += row.observations;
    value.anomaly_count += row.anomaly_count;
    groups.set(key, value);
  }
  return Object.fromEntries([...groups.entries()].map(([key, value]) => [
    groupField === "zone" ? `Zone ${key}` : key,
    value.observations ? 100 * value.anomaly_count / value.observations : null,
  ]));
}

const defaultPredictionFilter = { split: "test", zone: 1, model: "rf" };
const defaultPredictionMetrics = aggregatePrediction(defaultPredictionFilter);
const defaultPredictionSeries = predictionSeries
  .filter(row => row.split === "test" && row.zone === 1)
  .sort((a, b) => a.time.localeCompare(b.time));

const metadata = {
  schema_version: "1.0.0",
  source_schema_version: source.summary.schema_version,
  source_commit: source.summary.prediction.source_commit,
  source_file: "../Gavin_Week8_Dashboard_Delivery/dashboard_data.json",
  source_sha256: sourceHash,
  floor: source.summary.floor,
  units: {
    power: "kW",
    anomaly_rate_pct: "percent",
    time: "building-local naive timestamp; no timezone conversion",
  },
  join_policy: "Prediction and anomaly results are separate datasets with different coverage. Do not join them by row position or fill missing anomaly coverage as normal.",
  point_level_data: {
    anomaly: "data/anomaly_points.csv.gz",
    prediction: "data/prediction_points.csv.gz",
  },
  source_files: {
    anomaly: "../Gavin_Week8_Dashboard_Delivery/data/anomaly_results.csv.gz",
    prediction: "../Gavin_Week8_Dashboard_Delivery/data/prediction_results.csv.gz",
  },
};

const dashboardData = {
  metadata,
  filters: {
    zones: unique(predictionSeries.map(row => row.zone), numberSort),
    periods: unique([...predictionSeries, ...anomalySummary].map(row => row.period)),
    prediction_splits: unique(predictionSeries.map(row => row.split)),
    prediction_models: unique(predictionMetrics.map(row => row.model)),
    anomaly_models: unique(anomalySummary.map(row => row.model)),
  },
  overview: {
    total_records: source.summary.anomaly.observations,
    zones: source.summary.zones,
    date_start: source.summary.anomaly.date_start,
    date_end: source.summary.anomaly.date_end,
    primary_anomaly_model: source.summary.anomaly.primary_model,
    primary_anomaly_count: source.summary.anomaly.count,
    primary_anomaly_rate_pct: source.summary.anomaly.rate_pct,
    anomaly_event_count: source.summary.anomaly.event_count,
    prediction_model: source.summary.prediction.model,
    prediction_target: source.summary.prediction.target,
    prediction_horizon_minutes: source.summary.prediction.horizon_minutes,
  },
  prediction: {
    default_filter: defaultPredictionFilter,
    overall_metrics: source.summary.prediction.metrics,
    series: predictionSeries,
    metrics: predictionMetrics,
  },
  anomaly: {
    default_model: source.summary.anomaly.default_model_key,
    overall_metrics: source.summary.anomaly_overall,
    series: anomalySummary,
    events: source.anomaly_events,
  },
  comparison: {
    overall_metrics: source.summary.comparison,
    series: comparisonSummary,
  },
  representative_cases: source.representative_cases,
  limitations: source.summary.limitations,
};

const dashboardDisplay = {
  overview: {
    totalRecords: source.summary.anomaly.observations,
    zones: source.summary.zones.length,
    primaryIfAnomalies: source.summary.anomaly.count,
    primaryThresholdMode: source.summary.anomaly.threshold_mode,
    eventCount: source.summary.anomaly.event_count,
  },
  anomaly: {
    defaultModel: source.summary.anomaly.default_model_key,
    label: "Period-Specific IF (5% calibrated per period)",
    anomalyCount: source.summary.anomaly.count,
    anomalyRatePct: source.summary.anomaly.rate_pct,
    zoneRates: aggregateAnomaly("zone"),
    periodRates: aggregateAnomaly("period"),
  },
  prediction: {
    isDemo: false,
    model: source.summary.prediction.model,
    split: "test",
    zone: "Zone 1",
    forecastHorizon: "Mean AC power from t+1 to t+15 minutes",
    unit: "kW",
    mae: defaultPredictionMetrics.mae_kw,
    rmse: defaultPredictionMetrics.rmse_kw,
    n: defaultPredictionMetrics.n,
    labels: defaultPredictionSeries.map(row => row.time.replace("T", " ").replace(".000", "")),
    actual: defaultPredictionSeries.map(row => row.actual),
    predicted: defaultPredictionSeries.map(row => row.predicted),
    persistence: defaultPredictionSeries.map(row => row.persistence),
    chartCounts: defaultPredictionSeries.map(row => row.sample_count),
    metricScope: "All Zone 1 test minute rows; hourly means shown for chart only",
  },
  comparison: {
    models: [
      { name: "Period-Specific Isolation Forest", role: "Primary" },
      { name: "LOF", role: "Complementary" },
      { name: "Statistical Baseline", role: "Reference only" },
    ],
    pairs: source.summary.comparison,
  },
  data: {
    dataset: "CU-BEMS",
    floor: source.summary.floor,
    zones: source.summary.zones,
    anomalyRows: source.summary.anomaly.observations,
    dateStart: source.summary.anomaly.date_start,
    dateEnd: source.summary.anomaly.date_end,
    ifFeatures: source.summary.anomaly.if_features,
    lofFeatures: source.summary.anomaly.lof_features,
    forecastInputFieldsBeforeEncoding: source.summary.prediction.features_before_encoding,
    sourceCommit: source.summary.prediction.source_commit,
  },
};

function csvCell(value) {
  if (value == null) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filename, rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map(column => csvCell(row[column])).join(","));
  await writeFile(join(outputDataDir, filename), `${lines.join("\n")}\n`, "utf8");
}

function parseCsvLine(line) {
  const cells = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      cells.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  cells.push(value);
  return cells;
}

async function writeSelectedGzipCsv(sourceFilename, outputFilename, columns) {
  const input = createReadStream(join(sourceDir, "data", sourceFilename)).pipe(createGunzip());
  const reader = createInterface({ input, crlfDelay: Infinity });
  const gzip = createGzip({ level: 9 });
  const output = createWriteStream(join(outputDataDir, outputFilename));
  gzip.pipe(output);

  let sourceHeaders;
  let indexes;
  let rowCount = 0;
  gzip.write(`${columns.map(column => csvCell(column.output)).join(",")}\n`);

  for await (const line of reader) {
    if (!sourceHeaders) {
      sourceHeaders = parseCsvLine(line);
      indexes = columns.map(column => {
        const index = sourceHeaders.indexOf(column.source);
        if (index < 0) throw new Error(`${sourceFilename} missing source field ${column.source}`);
        return index;
      });
      continue;
    }
    const cells = parseCsvLine(line);
    const outputLine = `${indexes.map(index => csvCell(cells[index])).join(",")}\n`;
    if (!gzip.write(outputLine)) await once(gzip, "drain");
    rowCount += 1;
  }

  gzip.end();
  await once(output, "close");
  return rowCount;
}

await mkdir(outputDataDir, { recursive: true });
await writeFile(join(here, "dashboard_data.json"), `${JSON.stringify(dashboardData, null, 2)}\n`, "utf8");
await writeFile(join(here, "dashboard_data.js"), `window.LeeDashboardData = ${JSON.stringify(dashboardData)};\n`, "utf8");
await writeFile(join(here, "dashboard_display_values.json"), `${JSON.stringify(dashboardDisplay, null, 2)}\n`, "utf8");
await writeFile(join(here, "dashboard_display_values.js"), `window.LeeDashboardDisplay = ${JSON.stringify(dashboardDisplay)};\n`, "utf8");

await writeCsv("prediction_series.csv", predictionSeries,
  ["split", "zone", "period", "time", "date", "actual", "predicted", "persistence", "sample_count"]);
await writeCsv("anomaly_summary.csv", anomalySummary,
  ["date", "zone", "period", "model", "observations", "anomaly_count", "anomaly_rate_pct"]);
await writeCsv("prediction_metrics.csv", predictionMetrics,
  ["split", "date", "zone", "period", "model", "n", "sum_abs_error", "sum_squared_error", "sum_error", "mae_kw", "rmse_kw", "bias_kw"]);
await writeCsv("comparison_summary.csv", comparisonSummary,
  ["date", "zone", "period", "mode", "left_model", "right_model", "observations", "left_count", "right_count", "intersection", "union", "left_only", "right_only", "normal_both", "jaccard"]);

const anomalyPointRows = await writeSelectedGzipCsv("anomaly_results.csv.gz", "anomaly_points.csv.gz", [
  { source: "timestamp", output: "time" },
  { source: "floor", output: "floor" },
  { source: "zone", output: "zone" },
  { source: "period", output: "period" },
  { source: "if_anomaly", output: "anomaly" },
  { source: "if_score", output: "anomaly_score" },
  { source: "if_percentile", output: "anomaly_percentile" },
  { source: "if_decision_function", output: "anomaly_decision_function" },
  { source: "if_threshold_score", output: "anomaly_threshold_score" },
  { source: "lof_anomaly", output: "lof_anomaly" },
  { source: "lof_score", output: "lof_score" },
  { source: "baseline_reference_label", output: "baseline_anomaly" },
  { source: "total_AC_power_kw", output: "total_ac_power_kw" },
  { source: "light_power_kw", output: "light_power_kw" },
  { source: "plug_power_kw", output: "plug_power_kw" },
  { source: "temperature_c", output: "temperature_c" },
  { source: "humidity_pct", output: "humidity_pct" },
  { source: "lux", output: "lux" },
]);

const predictionPointRows = await writeSelectedGzipCsv("prediction_results.csv.gz", "prediction_points.csv.gz", [
  { source: "split", output: "split" },
  { source: "prediction_time", output: "time" },
  { source: "target_window_start", output: "target_window_start" },
  { source: "target_window_end", output: "target_window_end" },
  { source: "floor", output: "floor" },
  { source: "zone", output: "zone" },
  { source: "period", output: "period" },
  { source: "actual", output: "actual" },
  { source: "predicted", output: "predicted" },
  { source: "persistence", output: "persistence" },
  { source: "error", output: "error" },
  { source: "absolute_error", output: "absolute_error" },
  { source: "unit", output: "unit" },
]);

console.log(JSON.stringify({
  source_sha256: sourceHash,
  prediction_series_rows: predictionSeries.length,
  anomaly_summary_rows: anomalySummary.length,
  prediction_metric_rows: predictionMetrics.length,
  comparison_rows: comparisonSummary.length,
  anomaly_point_rows: anomalyPointRows,
  prediction_point_rows: predictionPointRows,
}, null, 2));
