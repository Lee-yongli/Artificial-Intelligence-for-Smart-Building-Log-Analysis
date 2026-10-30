import assert from "node:assert/strict";
import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { createGunzip } from "node:zlib";
import { anomalyMetrics, comparisonMetrics, predictionChart, predictionMetrics } from "./dashboard_adapter.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await readFile(join(here, "dashboard_data.json"), "utf8"));
const display = JSON.parse(await readFile(join(here, "dashboard_display_values.json"), "utf8"));

const checks = [];
function check(name, fn) {
  fn();
  checks.push({ name, status: "pass" });
}

check("required top-level sections", () => {
  assert.deepEqual(Object.keys(data).sort(), ["anomaly", "comparison", "filters", "limitations", "metadata", "overview", "prediction", "representative_cases"].sort());
});

check("prediction series fields", () => {
  assert.ok(data.prediction.series.length > 0);
  for (const field of ["split", "zone", "period", "time", "actual", "predicted", "persistence", "sample_count"]) {
    assert.ok(Object.hasOwn(data.prediction.series[0], field), `missing ${field}`);
  }
});

check("anomaly summary fields", () => {
  assert.ok(data.anomaly.series.length > 0);
  for (const field of ["date", "zone", "period", "model", "observations", "anomaly_count", "anomaly_rate_pct"]) {
    assert.ok(Object.hasOwn(data.anomaly.series[0], field), `missing ${field}`);
  }
});

check("official anomaly totals", () => {
  const metrics = anomalyMetrics(data);
  assert.equal(metrics.observations, 362880);
  assert.equal(metrics.anomaly_count, 18144);
  assert.equal(metrics.anomaly_rate_pct, 5);
});

check("test Zone 1 prediction metrics", () => {
  const metrics = predictionMetrics(data, { split: "test", zone: 1, model: "rf" });
  assert.equal(metrics.n, 12945);
  assert.ok(Math.abs(metrics.mae_kw - 0.27809719297647817) < 1e-12);
  assert.ok(Math.abs(metrics.rmse_kw - 0.9474701409855618) < 1e-12);
  assert.equal(display.prediction.actual.length, display.prediction.predicted.length);
});

check("prediction chart filter", () => {
  const rows = predictionChart(data, { split: "test", zone: "Zone 1" });
  assert.equal(rows.length, display.prediction.labels.length);
  assert.ok(rows.every(row => row.zone === 1 && row.split === "test"));
});

check("official IF versus LOF comparison", () => {
  const metrics = comparisonMetrics(data, { mode: "official", right_model: "lof" });
  assert.equal(metrics.intersection, 1253);
  assert.equal(metrics.union, 35035);
  assert.ok(Math.abs(metrics.jaccard - 0.035764235764236) < 1e-12);
});

const csvChecks = [
  ["prediction_series.csv", 1904, ["time", "actual", "predicted"]],
  ["anomaly_summary.csv", 3888, ["zone", "period", "anomaly_count"]],
  ["prediction_metrics.csv", 640, ["mae_kw", "rmse_kw"]],
  ["comparison_summary.csv", 3888, ["intersection", "jaccard"]],
];

for (const [filename, expectedRows, requiredColumns] of csvChecks) {
  const text = await readFile(join(here, "data", filename), "utf8");
  const lines = text.trimEnd().split(/\r?\n/);
  const columns = lines[0].split(",");
  check(`${filename} row count and columns`, () => {
    assert.equal(lines.length - 1, expectedRows, `${filename} row count`);
    for (const column of requiredColumns) assert.ok(columns.includes(column), `${filename} missing ${column}`);
  });
}

async function inspectGzipCsv(filename) {
  const reader = createInterface({
    input: createReadStream(join(here, "data", filename)).pipe(createGunzip()),
    crlfDelay: Infinity,
  });
  let header;
  let rows = 0;
  for await (const line of reader) {
    if (header == null) header = line.split(",");
    else rows += 1;
  }
  return { header, rows };
}

const anomalyPoints = await inspectGzipCsv("anomaly_points.csv.gz");
check("anomaly_points.csv.gz fields and row count", () => {
  assert.equal(anomalyPoints.rows, 362880);
  for (const column of ["time", "zone", "period", "anomaly", "anomaly_score"]) assert.ok(anomalyPoints.header.includes(column));
});

const predictionPoints = await inspectGzipCsv("prediction_points.csv.gz");
check("prediction_points.csv.gz fields and row count", () => {
  assert.equal(predictionPoints.rows, 113622);
  for (const column of ["time", "zone", "period", "actual", "predicted"]) assert.ok(predictionPoints.header.includes(column));
});

const report = {
  status: "pass",
  checks,
  row_counts: {
    prediction_series: data.prediction.series.length,
    anomaly_summary: data.anomaly.series.length,
    prediction_metrics: data.prediction.metrics.length,
    comparison_summary: data.comparison.series.length,
    anomaly_points: anomalyPoints.rows,
    prediction_points: predictionPoints.rows,
  },
};
await writeFile(join(here, "validation_report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
