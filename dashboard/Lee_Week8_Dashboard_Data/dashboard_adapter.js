(function (global) {
  "use strict";

  function normaliseZone(zone) {
    if (zone == null || zone === "All") return null;
    return Number(String(zone).replace(/^Zone\s+/i, ""));
  }

  function matches(row, filters) {
    const zone = normaliseZone(filters.zone);
    return (zone == null || Number(row.zone) === zone) &&
      (filters.period == null || filters.period === "All" || row.period === filters.period) &&
      (filters.from == null || row.date >= filters.from) &&
      (filters.to == null || row.date <= filters.to);
  }

  const sum = (rows, field) => rows.reduce((total, row) => total + Number(row[field] || 0), 0);

  function anomalyMetrics(data, filters = {}) {
    const model = filters.model || data.anomaly.default_model;
    const rows = data.anomaly.series.filter(row => row.model === model && matches(row, filters));
    const observations = sum(rows, "observations");
    const anomaly_count = sum(rows, "anomaly_count");
    return { model, observations, anomaly_count, anomaly_rate_pct: observations ? 100 * anomaly_count / observations : null };
  }

  function anomalyDistribution(data, dimension = "zone", filters = {}) {
    if (!["zone", "period", "date"].includes(dimension)) throw new Error(`Unsupported anomaly dimension: ${dimension}`);
    const model = filters.model || data.anomaly.default_model;
    const groups = new Map();
    for (const row of data.anomaly.series) {
      if (row.model !== model || !matches(row, filters)) continue;
      const key = row[dimension];
      const result = groups.get(key) || { label: key, observations: 0, anomaly_count: 0 };
      result.observations += row.observations;
      result.anomaly_count += row.anomaly_count;
      groups.set(key, result);
    }
    return [...groups.values()].map(row => ({ ...row, anomaly_rate_pct: row.observations ? 100 * row.anomaly_count / row.observations : null }));
  }

  function predictionMetrics(data, filters = {}) {
    const split = filters.split || data.prediction.default_filter.split;
    const model = filters.model || data.prediction.default_filter.model;
    const rows = data.prediction.metrics.filter(row => row.split === split && row.model === model && matches(row, filters));
    const n = sum(rows, "n");
    return { split, model, n, mae_kw: n ? sum(rows, "sum_abs_error") / n : null, rmse_kw: n ? Math.sqrt(sum(rows, "sum_squared_error") / n) : null, bias_kw: n ? sum(rows, "sum_error") / n : null };
  }

  function predictionChart(data, filters = {}) {
    const split = filters.split || data.prediction.default_filter.split;
    const zone = normaliseZone(filters.zone == null ? data.prediction.default_filter.zone : filters.zone);
    if (zone == null) throw new Error("Prediction chart requires one zone");
    return data.prediction.series.filter(row => row.split === split && matches(row, { ...filters, zone })).sort((a, b) => a.time.localeCompare(b.time));
  }

  function comparisonMetrics(data, filters = {}) {
    const mode = filters.mode || "official";
    const right_model = filters.right_model || "lof";
    const rows = data.comparison.series.filter(row => row.mode === mode && row.right_model === right_model && matches(row, filters));
    const result = { mode, left_model: "period_if", right_model };
    for (const field of ["observations", "left_count", "right_count", "intersection", "union", "left_only", "right_only", "normal_both"]) result[field] = sum(rows, field);
    result.jaccard = result.union ? result.intersection / result.union : null;
    return result;
  }

  global.LeeDashboard = { anomalyMetrics, anomalyDistribution, predictionMetrics, predictionChart, comparisonMetrics };
})(window);
