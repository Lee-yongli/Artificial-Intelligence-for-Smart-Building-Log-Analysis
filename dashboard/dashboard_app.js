(function () {
  "use strict";

  const data = window.LeeDashboardData;
  const display = window.LeeDashboardDisplay;
  const api = window.LeeDashboard;

  if (!data || !display || !api) {
    const main = document.getElementById("main-content");
    if (main) main.innerHTML = '<div class="warning-note"><span class="note-icon">!</span><span>Dashboard data failed to load. Check the Lee_Week8_Dashboard_Data file paths.</span></div>';
    throw new Error("Lee dashboard data or adapter is unavailable");
  }

  const modelLabels = {
    if_calibrated: "Period-Specific IF",
    lof: "LOF",
    if_auto_reference: "IF Auto Threshold (Reference)",
    baseline_reference: "Statistical Baseline (Reference)",
  };

  const formatInteger = value => new Intl.NumberFormat("en-AU").format(value);
  const formatMetric = value => value == null ? "No data" : Number(value).toFixed(3);
  const formatDateTime = value => String(value).replace("T", " ").slice(0, 16);
  const formatRate = value => value == null ? "No data" : `${Number(value).toFixed(2)}%`;
  const escapeHtml = value => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function renderKpi(container, items) {
    container.innerHTML = items.map(item => `
      <article class="kpi-card">
        <span class="kpi-label">${escapeHtml(item.label)}</span>
        <strong class="kpi-value">${escapeHtml(item.value)}</strong>
        ${item.caption ? `<span class="kpi-caption">${escapeHtml(item.caption)}</span>` : ""}
      </article>`).join("");
  }

  function renderBarChart(container, values, options = {}) {
    const entries = Object.entries(values).filter(([, value]) => value != null);
    if (!entries.length) {
      container.innerHTML = '<p class="chart-axis-note">No data for this selection.</p>';
      return;
    }
    const max = options.max ?? Math.max(...entries.map(([, value]) => Number(value)), 1);
    const formatter = options.formatter ?? (value => `${Number(value).toFixed(2)}%`);
    container.innerHTML = entries.map(([label, value]) => {
      const width = Math.max(0, Math.min(100, (Number(value) / max) * 100));
      return `<div class="bar-row">
        <span class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        <span class="bar-track" aria-hidden="true"><span class="bar-fill" style="--bar-width: ${width}%; --bar-color: ${options.color || "#1479b8"}"></span></span>
        <span class="bar-value">${escapeHtml(formatter(value))}</span>
      </div>`;
    }).join("");
  }

  function renderPredictionChart(container, rows) {
    if (!rows.length) {
      container.innerHTML = '<p class="chart-axis-note">No prediction data for this selection.</p>';
      return;
    }

    const width = 900;
    const height = 330;
    const pad = { left: 52, right: 20, top: 18, bottom: 48 };
    const values = rows.flatMap(row => [Number(row.actual), Number(row.predicted)]).filter(Number.isFinite);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const margin = Math.max((rawMax - rawMin) * 0.08, 0.25);
    const min = Math.max(0, rawMin - margin);
    const max = rawMax + margin;
    const x = index => pad.left + index * ((width - pad.left - pad.right) / Math.max(rows.length - 1, 1));
    const y = value => pad.top + (max - value) * ((height - pad.top - pad.bottom) / Math.max(max - min, 1));
    const points = field => rows.map((row, index) => `${x(index)},${y(Number(row[field]))}`).join(" ");
    const yTicks = Array.from({ length: 5 }, (_, index) => min + ((max - min) * index / 4));
    const tickStep = Math.max(1, Math.ceil(rows.length / 7));
    const xTicks = rows.map((row, index) => ({ row, index })).filter(({ index }) => index === 0 || index === rows.length - 1 || index % tickStep === 0);

    container.innerHTML = `<svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Line chart comparing actual and predicted AC power in kilowatts">
      ${yTicks.map(value => `<line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${y(value)}" y2="${y(value)}"></line><text class="axis-text" x="${pad.left - 9}" y="${y(value) + 3}" text-anchor="end">${value.toFixed(1)}</text>`).join("")}
      ${xTicks.map(({ row, index }) => `<text class="axis-text" x="${x(index)}" y="${height - 15}" text-anchor="middle">${escapeHtml(row.time.slice(5, 16).replace("T", " "))}</text>`).join("")}
      <polyline class="actual-line" points="${points("actual")}"></polyline>
      <polyline class="predicted-line" points="${points("predicted")}"></polyline>
    </svg>`;
  }

  function initialiseOverview() {
    renderKpi(document.getElementById("overview-kpis"), [
      { label: "Total Records", value: formatInteger(display.overview.totalRecords), caption: "Validated anomaly dataset" },
      { label: "Zones", value: display.overview.zones, caption: "Zones 1, 2, 4 and 5" },
      { label: "Period-Specific IF Anomalies", value: formatInteger(display.overview.primaryIfAnomalies), caption: "Candidate observations" },
      { label: "Sustained Events", value: formatInteger(data.overview.anomaly_event_count), caption: "At least 30 minutes" },
    ]);
  }

  function updateAnomaly() {
    const zone = document.getElementById("zone-filter").value;
    const period = document.getElementById("period-filter").value;
    const model = document.getElementById("model-filter").value;
    const label = modelLabels[model] || model;
    const filters = { zone, period, model };
    const metrics = api.anomalyMetrics(data, filters);
    const zoneRows = api.anomalyDistribution(data, "zone", filters);
    const periodRows = api.anomalyDistribution(data, "period", filters);
    const zoneValues = Object.fromEntries(zoneRows.map(row => [`Zone ${row.label}`, row.anomaly_rate_pct]));
    const periodValues = Object.fromEntries(periodRows.map(row => [row.label, row.anomaly_rate_pct]));

    document.getElementById("filter-summary").innerHTML = `Current selection: <strong>${escapeHtml(zone)}</strong> · <strong>${escapeHtml(period)}</strong> · <strong>${escapeHtml(label)}</strong>`;
    document.getElementById("zone-chart-model").textContent = label;
    document.getElementById("period-chart-model").textContent = label;
    renderKpi(document.getElementById("anomaly-kpis"), [
      { label: "Selected Model", value: label, caption: model.includes("reference") ? "Reference result" : "Filter selection" },
      { label: "Anomaly Count", value: formatInteger(metrics.anomaly_count), caption: `${formatInteger(metrics.observations)} observations` },
      { label: "Anomaly Rate", value: metrics.anomaly_rate_pct == null ? "No data" : `${metrics.anomaly_rate_pct.toFixed(2)}%`, caption: "Selected scope" },
    ]);
    renderBarChart(document.getElementById("zone-chart"), zoneValues);
    renderBarChart(document.getElementById("period-chart"), periodValues, { color: "#294b8f" });
    renderAnomalyExplanation({ metrics, zoneRows, periodRows, zone, period, model, label });
  }

  function rateAsFrequency(rate) {
    if (rate == null) return "No percentage is available for this selection.";
    const value = Number(rate);
    if (value < 1) return `about ${(value * 10).toFixed(1)} of every 1,000 records`;
    return `about ${value.toFixed(1)} of every 100 records`;
  }

  function highestRate(rows) {
    return rows
      .filter(row => row.anomaly_rate_pct != null)
      .sort((left, right) => Number(right.anomaly_rate_pct) - Number(left.anomaly_rate_pct))[0] || null;
  }

  function renderAnomalyExplanation({ metrics, zoneRows, periodRows, zone, period, model, label }) {
    const selectedRate = metrics.anomaly_rate_pct;
    const zoneLeader = highestRate(zoneRows);
    const periodLeader = highestRate(periodRows);
    const calibrated = model === "if_calibrated";
    const zoneValue = zoneLeader ? `Zone ${zoneLeader.label} · ${formatRate(zoneLeader.anomaly_rate_pct)}` : "No zone result";
    const periodValue = calibrated && period === "All"
      ? "5.00% per period"
      : periodLeader ? `${periodLeader.label} · ${formatRate(periodLeader.anomaly_rate_pct)}` : "No period result";
    const zoneMeaning = zoneRows.length > 1
      ? `The highest displayed zone rate is ${zoneValue}. This identifies where the current model creates the greatest review workload, so facilities staff can decide where to inspect first.`
      : `${zoneValue} is the share of records flagged within the selected zone. Use it to size that zone's review workload and compare it with the same model over time.`;
    const periodMeaning = calibrated
      ? "The primary Isolation Forest intentionally flags 5% within each operating period. Equal 5% bars are therefore a controlled alert budget, not evidence that every period has the same fault risk."
      : `${periodValue} is the highest displayed operating-period rate. It helps reveal when unusual behaviour is concentrated, such as after-hours or ramp-up operation.`;
    const summary = calibrated
      ? `For ${escapeHtml(zone)} and ${escapeHtml(period)}, the ${escapeHtml(label)} rate is a screening quota. Smart Facilities SA can use the count to plan investigation capacity, then use duration and model agreement in Priority Review to decide which events deserve attention first.`
      : `For ${escapeHtml(zone)} and ${escapeHtml(period)}, compare this percentage with the same model across zones and periods. A higher percentage means more records require review; it does not by itself prove poorer equipment condition.`;

    document.getElementById("anomaly-explanation").innerHTML = `
      <div class="explanation-grid">
        <section class="explanation-card">
          <span class="explanation-value">${escapeHtml(formatRate(selectedRate))}</span>
          <h4>Selected anomaly rate</h4>
          <p><strong>Meaning:</strong> ${escapeHtml(rateAsFrequency(selectedRate))} were flagged by ${escapeHtml(label)} in the selected scope.</p>
          <p><strong>Company use:</strong> Estimate alert volume and the staff time needed for review.</p>
        </section>
        <section class="explanation-card">
          <span class="explanation-value">${escapeHtml(zoneValue)}</span>
          <h4>Percentage by zone</h4>
          <p>${escapeHtml(zoneMeaning)}</p>
          <p><strong>Caution:</strong> A high value can also reflect schedules, sensor quality or unusual but valid operation.</p>
        </section>
        <section class="explanation-card">
          <span class="explanation-value">${escapeHtml(periodValue)}</span>
          <h4>Percentage by operating period</h4>
          <p>${escapeHtml(periodMeaning)}</p>
          <p><strong>Company use:</strong> Compare working hours, ramp periods and after-hours behaviour in context.</p>
        </section>
      </div>
      <p class="explanation-summary"><strong>Recommended interpretation:</strong> ${summary}</p>`;
  }

  function updatePrediction() {
    const split = "test";
    const zone = Number(document.getElementById("prediction-zone-filter").value);
    const period = document.getElementById("prediction-period-filter").value;
    const filters = { split, zone, period, model: "rf" };
    const metrics = api.predictionMetrics(data, filters);
    const rows = api.predictionChart(data, filters);
    const scope = `Test · Zone ${zone}${period === "All" ? "" : ` · ${period}`}`;

    document.getElementById("prediction-filter-summary").innerHTML = `Evaluation data: <strong>Test set only</strong> · <strong>Zone ${zone}</strong> · <strong>${escapeHtml(period)}</strong>`;
    document.getElementById("prediction-chart-scope").textContent = scope;
    renderPredictionChart(document.getElementById("prediction-chart"), rows);
    renderKpi(document.getElementById("prediction-kpis"), [
      { label: "MAE", value: metrics.mae_kw == null ? "No data" : `${formatMetric(metrics.mae_kw)} kW`, caption: "Minute-level absolute error" },
      { label: "RMSE", value: metrics.rmse_kw == null ? "No data" : `${formatMetric(metrics.rmse_kw)} kW`, caption: "Minute-level root mean square error" },
      { label: "Records", value: formatInteger(metrics.n), caption: "Underlying minute rows" },
      { label: "Forecast Horizon", value: "15 minutes", caption: "Mean AC power, t+1 to t+15" },
    ]);
    renderPredictionExplanation({ metrics, zone, period });
  }

  function renderPredictionExplanation({ metrics, zone, period }) {
    const mae = metrics.mae_kw == null ? null : Number(metrics.mae_kw);
    const rmse = metrics.rmse_kw == null ? null : Number(metrics.rmse_kw);
    const scope = `Zone ${zone}${period === "All" ? " across all operating periods" : ` during ${period}`}`;
    const errorReading = mae == null
      ? "There are no test records for this selection."
      : `On the unseen test data, the prediction differs from the observed AC power by ${mae.toFixed(3)} kW on average. RMSE is ${rmse.toFixed(3)} kW and gives extra weight to occasional larger misses.`;

    document.getElementById("prediction-explanation").innerHTML = `
      <article class="panel"><div class="panel-body">
        <h3>How to read the chart</h3>
        <ul class="interpretation-list">
          <li><strong>Blue line:</strong> AC power that was actually observed in the test period.</li>
          <li><strong>Red dashed line:</strong> the model's 15-minute-ahead estimate.</li>
          <li>Lines close together mean the model followed the demand pattern; wider gaps show under- or over-estimation.</li>
        </ul>
      </div></article>
      <article class="panel"><div class="panel-body">
        <h3>What this means for the company</h3>
        <p>${escapeHtml(errorReading)}</p>
        <p><strong>Business use:</strong> ${escapeHtml(scope)} can be compared with its short-term expected load to support planning and provide context for unusual changes.</p>
        <p><strong>Limit:</strong> A forecast gap is not automatically a fault; schedules, weather, occupancy and sensor issues should be checked.</p>
      </div></article>`;
  }

  function eventEvidence(event) {
    const lofPct = event.anomaly_points ? 100 * event.lof_overlap_points / event.anomaly_points : 0;
    const baselinePct = Number(event.baseline_overlap_pct || 0);
    const labels = [];
    if (lofPct > 0) labels.push(`LOF ${lofPct.toFixed(0)}%`);
    if (baselinePct > 0) labels.push(`Baseline ${baselinePct.toFixed(0)}%`);
    return {
      count: Number(lofPct > 0) + Number(baselinePct > 0),
      label: labels.length ? labels.join(" · ") : "IF only",
    };
  }

  function eventReviewTier(event) {
    const evidence = eventEvidence(event);
    if (evidence.count === 2 || (evidence.count === 1 && event.duration_minutes >= 120) || event.duration_minutes >= 240) {
      return { key: "review-first", label: "Review first", rank: 0 };
    }
    if (evidence.count === 1 || event.duration_minutes >= 60) {
      return { key: "review-next", label: "Review next", rank: 1 };
    }
    return { key: "monitor", label: "Monitor", rank: 2 };
  }

  function eventNextStep(event) {
    const evidence = eventEvidence(event);
    const afterHours = ["night_off", "evening_off", "weekend"].includes(event.operating_period);
    if (afterHours && event.duration_minutes >= 120) return "Check schedule overrides and after-hours operation.";
    if (evidence.count === 2) return "Inspect trend, equipment state and operator logs.";
    if (event.baseline_overlap_points > 0) return "Check expected range and operating schedule.";
    if (event.lof_overlap_points > 0) return "Compare with nearby operating states and sensors.";
    return "Monitor for recurrence; seek corroborating evidence.";
  }

  function updatePriorityReview() {
    const zone = document.getElementById("priority-zone-filter").value;
    const period = document.getElementById("priority-period-filter").value;
    const tier = document.getElementById("priority-tier-filter").value;
    const tierLabel = document.getElementById("priority-tier-filter").selectedOptions[0].textContent;
    const allEvents = data.anomaly.events.map(event => ({
      ...event,
      evidence: eventEvidence(event),
      tier: eventReviewTier(event),
    }));
    const events = allEvents
      .filter(event => zone === "All" || String(event.zone) === zone)
      .filter(event => period === "All" || event.operating_period === period)
      .filter(event => tier === "All" || event.tier.key === tier)
      .sort((left, right) => left.tier.rank - right.tier.rank || right.evidence.count - left.evidence.count || right.duration_minutes - left.duration_minutes || left.if_event_id - right.if_event_id);
    const durations = events.map(event => Number(event.duration_minutes)).sort((a, b) => a - b);
    const medianDuration = durations.length ? durations[Math.floor(durations.length / 2)] : 0;
    const reviewFirst = events.filter(event => event.tier.key === "review-first").length;

    document.getElementById("priority-filter-summary").innerHTML = `Current selection: <strong>${zone === "All" ? "All zones" : `Zone ${escapeHtml(zone)}`}</strong> · <strong>${escapeHtml(period === "All" ? "All periods" : period)}</strong> · <strong>${escapeHtml(tierLabel)}</strong>`;
    document.getElementById("priority-table-scope").textContent = `${formatInteger(events.length)} events · first 20 shown`;
    renderKpi(document.getElementById("priority-kpis"), [
      { label: "Matching Events", value: formatInteger(events.length), caption: "Sustained IF candidates" },
      { label: "Review First", value: formatInteger(reviewFirst), caption: "Duration and/or cross-method support" },
      { label: "Median Duration", value: `${formatInteger(medianDuration)} min`, caption: "Selected event queue" },
      { label: "Longest Event", value: `${formatInteger(durations.at(-1) || 0)} min`, caption: "Selected event queue" },
    ]);
    document.getElementById("priority-table-body").innerHTML = events.length ? events.slice(0, 20).map(event => `
      <tr>
        <td><span class="priority-badge ${event.tier.key}">${escapeHtml(event.tier.label)}</span></td>
        <td><strong>#${formatInteger(event.if_event_id)}</strong><small>${formatInteger(event.anomaly_points)} anomaly points</small></td>
        <td><strong>${escapeHtml(formatDateTime(event.start_time))}</strong><small>to ${escapeHtml(formatDateTime(event.end_time))}</small></td>
        <td><strong>Zone ${escapeHtml(event.zone)}</strong><small>${escapeHtml(event.operating_period)}</small></td>
        <td><strong>${formatInteger(event.duration_minutes)} min</strong><small>minimum: ${formatInteger(event.required_duration_minutes)} min</small></td>
        <td><strong>${escapeHtml(event.evidence.label)}</strong><small>Period-specific IF event</small></td>
        <td>${escapeHtml(eventNextStep(event))}</td>
      </tr>`).join("") : '<tr><td colspan="7">No events match this selection.</td></tr>';
  }

  function initialiseRepresentativeCases() {
    document.getElementById("representative-cases").innerHTML = data.representative_cases.map(item => {
      const caseLabel = item.case === "Both" ? "IF + LOF agreement" : item.case;
      return `<article class="case-card">
        <h3>${escapeHtml(caseLabel)}</h3>
        <p>${escapeHtml(formatDateTime(item.timestamp))} · Zone ${escapeHtml(item.zone)} · ${escapeHtml(item.period)}</p>
        <div class="case-metrics">
          <div class="case-metric"><span>AC power</span><strong>${Number(item.total_AC_power_kw).toFixed(2)} kW</strong></div>
          <div class="case-metric"><span>Temperature</span><strong>${Number(item.temperature_c).toFixed(2)} °C</strong></div>
          <div class="case-metric"><span>IF percentile</span><strong>${(100 * Number(item.if_percentile)).toFixed(1)}%</strong></div>
          <div class="case-metric"><span>Baseline evidence</span><strong>${item.baseline_reference_label ? "Yes" : "No"}</strong></div>
        </div>
      </article>`;
    }).join("");
  }

  function initialiseComparison() {
    document.getElementById("model-cards").innerHTML = display.comparison.models.map(model => `<article class="model-card"><h3>${escapeHtml(model.name)}</h3><p>${escapeHtml(model.role)}</p></article>`).join("");
    const pairs = display.comparison.pairs;
    document.getElementById("comparison-table-body").innerHTML = pairs.map(pair => {
      const mode = pair.mode === "official" ? "Official" : "Auto reference";
      const right = pair.right_model === "lof" ? "LOF" : "Statistical Baseline";
      return `<tr><td>${escapeHtml(`${mode} IF vs ${right}`)}</td><td>${formatInteger(pair.intersection)}</td><td>${pair.jaccard.toFixed(4)}</td></tr>`;
    }).join("");
    const chartValues = Object.fromEntries(pairs.map(pair => {
      const mode = pair.mode === "official" ? "Official" : "Auto";
      const right = pair.right_model === "lof" ? "LOF" : "Baseline";
      return [`${mode} / ${right}`, pair.jaccard];
    }));
    renderBarChart(document.getElementById("jaccard-chart"), chartValues, {
      color: "#52446f",
      formatter: value => Number(value).toFixed(4),
    });
  }

  function initialiseDataProfile() {
    const profile = [
      ["Dataset", display.data.dataset],
      ["Data role", "Public proxy / proof of concept"],
      ["Floor", display.data.floor],
      ["Zones", display.data.zones.join(", ")],
      ["Anomaly rows", formatInteger(display.data.anomalyRows)],
      ["Date range", `${display.data.dateStart} to ${display.data.dateEnd}`],
      ["IF physical features", display.data.ifFeatures],
      ["LOF features", display.data.lofFeatures],
      ["RF input fields", display.data.forecastInputFieldsBeforeEncoding],
      ["Source commit", display.data.sourceCommit.slice(0, 8)],
    ];
    document.getElementById("data-profile").innerHTML = profile.map(([label, value]) => `<div class="data-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
    const workflow = ["CU-BEMS", "Data Preparation", "ML-Ready Dataset", "Period-Specific IF / LOF", "RF Forecast", "Dashboard"];
    document.getElementById("workflow").innerHTML = workflow.map((step, index) => `${index ? '<div class="workflow-arrow" aria-hidden="true">↓</div>' : ""}<div class="workflow-step">${escapeHtml(step)}</div>`).join("");
  }

  function initialiseLimitations() {
    document.getElementById("limitations-list").innerHTML = data.limitations.slice(0, 5).map(item => `<li>${escapeHtml(item)}</li>`).join("");
  }

  function showSection(sectionName, options = {}) {
    const requested = document.getElementById(`${sectionName}-section`);
    if (!requested) return;
    document.querySelectorAll(".page-section").forEach(section => { section.hidden = section !== requested; });
    document.querySelectorAll(".nav-button").forEach(button => {
      const active = button.dataset.section === sectionName;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    document.getElementById("page-title").textContent = requested.dataset.title;
    document.title = `${requested.dataset.title} — Smart Building`;
    document.body.classList.remove("menu-open");
    document.getElementById("mobile-menu-button").setAttribute("aria-expanded", "false");
    if (!options.skipHistory) history.replaceState(null, "", `#${sectionName}`);
    if (!options.preserveScroll) window.scrollTo({ top: 0, behavior: "smooth" });
    if (options.focus) document.getElementById("main-content").focus({ preventScroll: true });
  }

  function initialiseInteractions() {
    document.querySelectorAll(".nav-button").forEach(button => button.addEventListener("click", () => showSection(button.dataset.section, { focus: true })));
    document.querySelectorAll(".module-card").forEach(card => card.addEventListener("click", () => showSection(card.dataset.targetSection, { focus: true })));
    ["zone-filter", "period-filter", "model-filter"].forEach(id => document.getElementById(id).addEventListener("change", updateAnomaly));
    ["priority-zone-filter", "priority-period-filter", "priority-tier-filter"].forEach(id => document.getElementById(id).addEventListener("change", updatePriorityReview));
    ["prediction-zone-filter", "prediction-period-filter"].forEach(id => document.getElementById(id).addEventListener("change", updatePrediction));

    const menuButton = document.getElementById("mobile-menu-button");
    menuButton.addEventListener("click", () => {
      const open = document.body.classList.toggle("menu-open");
      menuButton.setAttribute("aria-expanded", String(open));
    });
    document.getElementById("sidebar-overlay").addEventListener("click", () => {
      document.body.classList.remove("menu-open");
      menuButton.setAttribute("aria-expanded", "false");
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        document.body.classList.remove("menu-open");
        menuButton.setAttribute("aria-expanded", "false");
      }
    });

    const hashSection = location.hash.slice(1);
    const initialSection = document.getElementById(`${hashSection}-section`) ? hashSection : "dashboard";
    showSection(initialSection, { skipHistory: !location.hash, preserveScroll: true });
  }

  initialiseOverview();
  updateAnomaly();
  updatePriorityReview();
  initialiseRepresentativeCases();
  updatePrediction();
  initialiseComparison();
  initialiseDataProfile();
  initialiseLimitations();
  initialiseInteractions();
})();
