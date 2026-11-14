"use strict";
/* Meteo Accuracy - interfaccia del sito (statico, nessun server). */

const $ = s => document.querySelector(s);
const DATA = "data/";
const VAR_LABEL = { t2m: "Mean temperature", tmax: "Maximum temperature", tmin: "Minimum temperature", prcp: "Precipitation" };
const VAR_UNIT = { t2m: "°C", tmax: "°C", tmin: "°C", prcp: "mm" };
const AGG_LABEL = { "1h": "Hourly", "6h": "6-hourly", "24h": "Daily" };
const COMBO = { BEST: "★ Best", WEIGHTED: "⚖ Weighted mean", GPR: "◆ GPR" };
const PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
  "#7f7f7f", "#ff7f0e", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"];
const S = { meta: null, scores: [], hold: [], trend: [], weights: [], gpval: [], gpst: [], stations: [], maps: {},
  evScores: [], evList: [], evDef: [], region: "italia", italy: null };
const inR = r => !r.region || r.region === S.region;          // filtro per regione selezionata

const fetchJSON = (f, fallback = null) => fetch(DATA + f).then(r => (r.ok ? r.json() : fallback)).catch(() => fallback);
const fmt = (v, d = 2) => (v == null || !isFinite(v) ? "–" : (+v).toFixed(d));
const pname = id => COMBO[id] || (S.meta.providers.find(p => p.id === id) || {}).name || id;
const leadLabel = l => (l === 0 ? "Today" : l === 1 ? "Tomorrow (+1 d)" : `+${l} days`);
const leadShort = l => (l === 0 ? "d0" : `+${l}d`);
function opts(sel, items, value) {
  sel.innerHTML = items.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
  if (value != null && items.some(([v]) => String(v) === String(value))) sel.value = value;
}
const varsFor = agg => (agg === "24h" ? ["t2m", "tmax", "tmin", "prcp"] : ["t2m", "prcp"]);
const charts = {};
function drawChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
}

// ------------------------------------------------------------------ avvio
(async function init() {
  [S.meta, S.scores, S.hold, S.trend, S.weights, S.gpval, S.gpst, S.stations] = await Promise.all([
    fetchJSON("meta.json", {}), fetchJSON("scores.json", []), fetchJSON("scores_holdout.json", []),
    fetchJSON("trend.json", []), fetchJSON("weights.json", []), fetchJSON("gpr_validation.json", []),
    fetchJSON("gpr_station_errors.json", []), fetchJSON("stations.json", [])]);
  [S.evScores, S.evList, S.evDef, S.bt] = await Promise.all([fetchJSON("events_scores.json", []),
    fetchJSON("events_list.json", []), fetchJSON("events_def.json", []), fetchJSON("gpr_backtest.json", [])]);
  S.meta.providers = S.meta.providers || [];
  S.meta.regions = S.meta.regions || [{ id: "italia", name: "Italia", center: [42.2, 12.6, 6], providers: [] }];
  S.region = S.meta.regions[0].id;
  opts($("#region"), S.meta.regions.map(r => [r.id, r.name]), S.region);
  $("#region").onchange = () => {
    S.region = $("#region").value; refreshAll();
    const cur = document.querySelector("#tabs button.active")?.dataset.tab;
    if (!isVerified() && ["rank", "events", "valid"].includes(cur)) showTab("maps");   // nessuna verifica: si va alle mappe
  };
  S.italy = await fetch(DATA + "italy.geojson").then(r => r.json()).catch(() => null);
  $("#updated").textContent = S.meta.last_day
    ? `Verified up to ${S.meta.last_day} · updated ${S.meta.built?.replace("T", " ").slice(0, 16)} UTC` : "No data yet";
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
  initRank(); initEvents(); initMaps(); initPoint(); initValid(); initData();
  refreshAll();
  const h = location.hash.slice(1);
  if (h) showTab(h.split("?")[0]);
})();

const regionOf = id => S.meta.regions.find(r => r.id === id) || S.meta.regions[0];
const NOVER = "This region is not verified: the site only shows the model forecasts (Maps and Point forecast tabs); the values are archived in the data.";
const isVerified = () => regionOf(S.region).verify !== false;
function refreshAll() {
  renderRank(); renderEvents(); renderValid();
  $("#m-stations-wrap").style.display = isVerified() ? "" : "none";
  const c = regionOf(S.region).center || [42, 12, 5];
  S.map.setView([c[0], c[1]], c[2]);
  loadMap();
}

function showTab(t) {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + t));
  history.replaceState(null, "", "#" + t);
  if (t === "maps" && S.map) setTimeout(() => S.map.invalidateSize(), 50);
  if (t === "point" && S.pmap) setTimeout(() => S.pmap.invalidateSize(), 50);
}

// ================================================================== CLASSIFICA
function metricsFor(v, agg) {
  const m = [["rmse", "RMSE (root mean square error)"], ["mae", "MAE (mean absolute error)"], ["bias", "Bias (forecast − observed)"], ["corr", "Correlation"], ["r2", "R² (explained variance)"]];
  if (v === "prcp") {
    (S.meta.prcp_thresholds?.[agg] || []).forEach((t, i) => {
      m.push([`ets_${i + 1}`, `ETS ≥ ${t} mm`], [`csi_${i + 1}`, `CSI ≥ ${t} mm`],
        [`pod_${i + 1}`, `POD ≥ ${t} mm (hit rate)`], [`far_${i + 1}`, `FAR ≥ ${t} mm (false alarms)`]);
    });
  }
  return m;
}
const higherBetter = m => /^(corr|r2|ets|csi|pod)/.test(m);
const absBetter = m => m === "bias" || m.startsWith("fbias");

function initRank() {
  opts($("#r-agg"), S.meta.aggs.map(a => [a, AGG_LABEL[a]]), "24h");
  const wins = [...new Set(S.scores.map(r => r.window))];
  opts($("#r-win"), [...wins.map(w => [w, w === "all" ? "Whole record" : `Last ${parseInt(w)} days`]),
    ["holdout", `Last ${S.meta.holdout_days || ""} days + combinations`]], "90d");
  opts($("#r-trend-lead"), S.meta.leads.map(l => [l, leadLabel(l)]), 1);
  const upd = () => {
    opts($("#r-var"), varsFor($("#r-agg").value).map(v => [v, VAR_LABEL[v]]), $("#r-var").value || "t2m");
    opts($("#r-metric"), metricsFor($("#r-var").value, $("#r-agg").value), $("#r-metric").value || "rmse");
    renderRank();
  };
  ["#r-agg", "#r-var"].forEach(s => $(s).onchange = upd);
  ["#r-win", "#r-metric", "#r-trend-lead"].forEach(s => $(s).onchange = renderRank);
  upd();
}

function renderRank() {
  if (!isVerified()) { $("#r-hint").textContent = NOVER; $("#r-table").innerHTML = ""; drawChart("r-chart", { type: "line", data: { labels: [], datasets: [] } }); drawChart("r-trend", { type: "line", data: { labels: [], datasets: [] } }); return; }
  const v = $("#r-var").value, agg = $("#r-agg").value, win = $("#r-win").value, met = $("#r-metric").value;
  const rows = (win === "holdout" ? S.hold : S.scores.filter(r => r.window === win))
    .filter(r => r.var === v && r.agg === agg && inR(r));
  const leads = S.meta.leads;
  const provs = [...new Set(rows.map(r => r.provider))].sort((a, b) => (a in COMBO) - (b in COMBO) || pname(a).localeCompare(pname(b)));
  const get = (p, l) => rows.find(r => r.provider === p && r.lead_day === l);
  const unit = /^(rmse|mae|bias)$/.test(met) ? ` [${VAR_UNIT[v]}]` : "";
  $("#r-hint").textContent = rows.length
    ? `${VAR_LABEL[v]} · ${AGG_LABEL[agg]} · ${met.toUpperCase()}${unit}. Colours by column: green = best, red = worst; the black frame marks the best.`
      + (win === "holdout" ? " Combinations use weights chosen before these days (out-of-sample test)." : "")
    : "No data for this combination (a few weeks of record are needed).";
  // ranking per colonna
  const rank = {};
  for (const l of leads) {
    const vals = provs.map(p => [p, get(p, l)?.[met]]).filter(([, x]) => x != null && isFinite(x));
    const key = x => (absBetter(met) ? Math.abs(x) : higherBetter(met) ? -x : x);
    vals.sort((a, b) => key(a[1]) - key(b[1]));
    vals.forEach(([p], i) => (rank[p + "|" + l] = vals.length > 1 ? i / (vals.length - 1) : 0));
  }
  const color = t => `hsl(${120 - 120 * t}, 65%, ${88 - 8 * Math.abs(t - .5)}%)`;
  let html = `<thead><tr><th>Provider</th>${leads.map(l => `<th title="${leadLabel(l)}">${leadShort(l)}</th>`).join("")}<th>n</th></tr></thead><tbody>`;
  let sepDone = false;
  for (const p of provs) {
    const isCombo = p in COMBO;
    const cls = isCombo ? `combo${!sepDone ? " sep" : ""}` : "";
    if (isCombo) sepDone = true;
    const pr = S.meta.providers.find(x => x.id === p);
    const regProv = regionOf(S.region).providers || [];
    const tag = pr && !regProv.includes(p) ? '<span class="tag">solo stazioni</span>' : "";
    html += `<tr class="${cls}"><td>${pname(p)}${tag}</td>`;
    for (const l of leads) {
      const r = get(p, l), x = r?.[met], t = rank[p + "|" + l];
      html += x == null || !isFinite(x) ? "<td>–</td>"
        : `<td class="v${t === 0 ? " best" : ""}" style="background:${color(t)}">${fmt(x, /^(corr|r2)/.test(met) || /_/.test(met) ? 3 : 2)}</td>`;
    }
    html += `<td>${Math.max(0, ...leads.map(l => get(p, l)?.n || 0))}</td></tr>`;
  }
  $("#r-table").innerHTML = html + "</tbody>";
  // grafico metrica vs lead
  drawChart("r-chart", {
    type: "line",
    data: { labels: leads.map(leadShort), datasets: provs.map((p, i) => ({
      label: pname(p), data: leads.map(l => get(p, l)?.[met] ?? null), borderColor: PALETTE[i % PALETTE.length],
      backgroundColor: PALETTE[i % PALETTE.length], borderWidth: p in COMBO ? 3 : 1.5, pointRadius: 2, spanGaps: true,
      borderDash: p === "WEIGHTED" ? [6, 3] : p === "BEST" ? [2, 2] : [] })) },
    options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { x: { title: { display: true, text: "forecast lead time" } }, y: { title: { display: true, text: met.toUpperCase() + unit } } } },
  });
  // trend mensile
  const tl = +$("#r-trend-lead").value;
  const tr = S.trend.filter(r => r.var === v && r.agg === agg && r.lead_day === tl && inR(r));
  const months = [...new Set(tr.map(r => r.month))].sort();
  const tprov = [...new Set(tr.map(r => r.provider))];
  drawChart("r-trend", {
    type: "line",
    data: { labels: months, datasets: tprov.map((p, i) => ({ label: pname(p),
      data: months.map(m => tr.find(r => r.provider === p && r.month === m)?.[met] ?? null),
      borderColor: PALETTE[provs.indexOf(p) % PALETTE.length], borderWidth: 1.5, pointRadius: 2, spanGaps: true })) },
    options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: met.toUpperCase() + unit } } } },
  });
}

// ================================================================== EVENTI
function initEvents() {
  opts($("#e-type"), S.evDef.map(e => [e.id, `${e.name} (${VAR_LABEL[e.var].toLowerCase()} ${e.op} ${e.thr} ${VAR_UNIT[e.var]})`]));
  opts($("#e-lead"), S.meta.leads.map(l => [l, leadLabel(l)]), 1);
  $("#e-type").onchange = renderEvents; $("#e-lead").onchange = renderEvents;
  renderEvents();
}
function renderEvents() {
  if (!isVerified()) { $("#e-hint").textContent = NOVER; $("#e-table").innerHTML = ""; $("#e-list").innerHTML = ""; drawChart("e-chart", { type: "line", data: { labels: [], datasets: [] } }); return; }
  const ev = S.evDef.find(e => e.id === $("#e-type").value);
  if (!ev) { $("#e-hint").textContent = "No events defined."; return; }
  const lead = +$("#e-lead").value, u = VAR_UNIT[ev.var];
  const rows = S.evScores.filter(r => r.event === ev.id && r.lead_day === lead && inR(r)).sort((a, b) => (b.csi ?? -1) - (a.csi ?? -1));
  const nobs = Math.max(0, ...rows.map(r => r.observed));
  $("#e-hint").textContent = rows.length
    ? `${nobs} events observed in the whole record (station × day). Hit = the forecast also exceeds the threshold; false alarm = forecast but not observed. Sorted by CSI.`
    : "No event of this type observed yet.";
  const pct = x => (x == null || !isFinite(x) ? "–" : Math.round(100 * x) + "%");
  let html = `<thead><tr><th>Provider</th><th>Observed</th><th>Hits</th><th>Misses</th><th>False alarms</th><th>POD</th><th>FAR</th><th>CSI</th>
    <th>Mean error on event days</th><th>MAE on event days</th></tr></thead><tbody>`;
  rows.forEach((r, i) => {
    html += `<tr><td>${i === 0 ? "★ " : ""}${pname(r.provider)}</td><td>${r.observed}</td><td>${r.hits}</td><td>${r.misses}</td><td>${r.false_alarms}</td>
      <td>${pct(r.pod)}</td><td>${pct(r.far)}</td><td><b>${pct(r.csi)}</b></td><td>${fmt(r.bias_on_event, 1)} ${u}</td><td>${fmt(r.mae_on_event, 1)} ${u}</td></tr>`;
  });
  $("#e-table").innerHTML = html + "</tbody>";
  const provs = [...new Set(S.evScores.filter(r => r.event === ev.id && inR(r)).map(r => r.provider))];
  drawChart("e-chart", { type: "line",
    data: { labels: S.meta.leads.map(leadShort), datasets: provs.map((p, i) => ({ label: pname(p),
      data: S.meta.leads.map(l => S.evScores.find(r => r.event === ev.id && r.provider === p && r.lead_day === l && inR(r))?.csi ?? null),
      borderColor: PALETTE[i % PALETTE.length], borderWidth: 1.5, pointRadius: 2, spanGaps: true })) },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "right", labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { y: { min: 0, max: 1, title: { display: true, text: "CSI" } } } } });
  const list = S.evList.filter(r => r.event === ev.id && r.lead_day === lead && inR(r)).slice(0, 200);
  const hit = x => (x == null ? "" : (ev.op === ">=" ? x >= ev.thr : x <= ev.thr) ? "ok" : "bad");
  const lp = provs.filter(p => list.some(r => r[p] != null));
  let lh = `<thead><tr><th>Date</th><th>Station</th><th>Observed</th>${lp.map(p => `<th>${pname(p)}</th>`).join("")}</tr></thead><tbody>`;
  list.forEach(r => {
    lh += `<tr><td>${String(r.date).slice(0, 10)}</td><td>${r.name || r.station}</td><td><b>${fmt(r.osservato, 1)}</b></td>
      ${lp.map(p => `<td class="${hit(r[p])}">${fmt(r[p], 1)}</td>`).join("")}</tr>`;
  });
  $("#e-list").innerHTML = lh + "</tbody>";
}

// ================================================================== MAPPE
const SCALES = {
  temp: { stops: ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"] },
  prcp: { stops: ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#6a51a3", "#d4b9da"] },
  unc: { stops: ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"] },
  prob: { stops: ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"] },
};
function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function colorAt(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + f * (b[k] - v))).join(",")})`;
}

function initMaps() {
  S.map = L.map("map", { preferCanvas: true, worldCopyJump: true }).setView([42.2, 12.6], 6);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 12, attribution: "© OpenStreetMap" }).addTo(S.map);
  S.mapLayer = L.layerGroup().addTo(S.map);
  S.stLayer = L.layerGroup().addTo(S.map);
  opts($("#m-agg"), S.meta.aggs.map(a => [a, AGG_LABEL[a]]), "24h");
  const upd = async () => {
    opts($("#m-var"), varsFor($("#m-agg").value).map(v => [v, VAR_LABEL[v]]), $("#m-var").value || "t2m");
    await loadMap();
  };
  $("#m-agg").onchange = upd; $("#m-var").onchange = loadMap;
  $("#m-layer").onchange = renderMap; $("#m-time").oninput = renderMap; $("#m-stations").onchange = renderMap;
  S.map.on("click", e => mapPopup(e.latlng.wrap()));
  opts($("#m-var"), varsFor($("#m-agg").value).map(v => [v, VAR_LABEL[v]]), "t2m");
}

async function loadMap() {
  const v = $("#m-var").value, agg = $("#m-agg").value, key = `${S.region}/${v}_${agg}`;
  if (!(key in S.maps)) S.maps[key] = await fetchJSON(`maps/${key}.json`, null);
  const m = S.maps[key];
  if (!m) { S.mapLayer.clearLayers(); $("#m-time-label").textContent = "map not available"; return; }
  let items;
  if (m.verified === false) {                      // regione senza verifica: solo i modelli
    items = [["mean", "Model mean"], ["spread", "Model spread (±1σ)"]];
  } else {
    items = [["best", "★ Best provider"], ["weighted", "⚖ Weighted mean"], ["gpr", "◆ GPR – estimate"], ["gpr_unc", "◆ GPR – uncertainty (±1σ)"]];
    if (m.layers.gpr_p) items.push(["gpr_p", `◆ GPR – probability ≥ ${S.meta.prcp_thresholds[agg][0]} mm`]);
    items.push(["gpr_lo95", "◆ GPR – lower 95% bound"], ["gpr_hi95", "◆ GPR – upper 95% bound"]);
  }
  Object.keys(m.providers || {}).forEach(p => items.push(["p:" + p, pname(p)]));
  opts($("#m-layer"), items, $("#m-layer").value || (m.verified === false ? "mean" : "gpr"));
  const t = $("#m-time");
  t.max = m.periods.length - 1;
  if (+t.value > +t.max) t.value = 0;
  renderMap();
}

function layerValues(m, layer, i) {
  return layer.startsWith("p:") ? m.providers[layer.slice(2)][i] : m.layers[layer]?.[i] || [];
}
function scaleFor(v, layer, vals) {
  const fin = vals.filter(x => x != null);
  if (layer === "gpr_unc" || layer === "spread") return { s: SCALES.unc, lo: 0, hi: Math.max(...fin, 0.5), u: VAR_UNIT[v] };
  if (layer === "gpr_p") return { s: SCALES.prob, lo: 0, hi: 100, u: "%" };
  if (v === "prcp") { const hi = { "1h": 5, "6h": 15, "24h": 40 }[$("#m-agg").value]; return { s: SCALES.prcp, lo: 0, hi, u: "mm" }; }
  const lo = Math.floor(Math.min(...fin)), hi = Math.ceil(Math.max(...fin));
  return { s: SCALES.temp, lo, hi: hi > lo ? hi : lo + 1, u: "°C" };
}

function renderMap() {
  const m = S.maps[`${S.region}/${$("#m-var").value}_${$("#m-agg").value}`];
  S.mapLayer.clearLayers(); S.stLayer.clearLayers();
  if (!m) return;
  const i = +$("#m-time").value, layer = $("#m-layer").value, v = $("#m-var").value;
  const vals = layerValues(m, layer, i);
  const sc = scaleFor(v, layer, vals);
  const h = m.res / 2;
  vals.forEach((x, k) => {
    if (x == null) return;
    const c = colorAt(sc.s.stops, (x - sc.lo) / (sc.hi - sc.lo));
    if (!m.res) {                                    // regione a punti (es. siti RadCalNet)
      L.circleMarker([m.lat[k], m.lon[k]], { radius: 11, color: "#111", weight: 1.5, fillColor: c, fillOpacity: .95 })
        .bindTooltip(`${m.names?.[k] || ""}: ${fmt(x, 1)} ${sc.u}`, { permanent: true, direction: k % 2 ? "left" : "right",
          offset: [k % 2 ? -8 : 8, k % 2 ? 10 : -10], className: "site-label" })
        .addTo(S.mapLayer);
      return;
    }
    L.rectangle([[m.lat[k] - h, m.lon[k] - h], [m.lat[k] + h, m.lon[k] + h]],
      { stroke: false, fillColor: c, fillOpacity: .78, interactive: false }).addTo(S.mapLayer);
  });
  const per = m.periods[i].replace("T", " ");
  const best = m.best_provider[i] ? ` · best: ${pname(m.best_provider[i])}` : "";
  const tzl = m.timezone === "locale" ? " local time" : m.timezone && m.timezone !== "Europe/Rome" ? ` ${m.timezone}` : "";
  $("#m-time-label").textContent = `${per}${tzl} (${leadLabel(m.lead_day[i])})${layer === "best" ? best : ""}`;
  const ticks = [0, .25, .5, .75, 1].map(t => fmt(sc.lo + t * (sc.hi - sc.lo), sc.hi - sc.lo < 5 ? 1 : 0));
  $("#legend").innerHTML = `<b>${$("#m-layer").selectedOptions[0]?.text || ""}</b> [${sc.u}]
    <div class="bar" style="background:linear-gradient(90deg,${sc.s.stops.join(",")})"></div>
    <div class="ticks">${ticks.map(t => `<span>${t}</span>`).join("")}</div>`;
  if ($("#m-stations").checked) {
    const errs = S.gpst.filter(r => r.var === v && r.agg === $("#m-agg").value && r.lead_day === m.lead_day[i] && inR(r));
    const mx = Math.max(...errs.map(r => r.mae), 0.1);
    errs.forEach(r => {
      const st = S.stations.find(s => String(s.station) === String(r.station));
      if (!st) return;
      L.circleMarker([st.lat, st.lon], { radius: 4 + 10 * r.mae / mx, color: "#111", weight: 1, fillColor: "#fff", fillOpacity: .7 })
        .bindTooltip(`${st.name}: MAE GPR (holdout) ${fmt(r.mae)} ${VAR_UNIT[v]}`).addTo(S.stLayer);
    });
  }
}

function mapPopup(ll) {
  const m = S.maps[`${S.region}/${$("#m-var").value}_${$("#m-agg").value}`];
  let txt = "";
  if (m) {
    let bi = -1, bd = 1e9;
    m.lat.forEach((la, k) => { const d = (la - ll.lat) ** 2 + (m.lon[k] - ll.lng) ** 2; if (d < bd) { bd = d; bi = k; } });
    const i = +$("#m-time").value, u = VAR_UNIT[$("#m-var").value];
    if (bi >= 0 && bd < Math.max(m.res * m.res, m.res ? 0 : 1)) {
      const g = k => m.layers[k]?.[i]?.[bi];
      const nm = m.names?.[bi] ? `${m.names[bi]}<br>` : "";
      txt = m.verified === false
        ? `<b>${nm}${m.periods[i].replace("T", " ")}</b><br>Model mean: ${fmt(g("mean"), 1)} ± ${fmt(g("spread"), 1)} ${u}<br>`
          + Object.keys(m.providers || {}).map(p => `${pname(p)}: ${fmt(m.providers[p][i][bi], 1)} ${u}`).join("<br>") + "<br>"
        : `<b>${nm}${m.periods[i].replace("T", " ")}</b><br>Best: ${fmt(g("best"), 1)} ${u}<br>Weighted mean: ${fmt(g("weighted"), 1)} ${u}<br>
        GPR: ${fmt(g("gpr"), 1)} ± ${fmt(g("gpr_unc"), 1)} ${u}` + (m.layers.gpr_p ? `<br>P(≥ threshold): ${fmt(g("gpr_p"), 0)}%` : "") + "<br>";
    }
  }
  L.popup().setLatLng(ll).setContent(`${txt}<button class="link" id="go-point">Detailed forecast here →</button>`).openOn(S.map);
  setTimeout(() => { const b = document.getElementById("go-point"); if (b) b.onclick = () => { setPoint(ll.lat, ll.lng); showTab("point"); runPoint(); }; }, 0);
}

// ================================================================== PUNTO
function initPoint() {
  S.pmap = L.map("pmap", { worldCopyJump: true }).setView([30, 10], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 14, attribution: "© OpenStreetMap" }).addTo(S.pmap);
  S.pmarker = L.marker([41.9028, 12.4964]).addTo(S.pmap);
  S.pmap.on("click", e => { const ll = e.latlng.wrap(); setPoint(ll.lat, ll.lng); runPoint(); });
  opts($("#p-agg"), S.meta.aggs.map(a => [a, AGG_LABEL[a]]), "1h");
  const upd = () => { opts($("#p-var"), varsFor($("#p-agg").value).map(v => [v, VAR_LABEL[v]]), $("#p-var").value || "t2m"); renderPoint(); };
  $("#p-agg").onchange = upd; $("#p-var").onchange = renderPoint;
  $("#p-go").onclick = () => { setPoint(+$("#p-lat").value, +$("#p-lon").value); runPoint(); };
  // ricerca per indirizzo (OpenStreetMap Nominatim: max 1 richiesta al secondo, solo su invio)
  const search = async () => {
    const q = $("#p-addr").value.trim();
    if (!q) return;
    $("#p-results").innerHTML = '<span class="hint">Searching…</span>';
    try {
      const r = await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=6&accept-language=en&q=${encodeURIComponent(q)}`);
      const res = await r.json();
      if (!res.length) { $("#p-results").innerHTML = '<span class="hint">No place found: try adding the town or country.</span>'; return; }
      const pick = x => { setPoint(+x.lat, +x.lon, x.display_name); $("#p-results").innerHTML = ""; runPoint(); };
      if (res.length === 1) return pick(res[0]);
      $("#p-results").innerHTML = res.map((x, i) => `<button class="result" data-i="${i}">${x.display_name}</button>`).join("");
      document.querySelectorAll("#p-results .result").forEach(b => (b.onclick = () => pick(res[+b.dataset.i])));
    } catch (e) { $("#p-results").innerHTML = `<span class="error">Address search failed: ${e.message}</span>`; }
  };
  $("#p-search").onclick = search;
  $("#p-addr").addEventListener("keydown", e => { if (e.key === "Enter") search(); });
  // siti RadCalNet
  const sites = S.meta.regions.filter(r => r.type === "points").flatMap(r => r.sites || []);
  opts($("#p-site"), [["", "— choose —"], ...sites.map(x => [x.id, x.name])], "");
  $("#p-site").closest("label").style.display = sites.length ? "" : "none";
  $("#p-site").onchange = () => {
    const x = sites.find(z => z.id === $("#p-site").value);
    if (x) { setPoint(x.lat, x.lon, `${x.name} (${x.id})`); runPoint(); }
  };
  upd();
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  if (q.has("lat") && q.has("lon")) { setPoint(+q.get("lat"), +q.get("lon")); runPoint(); }
}
function setPoint(lat, lon, label = null) {
  S.pointLabel = label;
  $("#p-lat").value = lat.toFixed(4); $("#p-lon").value = lon.toFixed(4);
  S.pmarker.setLatLng([lat, lon]); S.pmap.setView([lat, lon], Math.max(S.pmap.getZoom(), 7));
}

// componenti di data/ora locali (fuso tz) per un istante UTC
const DTFS = {};
function localParts(ms, tz = "Europe/Rome") {
  let f = DTFS[tz];
  if (!f) {
    try { f = new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23" }); }
    catch { f = new Intl.DateTimeFormat("en-CA", { timeZone: "UTC", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23" }); }
    DTFS[tz] = f;
  }
  const p = Object.fromEntries(f.formatToParts(new Date(ms)).map(x => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, hour: +p.hour };
}
// punto dentro l'Italia? (ray casting sul poligono) -> regione con i modelli ad alta risoluzione
function inPoly(lat, lon, gj) {
  if (!gj) return false;
  const g = gj.features[0].geometry, polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
  const inRing = ring => { let c = false; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) c = !c; } return c; };
  return polys.some(p => inRing(p[0]) && !p.slice(1).some(inRing));
}
function distKm(a, b, c, d) {
  const r = Math.PI / 180, x = Math.sin((c - a) * r / 2) ** 2 + Math.cos(a * r) * Math.cos(c * r) * Math.sin((d - b) * r / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(x));
}
// regione del punto: Italia (poligono) -> regione a punti se vicino a un sito -> Mondo se attivo -> nessuna
function pointRegion(lat, lon) {
  if (S.meta.regions.some(r => r.id === "italia") && inPoly(lat, lon, S.italy)) return "italia";
  for (const r of S.meta.regions.filter(r => r.type === "points"))
    if (r.sites.some(s => distKm(lat, lon, s.lat, s.lon) <= (r.radius_km || 150))) return r.id;
  return S.meta.regions.find(r => r.id === "mondo")?.id || null;
}
const dayDiff = (a, b) => Math.round((Date.parse(a + "T00:00Z") - Date.parse(b + "T00:00Z")) / 86400000);

// stessa logica di meteoita/verify.py::aggregate (ora di fine -> blocco locale)
function aggregateSeries(times, t2m, prcp, agg, tz) {
  const blocks = new Map();
  times.forEach((ms, k) => {
    const lp = localParts(ms - 3600000, tz);
    let key;
    if (agg === "1h") key = `${lp.date}T${String(lp.hour).padStart(2, "0")}:00`;
    else if (agg === "6h") key = `${lp.date}T${String(Math.floor(lp.hour / 6) * 6).padStart(2, "0")}:00`;
    else key = `${lp.date}T00:00`;
    if (!blocks.has(key)) blocks.set(key, { t: [], p: [] });
    const b = blocks.get(key);
    if (t2m[k] != null) b.t.push(t2m[k]);
    if (prcp[k] != null) b.p.push(prcp[k]);
  });
  const minT = { "1h": 1, "6h": 5, "24h": 20 }[agg], minP = { "1h": 1, "6h": 6, "24h": 23 }[agg];
  const out = {};
  for (const [key, b] of blocks) {
    const okT = b.t.length >= minT;
    out[key] = {
      t2m: okT ? b.t.reduce((a, c) => a + c, 0) / b.t.length : null,
      tmax: okT ? Math.max(...b.t) : null, tmin: okT ? Math.min(...b.t) : null,
      prcp: b.p.length >= minP ? b.p.reduce((a, c) => a + c, 0) : null,
    };
  }
  return out;
}

async function runPoint() {
  const lat = +$("#p-lat").value, lon = +$("#p-lon").value;
  if (!isFinite(lat) || !isFinite(lon)) return;
  history.replaceState(null, "", `#point?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`);
  const rid = pointRegion(lat, lon);
  // fuori dalle regioni verificate: solo i modelli globali grezzi, senza combinazioni
  const globalIds = S.meta.providers.filter(p => p.gridded && p.coverage === "global").map(p => p.id);
  const rprov = rid ? (regionOf(rid).providers || []) : globalIds;
  const grid = S.meta.providers.filter(p => p.gridded && rprov.includes(p.id));
  $("#p-summary").innerHTML = `<p class="hint">Downloading model forecasts…</p>`;
  const url = `${S.meta.openmeteo_url}?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,precipitation`
    + `&models=${grid.map(p => p.model).join(",")}&forecast_days=8&timezone=auto&timeformat=unixtime`;
  let js;
  try { const r = await fetch(url); js = await r.json(); if (js.error) throw new Error(js.reason); }
  catch (e) { $("#p-summary").innerHTML = `<p class="error">Open-Meteo download error: ${e.message}</p>`; return; }
  const h = js.hourly, times = h.time.map(t => (typeof t === "number" ? t * 1000 : Date.parse(t + "Z")));
  const now = Date.now();
  const raw = {};
  for (const p of grid) {
    const kt = h[`temperature_2m_${p.model}`] ? `temperature_2m_${p.model}` : grid.length === 1 ? "temperature_2m" : null;
    const kp = h[`precipitation_${p.model}`] ? `precipitation_${p.model}` : grid.length === 1 ? "precipitation" : null;
    if (!kt) continue;
    raw[p.id] = { t: h[kt], p: kp ? h[kp] : h[kt].map(() => null) };
  }
  const keep = times.map(t => t > now);
  S.point = { lat, lon, elev: js.elevation, tz: js.timezone || "UTC", region: rid, label: S.pointLabel,
    times: times.filter((_, k) => keep[k]), raw, agg: {} };
  for (const id in raw) { raw[id].t = raw[id].t.filter((_, k) => keep[k]); raw[id].p = raw[id].p.filter((_, k) => keep[k]); }
  await renderPoint();
}

async function computePoint(agg, v) {
  const P = S.point, today = localParts(Date.now(), P.tz).date;
  const per = {};
  for (const id in P.raw) {
    const a = aggregateSeries(P.times, P.raw[id].t, P.raw[id].p, agg, P.tz);
    for (const k in a) (per[k] ||= {})[id] = a[k][v];
  }
  const rows = [];
  for (const k of Object.keys(per).sort()) {
    const lead = dayDiff(k.slice(0, 10), today);
    if (!S.meta.leads.includes(lead)) continue;
    const vals = per[k];
    if (!Object.values(vals).some(x => x != null && isFinite(x))) continue;   // periodo incompleto
    const ver = !!P.region && regionOf(P.region).verify !== false;
    const sel = ver ? S.weights.filter(w => w.agg === agg && w.var === v && w.lead_day === lead && (!w.region || w.region === P.region)) : [];
    // migliore = primo della classifica disponibile in questo punto
    const ranked = [...sel].sort((a, b) => a.score - b.score).map(w => w.provider);
    const bestId = ranked.find(p => vals[p] != null && isFinite(vals[p]));
    let ws = 0, wf = 0;
    sel.forEach(w => { const x = vals[w.provider]; if (x != null && isFinite(x)) { ws += w.weight; wf += w.weight * x; } });
    if (!ver) {                                    // nessuna verifica: media semplice dei modelli
      Object.values(vals).forEach(x => { if (x != null && isFinite(x)) { ws += 1; wf += x; } });
      ws = ws ? Math.max(ws, 1) : 0;
    }
    const m = ver ? await GP.load(`${DATA}gp/${P.region}/${v}_${agg}_d${lead}.json`) : null;
    const g = m ? GP.predict(m, { lat: P.lat, lon: P.lon, elev: P.elev, ...vals }) : null;
    const thr = S.meta.prcp_thresholds?.[agg]?.[0];
    rows.push({ period: k, lead, vals, best: bestId ? vals[bestId] : null, bestId,
      weighted: ws > 0.5 ? wf / ws : null, gpr: g,
      prob: g && v === "prcp" ? 100 * GP.probExceed(g, thr, m.transform) : null });
  }
  return rows;
}

async function renderPoint() {
  if (!S.point) return;
  const agg = $("#p-agg").value, v = $("#p-var").value, u = VAR_UNIT[v];
  const rows = await computePoint(agg, v);
  const P = S.point;
  const pver = !!P.region && regionOf(P.region).verify !== false;
  const wLabel = pver ? COMBO.WEIGHTED : "Model mean";
  const first = rows.find(r => r.gpr) || rows[0];
  $("#p-summary").innerHTML = `<dl class="kv">
      <dt>Point</dt><dd>${P.label ? P.label + "<br>" : ""}${P.lat.toFixed(4)}, ${P.lon.toFixed(4)} · elevation ${fmt(P.elev, 0)} m</dd>
      <dt>Region / time zone</dt><dd>${P.region ? regionOf(P.region).name : "outside the verified regions"} · ${P.tz} (local times)</dd>
      <dt>Next period</dt><dd>${first ? first.period.replace("T", " ") : "–"}</dd>
      <dt>GPR estimate</dt><dd><span class="big">${first?.gpr ? fmt(first.gpr.est, 1) : "–"}</span> ${u}
        ${first?.gpr ? ` (68%: ${fmt(first.gpr.lo68, 1)} – ${fmt(first.gpr.hi68, 1)})` : ""}</dd>
      <dt>Best</dt><dd>${fmt(first?.best, 1)} ${u} ${first?.bestId ? "· " + pname(first.bestId) : ""}</dd>
      <dt>${P.region && regionOf(P.region).verify !== false ? "Weighted mean" : "Model mean"}</dt><dd>${fmt(first?.weighted, 1)} ${u}</dd>
      ${v === "prcp" ? `<dt>Probability ≥ ${S.meta.prcp_thresholds[agg][0]} mm</dt><dd>${fmt(first?.prob, 0)} %</dd>` : ""}
    </dl><p class="hint">${!P.region ? "Point far from Italy and the RadCalNet sites: only the global models and their mean, no verified combinations."
      : regionOf(P.region).verify === false ? `Near a ${regionOf(P.region).name} site: model data only (simple mean), no verification or GPR.`
      : "Model forecasts downloaded now from Open-Meteo; weights and GPR model from the latest verification."}</p>`;
  const labels = rows.map(r => r.period.replace("T", " ").replace(" 00:00", agg === "24h" ? "" : " 00:00"));
  const provIds = Object.keys(P.raw);
  const ds = provIds.map(id => ({ label: pname(id), data: rows.map(r => r.vals[id] ?? null), borderColor: "#c3c8d0",
    borderWidth: 1, pointRadius: 0, spanGaps: true, type: v === "prcp" && agg !== "1h" ? "line" : "line" }));
  ds.push({ label: "GPR 95%", data: rows.map(r => r.gpr?.hi95 ?? null), borderWidth: 0, pointRadius: 0, backgroundColor: "rgba(17,24,39,.12)", fill: "+1" },
    { label: "_lo", data: rows.map(r => r.gpr?.lo95 ?? null), borderWidth: 0, pointRadius: 0 },
    { label: COMBO.GPR, data: rows.map(r => r.gpr?.est ?? null), borderColor: "#111827", borderWidth: 2.5, pointRadius: 0 },
    { label: COMBO.BEST, data: rows.map(r => r.best), borderColor: "#1f6feb", borderWidth: 2, pointRadius: 0, borderDash: [2, 2] },
    { label: wLabel, data: rows.map(r => r.weighted), borderColor: "#e8590c", borderWidth: 2, pointRadius: 0, borderDash: [6, 3] });
  drawChart("p-chart", { type: "line", data: { labels, datasets: ds },
    options: { animation: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { filter: i => !i.text.startsWith("_"), boxWidth: 10, font: { size: 11 } } } },
      scales: { y: { title: { display: true, text: `${VAR_LABEL[v]} [${u}]` }, ...(v === "prcp" ? { min: 0 } : {}) },
        x: { ticks: { maxTicksLimit: 16 } } } } });
  let html = `<thead><tr><th>Period</th><th>Lead</th><th>◆ GPR</th><th>±1σ</th><th>95%</th>${v === "prcp" ? "<th>P(≥ threshold)</th>" : ""}
    <th>★ Best</th><th>${pver ? "⚖ Weighted" : "Model mean"}</th>${provIds.map(id => `<th>${pname(id)}</th>`).join("")}</tr></thead><tbody>`;
  rows.forEach(r => {
    html += `<tr><td>${r.period.replace("T", " ")}</td><td>${leadShort(r.lead)}</td><td><b>${fmt(r.gpr?.est, 1)}</b></td>
      <td>${fmt(r.gpr?.unc, 1)}</td><td>${r.gpr ? fmt(r.gpr.lo95, 1) + " – " + fmt(r.gpr.hi95, 1) : "–"}</td>
      ${v === "prcp" ? `<td>${fmt(r.prob, 0)}%</td>` : ""}<td>${fmt(r.best, 1)}</td><td>${fmt(r.weighted, 1)}</td>
      ${provIds.map(id => `<td>${fmt(r.vals[id], 1)}</td>`).join("")}</tr>`;
  });
  $("#p-table").innerHTML = html + "</tbody>";
}

// ================================================================== VALIDAZIONE
function initValid() {
  opts($("#v-agg"), S.meta.aggs.map(a => [a, AGG_LABEL[a]]), "24h");
  const upd = () => { opts($("#v-var"), varsFor($("#v-agg").value).map(v => [v, VAR_LABEL[v]]), $("#v-var").value || "t2m"); renderValid(); };
  $("#v-agg").onchange = upd; $("#v-var").onchange = renderValid;
  upd();
}
function renderValid() {
  if (!isVerified()) { $("#v-table").innerHTML = `<tr><td>${NOVER}</td></tr>`; $("#b-table").innerHTML = ""; $("#b-hint").textContent = ""; return; }
  const v = $("#v-var").value, agg = $("#v-agg").value, u = VAR_UNIT[v];
  renderBacktest(v, agg, u);
  const rows = S.gpval.filter(r => r.var === v && r.agg === agg && inR(r)).sort((a, b) => a.lead_day - b.lead_day);
  const hs = (p, l) => S.hold.find(r => r.provider === p && r.var === v && r.agg === agg && r.lead_day === l && inR(r));
  const bestProv = l => S.hold.filter(r => r.var === v && r.agg === agg && r.lead_day === l && !(r.provider in COMBO) && inR(r))
    .sort((a, b) => a.rmse - b.rmse)[0];
  const cls = (x, nom) => (x == null ? "" : Math.abs(x - nom) < 0.07 ? "ok" : Math.abs(x - nom) < 0.15 ? "meh" : "bad");
  let html = `<thead><tr><th>Lead</th><th>n</th><th>MAE GPR</th><th>RMSE GPR</th><th>RMSE best provider (ex post)</th><th>RMSE ★ Best</th>
    <th>RMSE ⚖ Weighted</th><th>R² ◆ GPR</th><th>R² ★ Best</th><th>R² ⚖ Weighted</th><th>Bias GPR</th><th>Coverage 68%</th><th>Coverage 95%</th><th>Mean uncertainty</th></tr></thead><tbody>`;
  if (!rows.length) html += `<tr><td colspan="14">No validation available yet for this case (at least 14 days of record are needed).</td></tr>`;
  rows.forEach(r => {
    const bp = bestProv(r.lead_day);
    html += `<tr><td>${leadLabel(r.lead_day)}</td><td>${r.n}</td><td>${fmt(r.mae)}</td><td><b>${fmt(r.rmse)}</b></td>
      <td>${bp ? fmt(bp.rmse) + ` <span class="tag">${pname(bp.provider)}</span>` : "–"}</td>
      <td>${fmt(hs("BEST", r.lead_day)?.rmse)}</td><td>${fmt(hs("WEIGHTED", r.lead_day)?.rmse)}</td>
      <td><b>${fmt(r.r2, 3)}</b></td><td>${fmt(hs("BEST", r.lead_day)?.r2, 3)}</td><td>${fmt(hs("WEIGHTED", r.lead_day)?.r2, 3)}</td><td>${fmt(r.bias)}</td>
      <td class="${cls(r.cov68, .68)}">${fmt(100 * r.cov68, 0)}%</td><td class="${cls(r.cov95, .95)}">${fmt(100 * r.cov95, 0)}%</td>
      <td>${fmt(r.mean_unc)} ${u}</td></tr>`;
  });
  $("#v-table").innerHTML = html + "</tbody>";
}

function renderBacktest(v, agg, u) {
  const rows = (S.bt || []).filter(r => r.var === v && r.agg === agg && inR(r)).sort((a, b) => a.lead_day - b.lead_day);
  if (!rows.length) {
    $("#b-hint").textContent = "Not available yet: the back-test needs about two months of record. The record is rebuilt automatically "
      + "a few weeks at a time every day, and the back-test runs once a week.";
    $("#b-table").innerHTML = ""; return;
  }
  $("#b-hint").textContent = `Test blocks from ${rows[0].first_test} to ${rows[0].last_test} (${rows[0].folds} blocks of 14 days, each trained only on the previous 45 days). `
    + "Skill = RMSE reduction of the GPR with respect to the Best provider chosen in advance (positive = GPR better).";
  const cls = (x, nom) => (x == null ? "" : Math.abs(x - nom) < 0.07 ? "ok" : Math.abs(x - nom) < 0.15 ? "meh" : "bad");
  const sk = x => (x == null ? "" : x > 2 ? "ok" : x > -2 ? "meh" : "bad");
  let html = `<thead><tr><th>Lead</th><th>Blocks</th><th>n</th><th>RMSE ◆ GPR</th><th>RMSE ★ Best (ex ante)</th><th>Skill vs Best</th>
    <th>GPR better than Best</th><th>RMSE ⚖ Weighted</th><th>RMSE best ex post</th><th>R² ◆ GPR</th><th>R² ★ Best</th><th>R² ⚖ Weighted</th>
    <th>Coverage 68%</th><th>Coverage 95%</th></tr></thead><tbody>`;
  rows.forEach(r => {
    html += `<tr><td>${leadLabel(r.lead_day)}</td><td>${r.folds}</td><td>${r.n}</td><td><b>${fmt(r.rmse_gpr)}</b></td>
      <td>${fmt(r.rmse_best_exante)}</td><td class="${sk(r.skill_vs_best_pct)}">${fmt(r.skill_vs_best_pct, 1)}%</td>
      <td>${fmt(100 * r.share_gpr_beats_best, 0)}% of blocks</td><td>${fmt(r.rmse_weighted)}</td><td>${fmt(r.rmse_best_expost)}</td>
      <td><b>${fmt(r.r2_gpr, 3)}</b></td><td>${fmt(r.r2_best_exante, 3)}</td><td>${fmt(r.r2_weighted, 3)}</td>
      <td class="${cls(r.cov68, .68)}">${fmt(100 * r.cov68, 0)}%</td><td class="${cls(r.cov95, .95)}">${fmt(100 * r.cov95, 0)}%</td></tr>`;
  });
  $("#b-table").innerHTML = html + "</tbody>";
}

// ================================================================== DATI
function initData() {
  const d = S.meta.data_url || "../data/csv";
  const raw = S.meta.raw_url;
  const link = f => (raw ? `<a href="${raw}/${f}">${f}</a>` : `<code>${f}</code>`);
  $("#data-text").innerHTML = `
  <h3>Download the data</h3>
  <p>All results are stored in the GitHub repository, folder <a href="${d}">data/csv</a>, as CSV files (comma separator, dot decimal)
  plus an Excel summary (file names are in Italian):</p>
  <ul>
    <li>${link("riepilogo.xlsx")} – all main tables in a single Excel file</li>
    <li>${link("classifica.csv")} – scores by provider, variable, aggregation, lead time and time window</li>
    <li>${link("classifica_holdout.csv")} – last days, including Best, Weighted mean and GPR</li>
    <li>${link("migliore_e_pesi.csv")} – best provider and weights in use</li>
    <li>${link("gpr_validazione.csv")} – GPR errors and calibration on the latest held-out days</li>
    <li>${link("gpr_backtest.csv")} / ${link("gpr_backtest_sintesi.csv")} – historical back-test of the GPR (every fold / summary)</li>
    <li>${link("trend_mensile.csv")} – monthly scores</li>
    <li><code>coppie_giornaliere/YYYY-MM.csv</code> – for each day, station, variable and lead time: observation and forecast of every provider</li>
    <li>${link("eventi_punteggi.csv")} – significant-event verification (hits, misses, false alarms, error on event days)</li>
    <li>${link("eventi_elenco.csv")} – every observed event with the forecast of each provider</li>
    <li><code>radcalnet/YYYY-MM.csv</code> – hourly forecasts of every model at the RadCalNet sites, for every lead time (0–7 days)</li>
    <li>${link("stazioni.csv")} – verification stations</li>
  </ul>
  <h3>How it works</h3>
  <p><b>Regions.</b> “Italy” uses all models covering Italy (including ICON-2I, ICON-EU, ARPEGE) on a dense grid.
  “RadCalNet” covers the calibration sites Railroad Valley, Gobabeb, La Crau and Baotou: here the site only shows the global models
  (ECMWF, ICON, GFS, UKMO, GEM), their mean and spread, without verification; the values are archived every day
  (<code>data/csv/radcalnet</code>) for comparison with the site measurements. When you query a point the region is chosen automatically;
  days and 6-hour blocks are in the point's local time.</p>
  <p><b>Providers.</b> Numerical models (ItaliaMeteo ICON-2I, ECMWF, ICON, GFS, ARPEGE, UKMO, …) come from Open-Meteo,
  which archives, for every hour, the forecast issued 0, 1, …, 7 days earlier, so the record can be rebuilt backwards. Station-only services
  (MET Norway and, if API keys are configured, OpenWeatherMap, WeatherAPI, Tomorrow.io, Visual Crossing) are archived every day.</p>
  <p><b>Observations.</b> Synoptic stations (Meteostat). Instantaneous temperature, hourly precipitation; 6-hour blocks and days in each station's local time.
  A block is verified only if complete.</p>
  <p><b>Lead time.</b> “Today” = forecast issued a few hours earlier; “+1 d” = issued about 24 hours earlier; … up to one week.</p>
  <p><b>Events.</b> Besides average errors, significant observed events are verified (heavy rain, heat, frost; thresholds in
  <code>config.yaml</code>): for each provider, how many were forecast, missed, and how many false alarms were issued.</p>
  <p><b>Best</b> = provider with the lowest error over the last ${S.meta.window_days || 90} days, for that variable, aggregation and lead time.
  <b>Weighted mean</b> = mean of the providers with weights proportional to 1/MSE. <b>GPR</b> = Gaussian-process regression using
  location, elevation and the forecasts of all models to estimate the value and its uncertainty. It is checked on days excluded from training
  and, every week, with a historical back-test over the whole record (“GPR reliability” tab).</p>
  <p><b>Address search</b> uses <a href="https://nominatim.openstreetmap.org/">OpenStreetMap Nominatim</a>.</p>
  <p><b>Limitations.</b> Best, Weighted mean and GPR only use models available everywhere (Open-Meteo), so they can be computed
  at any point. Stations are mostly in lowlands or on the coast: in the mountains estimates and uncertainties are less verified.</p>`;
}
