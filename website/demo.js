// ── Configuration ─────────────────────────────────────────────────────────
const API_BASE_URL = (window.API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');

// ── Browser-side LSTM-style drift simulation ───────────────────────────────
// Approximates quantum hardware T1 drift using exponential smoothing + noise,
// then applies a simple threshold detector to predict drift probability.
// Runs entirely in the browser when the Flask API is unavailable.

function generateQuantumSeries(steps, noiseLevel, driftRate, seed) {
  // Seeded pseudo-random (LCG)
  let state = seed || 12345;
  function rand() {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state >>> 0) / 4294967296;
  }
  function randGauss() {
    const u = Math.max(rand(), 1e-10), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const series = [];
  let t1 = 80.0;
  let trend = -driftRate;

  for (let i = 0; i < steps; i++) {
    const t = i * 0.5;
    const seasonal = 20 * Math.sin(2 * Math.PI * t / 48);
    const noise    = randGauss() * noiseLevel;
    t1 = Math.max(10, t1 + trend + seasonal * 0.05 + noise);
    series.push(t1);
  }
  return series;
}

function exponentialSmoothing(series, alpha) {
  const smoothed = [series[0]];
  for (let i = 1; i < series.length; i++) {
    smoothed.push(alpha * series[i] + (1 - alpha) * smoothed[i - 1]);
  }
  return smoothed;
}

function computeDriftIndicator(series, windowSize) {
  // Rolling mean + rolling std → z-score based drift probability
  const probs = new Array(series.length).fill(0);
  for (let i = windowSize; i < series.length; i++) {
    const window = series.slice(i - windowSize, i);
    const mean   = window.reduce((a, b) => a + b, 0) / windowSize;
    const std    = Math.sqrt(window.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / windowSize) || 1;
    // How many std-devs has the latest value descended from the window mean?
    const z      = (series[i] - mean) / std;
    // Convert to prob using sigmoid of negative z (more negative = more drift)
    probs[i] = 1 / (1 + Math.exp(z * 1.5));
  }
  return probs;
}

function browserForecast(series, noiseLevel) {
  const alpha    = 0.15;
  const smoothed = exponentialSmoothing(series, alpha);
  const probs    = computeDriftIndicator(smoothed, 8);
  const driftProb = probs[probs.length - 1];

  // 8-step linear extrapolation for forecast
  const n = series.length;
  const last4 = series.slice(n - 4);
  const slope = (last4[3] - last4[0]) / 3;
  const forecast = [];
  for (let h = 1; h <= 8; h++) {
    forecast.push(Math.max(10, series[n - 1] + slope * h + (Math.random() - 0.5) * noiseLevel));
  }

  const meanT1 = series.reduce((a, b) => a + b, 0) / series.length;
  const minT1  = Math.min(...series);
  const anomalyScore = Math.max(0, (80 - minT1) / 80);  // proxy: how far below nominal

  return {
    series: smoothed,
    driftProbs: probs,
    forecast,
    driftProb:    parseFloat(driftProb.toFixed(4)),
    anomalyScore: parseFloat(anomalyScore.toFixed(6)),
    meanT1:       parseFloat(meanT1.toFixed(2)),
    minT1:        parseFloat(minT1.toFixed(2)),
  };
}

// ── DOM references ─────────────────────────────────────────────────────────
const svgEl          = document.getElementById('drift-svg');
const hintEl         = document.getElementById('chart-hint');
const resultPanel    = document.getElementById('result-panel');
const errorPanel     = document.getElementById('error-panel');
const errorMsg       = document.getElementById('error-msg');
const driftStatusEl  = document.getElementById('drift-status');

const stepsRangeEl   = document.getElementById('steps-range');
const stepsValEl     = document.getElementById('steps-val');
const noiseRangeEl   = document.getElementById('noise-range');
const noiseValEl     = document.getElementById('noise-val');
const driftRangeEl   = document.getElementById('drift-range');
const driftValEl     = document.getElementById('drift-val');

let currentSeries   = [];
let currentProbs    = [];
let currentForecast = [];

// Sync sliders
function syncSlider(rangeEl, valEl, fmt) {
  rangeEl.addEventListener('input', () => { valEl.textContent = fmt(rangeEl.value); });
}
syncSlider(stepsRangeEl,  stepsValEl,  v => v);
syncSlider(noiseRangeEl,  noiseValEl,  v => parseFloat(v).toFixed(1));
syncSlider(driftRangeEl,  driftValEl,  v => parseFloat(v).toFixed(3));

// ── Chart rendering ────────────────────────────────────────────────────────
function renderChart(series, probs, forecast, threshold) {
  const isLightTheme = document.body.classList.contains('light-theme');
  const palette = isLightTheme
    ? {
        background: '#ffffff',
        shade: 'rgba(248,113,113,0.10)',
        grid: '#d9e2ec',
        axis: '#64748b',
        area: 'rgba(52,211,153,0.12)',
        separator: '#94a3b8',
      }
    : {
        background: '#151824',
        shade: 'rgba(248,113,113,0.07)',
        grid: '#2d3148',
        axis: '#64748b',
        area: 'rgba(52,211,153,0.06)',
        separator: '#475569',
      };

  const W  = svgEl.parentElement.clientWidth || 640;
  const H  = 260;
  const pd = { top: 20, right: 20, bottom: 35, left: 52 };
  const cW = W - pd.left - pd.right;
  const cH = H - pd.top  - pd.bottom;

  svgEl.setAttribute('width',  W);
  svgEl.setAttribute('height', H);
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svgEl.innerHTML = '';

  const ns = 'http://www.w3.org/2000/svg';
  const all = [...series, ...forecast];
  const minV = Math.min(...all) * 0.95;
  const maxV = Math.max(...all) * 1.05;

  function xScale(i, total) { return pd.left + (i / (total - 1)) * cW; }
  function yScale(v)         { return pd.top + (1 - (v - minV) / (maxV - minV)) * cH; }

  // Background
  const bg = document.createElementNS(ns, 'rect');
  bg.setAttribute('x', 0); bg.setAttribute('y', 0);
  bg.setAttribute('width', W); bg.setAttribute('height', H);
  bg.setAttribute('fill', palette.background);
  svgEl.appendChild(bg);

  // Drift alert shading (where prob > threshold)
  if (probs.length === series.length) {
    for (let i = 0; i < series.length; i++) {
      if (probs[i] > threshold) {
        const r = document.createElementNS(ns, 'rect');
        const xL = xScale(i > 0 ? i - 0.5 : 0, series.length);
        const xR = xScale(Math.min(i + 0.5, series.length - 1), series.length);
        r.setAttribute('x',      xL);
        r.setAttribute('y',      pd.top);
        r.setAttribute('width',  Math.max(1, xR - xL));
        r.setAttribute('height', cH);
        r.setAttribute('fill', palette.shade);
        svgEl.appendChild(r);
      }
    }
  }

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const yv  = minV + i * (maxV - minV) / 4;
    const y   = yScale(yv);
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', pd.left); line.setAttribute('x2', W - pd.right);
    line.setAttribute('y1', y); line.setAttribute('y2', y);
    line.setAttribute('stroke', palette.grid); line.setAttribute('stroke-width', '1');
    svgEl.appendChild(line);
    const lbl = document.createElementNS(ns, 'text');
    lbl.setAttribute('x', pd.left - 6); lbl.setAttribute('y', y + 4);
    lbl.setAttribute('text-anchor', 'end'); lbl.setAttribute('fill', palette.axis);
    lbl.setAttribute('font-size', '10'); lbl.setAttribute('font-family', 'Inter, sans-serif');
    lbl.textContent = yv.toFixed(0);
    svgEl.appendChild(lbl);
  }

  // X-axis label
  const xlabel = document.createElementNS(ns, 'text');
  xlabel.setAttribute('x', pd.left + cW / 2);
  xlabel.setAttribute('y', H - 5);
  xlabel.setAttribute('text-anchor', 'middle');
  xlabel.setAttribute('fill', palette.axis);
  xlabel.setAttribute('font-size', '10');
  xlabel.setAttribute('font-family', 'Inter, sans-serif');
  xlabel.textContent = 'Time step (0.5 h intervals)';
  svgEl.appendChild(xlabel);

  // Y-axis label
  const ylabel = document.createElementNS(ns, 'text');
  ylabel.setAttribute('x', 14);
  ylabel.setAttribute('y', pd.top + cH / 2);
  ylabel.setAttribute('text-anchor', 'middle');
  ylabel.setAttribute('fill', palette.axis);
  ylabel.setAttribute('font-size', '10');
  ylabel.setAttribute('font-family', 'Inter, sans-serif');
  ylabel.setAttribute('transform', `rotate(-90, 14, ${pd.top + cH / 2})`);
  ylabel.textContent = 'T1 (µs)';
  svgEl.appendChild(ylabel);

  // Historical T1 line
  function makePath(pts, style) {
    const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    Object.entries(style).forEach(([k, v]) => path.setAttribute(k, v));
    svgEl.appendChild(path);
    return path;
  }

  const histPts = series.map((v, i) => ({
    x: xScale(i, series.length),
    y: yScale(v),
  }));
  makePath(histPts, { stroke: '#38bdf8', 'stroke-width': '1.8' });

  // Forecast continuation (dashed)
  const nHist = series.length;
  const fcPts = [histPts[histPts.length - 1], ...forecast.map((v, h) => ({
    x: xScale(nHist + h, nHist + forecast.length - 1 + 1) * (nHist / (nHist + 8)) + xScale(nHist + h, nHist + 8),
    // approximate x positioning for forecast
    y: yScale(v),
  }))];

  // Simpler: render forecast in the right portion of the chart
  const fcXStart = xScale(nHist - 1, nHist);
  const fcXEnd   = W - pd.right;
  const fcWidth  = fcXEnd - fcXStart;
  const fcPts2   = [histPts[histPts.length - 1], ...forecast.map((v, h) => ({
    x: fcXStart + (h + 1) * fcWidth / forecast.length,
    y: yScale(v),
  }))];
  makePath(fcPts2, { stroke: '#34d399', 'stroke-width': '1.5', 'stroke-dasharray': '5 3' });

  // Forecast area fill
  const areaD = fcPts2.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
    + ` L${fcPts2[fcPts2.length-1].x.toFixed(1)},${(pd.top+cH).toFixed(1)}`
    + ` L${fcPts2[0].x.toFixed(1)},${(pd.top+cH).toFixed(1)} Z`;
  const areaPath = document.createElementNS(ns, 'path');
  areaPath.setAttribute('d', areaD);
  areaPath.setAttribute('fill', palette.area);
  svgEl.appendChild(areaPath);

  // Forecast separator line
  const sep = document.createElementNS(ns, 'line');
  sep.setAttribute('x1', fcXStart); sep.setAttribute('x2', fcXStart);
  sep.setAttribute('y1', pd.top);   sep.setAttribute('y2', pd.top + cH);
  sep.setAttribute('stroke', palette.separator); sep.setAttribute('stroke-width', '1');
  sep.setAttribute('stroke-dasharray', '3 3');
  svgEl.appendChild(sep);

  // "Forecast →" label
  const fcLabel = document.createElementNS(ns, 'text');
  fcLabel.setAttribute('x', fcXStart + 5);
  fcLabel.setAttribute('y', pd.top + 14);
  fcLabel.setAttribute('fill', '#34d399');
  fcLabel.setAttribute('font-size', '9');
  fcLabel.setAttribute('font-family', 'Inter, sans-serif');
  fcLabel.textContent = 'Forecast →';
  svgEl.appendChild(fcLabel);

  hintEl.textContent = `${series.length} steps · ${forecast.length}-step forecast`;
}

// ── Drift status indicator ─────────────────────────────────────────────────
function updateDriftStatus(prob) {
  driftStatusEl.classList.remove('stable', 'warning', 'alert');
  if (prob < 0.35) {
    driftStatusEl.textContent = `⬤ STABLE — drift probability ${(prob * 100).toFixed(1)}%`;
    driftStatusEl.classList.add('stable');
  } else if (prob < 0.65) {
    driftStatusEl.textContent = `⬤ CAUTION — drift probability ${(prob * 100).toFixed(1)}%`;
    driftStatusEl.classList.add('warning');
  } else {
    driftStatusEl.textContent = `⬤ DRIFT ALERT — drift probability ${(prob * 100).toFixed(1)}%`;
    driftStatusEl.classList.add('alert');
  }
}

// ── Error helper ───────────────────────────────────────────────────────────
function showError(html) {
  errorMsg.innerHTML = html;
  errorPanel.classList.remove('hidden');
  resultPanel.classList.add('hidden');
}

// ── Generate button ────────────────────────────────────────────────────────
document.getElementById('generate').addEventListener('click', () => {
  const steps     = parseInt(stepsRangeEl.value);
  const noise     = parseFloat(noiseRangeEl.value);
  const driftRate = parseFloat(driftRangeEl.value);
  const res = browserForecast(
    generateQuantumSeries(steps, noise, driftRate, Date.now() & 0xfffff),
    noise
  );
  currentSeries   = res.series;
  currentProbs    = res.driftProbs;
  currentForecast = res.forecast;

  renderChart(currentSeries, currentProbs, currentForecast, 0.5);
  updateDriftStatus(res.driftProb);
  resultPanel.classList.add('hidden');
  errorPanel.classList.add('hidden');
});

// ── Forecast button ────────────────────────────────────────────────────────
document.getElementById('forecast-btn').addEventListener('click', async () => {
  if (currentSeries.length === 0) {
    showError('Generate a time series first.');
    return;
  }
  errorPanel.classList.add('hidden');

  const noise     = parseFloat(noiseRangeEl.value);
  const driftRate = parseFloat(driftRangeEl.value);

  let data   = null;
  let source = 'api';

  // 1. Try the local Flask API
  try {
    // Build a feature matrix: just T1 repeated across all 7 feature dims (simplified)
    const sequence = currentSeries.map(t1 => [
      t1,
      t1 * 0.9,
      0.999 - t1 * 0.00002,
      0.985 - t1 * 0.00004,
      0.012 + t1 * 0.00003 * 0.01,
      0.0005,
      1.5708,
    ]);
    const res = await fetch(`${API_BASE_URL}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, model: 'lstm' }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    data = await res.json();
  } catch (_) {
    // 2. Fallback: browser-side computation
    const res = browserForecast(currentSeries, noise);
    data   = res;
    source = 'browser';
    data.forecast_8step = res.forecast;
  }

  // Show results
  const fc       = data.forecast || data.forecast_8step || currentForecast;
  const driftP   = data.drift_prob ?? data.driftProb ?? 0;
  const anomaly  = data.anomaly_score ?? data.anomalyScore ?? 0;
  const meanT1   = data.meanT1 ?? (currentSeries.reduce((a,b)=>a+b,0)/currentSeries.length).toFixed(2);
  const minT1    = data.minT1 ?? Math.min(...currentSeries).toFixed(2);

  document.getElementById('res-forecast').textContent =
    Array.isArray(fc) ? fc.slice(0, 4).map(v => parseFloat(v).toFixed(2)).join(', ') + ' …' : fc;

  const driftPEl = document.getElementById('res-drift-prob');
  driftPEl.textContent = `${(driftP * 100).toFixed(1)}%`;
  driftPEl.parentElement.className =
    'result-row ' + (driftP > 0.65 ? 'alert' : driftP > 0.35 ? 'warn' : 'highlight');

  document.getElementById('res-anomaly').textContent  = parseFloat(anomaly).toFixed(6);
  document.getElementById('res-mean-t1').textContent  = parseFloat(meanT1).toFixed(2);
  document.getElementById('res-min-t1').textContent   = parseFloat(minT1).toFixed(2);

  const badge = document.getElementById('source-badge');
  badge.textContent  = source === 'browser'
    ? '⚡ computed in browser (exponential smoothing)'
    : '🖥 computed by local API (LSTM model)';
  badge.className = `source-badge ${source}`;
  badge.classList.remove('hidden');

  resultPanel.classList.remove('hidden');
  updateDriftStatus(driftP);

  if (Array.isArray(fc)) {
    currentForecast = fc.slice(0, 8).map(Number);
    renderChart(currentSeries, currentProbs, currentForecast, 0.5);
  }
});

// ── Initial chart render ───────────────────────────────────────────────────
(function() {
  const initSeries = generateQuantumSeries(60, 3.0, 0.05, 42);
  const res = browserForecast(initSeries, 3.0);
  currentSeries   = res.series;
  currentProbs    = res.driftProbs;
  currentForecast = res.forecast;
  renderChart(currentSeries, currentProbs, currentForecast, 0.5);
  updateDriftStatus(res.driftProb);
})();
