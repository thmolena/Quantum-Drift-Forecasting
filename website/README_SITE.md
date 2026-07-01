# Quantum Drift Forecasting — Website

This directory contains the front-end assets for the interactive research demo hosted via GitHub Pages.

## File Structure

```
website/
├── style.css              Dark-theme CSS (blue quantum palette)
├── demo.js                Browser-side drift simulation + Flask API client
├── index.html             Local development copy (paths relative to website/)
└── notebooks_html/        Pre-rendered Jupyter notebook HTML exports
    ├── rnn_drift_forecast.html
    ├── transformer_calibration.html
    └── quantum_drift_combined.html
```

The root [`index.html`](../index.html) is the GitHub Pages entry point and references assets as `website/style.css` and `website/demo.js`.

## Generating Notebook HTML Exports

```bash
cd ..   # project root
jupyter nbconvert --to html notebooks/rnn_drift_forecast.ipynb \
    --output website/notebooks_html/rnn_drift_forecast.html

jupyter nbconvert --to html notebooks/transformer_calibration.ipynb \
    --output website/notebooks_html/transformer_calibration.html

jupyter nbconvert --to html notebooks/quantum_drift_combined.ipynb \
    --output website/notebooks_html/quantum_drift_combined.html
```

## Local Development

```bash
# Start the Flask inference API (optional, enables LSTM model queries)
python -m src.server --port 5000

# Serve the site locally (any static file server works)
python -m http.server 8080 --directory ..
# then open http://localhost:8080
```

## Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--accent`       | `#0ea5e9` | Primary CTA, links, badges |
| `--accent-light` | `#38bdf8` | Chart lines, hover |
| `--accent-green` | `#34d399` | Forecast line, STABLE status |
| `--accent-warn`  | `#fbbf24` | CAUTION drift status |
| `--accent-alert` | `#f87171` | DRIFT ALERT status |
| `--bg-base`      | `#0f1117` | Page background |
| `--bg-card`      | `#151824` | Card / panel background |
