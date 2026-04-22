# MLOps Batch Pipeline — Rolling-Mean Signal

A minimal, production-style MLOps batch job that generates a binary trading signal from OHLCV data using a configurable rolling mean.Demonstrates
**reproducibility** (seeded NumPy), **observability** (structured logging +guaranteed metrics JSON), and **deployment readiness** (Docker).

## Repository layout

```
.
├── run.py            # Pipeline entry point
├── config.yaml       # Runtime configuration (seed, window, version)
├── data.csv          # Sample OHLCV input
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition

```

---

## How it works

| Step | Detail |
|------|--------|
| **Config** | Reads `seed`, `window`, and `version` from `config.yaml` |
| **Seed** | Sets `numpy.random.seed(seed)` for full reproducibility |
| **Data** | Loads `data.csv`, normalises column names, validates OHLCV schema |
| **Feature** | Computes a `window`-period rolling mean on the `close` column |
| **Signal** | Binary: `1` if `close > rolling_mean`, else `0`; NaN warm-up rows → `0` |
| **Output** | Writes `metrics.json` and `run.log`; prints JSON to stdout |

---

## Local run

### 1 — Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Execute the pipeline

```bash
python run.py \
  --input    data.csv    \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

### 3 — Inspect outputs

```bash
cat metrics.json   # structured JSON result
cat run.log        # full debug log
```

**Expected `metrics.json` shape**

```json
{
  "version": "v1",
  "rows_processed": 15,
  "metric": "signal_ratio",
  "value": 0.6,
  "latency_ms": 12.5,
  "seed": 42,
  "status": "success"
}
```

---

## Docker run

### 1 — Build the image

```bash

sudo docker build -t mlops-task .

```

### 2 — Run the container

```bash
sudo docker run --rm mlops-task
```

The final metrics JSON is printed to **stdout** and also written to
`metrics.json`.  The full log is written to `outputs/run.log`.

---

## Configuration reference (`config.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `seed` | `int ≥ 0` | NumPy random seed |
| `window` | `int ≥ 1` | Rolling-mean look-back period (rows) |
| `version` | `string` | Pipeline version tag; written to `metrics.json` |

---

## Metrics JSON schema

| Key | Type | Notes |
|-----|------|-------|
| `version` | string | From `config.yaml` |
| `rows_processed` | int | Rows after cleaning |
| `metric` | string | `"signal_ratio"` |
| `value` | float | Fraction of bullish-signal rows |
| `latency_ms` | float | Wall-clock time for the full run |
| `seed` | int | NumPy seed used |
| `status` | string | `"success"` or `"error: <message>"` |

`metrics.json` is **always written**, even when the pipeline exits with an
error, so downstream monitoring can detect and report failures.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Pipeline completed successfully |
| `1` | Pipeline failed (see `run.log` and `metrics.json` for details) |
