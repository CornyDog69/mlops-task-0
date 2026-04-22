"""
run.py — Minimal MLOps batch pipeline: rolling-mean signal generation on OHLCV data.

Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps batch pipeline: rolling-mean signal on OHLCV data."
    )
    parser.add_argument("--input",    required=True, help="Path to input CSV (OHLCV).")
    parser.add_argument("--config",   required=True, help="Path to YAML config file.")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON.")
    parser.add_argument("--log-file", required=True, help="Path for log file output.")
    return parser.parse_args()

def configure_logging(log_file: str) -> logging.Logger:
    """Set up a logger that writes to both a file and stdout."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    logger.info("Loading config from '%s'.", path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must be YAML (.yaml/.yml), got: {path.suffix}")

    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a top-level mapping.")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise KeyError(f"Config is missing required keys: {sorted(missing)}")

    seed = cfg["seed"]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"'seed' must be a non-negative integer, got: {seed!r}")

    window = cfg["window"]
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"'window' must be a positive integer, got: {window!r}")

    version = cfg["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"'version' must be a non-empty string, got: {version!r}")

    logger.debug(
        "Config loaded — seed=%d  window=%d  version=%s", seed, window, version
    )
    return cfg

REQUIRED_COLUMNS = {"open", "high", "low", "close"}
VOLUME_COLUMN_ALIASES = ["volume", "volume_btc", "volume_usd", "vol"]


def _detect_csv_quoting(path: Path) -> dict:
    """
    Detect whether the CSV rows are wrapped in an extra layer of quotes.
    Some exporters (e.g. certain crypto data vendors) write each row as a
    single quoted string:  "timestamp,open,high,low,close,volume_btc,..."
    In that case pandas sees only one column.  We re-read with quoting=QUOTE_NONE
    and sep auto-detected from the inner content.
    """
    import csv as _csv
    with path.open("r", newline="") as fh:
        sample = fh.read(4096)
    reader = _csv.reader(sample.splitlines())
    try:
        first_row = next(reader)
    except StopIteration:
        return {}
   
    if len(first_row) == 1 and "," in first_row[0]:
        return {"quoting": __import__("csv").QUOTE_NONE, "escapechar": "\\"}
    return {}


def load_ohlcv(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    logger.info("Loading OHLCV data from '%s'.", path)

    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a CSV file, got: {path.suffix}")

    df = pd.read_csv(path)
    logger.debug("Raw CSV shape: %s  columns: %s", df.shape, df.columns.tolist())

    df.columns = [c.strip().lower() for c in df.columns]

    
    if len(df.columns) == 1 and "," in df.columns[0]:
        logger.warning(
            "CSV appears to have extra row-level quoting; re-parsing with QUOTE_NONE."
        )
        extra_kwargs = _detect_csv_quoting(path)
        df = pd.read_csv(path, **extra_kwargs)
        df.columns = [c.strip().lower() for c in df.columns]
        logger.debug(
            "Re-parsed CSV shape: %s  columns: %s", df.shape, df.columns.tolist()
        )

    for alias in VOLUME_COLUMN_ALIASES:
        if alias in df.columns and alias != "volume":
            df = df.rename(columns={alias: "volume"})
            logger.debug("Renamed column '%s' → 'volume'.", alias)
            break

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_cols)}")

    if df.empty:
        raise ValueError("Input CSV contains no data rows.")


    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad_rows = df["close"].isna().sum()
    if bad_rows > 0:
        logger.warning(
            "%d row(s) have non-numeric 'close' values and will be dropped.", bad_rows
        )
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning 'close' column.")

    logger.info("Data loaded: %d rows, %d columns.", len(df), len(df.columns))
    return df

def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Add 'rolling_mean' and binary 'signal' columns.

    NaN rows from the rolling window are treated as 0.
    """
    logger.info("Computing %d-period rolling mean on 'close'.", window)

    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window).mean()

    df["signal"] = np.where(
        df["rolling_mean"].notna() & (df["close"] > df["rolling_mean"]), 1, 0
    )

    signal_sum   = int(df["signal"].sum())
    signal_rate = signal_sum / len(df)
    logger.debug(
        "Signal distribution — bullish: %d (%.1f%%)  neutral/bearish: %d (%.1f%%)",
        signal_sum,
        signal_rate * 100,
        len(df) - signal_sum,
        (1 - signal_rate) * 100,
    )
    return df


def compute_metric(df: pd.DataFrame, logger: logging.Logger) -> dict:
    """
    Primary metric: fraction of rows with a bullish signal (signal == 1).
    """
    bullish_ratio = float(df["signal"].mean())
    logger.info("Primary metric — signal_rate: %.4f", bullish_ratio)
    return {"metric": "signal_rate", "value": round(bullish_ratio, 6)}


def write_metrics(
    output_path: str,
    payload: dict,
    logger: logging.Logger,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Metrics written to '%s'.", path)


def run_pipeline(args: argparse.Namespace, logger: logging.Logger) -> dict:
    t_start = time.perf_counter()

    # 1. Load config
    cfg = load_config(args.config, logger)
    seed    = cfg["seed"]
    window  = cfg["window"]
    version = cfg["version"]

    # 2. Set random seed
    np.random.seed(seed)
    logger.info("NumPy random seed set to %d.", seed)

    # 3. Load data
    df = load_ohlcv(args.input, logger)
    rows_processed = len(df)

    # 4. Compute rolling mean + signal
    df = compute_signal(df, window, logger)

    # 5. Compute metric
    metric_info = compute_metric(df, logger)

    latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
    logger.info("Pipeline completed in %.1f ms.", latency_ms)

    return {
        "version":        version,
        "rows_processed": rows_processed,
        "metric":         metric_info["metric"],
        "value":          metric_info["value"],
        "latency_ms":     latency_ms,
        "seed":           seed,
        "status":         "success",
    }


def main() -> None:
    args = parse_args()

    logger = configure_logging(args.log_file)
    logger.info("=== Pipeline starting ===")
    logger.debug("CLI args: %s", vars(args))

    metrics: dict = {}
    try:
        metrics = run_pipeline(args, logger)
        logger.info("Status: success")

    except Exception as exc:  
        logger.exception("Pipeline failed: %s", exc)
        metrics = {
            "version":        "unknown",
            "rows_processed": 0,
            "metric":         "n/a",
            "value":          None,
            "latency_ms":     None,
            "seed":           None,
            "status":         f"error: {exc}",
        }
    
        try:
            with Path(args.config).open("r") as fh:
                cfg_partial = yaml.safe_load(fh) or {}
            if "version" in cfg_partial:
                metrics["version"] = cfg_partial["version"]
            if "seed" in cfg_partial:
                metrics["seed"] = cfg_partial["seed"]
        except Exception:  
            pass

    finally:
        
        try:
            write_metrics(args.output, metrics, logger)
        except Exception as write_exc: 
            logger.error("Could not write metrics file: %s", write_exc)

        
        print(json.dumps(metrics, indent=2))

    if metrics.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()