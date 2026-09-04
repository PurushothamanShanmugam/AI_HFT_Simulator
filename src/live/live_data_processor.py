"""
Live Data Processor
====================
Watches the data/live_data/ folder for newly uploaded CSV or JSON files,
normalises them into the standard HFT AI Simulator schema, and streams
every tick to the Kafka topic "market-ticks".

Can be called two ways:
  1. From the dashboard (programmatic) via process_live_folder()
  2. As a standalone watcher via: python -m src.live.live_data_processor

Supported upload formats
------------------------
CSV  — any file with price columns (mid_price / midpoint / close / bid+ask)
JSON — array of tick objects, or newline-delimited JSON

Standard output schema (matches kafka_consumer.py expectations)
---------------------------------------------------------------
  asset, mid_price, bid_price, ask_price, spread,
  returns, volatility, order_flow_imbalance,
  depth_imbalance, microstructure_pressure,
  imbalance, timestamp, global_tick, source
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
LIVE_DIR  = ROOT / "data" / "live_data"
DONE_DIR  = LIVE_DIR / ".processed"

LIVE_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)

# Column name aliases accepted from uploaded files
COLUMN_ALIASES = {
    "midpoint":       "mid_price",
    "mid":            "mid_price",
    "price":          "mid_price",
    "close":          "mid_price",
    "last":           "mid_price",
    "ltp":            "mid_price",
    "bid":            "bid_price",
    "best_bid":       "bid_price",
    "ask":            "ask_price",
    "best_ask":       "ask_price",
    "offer":          "ask_price",
    "bid_ask_spread": "spread",
    "vol":            "volume",
    "qty":            "volume",
    "quantity":       "volume",
    "system_time":    "timestamp",
    "datetime":       "timestamp",
    "date":           "timestamp",
    "time":           "timestamp",
    "symbol":         "asset",
    "ticker":         "asset",
    "coin":           "asset",
    "pair":           "asset",
    "buys":           "buy_volume",
    "sells":          "sell_volume",
}

REQUIRED_OUTPUT_COLS = [
    "asset", "mid_price", "bid_price", "ask_price", "spread",
    "returns", "volatility", "order_flow_imbalance",
    "depth_imbalance", "microstructure_pressure",
    "imbalance", "timestamp", "global_tick", "source",
]


# ── File readers ──────────────────────────────────────────────────────────────

def _read_file(path: Path) -> pd.DataFrame:
    """Read CSV or JSON file into a raw DataFrame."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".json":
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                # Try common wrappers: {"data": [...]} or {"ticks": [...]}
                for key in ("data", "ticks", "records", "rows", "items"):
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
        except json.JSONDecodeError:
            # Try newline-delimited JSON
            rows = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if rows:
                return pd.DataFrame(rows)

    raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .json")


# ── Schema normalisation ──────────────────────────────────────────────────────

def normalise(df: pd.DataFrame, source_name: str = "LIVE") -> pd.DataFrame:
    """
    Normalise an arbitrarily-formatted upload into the standard schema.
    Returns a clean DataFrame ready for Kafka streaming.
    """
    df = df.copy()

    # Lowercase all column names, strip whitespace
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Apply aliases
    df = df.rename(columns=COLUMN_ALIASES)

    # ── mid_price ──────────────────────────────────────────────────────────
    if "mid_price" not in df.columns:
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["mid_price"] = (
                pd.to_numeric(df["bid_price"], errors="coerce") +
                pd.to_numeric(df["ask_price"], errors="coerce")
            ) / 2
        else:
            raise ValueError(
                "Cannot find price column. Expected one of: "
                "mid_price, midpoint, price, close, ltp, or both bid and ask columns."
            )

    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce").ffill().bfill()
    df = df[df["mid_price"] > 0].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows with positive mid_price found.")

    # ── bid / ask ──────────────────────────────────────────────────────────
    if "bid_price" not in df.columns:
        df["bid_price"] = df["mid_price"] * 0.9999
    else:
        df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce").fillna(
            df["mid_price"] * 0.9999
        )

    if "ask_price" not in df.columns:
        df["ask_price"] = df["mid_price"] * 1.0001
    else:
        df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce").fillna(
            df["mid_price"] * 1.0001
        )

    # ── spread ─────────────────────────────────────────────────────────────
    if "spread" not in df.columns:
        df["spread"] = (df["ask_price"] - df["bid_price"]).abs()
    else:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(
            (df["ask_price"] - df["bid_price"]).abs()
        )
    df["spread"] = df["spread"].clip(lower=0)

    # ── asset ──────────────────────────────────────────────────────────────
    if "asset" not in df.columns:
        # Try to infer from filename
        df["asset"] = source_name.split("_")[0].upper()[:6]
    else:
        df["asset"] = df["asset"].astype(str).str.upper().str.strip()

    # ── timestamp ──────────────────────────────────────────────────────────
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.RangeIndex(len(df)).astype(str)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype(str)

    # ── returns & volatility ───────────────────────────────────────────────
    df["returns"] = (
        df.groupby("asset")["mid_price"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    df["volatility"] = (
        df.groupby("asset")["returns"]
        .transform(lambda x: x.rolling(20, min_periods=2).std())
        .fillna(0)
    )

    # ── order flow imbalance ───────────────────────────────────────────────
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        bv = pd.to_numeric(df["buy_volume"], errors="coerce").fillna(0)
        sv = pd.to_numeric(df["sell_volume"], errors="coerce").fillna(0)
        df["order_flow_imbalance"] = ((bv - sv) / (bv + sv + 1e-9)).clip(-1, 1)
    elif "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        shifted = vol.shift(1).fillna(0)
        df["order_flow_imbalance"] = (
            (vol - shifted) / (vol + shifted + 1e-9)
        ).clip(-1, 1)
    else:
        # Approximate from bid/ask sizes if present
        if "bid_size" in df.columns and "ask_size" in df.columns:
            bs = pd.to_numeric(df["bid_size"], errors="coerce").fillna(100)
            as_ = pd.to_numeric(df["ask_size"], errors="coerce").fillna(100)
            df["order_flow_imbalance"] = ((bs - as_) / (bs + as_ + 1e-9)).clip(-1, 1)
        else:
            # Fall back to return-direction proxy
            df["order_flow_imbalance"] = np.sign(df["returns"]) * 0.3

    # ── depth imbalance & microstructure pressure ──────────────────────────
    df["depth_imbalance"] = (df["order_flow_imbalance"] * 0.7).clip(-1, 1)
    df["microstructure_pressure"] = (
        0.5 * df["order_flow_imbalance"] + 0.5 * df["depth_imbalance"]
    ).clip(-1, 1)
    df["imbalance"] = df["order_flow_imbalance"]

    # ── global_tick & source ───────────────────────────────────────────────
    df["global_tick"] = np.arange(len(df))
    df["source"]      = "LIVE"

    # ── final cleanup ──────────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    df = df.reset_index(drop=True)

    return df


# ── Kafka streaming ───────────────────────────────────────────────────────────

def stream_dataframe_to_kafka(
    df: pd.DataFrame,
    topic: str = "market-ticks",
    bootstrap: Optional[str] = None,
    interval: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    Stream a normalised DataFrame to Kafka row by row.
    Returns number of rows sent.
    interval=0.0 means send as fast as possible (batch mode for dashboard).
    """
    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable
    except ImportError:
        raise RuntimeError(
            "kafka-python not installed. Run: pip install kafka-python"
        )

    bootstrap_servers = bootstrap or os.environ.get(
        "KAFKA_BOOTSTRAP", "localhost:9092"
    )

    for attempt in range(5):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                linger_ms=10,
                batch_size=16384,
            )
            break
        except NoBrokersAvailable:
            if verbose:
                print(f"[LIVE] Waiting for Kafka... attempt {attempt+1}/5")
            time.sleep(2)
    else:
        raise RuntimeError(
            f"Cannot connect to Kafka at {bootstrap_servers}. "
            "Is the broker running? Start with: docker compose --profile kafka up"
        )

    sent = 0
    for _, row in df.iterrows():
        msg = {}
        for k, v in row.to_dict().items():
            if isinstance(v, (np.floating, float)):
                msg[k] = float(v) if np.isfinite(v) else 0.0
            elif isinstance(v, (np.integer, int)):
                msg[k] = int(v)
            else:
                msg[k] = str(v)

        producer.send(topic, value=msg)
        sent += 1

        if interval > 0:
            time.sleep(interval)

        if verbose and sent % 1000 == 0:
            print(f"[LIVE] Sent {sent:,} / {len(df):,} ticks")

    producer.flush()
    producer.close()

    if verbose:
        print(f"[LIVE] Done — streamed {sent:,} ticks to topic '{topic}'")

    return sent


# ── High-level: process one file ──────────────────────────────────────────────

def process_file(
    path: Path,
    topic: str = "market-ticks",
    bootstrap: Optional[str] = None,
    interval: float = 0.0,
    move_to_done: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Read → normalise → stream one uploaded file.
    Returns (normalised_df, rows_sent).
    """
    if verbose:
        print(f"[LIVE] Processing: {path.name}")

    raw = _read_file(path)

    if verbose:
        print(f"[LIVE]   Raw rows   : {len(raw):,}")
        print(f"[LIVE]   Raw columns: {list(raw.columns)}")

    df = normalise(raw, source_name=path.stem)

    if verbose:
        print(f"[LIVE]   Clean rows : {len(df):,}")
        print(f"[LIVE]   Assets     : {df['asset'].unique().tolist()}")

    sent = stream_dataframe_to_kafka(
        df, topic=topic, bootstrap=bootstrap,
        interval=interval, verbose=verbose,
    )

    if move_to_done:
        dest = DONE_DIR / path.name
        path.rename(dest)
        if verbose:
            print(f"[LIVE]   Moved to   : {dest}")

    return df, sent


# ── High-level: process whole folder ─────────────────────────────────────────

def process_live_folder(
    topic: str = "market-ticks",
    bootstrap: Optional[str] = None,
    interval: float = 0.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame | None, int]:
    """
    Process ALL unprocessed files in data/live_data/.
    Called by the dashboard when Live Data mode is activated.
    Returns (combined_df, total_rows_sent).
    Raises RuntimeError if no files found or Kafka unreachable.
    """
    files = sorted([
        f for f in LIVE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".csv", ".json")
        and not f.name.startswith(".")
    ])

    if not files:
        raise FileNotFoundError(
            f"No CSV or JSON files found in {LIVE_DIR}. "
            "Please upload data files first."
        )

    if verbose:
        print(f"[LIVE] Found {len(files)} file(s) to process")

    all_frames = []
    total_sent = 0

    for f in files:
        try:
            df, sent = process_file(
                f, topic=topic, bootstrap=bootstrap,
                interval=interval, move_to_done=True, verbose=verbose,
            )
            all_frames.append(df)
            total_sent += sent
        except Exception as e:
            if verbose:
                print(f"[LIVE] ⚠️  Skipped {f.name}: {e}")

    if not all_frames:
        raise RuntimeError("All files failed to process. Check file format.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.reset_index(drop=True)

    return combined, total_sent


# ── Folder watcher (standalone mode) ─────────────────────────────────────────

def watch_folder(
    poll_interval: float = 5.0,
    topic: str = "market-ticks",
    bootstrap: Optional[str] = None,
    tick_interval: float = 0.0,
):
    """
    Continuously watch data/live_data/ and process new files as they appear.
    Run as: python -m src.live.live_data_processor
    """
    print(f"[LIVE WATCHER] Watching: {LIVE_DIR}")
    print(f"[LIVE WATCHER] Topic   : {topic}")
    print(f"[LIVE WATCHER] Kafka   : {bootstrap or 'localhost:9092'}")
    print(f"[LIVE WATCHER] Press Ctrl+C to stop\n")

    seen = set()

    while True:
        files = [
            f for f in LIVE_DIR.iterdir()
            if f.is_file()
            and f.suffix.lower() in (".csv", ".json")
            and not f.name.startswith(".")
            and f not in seen
        ]

        for f in sorted(files):
            seen.add(f)
            try:
                process_file(
                    f, topic=topic, bootstrap=bootstrap,
                    interval=tick_interval, move_to_done=True, verbose=True,
                )
            except Exception as e:
                print(f"[LIVE WATCHER] ⚠️  {f.name} failed: {e}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HFT AI Simulator — Live Data Processor"
    )
    parser.add_argument("--watch",    action="store_true",
                        help="Continuously watch folder for new files")
    parser.add_argument("--topic",    default="market-ticks")
    parser.add_argument("--bootstrap", default=None,
                        help="Kafka bootstrap server (default: localhost:9092)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Seconds between ticks (0 = send immediately)")
    parser.add_argument("--poll",     type=float, default=5.0,
                        help="Folder poll interval in seconds (watch mode only)")
    args = parser.parse_args()

    if args.watch:
        watch_folder(
            poll_interval=args.poll,
            topic=args.topic,
            bootstrap=args.bootstrap,
            tick_interval=args.interval,
        )
    else:
        combined, total = process_live_folder(
            topic=args.topic,
            bootstrap=args.bootstrap,
            interval=args.interval,
        )
        print(f"\n[LIVE] Total rows streamed: {total:,}")