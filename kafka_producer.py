"""
Kafka Producer
==============
Reads a CSV file and streams it row by row to Kafka topic "market-ticks".
Supports both user-supplied CSVs and the synthetic test data.

Usage:
  # Stream synthetic test data (default)
  python kafka_producer.py

  # Stream a specific CSV
  python kafka_producer.py --file data/synthetic/kafka_test_data.csv

  # Stream with custom interval
  python kafka_producer.py --file data/raw/crypto/BTC_1sec.csv --interval 0.01

  # Docker
  docker compose --profile kafka up
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

COLUMN_MAP = {
    "midpoint":       "mid_price",
    "mid":            "mid_price",
    "price":          "mid_price",
    "close":          "mid_price",
    "last":           "mid_price",
    "bid_ask_spread": "spread",
    "bid":            "bid_price",
    "best_bid":       "bid_price",
    "ask":            "ask_price",
    "best_ask":       "ask_price",
    "offer":          "ask_price",
    "vol":            "volume",
    "qty":            "volume",
    "system_time":    "timestamp",
    "datetime":       "timestamp",
    "date":           "timestamp",
    "symbol":         "asset",
    "ticker":         "asset",
}


def normalise_dataframe(df: pd.DataFrame, asset_name: str = "CUSTOM") -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    if "mid_price" not in df.columns:
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
        else:
            raise ValueError(
                "Cannot find price column. Expected: mid_price, midpoint, price, close"
            )

    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce").ffill()

    if "bid_price" not in df.columns:
        df["bid_price"] = df["mid_price"] * 0.9999
    if "ask_price" not in df.columns:
        df["ask_price"] = df["mid_price"] * 1.0001
    if "spread" not in df.columns:
        df["spread"] = df["ask_price"] - df["bid_price"]

    df["returns"]   = df["mid_price"].pct_change().fillna(0)
    df["volatility"] = df["returns"].rolling(20, min_periods=2).std().fillna(0)

    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["order_flow_imbalance"] = (
            (vol - vol.shift(1)).fillna(0) / (vol + vol.shift(1) + 1e-9)
        ).clip(-1, 1)
    else:
        rng = np.random.RandomState(42)
        df["order_flow_imbalance"] = rng.uniform(-0.5, 0.5, len(df))

    df["depth_imbalance"]         = df["order_flow_imbalance"] * 0.7
    df["microstructure_pressure"] = (
        0.5 * df["order_flow_imbalance"] + 0.5 * df["depth_imbalance"]
    )
    df["imbalance"]  = df["order_flow_imbalance"]
    df["bid_size"]   = 100.0
    df["ask_size"]   = 100.0

    if "asset" not in df.columns:
        df["asset"] = asset_name
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.RangeIndex(len(df)).astype(str)

    df["global_tick"]  = np.arange(len(df))
    df["source_file"]  = "kafka_feed"

    df = df.dropna(subset=["mid_price"]).reset_index(drop=True)
    return df


def stream_to_kafka(
    csv_path: str,
    interval: float = 0.05,
    topic: str = "market-ticks",
    bootstrap: str = None,
):
    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable
    except ImportError:
        print("[ERROR] kafka-python not installed. Run: pip install kafka-python")
        sys.exit(1)

    print(f"[PRODUCER] Loading: {csv_path}")
    df          = pd.read_csv(csv_path)
    asset_name  = Path(csv_path).stem.split("_")[0].upper()
    df          = normalise_dataframe(df, asset_name=asset_name)
    print(f"[PRODUCER] Rows   : {len(df):,}")
    print(f"[PRODUCER] Schema : {list(df.columns)}")

    bootstrap_servers = bootstrap or os.environ.get(
        "KAFKA_BOOTSTRAP", "localhost:9092"
    )
    print(f"[PRODUCER] Connecting to Kafka at {bootstrap_servers}...")

    for attempt in range(10):
        try:
            producer = KafkaProducer(
                bootstrap_servers = [bootstrap_servers],
                value_serializer  = lambda v: json.dumps(v).encode("utf-8"),
                acks              = "all",
            )
            print("[PRODUCER] Connected.")
            break
        except NoBrokersAvailable:
            print(f"[PRODUCER] Waiting for Kafka... attempt {attempt+1}/10")
            time.sleep(3)
    else:
        print("[PRODUCER] Could not connect to Kafka after 10 attempts.")
        sys.exit(1)

    print(f"[PRODUCER] Streaming {len(df):,} rows to topic '{topic}'...")

    for i, (_, row) in enumerate(df.iterrows()):
        msg = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
            for k, v in row.to_dict().items()
        }
        producer.send(topic, value=msg)

        if (i + 1) % 1000 == 0:
            print(f"[PRODUCER] Sent {i+1:,} / {len(df):,} ticks")

        time.sleep(interval)

    producer.flush()
    producer.close()
    print(f"[PRODUCER] Done. Streamed {len(df):,} ticks to '{topic}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFT AI Simulator — Kafka Producer")
    parser.add_argument(
        "--file",
        default=os.environ.get(
            "CSV_FILE",
            str(ROOT / "data" / "synthetic" / "kafka_test_data.csv"),
        ),
        help="Path to CSV file to stream",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("TICK_INTERVAL", "0.05")),
        help="Seconds between ticks (default 0.05 = 20 ticks/sec)",
    )
    parser.add_argument(
        "--topic",
        default="market-ticks",
        help="Kafka topic name",
    )
    args = parser.parse_args()

    # Auto-generate synthetic data if file doesn't exist
    if not Path(args.file).exists():
        print(f"[PRODUCER] File not found: {args.file}")
        print("[PRODUCER] Generating synthetic test data...")
        sys.path.insert(0, str(ROOT))
        from src.data.synthetic_live_generator import generate_kafka_test_data
        generate_kafka_test_data(save=True)

    stream_to_kafka(args.file, interval=args.interval, topic=args.topic)