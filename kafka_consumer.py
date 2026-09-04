"""
Kafka Consumer
==============
Reads the latest N ticks from Kafka topic "market-ticks"
and returns them as a DataFrame in the standard HFT AI Simulator schema.

Called by the dashboard when API Feed mode is enabled.
Falls back gracefully if Kafka is unavailable.
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd


def kafka_available(bootstrap: Optional[str] = None) -> bool:
    """Quick check whether Kafka broker is reachable."""
    try:
        from kafka import KafkaConsumer
        bs = bootstrap or os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
        c  = KafkaConsumer(bootstrap_servers=[bs], consumer_timeout_ms=2000)
        c.close()
        return True
    except Exception:
        return False


def consume_ticks(
    n: int = 500,
    topic: str = "market-ticks",
    timeout_ms: int = 8_000,
    bootstrap: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Read the latest N ticks from a Kafka topic.
    Returns a DataFrame or None if Kafka is unavailable.
    """
    try:
        from kafka import KafkaConsumer
        from kafka.errors import NoBrokersAvailable

        bootstrap_servers = bootstrap or os.environ.get(
            "KAFKA_BOOTSTRAP", "localhost:9092"
        )

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers    = [bootstrap_servers],
            auto_offset_reset    = "earliest",
            enable_auto_commit   = False,
            consumer_timeout_ms  = timeout_ms,
            value_deserializer   = lambda x: json.loads(x.decode("utf-8")),
            group_id             = None,
        )

        rows = []
        for message in consumer:
            rows.append(message.value)
            if len(rows) >= n:
                break
        consumer.close()

        if len(rows) < 10:
            print(f"[CONSUMER] Only {len(rows)} ticks available — need at least 10.")
            return None

        df = pd.DataFrame(rows)

        for col in [
            "mid_price", "bid_price", "ask_price", "spread",
            "returns", "volatility", "order_flow_imbalance",
            "depth_imbalance", "microstructure_pressure",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0.0

        if "global_tick" not in df.columns:
            df["global_tick"] = np.arange(len(df))
        else:
            df["global_tick"] = pd.to_numeric(
                df["global_tick"], errors="coerce"
            ).fillna(0).astype(int)

        if "asset" not in df.columns:
            df["asset"] = "CUSTOM"
        if "source" not in df.columns:
            df["source"] = "KAFKA"

        df = df.dropna(subset=["mid_price"])
        df = df[df["mid_price"] > 0].reset_index(drop=True)

        print(f"[CONSUMER] Received {len(df):,} ticks from topic '{topic}'.")
        return df

    except Exception as e:
        print(f"[CONSUMER] Error reading from Kafka: {e}")
        return None