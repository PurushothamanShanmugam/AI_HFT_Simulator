#!/bin/bash
set -e

echo "============================================"
echo " HFT AI Simulator"
echo "============================================"

# Run data pipeline on first launch only
if [ ! -f "data/splits/train_70.parquet" ]; then
    echo "[STARTUP] First run detected — running data pipeline..."
    python -m src.data.data_pipeline
    echo "[STARTUP] Data pipeline complete."
else
    echo "[STARTUP] Data splits already exist — skipping pipeline."
fi

# Generate synthetic Kafka test data if missing
if [ ! -f "data/synthetic/kafka_test_data.csv" ]; then
    echo "[STARTUP] Generating synthetic Kafka test data..."
    python -m src.data.synthetic_live_generator
fi

echo "[STARTUP] Starting dashboard..."
exec streamlit run src/dashboard/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false