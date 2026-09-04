"""
HFT AI Simulator — Main Entry Point
=====================================
Runs the full pipeline end to end:
  1. Data pipeline   (load + split + augment)
  2. Train all models
  3. Run simulation
  4. Print results

Usage:
  python main.py                  # full pipeline
  python main.py --skip-train     # skip training (use saved models)
  python main.py --pipeline-only  # data pipeline only
  python main.py --simulate-only  # simulation only (needs trained models)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="HFT AI Simulator")
    parser.add_argument("--skip-train",     action="store_true",
                        help="Skip training — use saved models")
    parser.add_argument("--pipeline-only",  action="store_true",
                        help="Run data pipeline only")
    parser.add_argument("--simulate-only",  action="store_true",
                        help="Run simulation only")
    parser.add_argument("--generate-kafka", action="store_true",
                        help="Generate synthetic Kafka test data")
    args = parser.parse_args()

    print("=" * 60)
    print("  HFT AI Simulator")
    print("=" * 60)

    # ── Step 1: Data Pipeline ─────────────────────────────────────────────
    if not args.simulate_only:
        from src.data.data_pipeline import run_pipeline, splits_exist
        if not splits_exist():
            print("\n[STEP 1] Running data pipeline...")
            run_pipeline()
        else:
            print("\n[STEP 1] Data splits already exist — skipping.")

        if args.pipeline_only:
            print("\nDone.")
            return

    # ── Step 2: Generate Kafka test data ──────────────────────────────────
    if args.generate_kafka:
        print("\n[KAFKA] Generating synthetic live data...")
        from src.data.synthetic_live_generator import generate_kafka_test_data
        generate_kafka_test_data()

    # ── Step 3: Train models ──────────────────────────────────────────────
    if not args.skip_train and not args.simulate_only:
        print("\n[STEP 2] Training models...")

        print("\n  Training DQN...")
        from src.models.train_dqn import train_dqn
        train_dqn()

        print("\n  Training PPO...")
        from src.models.train_ppo import train_ppo
        train_ppo()

        print("\n  Training LSTM...")
        from src.models.train_lstm import train_lstm
        train_lstm()

        print("\n  Training XGBoost...")
        from src.models.train_xgboost import train_xgboost
        train_xgboost()

        print("\n  Training Transformer...")
        from src.models.train_transformer import train_transformer
        train_transformer()

    # ── Step 4: Run simulation ────────────────────────────────────────────
    print("\n[STEP 3] Running multi-agent simulation...")
    from src.simulation.multi_agent_sim import run_simulation
    results = run_simulation()

    if results:
        print("\n[DONE] Results saved to outputs/reports/ and outputs/metrics/")
        print("[DONE] Run dashboard: streamlit run src/dashboard/dashboard.py")


if __name__ == "__main__":
    main()