"""
DQN Trainer — Two-Stage Curriculum Learning
============================================
Stage 1: Learn basic trading on small dataset (fast)
Stage 2: Refine strategy on larger dataset (better quality)
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.data.data_pipeline import load_splits, splits_exist, run_pipeline
from src.envs.multi_asset_env import MultiAssetTradingEnv

MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ProgressCallback(BaseCallback):
    """Prints training progress every N steps."""

    def __init__(self, print_every=5_000, stage=1, total_steps=150_000):
        super().__init__(verbose=0)
        self.print_every  = print_every
        self.stage        = stage
        self.total_steps  = total_steps
        self.start_time   = time.time()
        self.best_pv      = 100_000
        self.best_wr      = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            info    = self.locals.get("infos", [{}])[0]
            pv      = info.get("portfolio_value", 100_000)
            wr      = info.get("win_rate_pct", 0.0)
            trades  = info.get("total_trades", 0)
            dd      = info.get("drawdown", 0.0) * 100
            elapsed = time.time() - self.start_time
            pnl     = pv - 100_000
            pct     = self.n_calls / self.total_steps * 100

            # Track personal bests
            improved = ""
            if pv > self.best_pv:
                self.best_pv = pv
                improved = "  ← best portfolio"
            if wr > self.best_wr and trades > 10:
                self.best_wr = wr
                improved = "  ← best win rate"

            # Status indicator
            if pv >= 105_000:
                status = "PROFIT ✅"
            elif pv >= 95_000:
                status = "BREAK-EVEN"
            elif pv >= 80_000:
                status = "LOSS ⚠️"
            else:
                status = "HEAVY LOSS ❌"

            print(
                f"  [Stage {self.stage} | {pct:>5.1f}%] "
                f"Step {self.n_calls:>7,}/{self.total_steps:,} | "
                f"Portfolio: ${pv:>10,.0f} ({status}) | "
                f"P&L: ${pnl:>+9,.0f} | "
                f"Win rate: {wr:>5.1f}% | "
                f"Trades: {trades:>5,} | "
                f"Drawdown: {dd:>5.1f}% | "
                f"Time: {elapsed/60:.1f}min"
                f"{improved}"
            )
        return True


def make_env(df, rows_per_asset, label):
    """Build trading environment with N rows per asset."""
    frames = []
    for asset in df["asset"].unique():
        frames.append(df[df["asset"] == asset].head(rows_per_asset))
    subset = pd.concat(frames, ignore_index=True)
    n_assets = subset["asset"].nunique()
    print(f"  {label} environment: {len(subset):,} rows | "
          f"{n_assets} assets | {rows_per_asset:,} rows/asset")
    return Monitor(MultiAssetTradingEnv(
        data=subset, initial_cash=100_000.0,
        position_size_pct=0.10, max_inventory=10,
        transaction_cost_pct=0.001, slippage_pct=0.0005,
    ))


def evaluate_model(model, env, label):
    """Run one full episode and print results."""
    print(f"\n  Evaluating: {label}...")
    obs, _ = env.reset()
    done   = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)

    pv     = info.get("portfolio_value", 100_000)
    wr     = info.get("win_rate_pct", 0)
    tr     = info.get("total_trades", 0)
    dd     = info.get("drawdown", 0) * 100
    pnl    = pv - 100_000
    pnl_pct = pnl / 100_000 * 100

    print(f"  ┌─────────────────────────────────────────")
    print(f"  │ Portfolio value : ${pv:>10,.2f}")
    print(f"  │ P&L             : ${pnl:>+10,.2f}  ({pnl_pct:+.2f}%)")
    print(f"  │ Win rate        : {wr:>6.1f}%")
    print(f"  │ Total trades    : {tr:>6,}")
    print(f"  │ Max drawdown    : {dd:>6.1f}%")
    print(f"  └─────────────────────────────────────────")
    return pv, wr, tr


def train_dqn():
    print("\n" + "=" * 65)
    print("  HFT AI Simulator — DQN Two-Stage Curriculum Training")
    print("=" * 65)
    print("  Goal: Train DQN agent to trade 8 assets with $100K budget")
    print("  Method: Stage 1 = learn basics, Stage 2 = refine strategy")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[DATA] Loading training splits...")
    if not splits_exist():
        print("  Splits not found — running data pipeline first...")
        run_pipeline()

    SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"
    try:
        train_df = pd.read_parquet(SPLITS_DIR / "train_70.parquet", engine="fastparquet")
        val_df   = pd.read_parquet(SPLITS_DIR / "val_20.parquet",   engine="fastparquet")
    except Exception:
        train_df, val_df, _ = load_splits()

    if "asset" not in train_df.columns:
        raise ValueError("ERROR: asset column missing from parquet file")

    print(f"  Training data   : {len(train_df):,} rows")
    print(f"  Validation data : {len(val_df):,} rows")
    print(f"  Assets          : {sorted(train_df['asset'].unique().tolist())}")
    print(f"  Features        : {sum(1 for c in train_df.columns if c.startswith('f_'))} market features")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  STAGE 1 — Basic Learning (Small Dataset)")
    print("  Purpose : Agent explores the market and learns basic signals")
    print("  Data    : 5,000 rows per asset = 40,000 total rows")
    print("  Steps   : 150,000 training steps")
    print("  Episodes: ~3,750 complete episodes (40K ÷ 5K × 150K/steps)")
    print("  Expected: Win rate 30-50% by end of Stage 1")
    print("═" * 65)

    print("\n  Building Stage 1 environments...")
    t_env = make_env(train_df, 5_000, "Training")
    v_env = make_env(val_df,   2_000, "Validation")

    print(f"\n  DQN Model configuration:")
    print(f"    Network       : [256 → 256 → 128] neurons")
    print(f"    Learning rate : 3e-4 (aggressive — learning fast)")
    print(f"    Exploration   : 50% random → 5% random over 150K steps")
    print(f"    Buffer size   : 10,000 experiences")
    print(f"    Regularisation: L2 weight decay = 1e-5")
    print(f"\n  Training started — progress every 5,000 steps:\n")

    model = DQN(
        "MlpPolicy", t_env, verbose=0,
        learning_rate=3e-4,
        buffer_size=10_000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=200,
        exploration_fraction=0.50,
        exploration_final_eps=0.05,
        policy_kwargs={
            "net_arch": [256, 256, 128],
            "optimizer_kwargs": {"weight_decay": 1e-5},
        },
    )

    s1_start = time.time()
    model.learn(
        150_000,
        callback=ProgressCallback(5_000, stage=1, total_steps=150_000),
        progress_bar=False,
    )
    s1_elapsed = time.time() - s1_start

    print(f"\n  ✅ Stage 1 training complete in {s1_elapsed/60:.1f} minutes")
    evaluate_model(model, v_env, "Stage 1 — Validation")

    s1_path = MODELS_DIR / "dqn_stage1"
    model.save(str(s1_path))
    print(f"  Stage 1 model saved → {s1_path}.zip")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  STAGE 2 — Strategy Refinement (Larger Dataset)")
    print("  Purpose : Agent refines what it learned on more diverse data")
    print("  Data    : 25,000 rows per asset = 200,000 total rows")
    print("  Steps   : 300,000 training steps")
    print("  Weights : Loaded from Stage 1 (not starting from scratch)")
    print("  Expected: Win rate 45-65% by end of Stage 2")
    print("═" * 65)

    print("\n  Building Stage 2 environments...")
    t_env2 = make_env(train_df, 25_000, "Training")
    v_env2 = make_env(val_df,    6_000, "Validation")

    print(f"\n  Loading Stage 1 weights...")
    model2 = DQN.load(str(s1_path), env=t_env2, learning_rate=3e-5)
    model2.exploration_fraction  = 0.10  # less exploration — agent knows basics
    model2.exploration_final_eps = 0.01
    print(f"  ✅ Stage 1 weights loaded successfully")
    print(f"\n  Stage 2 configuration:")
    print(f"    Learning rate : 3e-5 (lower — fine-tuning, not relearning)")
    print(f"    Exploration   : 10% → 1% (mostly exploiting learned policy)")
    print(f"\n  Training started — progress every 10,000 steps:\n")

    s2_start = time.time()
    model2.learn(
        300_000,
        callback=ProgressCallback(10_000, stage=2, total_steps=300_000),
        progress_bar=False,
        reset_num_timesteps=False,
    )
    s2_elapsed = time.time() - s2_start

    print(f"\n  ✅ Stage 2 training complete in {s2_elapsed/60:.1f} minutes")
    evaluate_model(model2, v_env2, "Stage 2 — Final Validation")

    final_path = MODELS_DIR / "dqn_multi_asset"
    model2.save(str(final_path))

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = s1_elapsed + s2_elapsed
    print(f"\n{'═'*65}")
    print(f"  DQN TRAINING COMPLETE")
    print(f"{'═'*65}")
    print(f"  Stage 1 time  : {s1_elapsed/60:.1f} minutes")
    print(f"  Stage 2 time  : {s2_elapsed/60:.1f} minutes")
    print(f"  Total time    : {total_time/60:.1f} minutes")
    print(f"  Model saved   : {final_path}.zip")
    print(f"\n  Next step: python -m src.models.train_xgboost")
    print(f"{'═'*65}\n")
    return model2


if __name__ == "__main__":
    train_dqn()