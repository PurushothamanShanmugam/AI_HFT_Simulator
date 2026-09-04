"""
PPO Trainer — Fast Two-Stage Curriculum
========================================
Stage 1: 5,000 rows/asset, 100K steps  (~10 min)
Stage 2: 10,000 rows/asset, 200K steps (~25 min)
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.data.data_pipeline import load_splits, splits_exist, run_pipeline
from src.envs.multi_asset_env import MultiAssetTradingEnv

MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ProgressCallback(BaseCallback):
    def __init__(self, print_every=5_000, stage=1):
        super().__init__(verbose=0)
        self.print_every = print_every
        self.stage       = stage
        self.start_time  = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            info    = self.locals.get("infos", [{}])[0]
            pv      = info.get("portfolio_value", 100_000)
            wr      = info.get("win_rate_pct", 0.0)
            trades  = info.get("total_trades", 0)
            dd      = info.get("drawdown", 0.0) * 100
            elapsed = time.time() - self.start_time
            pnl     = pv - 100_000
            print(f"  [S{self.stage}] Step {self.n_calls:>7,} | "
                  f"PV: ${pv:>10,.0f} | P&L: ${pnl:>+8,.0f} | "
                  f"Win: {wr:>5.1f}% | Trades: {trades:>4} | "
                  f"DD: {dd:>4.1f}% | {elapsed/60:.1f}m")
        return True


def make_env(df, rows_per_asset, label):
    frames = []
    for asset in df["asset"].unique():
        frames.append(df[df["asset"] == asset].head(rows_per_asset))
    subset = pd.concat(frames, ignore_index=True)
    print(f"  {label}: {len(subset):,} rows | assets: {sorted(subset['asset'].unique().tolist())}")
    return Monitor(MultiAssetTradingEnv(
        data=subset, initial_cash=100_000.0,
        position_size_pct=0.10, max_inventory=10,
        transaction_cost_pct=0.001, slippage_pct=0.0005,
    ))


def quick_eval(model, env, label):
    obs, _ = env.reset()
    done   = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
    pv  = info.get("portfolio_value", 100_000)
    wr  = info.get("win_rate_pct", 0)
    tr  = info.get("total_trades", 0)
    pnl = pv - 100_000
    print(f"  {label}: PV=${pv:,.0f} | P&L=${pnl:+,.0f} | "
          f"Win={wr:.1f}% | Trades={tr}")
    return pv


def train_ppo():
    print("=" * 60)
    print("  PPO — Two-Stage Curriculum (Fast Mode)")
    print("  Stage 1: 5K rows/asset, 100K steps  (~10 min)")
    print("  Stage 2: 10K rows/asset, 200K steps (~25 min)")
    print("=" * 60)

    if not splits_exist():
        run_pipeline()
    from pathlib import Path as _P
    _SD = _P(__file__).resolve().parents[2] / "data" / "splits"
    try:
        train_df = pd.read_parquet(_SD / "train_70.parquet", engine="fastparquet")
        val_df   = pd.read_parquet(_SD / "val_20.parquet",   engine="fastparquet")
        _        = pd.read_parquet(_SD / "test_10.parquet",  engine="fastparquet")
    except Exception:
        train_df, val_df, _ = load_splits()
    print(f"\n  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Stage 1 ───────────────────────────────────────────────────────────
    print("\n── STAGE 1 ──────────────────────────────────────────────")
    t_env = make_env(train_df, 5_000, "Train")  # 5K = many episodes
    v_env = make_env(val_df,   2_000, "Val  ")

    model = PPO("MlpPolicy", t_env, verbose=0,
                learning_rate=3e-4, n_steps=512,
                batch_size=64, n_epochs=5,
                gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs={
                    "net_arch": dict(pi=[256,256,128], vf=[256,256,128]),
                    "optimizer_kwargs": {"weight_decay": 1e-5},
                })

    print(f"  Steps: 100,000 | LR: 3e-4 | n_steps: 512\n")
    t0 = time.time()
    model.learn(150_000, callback=ProgressCallback(5_000, 1),
                progress_bar=False)
    print(f"\n  ✅ Stage 1 done in {(time.time()-t0)/60:.1f} min")
    quick_eval(model, v_env, "Val S1")
    s1_path = MODELS_DIR / "ppo_stage1"
    model.save(str(s1_path))

    # ── Stage 2 ───────────────────────────────────────────────────────────
    print("\n── STAGE 2 ──────────────────────────────────────────────")
    t_env2 = make_env(train_df, 25_000, "Train")
    v_env2 = make_env(val_df,    3_000, "Val  ")

    model2 = PPO.load(str(s1_path), env=t_env2,
                      learning_rate=1e-4, clip_range=0.1)
    print(f"  Stage 1 weights loaded ✅")
    print(f"  Steps: 200,000 | LR: 1e-4 | clip: 0.1\n")

    t0 = time.time()
    model2.learn(300_000, callback=ProgressCallback(10_000, 2),
                 progress_bar=False, reset_num_timesteps=False)
    print(f"\n  ✅ Stage 2 done in {(time.time()-t0)/60:.1f} min")
    quick_eval(model2, v_env2, "Val Final")

    final = MODELS_DIR / "ppo_multi_asset"
    model2.save(str(final))
    print(f"\n  Model saved: {final}.zip")
    print("=" * 60)
    return model2


if __name__ == "__main__":
    train_ppo()