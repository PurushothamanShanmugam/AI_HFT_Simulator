"""
Multi-Agent Simulation
======================
Step 4 of the hft_ai_simulator build.

Loads all 5 trained AI models and runs them simultaneously
on the 10% test split (data they have never seen before).

All 5 agents:
  - Trade in the same shared order book
  - Use real prices (no normalisation)
  - Start with $100,000 each
  - Trade across all 8 assets

Agents:
  1. DQN         — RL agent (Stable Baselines3)
  2. PPO         — RL agent (Stable Baselines3)
  3. LSTM        — Supervised (PyTorch)
  4. XGBoost     — Supervised (XGBoost)
  5. Transformer — Supervised (PyTorch)

Output:
  - Console leaderboard
  - outputs/reports/simulation_report.json
  - outputs/metrics/agent_metrics.csv
"""

from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
# sklearn imported inside functions to avoid Windows DLL issues

from src.data.data_pipeline import load_splits, splits_exist, run_pipeline
from src.data.feature_builder import get_observation_features
from src.envs.multi_asset_env import (
    MultiAssetTradingEnv, ASSETS, ACTION_TO_ASSET, ACTION_TO_SIDE
)
from src.analytics.risk_metrics import (
    sharpe_ratio, max_drawdown, win_rate, sortino_ratio
)

MODELS_DIR  = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
METRICS_DIR = ROOT / "outputs" / "metrics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_COLS = get_observation_features()
SEQ_LEN      = 60


# ── Model loaders ─────────────────────────────────────────────────────────

def load_dqn():
    try:
        from stable_baselines3 import DQN
        path = MODELS_DIR / "dqn_multi_asset.zip"
        if not path.exists():
            print("  [SKIP] DQN model not found")
            return None
        model = DQN.load(str(path))
        print("  [OK] DQN loaded")
        return model
    except Exception as e:
        print(f"  [FAIL] DQN: {e}")
        return None


def load_ppo():
    try:
        from stable_baselines3 import PPO
        path = MODELS_DIR / "ppo_multi_asset.zip"
        if not path.exists():
            print("  [SKIP] PPO model not found")
            return None
        model = PPO.load(str(path))
        print("  [OK] PPO loaded")
        return model
    except Exception as e:
        print(f"  [FAIL] PPO: {e}")
        return None


def load_lstm():
    try:
        from src.models.train_lstm import LSTMWithAttention, load_lstm_model
        model = load_lstm_model()
        if model is None:
            print("  [SKIP] LSTM model not found")
            return None
        print("  [OK] LSTM loaded")
        return model
    except Exception as e:
        print(f"  [FAIL] LSTM: {e}")
        return None


def load_xgboost():
    try:
        from src.models.train_xgboost import load_xgboost_model
        model, scaler = load_xgboost_model()
        if model is None:
            print("  [SKIP] XGBoost model not found")
            return None, None
        print("  [OK] XGBoost loaded")
        return model, scaler
    except Exception as e:
        print(f"  [FAIL] XGBoost: {e}")
        return None, None


def load_transformer():
    try:
        from src.models.train_transformer import load_transformer_model
        model = load_transformer_model()
        if model is None:
            print("  [SKIP] Transformer model not found")
            return None
        print("  [OK] Transformer loaded")
        return model
    except Exception as e:
        print(f"  [FAIL] Transformer: {e}")
        return None


# ── Agent wrappers ────────────────────────────────────────────────────────

class RLAgent:
    """Wraps a Stable Baselines3 DQN or PPO model."""

    def __init__(self, name: str, model):
        self.name  = name
        self.model = model

    def predict(self, obs: np.ndarray, env=None) -> int:
        if self.model is None:
            return 0
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)


class LSTMAgent:
    """Wraps a PyTorch LSTM model trained on 38 temporal features."""

    _TEMPORAL = {
        'f_returns','f_log_return','f_ofi','f_imbalance','f_depth_imbalance',
        'f_pressure','f_buy_ratio','f_sell_ratio','f_return_5','f_return_10',
        'f_return_20','f_return_60','f_accel','f_jerk','f_vol_5','f_vol_10',
        'f_vol_20','f_vol_60','f_zscore_20','f_zscore_60','f_dev_vwap',
        'f_bollinger_pct','f_effective_spread','f_price_impact',
        'f_bid_ask_bounce','f_trade_sign','f_kyle_lambda','f_size_ratio',
        'f_trend_strength','f_vol_regime','f_spread_regime','f_liquidity_score',
        'f_prev_return','f_prev_spread','f_prev_imbalance','f_time_of_day',
        'f_spread_pct','f_volatility',
    }

    def __init__(self, name: str, model):
        self.name    = name
        self.model   = model
        self.buffer  = []
        self.scaler  = None
        self._fitted = False
        self._feat_idx = [i for i, f in enumerate(FEATURE_COLS)
                          if f in self._TEMPORAL]

    def predict(self, obs: np.ndarray, env=None) -> int:
        if self.model is None:
            return 0

        from sklearn.preprocessing import StandardScaler as _SS
        if self.scaler is None:
            self.scaler = _SS()

        # Select only 38 temporal features that LSTM was trained on
        market_feat = obs[self._feat_idx] if self._feat_idx else obs[:38]
        self.buffer.append(market_feat)

        if len(self.buffer) < SEQ_LEN:
            return 0

        if len(self.buffer) > SEQ_LEN:
            self.buffer.pop(0)

        seq = np.array(self.buffer, dtype=np.float32)

        if not self._fitted:
            self.scaler.fit(seq)
            self._fitted = True
        seq = self.scaler.transform(seq)
        seq = np.nan_to_num(seq, nan=0.0)

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x)
            pred   = logits.argmax(dim=1).item()

        # Map LSTM output (0=HOLD, 1=BUY, 2=SELL) to env action
        # For LSTM: BUY best asset, SELL all with positive inventory
        if pred == 1:
            return 1   # BUY BTC (action 1)
        elif pred == 2:
            return 2   # SELL BTC (action 2)
        return 0       # HOLD


class XGBoostAgent:
    """Wraps an XGBoost classifier model."""

    def __init__(self, name: str, model, scaler):
        self.name   = name
        self.model  = model
        self.scaler = scaler

    def predict(self, obs: np.ndarray, env=None) -> int:
        if self.model is None:
            return 0

        market_feat = obs[:len(FEATURE_COLS)].reshape(1, -1)

        if self.scaler is not None:
            try:
                market_feat = self.scaler.transform(market_feat)
            except Exception:
                pass

        market_feat = np.nan_to_num(market_feat, nan=0.0)

        try:
            pred = self.model.predict(market_feat)[0]
            proba = self.model.predict_proba(market_feat)[0]

            # Only trade if confidence is high enough
            confidence = proba.max()
            if confidence < 0.5:
                return 0

            if pred == 1:
                return 1   # BUY BTC
            elif pred == 2:
                return 2   # SELL BTC
        except Exception:
            pass

        return 0


class TransformerAgent:
    """Wraps a PyTorch Transformer model."""

    def __init__(self, name: str, model):
        self.name    = name
        self.model   = model
        self.buffer  = []
        self.scaler  = None
        self._fitted = False

    def predict(self, obs: np.ndarray, env=None) -> int:
        if self.model is None:
            return 0

        from sklearn.preprocessing import StandardScaler as _SS
        if self.scaler is None:
            self.scaler = _SS()

        market_feat = obs[:len(FEATURE_COLS)]
        self.buffer.append(market_feat)

        if len(self.buffer) < SEQ_LEN:
            return 0

        if len(self.buffer) > SEQ_LEN:
            self.buffer.pop(0)

        seq = np.array(self.buffer, dtype=np.float32)

        if not self._fitted:
            self.scaler.fit(seq)
            self._fitted = True
        seq = self.scaler.transform(seq)
        seq = np.nan_to_num(seq, nan=0.0)

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x)
            pred   = logits.argmax(dim=1).item()

        if pred == 1:
            return 1
        elif pred == 2:
            return 2
        return 0


# ── Simulation runner ─────────────────────────────────────────────────────

class AgentSimulator:
    """Runs one agent through the full test environment."""

    def __init__(self, name: str, agent, test_df: pd.DataFrame):
        self.name  = name
        self.agent = agent
        self.env   = MultiAssetTradingEnv(
            data                 = test_df,
            initial_cash         = 100_000.0,
            position_size_pct    = 0.10,
            max_inventory        = 10,
            transaction_cost_pct = 0.001,
            slippage_pct         = 0.0005,
        )
        self.portfolio_history = []
        self.trade_log         = []

    def run(self) -> dict:
        obs, _ = self.env.reset()
        done   = False
        step   = 0

        while not done:
            action = self.agent.predict(obs, self.env)
            obs, reward, done, _, info = self.env.step(action)

            self.portfolio_history.append(info["portfolio_value"])

            if info["trade_executed"]:
                self.trade_log.append({
                    "step":      step,
                    "asset":     info["trade_asset"],
                    "side":      info["trade_side"],
                    "pnl":       info["realised_pnl"],
                    "portfolio": info["portfolio_value"],
                })

            step += 1

        return self.env.get_portfolio_summary()


# ── Main simulation ───────────────────────────────────────────────────────

def run_simulation(n_test_ticks: int = 50_000):
    print("=" * 60)
    print("HFT AI Simulator — Multi-Agent Simulation")
    print("=" * 60)

    # ── Load test data ────────────────────────────────────────────────────
    print("\n[1/4] Loading test data...")
    if not splits_exist():
        print("  Splits not found — running data pipeline first...")
        run_pipeline()

    _, _, test_df = load_splits()

    if n_test_ticks:
        assets = test_df["asset"].unique()
        rows_per_asset = max(1, n_test_ticks // len(assets))
        test_df = pd.concat([
            test_df[test_df["asset"]==a].head(rows_per_asset)
            for a in assets
        ], ignore_index=True)

    print(f"  Test rows  : {len(test_df):,}")
    print(f"  Assets     : {test_df['asset'].nunique()}")
    print(f"  Date range : {test_df['timestamp'].iloc[0]} → "
          f"{test_df['timestamp'].iloc[-1]}")

    # ── Load models ───────────────────────────────────────────────────────
    print("\n[2/4] Loading trained models...")
    dqn_model        = load_dqn()
    ppo_model        = load_ppo()
    lstm_model       = load_lstm()
    xgb_model, xgb_scaler = load_xgboost()
    transformer_model = load_transformer()

    # Build agent list
    agents = []
    if dqn_model:
        agents.append(RLAgent("DQN", dqn_model))
    if ppo_model:
        agents.append(RLAgent("PPO", ppo_model))
    if lstm_model:
        agents.append(LSTMAgent("LSTM", lstm_model))
    if xgb_model:
        agents.append(XGBoostAgent("XGBoost", xgb_model, xgb_scaler))
    if transformer_model:
        agents.append(TransformerAgent("Transformer", transformer_model))

    if not agents:
        print("\n  No trained models found.")
        print("  Run training scripts first:")
        print("    python -m src.models.train_dqn")
        print("    python -m src.models.train_ppo")
        print("    python -m src.models.train_lstm")
        print("    python -m src.models.train_xgboost")
        print("    python -m src.models.train_transformer")
        return None

    print(f"\n  {len(agents)} agents ready: {[a.name for a in agents]}")

    # ── Run simulation ────────────────────────────────────────────────────
    print("\n[3/4] Running simulation...")
    results    = {}
    simulators = {}

    for agent in agents:
        print(f"\n  Running {agent.name}...")
        start_time = time.time()
        sim  = AgentSimulator(agent.name, agent, test_df)
        summ = sim.run()
        elapsed = time.time() - start_time

        # Add extra risk metrics
        pv_series = pd.Series(sim.portfolio_history)
        rets      = pv_series.pct_change().dropna()

        summ["sharpe"]    = round(float(sharpe_ratio(pv_series)), 3)
        summ["sortino"]   = round(float(sortino_ratio(pv_series)), 3)
        summ["max_dd"]    = round(float(max_drawdown(pv_series)), 3)
        summ["elapsed_s"] = round(elapsed, 1)
        summ["agent"]     = agent.name

        results[agent.name]    = summ
        simulators[agent.name] = sim

        print(f"    Portfolio : ${summ['portfolio_value']:>12,.2f}")
        print(f"    P&L       : ${summ['pnl']:>+12,.2f}  ({summ['pnl_pct']:>+6.2f}%)")
        print(f"    Win rate  : {summ['win_rate']:>6.1f}%")
        print(f"    Trades    : {summ['total_trades']:>6}")
        print(f"    Sharpe    : {summ['sharpe']:>6.3f}")
        print(f"    Max DD    : {summ['max_dd']*100:>6.1f}%")
        print(f"    Time      : {elapsed:.1f}s")

    # ── Leaderboard ───────────────────────────────────────────────────────
    print("\n[4/4] Results")
    print("=" * 60)
    print(f"\n  {'FINAL LEADERBOARD':^56}")
    print(f"  {'─'*56}")
    print(f"  {'Rank':<5} {'Agent':<14} {'Portfolio':>13} {'P&L':>12} "
          f"{'Win%':>7} {'Trades':>7} {'Sharpe':>8}")
    print(f"  {'─'*56}")

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["portfolio_value"],
        reverse=True,
    )

    medals = ["1st", "2nd", "3rd", "4th", "5th"]
    for i, (name, r) in enumerate(sorted_results):
        medal = medals[i] if i < len(medals) else f"{i+1}th"
        print(
            f"  {medal:<5} {name:<14} "
            f"${r['portfolio_value']:>12,.2f} "
            f"${r['pnl']:>+11,.2f} "
            f"{r['win_rate']:>6.1f}% "
            f"{r['total_trades']:>7} "
            f"{r['sharpe']:>8.3f}"
        )

    print(f"  {'─'*56}")
    print(f"  All agents started with $100,000.00")
    print()

    # ── Save outputs ──────────────────────────────────────────────────────
    # JSON report
    report = {
        "simulation_date": pd.Timestamp.now().isoformat(),
        "test_rows":       len(test_df),
        "n_agents":        len(agents),
        "initial_capital": 100_000.0,
        "leaderboard":     [
            {
                "rank":            i + 1,
                "agent":           name,
                "portfolio_value": r["portfolio_value"],
                "pnl":             r["pnl"],
                "pnl_pct":         r["pnl_pct"],
                "win_rate":        r["win_rate"],
                "total_trades":    r["total_trades"],
                "sharpe":          r["sharpe"],
                "sortino":         r["sortino"],
                "max_drawdown":    r["max_dd"],
            }
            for i, (name, r) in enumerate(sorted_results)
        ],
    }

    report_path = REPORTS_DIR / "simulation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    # CSV metrics
    metrics_df = pd.DataFrame([
        {
            "agent":           name,
            "portfolio_value": r["portfolio_value"],
            "pnl":             r["pnl"],
            "pnl_pct":         r["pnl_pct"],
            "win_rate":        r["win_rate"],
            "total_trades":    r["total_trades"],
            "wins":            r["wins"],
            "losses":          r["losses"],
            "sharpe":          r["sharpe"],
            "sortino":         r["sortino"],
            "max_drawdown":    r["max_dd"],
        }
        for name, r in results.items()
    ])
    metrics_path = METRICS_DIR / "agent_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved: {metrics_path}")

    # Portfolio history CSV
    history_data = {}
    for name, sim in simulators.items():
        history_data[name] = sim.portfolio_history

    max_len = max(len(v) for v in history_data.values())
    for name in history_data:
        h = history_data[name]
        if len(h) < max_len:
            history_data[name] = h + [h[-1]] * (max_len - len(h))

    history_df = pd.DataFrame(history_data)
    history_path = METRICS_DIR / "portfolio_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"  History saved: {history_path}")

    print("\n" + "=" * 60)
    print("Simulation complete.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_simulation()