"""
Multi-Asset Trading Environment
================================
Step 2 of the hft_ai_simulator build.

Option B — realistic multi-asset trading:
  - 8 assets trading simultaneously at REAL prices
  - $100,000 starting budget per agent
  - 17 actions: HOLD + BUY/SELL × 8 assets
  - Fixed 10% position sizing per trade
  - Portfolio tracking across all 8 assets
  - Sharpe-based reward with drawdown penalty

Action space (17 discrete actions):
  0  = HOLD
  1  = BUY  BTC      2  = SELL BTC
  3  = BUY  ETH      4  = SELL ETH
  5  = BUY  ADA      6  = SELL ADA
  7  = BUY  AAPL     8  = SELL AAPL
  9  = BUY  AMZN     10 = SELL AMZN
  11 = BUY  GOOG     12 = SELL GOOG
  13 = BUY  INTC     14 = SELL INTC
  15 = BUY  MSFT     16 = SELL MSFT

Observation space (57 features):
  48 market features (from feature_builder)
   + 9 portfolio features (cash_ratio + 8 inventory ratios)
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from src.data.feature_builder import get_observation_features

ASSETS = ["BTC", "ETH", "ADA", "AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
N_ASSETS = len(ASSETS)

# Action mapping
HOLD = 0
BUY_ACTIONS  = {i: ASSETS[(i - 1) // 2] for i in range(1, 17, 2)}
SELL_ACTIONS = {i: ASSETS[(i - 2) // 2] for i in range(2, 17, 2)}

ACTION_TO_ASSET = {}
ACTION_TO_SIDE  = {}
for a in range(1, 17, 2):
    asset = ASSETS[(a - 1) // 2]
    ACTION_TO_ASSET[a]   = asset
    ACTION_TO_SIDE[a]    = "BUY"
    ACTION_TO_ASSET[a+1] = asset
    ACTION_TO_SIDE[a+1]  = "SELL"


class MultiAssetTradingEnv(gym.Env):
    """
    Multi-asset RL trading environment.

    Each step:
      1. Agent observes 57-dimensional state
      2. Agent selects one of 17 actions
      3. Trade executes at real market price
      4. Portfolio updates
      5. Reward calculated
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        position_size_pct: float = 0.10,
        max_inventory: int = 10,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        reward_scaling: float = 1.0,
    ):
        """
        Args:
            data:                 DataFrame with feature columns + asset + mid_price
            initial_cash:         Starting cash per agent ($100,000)
            position_size_pct:    Fraction of cash used per trade (10%)
            max_inventory:        Max units held per asset
            transaction_cost_pct: Transaction cost as % of trade value
            slippage_pct:         Slippage as % of trade value
            reward_scaling:       Scale factor for reward normalisation
        """
        super().__init__()

        self.data               = data.reset_index(drop=True)
        self.initial_cash       = initial_cash
        self.position_size_pct  = position_size_pct
        self.max_inventory      = max_inventory
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct       = slippage_pct
        self.reward_scaling     = reward_scaling

        self.feature_cols = get_observation_features()
        self.n_features   = len(self.feature_cols) + N_ASSETS + 1  # 48 + 8 + 1 = 57

        # Spaces
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # Episode state
        self.current_step        = 0
        self.cash                = initial_cash
        self.inventory           = {a: 0.0 for a in ASSETS}
        self.avg_entry_price     = {a: 0.0 for a in ASSETS}
        self.portfolio_history   = []
        self.trade_history       = []
        self.wins                = 0
        self.losses              = 0
        self.total_trades        = 0
        self.prev_portfolio_value = initial_cash
        self.peak_portfolio_value = initial_cash

        # Pre-build price lookup
        self._price_lookup = {}
        self._price_length = {}
        self._build_price_lookup()

    # ── helpers ───────────────────────────────────────────────────────────

    def _safe(self, v, default=0.0):
        try:
            f = float(v)
            return default if (np.isnan(f) or np.isinf(f)) else f
        except Exception:
            return default

    def _build_price_lookup(self):
        """Pre-build per-asset price arrays for O(1) lookup during steps."""
        self._price_lookup = {}
        self._price_length = {}
        if "asset" in self.data.columns:
            for asset in ASSETS:
                rows = self.data[self.data["asset"] == asset]["mid_price"].values
                self._price_lookup[asset] = rows if len(rows) > 0 else np.array([1.0])
                self._price_length[asset] = len(rows)
        else:
            prices = self.data["mid_price"].values
            for asset in ASSETS:
                self._price_lookup[asset] = prices
                self._price_length[asset] = len(prices)

    def _get_price(self, asset: str) -> float:
        """O(1) price lookup using pre-built arrays."""
        arr = self._price_lookup.get(asset, np.array([1.0]))
        idx = min(self.current_step, len(arr) - 1)
        return self._safe(arr[idx], 1.0)

    def _get_prices(self) -> dict:
        """Get current prices for all 8 assets — O(1) per asset."""
        return {asset: self._get_price(asset) for asset in ASSETS}

    def _portfolio_value(self, prices: dict) -> float:
        """Total portfolio value = cash + sum(inventory × price)."""
        holdings = sum(
            self.inventory[a] * prices[a] for a in ASSETS
        )
        return self.cash + holdings

    def _get_observation(self) -> np.ndarray:
        """Build 57-dimensional observation vector."""
        row = self.data.iloc[self.current_step]

        # 48 market features
        market_obs = np.array(
            [self._safe(row.get(f, 0.0)) for f in self.feature_cols],
            dtype=np.float32,
        )

        # 9 portfolio features
        prices = self._get_prices()
        pv     = max(self._portfolio_value(prices), 1.0)

        cash_ratio = np.clip(self.cash / pv, 0, 1)
        inv_ratios = np.array(
            [np.clip(
                self.inventory[a] * prices[a] / pv, 0, 1
            ) for a in ASSETS],
            dtype=np.float32,
        )

        portfolio_obs = np.concatenate([[cash_ratio], inv_ratios])

        obs = np.concatenate([market_obs, portfolio_obs]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step         = 0
        self.cash                 = self.initial_cash
        self.inventory            = {a: 0.0 for a in ASSETS}
        self.avg_entry_price      = {a: 0.0 for a in ASSETS}
        self.portfolio_history    = [self.initial_cash]
        self.trade_history        = []
        self.wins                 = 0
        self.losses               = 0
        self.total_trades         = 0
        self.prev_portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash

        # Build fast price lookup for this episode
        self._build_price_lookup()

        return self._get_observation(), {}

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: int):
        action = int(action)
        prices = self._get_prices()

        trade_executed = False
        invalid_action = False
        realised_pnl   = 0.0
        trade_asset    = None
        trade_side     = None

        # ── Execute trade ─────────────────────────────────────────────────
        if action == HOLD:
            pass

        elif action in ACTION_TO_ASSET:
            trade_asset = ACTION_TO_ASSET[action]
            trade_side  = ACTION_TO_SIDE[action]
            price       = prices[trade_asset]

            cost_pct    = self.transaction_cost_pct + self.slippage_pct
            trade_value = self.cash * self.position_size_pct

            if trade_side == "BUY":
                total_cost = trade_value * (1 + cost_pct)

                if (self.cash >= total_cost and
                        self.inventory[trade_asset] < self.max_inventory and
                        price > 0):

                    units = trade_value / price
                    self.cash -= total_cost

                    # Update average entry price (FIFO weighted)
                    prev_units = self.inventory[trade_asset]
                    prev_avg   = self.avg_entry_price[trade_asset]
                    total_units = prev_units + units
                    if total_units > 0:
                        self.avg_entry_price[trade_asset] = (
                            (prev_avg * prev_units + price * units)
                            / total_units
                        )

                    self.inventory[trade_asset] += units
                    self.inventory[trade_asset] = float(
                        np.clip(self.inventory[trade_asset], 0, 1e9))
                    trade_executed = True
                    self.total_trades += 1
                else:
                    invalid_action = True

            elif trade_side == "SELL":
                units = self.inventory[trade_asset]

                if units > 0 and price > 0:
                    # Sell ALL units held for this asset
                    units_sold   = units
                    proceeds     = units_sold * price
                    total_gain   = proceeds * (1 - cost_pct)
                    entry_price  = self.avg_entry_price[trade_asset]
                    realised_pnl = (price - entry_price) * units_sold

                    # Clip to prevent overflow
                    total_gain   = float(np.clip(total_gain, 0, self.initial_cash * 10))
                    realised_pnl = float(np.clip(realised_pnl,
                                                  -self.initial_cash,
                                                   self.initial_cash))

                    self.cash += total_gain
                    self.cash  = float(np.clip(self.cash, 0, self.initial_cash * 100))
                    self.inventory[trade_asset] = 0.0
                    self.avg_entry_price[trade_asset] = 0.0

                    trade_executed = True
                    self.total_trades += 1

                    if realised_pnl > 0:
                        self.wins += 1
                    else:
                        self.losses += 1
                else:
                    invalid_action = True
        else:
            invalid_action = True

        # ── Advance step ──────────────────────────────────────────────────
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # ── Portfolio update ──────────────────────────────────────────────
        new_prices = self._get_prices()
        pv         = self._portfolio_value(new_prices)
        pv         = max(pv, 0.01)

        self.portfolio_history.append(pv)
        self.peak_portfolio_value = max(self.peak_portfolio_value, pv)

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._calculate_reward(
            pv, realised_pnl, invalid_action, trade_executed,
            trade_side, trade_asset, new_prices,
        )

        # ── Info ──────────────────────────────────────────────────────────
        win_rate = (self.wins / max(self.total_trades, 1)) * 100
        drawdown = (self.peak_portfolio_value - pv) / self.peak_portfolio_value

        info = {
            "portfolio_value":  pv,
            "cash":             self.cash,
            "inventory":        dict(self.inventory),
            "realised_pnl":     realised_pnl,
            "total_trades":     self.total_trades,
            "wins":             self.wins,
            "losses":           self.losses,
            "win_rate_pct":     round(win_rate, 2),
            "drawdown":         round(drawdown, 4),
            "trade_executed":   trade_executed,
            "trade_asset":      trade_asset,
            "trade_side":       trade_side,
            "invalid_action":   invalid_action,
            "step":             self.current_step,
        }

        self.prev_portfolio_value = pv

        return self._get_observation(), float(reward), done, False, info

    # ── Reward function ───────────────────────────────────────────────────

    def _calculate_reward(
        self, pv, realised_pnl, invalid_action,
        trade_executed, trade_side, trade_asset, prices,
    ) -> float:
        """
        Multi-component reward function:
          + Sharpe component     — rewards consistent risk-adjusted returns
          + Realised P&L         — rewards profitable closed trades
          - Drawdown penalty     — punishes losing streaks
          - Transaction costs    — discourages overtrading
          + Diversification      — rewards spreading across assets
          - Inventory risk       — penalises concentrated positions
          - Invalid action       — penalises illegal moves
          + Competitive          — rewards outperforming initial capital
        """
        reward = 0.0

        # ── 1. Sharpe component (primary signal) ─────────────────────────
        if len(self.portfolio_history) >= 10:
            returns = pd.Series(self.portfolio_history[-20:]).pct_change().dropna()
            if len(returns) >= 2 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                reward += np.clip(sharpe, -3, 3) * 0.01
            else:
                pnl_change = (pv - self.prev_portfolio_value) / max(self.prev_portfolio_value, 1)
                reward += pnl_change * 0.1

        # ── 2. Realised P&L on closed trade ──────────────────────────────
        if trade_executed and trade_side == "SELL" and trade_asset:
            price      = prices.get(trade_asset, 1.0)
            pnl_norm   = realised_pnl / max(abs(price), 1.0)
            if realised_pnl > 0:
                reward += pnl_norm * 15.0
            else:
                reward += pnl_norm * 20.0

        # ── 3. Drawdown penalty ───────────────────────────────────────────
        drawdown = (self.peak_portfolio_value - pv) / max(self.peak_portfolio_value, 1)
        reward  -= drawdown * 0.3

        # ── 4. Transaction cost penalty (heavy to prevent overtrading) ────
        if trade_executed:
            cost = self.transaction_cost_pct + self.slippage_pct
            reward -= cost * 0.5   # minimal cost penalty

        # ── 4b. Overtrading penalty (minimal) ────────────────────────────
        if self.total_trades > self.current_step / 2:
            reward -= 0.001

        # ── 5. Diversification bonus ──────────────────────────────────────
        total_inv_value = sum(
            self.inventory[a] * prices.get(a, 1.0) for a in ASSETS
        )
        if total_inv_value > 0:
            weights = [
                self.inventory[a] * prices.get(a, 1.0) / (total_inv_value + 1e-9)
                for a in ASSETS
            ]
            concentration  = max(weights)
            diversification = 1 - concentration
            reward += diversification * 0.005

        # ── 6. Inventory risk penalty ─────────────────────────────────────
        total_inv_pct = total_inv_value / max(pv, 1.0)
        reward -= total_inv_pct * 0.001

        # ── 7. Invalid action penalty ─────────────────────────────────────
        if invalid_action:
            reward -= 0.02

        # ── 8. Outperformance bonus ───────────────────────────────────────
        outperformance = (pv - self.initial_cash) / self.initial_cash
        reward += outperformance * 0.001

        return reward * self.reward_scaling

    # ── utility ───────────────────────────────────────────────────────────

    def get_portfolio_summary(self) -> dict:
        """Return current portfolio state as a dict."""
        prices = self._get_prices()
        pv     = self._portfolio_value(prices)

        return {
            "portfolio_value": pv,
            "cash":            self.cash,
            "pnl":             pv - self.initial_cash,
            "pnl_pct":         (pv - self.initial_cash) / self.initial_cash * 100,
            "win_rate":        self.wins / max(self.total_trades, 1) * 100,
            "total_trades":    self.total_trades,
            "wins":            self.wins,
            "losses":          self.losses,
            "drawdown":        (self.peak_portfolio_value - pv) /
                               max(self.peak_portfolio_value, 1) * 100,
            "inventory":       {
                a: {"units": self.inventory[a], "value": self.inventory[a] * prices[a]}
                for a in ASSETS
            },
        }

    def action_name(self, action: int) -> str:
        """Human-readable name for an action."""
        if action == HOLD:
            return "HOLD"
        asset = ACTION_TO_ASSET.get(action, "?")
        side  = ACTION_TO_SIDE.get(action, "?")
        return f"{side} {asset}"

    def render(self):
        summary = self.get_portfolio_summary()
        print(f"Step {self.current_step:>6} | "
              f"PV: ${summary['portfolio_value']:>12,.2f} | "
              f"Cash: ${summary['cash']:>10,.2f} | "
              f"P&L: ${summary['pnl']:>+10,.2f} | "
              f"Trades: {summary['total_trades']:>5} | "
              f"WR: {summary['win_rate']:>5.1f}%")