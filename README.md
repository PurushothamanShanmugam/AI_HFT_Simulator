<div align="center">

# HFT AI Simulator

### Multi-Asset AI Trading System — 5 AI Models · 8 Assets · Real Prices · $100K Budget

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-informational)](https://xgboost.readthedocs.io)
[![Stable Baselines3](https://img.shields.io/badge/SB3-DQN%20%7C%20PPO-informational)](https://stable-baselines3.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com)
[![License](https://img.shields.io/badge/License-IIT%20Jodhpur-orange)](LICENSE)

*A multi-asset AI trading system where 5 independently trained models compete simultaneously in a simulated exchange — using real prices, a genuine $100K budget, and 5.2 million rows of real market data.*

[Features](#features) · [Architecture](#architecture) · [Quickstart](#quickstart) · [Docker](#docker) · [Results](#results) · [Dashboard](#dashboard) · [Testing](#testing)

</div>

---

## Overview

HFT AI Simulator is a production-grade multi-asset AI trading research platform built for MTech-level research and portfolio demonstration. It combines:

- **5 competing AI models** — DQN, PPO, LSTM, XGBoost, Transformer — each independently trained and competing in the same simulated exchange
- **Real open prices** — BTC at $56,035, AAPL at $585, MSFT at $30 — no normalisation to an artificial base value
- **$100,000 starting budget** per agent with genuine 10% position sizing across 8 assets simultaneously
- **5.2 million rows** of real market data — crypto 1-second LOB and LOBSTER equity orderbook depth-10
- **70/20/10 train/val/test split** with synthetic augmentation for bear, crash, and flat market regimes
- **Exchange-grade order book** with price-time-latency priority matching
- **57-feature observation space** covering price, order flow, momentum, microstructure, regime, and portfolio state
- **Streamlit dashboard** with leaderboard, equity curves, risk metrics, capital allocation, and Kafka live feed

---

## Features

- **5 AI model types** — DQN (RL), PPO (RL), LSTM (supervised), XGBoost (supervised), Transformer (supervised)
- **Real prices throughout** — no normalisation; BTC trades at $56,035, MSFT at $30.97
- **Multi-asset portfolio management** — agents allocate $100K across 8 assets with 10% position sizing
- **17-action discrete space** — HOLD + BUY/SELL × 8 assets per tick
- **48 market features** — spread, OFI, momentum, volatility, Z-score, VWAP, Kyle lambda, Bollinger, liquidity score and more
- **Exchange-grade matching engine** — price-time-latency priority, partial fills, stale order cancellation
- **Dual reward functions** — trade-level reward for DQN; Sharpe + portfolio reward for PPO
- **Synthetic augmentation** — bear market, high-volatility crash, and sideways regime data added to training
- **Kafka live feed** — stream your own CSV or the 100K synthetic test data tick by tick
- **Dynamic simulation** — 4 market regimes (bull, flat, bear, volatile) in the synthetic test stream
- **Annualised risk metrics** — Sharpe, Sortino, max drawdown, win rate, profit factor per agent
- **Capital allocation chart** — shows how $10K per trade buys very different unit counts at real prices
- **Market regime indicator** — detects bull / bear / volatile / sideways per agent in real time
- **Docker support** — Dockerfile, docker-compose.yml, entrypoint.sh for one-command deployment
- **105 tests** — pytest suite covering data pipeline, features, environment, models, and simulation

---

## Results

> Results shown are from an actual simulation run on the 10% test split
> (data the models never saw during training) — 1,000 ticks, 6 assets, 482 total trades.
> Initial capital: $100,000 per agent.

### Agent Leaderboard (Actual Run)

| Rank | Agent | Type | Final Portfolio | P&L | Trades | Win Rate | Sharpe (Ann.) | Max DD |
|------|-------|------|----------------|-----|--------|----------|---------------|--------|
| 1 | LSTM | Supervised | $100,026 | +$26 | 42 | 4.8% | 0.096 | 0.06% |
| 2 | XGBoost | Supervised | $99,997 | -$3 | 40 | 2.5% | -1.805 | 0% |
| 3 | Transformer | Supervised | $99,997 | -$3 | 40 | 2.5% | -1.805 | 0% |
| 4 | DQN | Reinforcement Learning | $99,923 | -$77 | 192 | 2.1% | -0.103 | 0.09% |
| 5 | PPO | Reinforcement Learning | $99,914 | -$86 | 168 | 0.6% | -0.736 | 0.09% |

> Across all 482 trades: 9 closed profitable, 232 closed at a loss, 241 (50%) were
> still open at end of run. Only LSTM finished above the $100,000 benchmark, and by a
> margin close to statistical noise. Win rates (0.6%–4.8%) are far below what any
> agent would need for a working directional edge, and 4 of 5 agents show negative
> Sharpe. This gap versus the RL agents' training-time objectives is discussed under
> [Limitations](#limitations) below — in particular, LSTM/Transformer's restricted
> BTC-only routing during simulation, single-day/12-day source data, and the lack of
> modelled execution costs beyond spread and slippage.

### What $10,000 Buys at Real Prices (10% position sizing)

| Asset | Real Price | Units per $10K trade |
|-------|-----------|---------------------|
| BTC | $56,035 | 0.18 units |
| ETH | $1,970 | 5.07 units |
| ADA | $1.17 | 8,547 units |
| AAPL | $585 | 17 units |
| AMZN | $223 | 44 units |
| GOOG | $579 | 17 units |
| INTC | $27 | 370 units |
| MSFT | $30 | 333 units |

This table shows why normalisation is wrong — a trader buying BTC spends 56% of their entire budget on one unit, while a trader buying ADA buys 8,000 units with the same $10K. The simulator handles all of this correctly.

---

## Architecture

```
hft_ai_simulator/
├── src/
│   ├── data/
│   │   ├── lob_loader.py              # loads 8 assets at real prices
│   │   ├── feature_builder.py         # 48 market features per tick
│   │   ├── data_pipeline.py           # 70/20/10 split + synthetic augmentation
│   │   └── synthetic_live_generator.py# 100K rows · 4 regimes · Kafka test data
│   │
│   ├── envs/
│   │   └── multi_asset_env.py         # $100K budget · 17 actions · real prices
│   │
│   ├── models/
│   │   ├── train_dqn.py               # DQN · 300K steps · trade-level reward
│   │   ├── train_ppo.py               # PPO · 500K steps · Sharpe reward
│   │   ├── train_lstm.py              # LSTM · 60-tick sequences · supervised
│   │   ├── train_xgboost.py           # XGBoost · current tick · supervised
│   │   └── train_transformer.py       # Transformer · 4 heads · cross-asset attention
│   │
│   ├── simulation/
│   │   └── multi_agent_sim.py         # runs all 5 agents on unseen test data
│   │
│   ├── execution/
│   │   ├── matching_engine.py         # exchange-grade order matching
│   │   ├── order_book_engine.py       # price-time-latency priority engine
│   │   ├── order_book.py              # in-memory bid/ask order book
│   │   ├── order.py                   # order dataclass
│   │   └── latency_model.py           # simulated network latency
│   │
│   ├── analytics/
│   │   └── risk_metrics.py            # Sharpe · Sortino · drawdown · win rate
│   │
│   └── dashboard/
│       └── dashboard.py               # Streamlit interactive dashboard
│
├── data/
│   ├── raw/crypto/                    # BTC_1sec · ETH_1sec · ADA_1sec
│   ├── raw/lobster/                   # AAPL · AMZN · GOOG · INTC · MSFT (orderbook_10)
│   ├── splits/                        # train_70 · val_20 · test_10 (parquet)
│   └── synthetic/                     # kafka_test_data.csv (100K rows)
│
├── outputs/
│   ├── models/                        # trained weights (zip · pt · json · pkl)
│   ├── reports/                       # simulation_report.json
│   └── metrics/                       # agent_metrics.csv · portfolio_history.csv
│
├── tests/                             # 105 tests across 5 test files
├── kafka_producer.py                  # streams CSV or synthetic data to Kafka
├── kafka_consumer.py                  # reads ticks from Kafka topic
├── main.py                            # runs full pipeline end to end
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## Quickstart

### Option A — Docker (easiest, no setup required)

```bash
git clone https://github.com/yourusername/hft_ai_simulator.git
cd hft_ai_simulator

# Build and launch the dashboard
docker compose up --build

# Open browser at http://localhost:8501
```

### Option B — Conda

```bash
git clone https://github.com/yourusername/hft_ai_simulator.git
cd hft_ai_simulator
conda env create -f environment.yml
conda activate hft_ai_env
python main.py
```

### Option C — pip

```bash
git clone https://github.com/yourusername/hft_ai_simulator.git
cd hft_ai_simulator
pip install -r requirements.txt
```

### Adding Your Own Data

```
data/raw/crypto/
  BTC_1sec.csv     # 1-second resolution · columns: system_time, midpoint, spread, buys, sells
  ETH_1sec.csv
  ADA_1sec.csv

data/raw/lobster/
  AAPL/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
  AMZN/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv
  GOOG/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv
  INTC/INTC_2012-06-21_34200000_57600000_orderbook_10.csv
  MSFT/MSFT_2012-06-21_34200000_57600000_orderbook_10.csv
```

The data loader auto-detects format from filename. No config changes needed.

### Run Step by Step

```bash
# 1. Data pipeline — load, split 70/20/10, augment
python -m src.data.data_pipeline

# 2. Train each model
python -m src.models.train_dqn           # ~15 min on CPU
python -m src.models.train_ppo           # ~25 min on CPU
python -m src.models.train_lstm          # ~30 min on CPU
python -m src.models.train_xgboost       # ~10 min on CPU
python -m src.models.train_transformer   # ~30 min on CPU

# 3. Run simulation on unseen test data
python -m src.simulation.multi_agent_sim

# 4. Launch dashboard
streamlit run src/dashboard/dashboard.py

# Or run everything at once
python main.py
```

---

## Docker

```bash
# Dashboard only
docker compose up --build

# With Kafka live feed
docker compose --profile kafka up --build

# Train individual models
docker compose run --rm train-dqn
docker compose run --rm train-ppo
docker compose run --rm train-lstm
docker compose run --rm train-xgboost
docker compose run --rm train-transformer

# Run simulation
docker compose run --rm simulate

# Run all tests
docker compose run --rm test
```

PyTorch is installed as **CPU-only** to keep the image ~2 GB. `data/` and `outputs/` are mounted as volumes so your data and trained models persist between runs.

---

## Dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

| Feature | Description |
|---------|-------------|
| Agent leaderboard | Ranked by final portfolio value with P&L, win rate, Sharpe |
| Equity curves | Portfolio value over time for all 5 agents on one chart |
| Risk metrics panel | Sharpe · Sortino · max drawdown · win rate · trades per agent |
| Capital allocation | Units purchasable per $10K at real prices across all 8 assets |
| Regime indicator | Detects bull / bear / volatile / sideways per agent |
| Real open prices | Sidebar shows actual open price for all 8 assets |
| Live crypto prices | Real-time BTC, ETH, ADA prices via CoinGecko API |
| Kafka feed toggle | Enable live data streaming from external source |
| Run simulation button | Trigger new simulation run from the dashboard |
| Run pipeline button | Trigger data pipeline from the dashboard |
| Download report | simulation_report.json and agent_metrics.csv |

---

## AI Model Details

### DQN — Deep Q-Network

```python
DQN(
    policy            = "MlpPolicy",
    learning_rate     = 3e-5,
    buffer_size       = 100_000,
    learning_starts   = 5_000,
    batch_size        = 64,
    gamma             = 0.99,
    exploration_fraction = 0.30,
    total_timesteps   = 300_000,
    policy_kwargs     = {"net_arch": [256, 256, 128]},
)
```

**Reward:** realised P&L × 15 (profit) or × 20 (loss) − transaction costs − drawdown − inventory risk

### PPO — Proximal Policy Optimisation

```python
PPO(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 2_048,
    batch_size    = 64,
    n_epochs      = 10,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    total_timesteps = 500_000,
    policy_kwargs = {"net_arch": dict(pi=[256,256,128], vf=[256,256,128])},
)
```

**Reward:** Sharpe ratio + realised P&L − drawdown − costs + diversification bonus

### LSTM

- 2 layers × 256 hidden units
- 60-tick input sequences
- Supervised labels: price direction over next 10 ticks
- Class-weighted cross-entropy loss
- Early stopping on validation accuracy

### XGBoost

- 500 trees · max depth 6 · learning rate 0.05
- Single-tick features (no sequence)
- 3-class output: BUY / SELL / HOLD
- Confidence threshold 0.5 — only trades when sure
- Feature importance printed after training

### Transformer

- 4 encoder layers · 4 attention heads · d_model = 128
- 60-tick input sequences
- Positional encoding preserves tick order
- AdamW optimiser + CosineAnnealingLR
- Early stopping on validation accuracy

---

## Observation Space (57 features)

| Group | Features | Count |
|-------|---------|-------|
| Price | mid, spread, spread_pct, returns, volatility, log_return | 6 |
| Order flow | imbalance, OFI, depth_imbalance, pressure, buy_ratio, sell_ratio | 6 |
| Momentum | return_5/10/20/60, acceleration, jerk | 6 |
| Volatility | vol_5/10/20/60 | 4 |
| Mean reversion | zscore_20/60, VWAP deviation, Bollinger position | 4 |
| Microstructure | effective_spread, price_impact, bid_ask_bounce, trade_sign, Kyle lambda | 5 |
| Level | bid1, ask1, bid_size1, ask_size1, size_ratio, mid_return | 6 |
| Regime | trend_strength, vol_regime, spread_regime, liquidity_score | 4 |
| Meta | global_tick_norm, asset_encoded, source_encoded, prev_return, prev_spread, prev_imbalance, time_of_day | 7 |
| Portfolio | cash_ratio + 8 inventory ratios | 9 |

---

## Order Book Engine

The `FinalExecutionEngine` implements:

- **Price-time-latency priority** — best price, earliest tick, lowest latency
- **Persistent resting orders** — survive across ticks until filled or cancelled
- **Price-tolerance crossing** — matches within 10 bps
- **Liquidity provider quotes** — synthetic quotes improve matching efficiency
- **Partial fills** — handles orders larger than opposing resting quantity
- **Stale order cancellation** — removes orders older than 20 ticks
- **Self-match prevention** — a trader cannot buy from itself

---

## Kafka Live Feed

Stream real-time or synthetic data into the simulator.

```bash
# Start with Kafka enabled
docker compose --profile kafka up --build

# Stream your own CSV
CSV_FILE=data/raw/crypto/BTC_1sec.csv \
TICK_INTERVAL=0.01 \
docker compose --profile kafka run --rm producer

# Auto-generate 100K synthetic ticks and stream
python kafka_producer.py
```

**Synthetic test stream — 4 market regimes (100K rows):**

| Regime | Rows | Description |
|--------|------|-------------|
| Bull | 1 – 25,000 | Steady uptrend across all 8 assets |
| Flat | 25,001 – 50,000 | Sideways ranging market |
| Bear | 50,001 – 75,000 | Steady downtrend |
| Volatile | 75,001 – 100,000 | High volatility crash and recovery |

---

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# Inside Docker
docker compose run --rm test
```

**105 tests across 5 test files:**

| File | What it covers | Tests |
|------|---------------|-------|
| test_data_pipeline.py | lob_loader · feature_builder · data structure | 21 |
| test_feature_builder.py | all 9 feature groups · range checks · correctness | 18 |
| test_multi_asset_env.py | env init · reset · step · reward · portfolio | 32 |
| test_models.py | LSTM · Transformer · labels · synthetic regimes | 17 |
| test_simulation.py | agent wrappers · episode runs · Kafka generator | 17 |

---

## Requirements

```
numpy · pandas · scikit-learn · scipy
torch · xgboost · stable-baselines3 · gymnasium
streamlit · plotly · kafka-python
pyarrow · joblib · tqdm · matplotlib
```

Python 3.10+ required. CUDA optional — all models train on CPU.

---

## Limitations

- Crypto data covers only 12 days (April 7–19, 2021) — a bull market period. Despite synthetic augmentation, models may underperform in sustained bear markets not represented in the real data.
- LOBSTER equity data covers only one trading day (June 21, 2012). This limits regime diversity for equity assets.
- LSTM and Transformer agents use simplified action routing during simulation (BUY/SELL BTC only). Full 17-action routing requires per-asset signal decomposition — identified as future work.
- Training times are significant on CPU. GPU strongly recommended for LSTM and Transformer (30+ min on CPU vs ~5 min on GPU).
- No real execution costs modelled beyond spread and slippage approximations.
- **Observed in the actual run above:** win rates (0.6%–4.8%) and near-zero/negative
  Sharpe across 4 of 5 agents suggest the models are not yet extracting a reliable
  directional edge on the test window, and the high share of `SL auto-exit` closes
  (stop-loss triggers rather than model-driven exits) points at position/risk sizing
  as a likely next area to tune, alongside the data-coverage and action-routing
  limitations above.
- 241 of 482 trades (50%) remained open at the end of the run — end-of-episode
  forced-close handling and open-position accounting are candidates for review
  before reporting these numbers as final.

---

## Future Work

- Join `message_10` LOBSTER files with `orderbook_10` to unlock 10 new microstructure features: VPIN, cancellation rate, order arrival intensity, aggressive vs passive ratio, hidden trade ratio
- Extend LSTM and Transformer to full 17-action multi-asset routing
- Add multi-year crypto data to cover bear market and high-volatility regimes in real data
- Implement multi-agent reinforcement learning (MARL) with competitive reward shaping across all 5 agents
- Publish benchmark results comparing all 5 model architectures on arXiv

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Cont, R. (2001). Empirical characteristics of asset returns. *Quantitative Finance*, 1(2), 223–236.
- Easley, D. et al. (2012). Flow Toxicity and Liquidity in a High-frequency World. *Review of Financial Studies*, 25(5).
- Bouchaud, J-P. et al. (2018). *Trades, Quotes and Prices*. Cambridge University Press.

---

## License

Indian Institute of Technology Jodhpur — Final Year MTech Project

---

## Author

**Purushothaman S** (M25DE1033)
M.Tech — Data Engineering
Indian Institute of Technology Jodhpur
NH 62, Surpura Bypass Rd, Karwar, Jheepasani, Rajasthan 342030, India
📧 m25de1033@iitj.ac.in

---

<div align="center">
<strong>HFT AI Simulator</strong> · Built as part of MTech Data Engineering · IIT Jodhpur · 2024–2026
</div>