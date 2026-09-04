"""Tests for AI model training utilities."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.feature_builder import get_observation_features

FEATURE_COLS = get_observation_features()


class TestLSTMModel:
    def test_model_instantiation(self):
        from src.models.train_lstm import LSTMTrader
        model = LSTMTrader(input_size=48, hidden_size=64, num_layers=2, n_classes=3)
        assert model is not None

    def test_forward_pass_shape(self):
        from src.models.train_lstm import LSTMTrader
        model = LSTMTrader(input_size=48, hidden_size=64, num_layers=2, n_classes=3)
        model.eval()
        x   = torch.randn(4, 60, 48)   # batch=4, seq=60, features=48
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_is_logits(self):
        from src.models.train_lstm import LSTMTrader
        model = LSTMTrader(input_size=48, hidden_size=64, num_layers=2, n_classes=3)
        model.eval()
        x   = torch.randn(2, 60, 48)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_parameter_count_reasonable(self):
        from src.models.train_lstm import LSTMTrader
        model  = LSTMTrader(input_size=48, hidden_size=256, num_layers=2, n_classes=3)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 10_000
        assert n_params < 10_000_000


class TestTransformerModel:
    def test_model_instantiation(self):
        from src.models.train_transformer import TransformerTrader
        model = TransformerTrader(
            input_size=48, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=128, n_classes=3, seq_len=60,
        )
        assert model is not None

    def test_forward_pass_shape(self):
        from src.models.train_transformer import TransformerTrader
        model = TransformerTrader(
            input_size=48, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=128, n_classes=3, seq_len=60,
        )
        model.eval()
        x   = torch.randn(4, 60, 48)
        out = model(x)
        assert out.shape == (4, 3)

    def test_positional_encoding(self):
        from src.models.train_transformer import PositionalEncoding
        pe  = PositionalEncoding(d_model=64, max_len=100)
        x   = torch.randn(2, 60, 64)
        out = pe(x)
        assert out.shape == (2, 60, 64)

    def test_no_nan_output(self):
        from src.models.train_transformer import TransformerTrader
        model = TransformerTrader(
            input_size=48, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=128, n_classes=3, seq_len=60,
        )
        model.eval()
        x   = torch.randn(2, 60, 48)
        out = model(x)
        assert not torch.isnan(out).any()


class TestLabelGeneration:
    def test_buy_label_when_price_rises(self):
        from src.models.train_lstm import generate_labels
        prices = [100.0] * 20 + [101.5] * 10
        df = pd.DataFrame({
            "mid_price": prices,
            "asset": "BTC",
        })
        labels = generate_labels(df)
        assert 1 in labels.values  # BUY label exists

    def test_sell_label_when_price_falls(self):
        from src.models.train_lstm import generate_labels
        prices = [100.0] * 20 + [98.0] * 10
        df = pd.DataFrame({
            "mid_price": prices,
            "asset": "BTC",
        })
        labels = generate_labels(df)
        assert 2 in labels.values  # SELL label exists

    def test_hold_label_for_flat_prices(self):
        from src.models.train_lstm import generate_labels
        prices = [100.0] * 50
        df = pd.DataFrame({
            "mid_price": prices,
            "asset": "BTC",
        })
        labels = generate_labels(df)
        assert 0 in labels.values  # HOLD label exists

    def test_label_values_valid(self):
        from src.models.train_lstm import generate_labels
        np.random.seed(42)
        prices = 100 + np.random.normal(0, 1, 100).cumsum()
        df = pd.DataFrame({"mid_price": prices, "asset": "BTC"})
        labels = generate_labels(df)
        assert set(labels.unique()).issubset({0, 1, 2})


class TestSyntheticDataPipeline:
    def test_synthetic_regime_bull(self):
        from src.data.data_pipeline import generate_synthetic_regime
        df = generate_synthetic_regime(200, "bull", base_price=100.0, seed=42)
        assert len(df) == 200
        assert df["mid_price"].iloc[-1] > df["mid_price"].iloc[0]

    def test_synthetic_regime_bear(self):
        from src.data.data_pipeline import generate_synthetic_regime
        df = generate_synthetic_regime(200, "bear", base_price=100.0, seed=42)
        assert len(df) == 200
        assert df["mid_price"].iloc[-1] < df["mid_price"].iloc[0]

    def test_synthetic_regime_flat(self):
        from src.data.data_pipeline import generate_synthetic_regime
        df = generate_synthetic_regime(200, "flat", base_price=100.0, seed=42)
        assert len(df) == 200
        price_range = df["mid_price"].max() - df["mid_price"].min()
        assert price_range < 50

    def test_synthetic_regime_volatile(self):
        from src.data.data_pipeline import generate_synthetic_regime
        df = generate_synthetic_regime(200, "volatile", base_price=100.0, seed=42)
        assert len(df) == 200

    def test_synthetic_has_required_columns(self):
        from src.data.data_pipeline import generate_synthetic_regime
        df = generate_synthetic_regime(100, "bull", base_price=1000.0, seed=1)
        for col in ["mid_price", "spread", "returns", "volatility", "imbalance"]:
            assert col in df.columns

    def test_synthetic_no_negative_prices(self):
        from src.data.data_pipeline import generate_synthetic_regime
        for regime in ["bull", "bear", "flat", "volatile"]:
            df = generate_synthetic_regime(200, regime, base_price=1.0, seed=42)
            assert (df["mid_price"] > 0).all(), f"Negative prices in {regime}"

    def test_unknown_regime_raises(self):
        from src.data.data_pipeline import generate_synthetic_regime
        with pytest.raises(ValueError):
            generate_synthetic_regime(100, "unknown_regime")