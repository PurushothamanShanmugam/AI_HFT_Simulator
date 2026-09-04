"""Tests for data pipeline — lob_loader and feature_builder."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_builder import build_features, get_feature_columns, get_observation_features


# ── lob_loader tests ──────────────────────────────────────────────────────

class TestCryptoDataframe:
    def test_has_required_columns(self, sample_crypto_df):
        required = [
            "market_id", "asset", "source", "resolution",
            "timestamp", "global_tick", "mid_price", "spread",
            "returns", "volatility", "imbalance",
            "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1",
        ]
        for col in required:
            assert col in sample_crypto_df.columns, f"Missing: {col}"

    def test_asset_name_correct(self, sample_crypto_df):
        assert (sample_crypto_df["asset"] == "BTC").all()

    def test_source_is_crypto(self, sample_crypto_df):
        assert (sample_crypto_df["source"] == "CRYPTO").all()

    def test_mid_price_positive(self, sample_crypto_df):
        assert (sample_crypto_df["mid_price"] > 0).all()

    def test_spread_non_negative(self, sample_crypto_df):
        assert (sample_crypto_df["spread"] >= 0).all()

    def test_ask_greater_than_bid(self, sample_crypto_df):
        assert (sample_crypto_df["ask_price_1"] >= sample_crypto_df["bid_price_1"]).all()

    def test_no_null_mid_price(self, sample_crypto_df):
        assert sample_crypto_df["mid_price"].isnull().sum() == 0

    def test_global_tick_monotonic(self, sample_crypto_df):
        assert (sample_crypto_df["global_tick"].diff().dropna() >= 0).all()

    def test_returns_first_row_zero(self, sample_crypto_df):
        assert sample_crypto_df["returns"].iloc[0] == 0.0


class TestLobsterDataframe:
    def test_has_required_columns(self, sample_lobster_df):
        assert "mid_price" in sample_lobster_df.columns
        assert "spread" in sample_lobster_df.columns

    def test_asset_name_correct(self, sample_lobster_df):
        assert (sample_lobster_df["asset"] == "MSFT").all()

    def test_source_is_lobster(self, sample_lobster_df):
        assert (sample_lobster_df["source"] == "LOBSTER").all()

    def test_prices_realistic(self, sample_lobster_df):
        assert sample_lobster_df["mid_price"].mean() > 1.0
        assert sample_lobster_df["mid_price"].mean() < 10_000.0

    def test_no_infinities(self, sample_lobster_df):
        numeric = sample_lobster_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


# ── feature_builder tests ─────────────────────────────────────────────────

class TestFeatureBuilder:
    def test_returns_dataframe(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert isinstance(result, pd.DataFrame)

    def test_feature_count(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        feat_cols = [c for c in result.columns if c.startswith("f_")]
        assert len(feat_cols) == 48

    def test_no_nan_in_features(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        feat_cols = [c for c in result.columns if c.startswith("f_")]
        assert result[feat_cols].isnull().sum().sum() == 0

    def test_no_inf_in_features(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        feat_cols = [c for c in result.columns if c.startswith("f_")]
        assert not np.isinf(result[feat_cols].values).any()

    def test_original_columns_preserved(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert "mid_price" in result.columns
        assert "asset" in result.columns

    def test_pressure_range(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_pressure"].between(-2, 2).all()

    def test_bollinger_pct_range(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_bollinger_pct"].between(-0.5, 1.5).all()

    def test_feature_columns_list_length(self):
        assert len(get_feature_columns()) == 48

    def test_observation_features_length(self):
        assert len(get_observation_features()) == 48

    def test_buy_sell_ratio_sum_to_one(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        total  = result["f_buy_ratio"] + result["f_sell_ratio"]
        assert total.between(0.99, 1.01).all()

    def test_does_not_modify_input(self, sample_crypto_df):
        original_len = len(sample_crypto_df)
        original_cols = list(sample_crypto_df.columns)
        _ = build_features(sample_crypto_df)
        assert len(sample_crypto_df) == original_len
        assert list(sample_crypto_df.columns) == original_cols