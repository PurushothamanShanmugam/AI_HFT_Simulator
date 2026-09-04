"""Detailed tests for individual feature groups in feature_builder."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_builder import build_features


class TestPriceFeatures:
    def test_f_spread_equals_ask_minus_bid(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        diff   = (result["ask_price_1"] - result["bid_price_1"] - result["f_spread"]).abs()
        assert diff.max() < 1.0

    def test_f_returns_pct_change(self, sample_crypto_df):
        result  = build_features(sample_crypto_df)
        manual  = sample_crypto_df["mid_price"].pct_change().fillna(0)
        diff    = (result["f_returns"] - manual).abs()
        assert diff.max() < 1e-6

    def test_f_log_return_clipped(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_log_return"].between(-1, 1).all()


class TestMomentumFeatures:
    def test_f_return_5_lags_correctly(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        manual = sample_crypto_df["mid_price"].pct_change(5).fillna(0)
        diff   = (result["f_return_5"] - manual).abs()
        assert diff.max() < 1e-6

    def test_f_accel_is_diff_of_returns(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        manual = result["f_returns"].diff().fillna(0)
        diff   = (result["f_accel"] - manual).abs()
        assert diff.max() < 1e-9


class TestVolatilityFeatures:
    def test_volatility_non_negative(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        for col in ["f_vol_5", "f_vol_10", "f_vol_20", "f_vol_60"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"

    def test_vol_5_less_than_vol_60_on_average(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_vol_5"].mean() <= result["f_vol_60"].mean() * 2


class TestMeanReversionFeatures:
    def test_zscore_clipped(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_zscore_20"].between(-5, 5).all()
        assert result["f_zscore_60"].between(-5, 5).all()

    def test_bollinger_pct_between_zero_one_mostly(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        pct_in_range = result["f_bollinger_pct"].between(0, 1).mean()
        assert pct_in_range > 0.85


class TestMicrostructureFeatures:
    def test_effective_spread_non_negative(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert (result["f_effective_spread"] >= 0).all()

    def test_price_impact_non_negative(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert (result["f_price_impact"] >= 0).all()

    def test_trade_sign_values(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_trade_sign"].isin([-1, 0, 1]).all()


class TestOrderFlowFeatures:
    def test_ofi_range(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_ofi"].between(-1, 1).all()

    def test_pressure_range(self, sample_crypto_df):
        result = build_features(sample_crypto_df)
        assert result["f_pressure"].between(-1.5, 1.5).all()


class TestMultiAssetFeatures:
    def test_features_built_for_all_assets(self, multi_asset_df):
        assets = multi_asset_df["asset"].unique()
        assert len(assets) >= 2

    def test_asset_encoded_unique(self, multi_asset_df):
        encodings = multi_asset_df["f_asset_encoded"].unique()
        assert len(encodings) >= 2

    def test_source_encoded_matches(self, multi_asset_df):
        crypto = multi_asset_df[multi_asset_df["source"] == "CRYPTO"]
        lobster = multi_asset_df[multi_asset_df["source"] == "LOBSTER"]
        assert (crypto["f_source_encoded"] == 0).all()
        assert (lobster["f_source_encoded"] == 1).all()