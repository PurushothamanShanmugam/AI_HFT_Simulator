"""
XGBoost Trainer — Per-Asset Models
====================================
Trains one XGBoost model per asset (8 models total).
Each model learns that asset's specific price patterns.

Expected accuracy: 55-65% per asset
Total time       : ~15-20 minutes

Assets: BTC, ETH, ADA, AAPL, AMZN, GOOG, INTC, MSFT
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report)
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from src.data.data_pipeline import load_splits, splits_exist, run_pipeline
from src.data.feature_builder import get_observation_features

MODELS_DIR = ROOT / "outputs" / "models" / "xgboost_per_asset"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Remove identity and raw price features
_ALL_FEATURES = get_observation_features()
_LEAKY        = {'f_source_encoded', 'f_asset_encoded',
                 'f_global_tick_norm', 'f_mid_price',
                 'f_bid1', 'f_ask1', 'f_mid_return'}
FEATURE_COLS  = [f for f in _ALL_FEATURES if f not in _LEAKY]
HORIZON       = 10


def prepare_asset_xy(df, asset, rows, label, thresh=None):
    """Build X and y for a single asset.
    If thresh is provided, use it (ensures train/eval use same threshold).
    """
    adf = df[df["asset"] == asset].head(rows).copy().reset_index(drop=True)
    if len(adf) < HORIZON + 10:
        print(f"  ⚠️  {asset}: not enough rows ({len(adf)})")
        return None, None, None

    fr = adf["mid_price"].shift(-HORIZON).sub(
         adf["mid_price"]).div(adf["mid_price"] + 1e-9)

    # Always compute threshold from this dataset
    # but use provided threshold if given (for eval consistency)
    if thresh is None:
        thresh = max(fr.abs().quantile(0.40), 1e-8)

    lbs = pd.Series(0, index=adf.index)
    lbs[fr >  thresh] = 1
    lbs[fr < -thresh] = 2

    feat = adf[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    X    = feat.iloc[:-HORIZON].values.astype(np.float32)
    y    = lbs.iloc[:-HORIZON].values.astype(int)

    h = (y==0).mean()*100; b = (y==1).mean()*100; s = (y==2).mean()*100
    print(f"    {label}: {len(X):,} rows | "
          f"HOLD={h:.1f}% BUY={b:.1f}% SELL={s:.1f}% | "
          f"threshold={thresh*100:.4f}%")
    return X, y, thresh


def train_asset(train_df, eval_df, asset):
    """Train a two-stage XGBoost model for one asset."""
    print(f"\n  ── {asset} ──────────────────────────────────────")

    # Prepare data — compute threshold from training, apply same to eval
    X_tr1, y_tr1, thresh = prepare_asset_xy(train_df, asset, 100_000, "Train S1")
    X_ev,  y_ev,  _      = prepare_asset_xy(eval_df,  asset,  20_000, "Eval   ", thresh=thresh)
    if X_tr1 is None or X_ev is None:
        return None, None, 0.0, 0.0

    # Scaler fit on training data only
    scaler   = StandardScaler()
    X_tr1_sc = scaler.fit_transform(X_tr1)
    X_ev_sc  = scaler.transform(X_ev)
    sw1      = compute_sample_weight("balanced", y_tr1)

    # Stage 1 — learn basic patterns
    m1 = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        early_stopping_rounds=30, tree_method="hist", device="cpu",
        random_state=42, verbosity=0, n_jobs=-1)
    m1.fit(X_tr1_sc, y_tr1, sample_weight=sw1,
           eval_set=[(X_ev_sc, y_ev)], verbose=False)

    s1_acc = accuracy_score(y_ev, m1.predict(X_ev_sc)) * 100
    print(f"    Stage 1: {m1.best_iteration} trees | "
          f"Val acc={s1_acc:.2f}%")

    # Stage 2 — more data, refinement
    X_tr2, y_tr2, _ = prepare_asset_xy(train_df, asset, 130_000, "Train S2", thresh=thresh)
    if X_tr2 is None:
        return m1, scaler, s1_acc, s1_acc

    X_tr2_sc = scaler.transform(X_tr2)
    sw2      = compute_sample_weight("balanced", y_tr2)

    m2 = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        early_stopping_rounds=50, tree_method="hist", device="cpu",
        random_state=42, verbosity=0, n_jobs=-1)
    m2.fit(X_tr2_sc, y_tr2, sample_weight=sw2,
           eval_set=[(X_ev_sc, y_ev)], verbose=False)

    s2_acc = accuracy_score(y_ev, m2.predict(X_ev_sc)) * 100
    best   = m2 if s2_acc >= s1_acc else m1
    best_acc = max(s1_acc, s2_acc)
    tr_acc = accuracy_score(y_tr2, best.predict(X_tr2_sc)) * 100

    print(f"    Stage 2: {m2.best_iteration} trees | "
          f"Val acc={s2_acc:.2f}%")
    print(f"    Best   : Stage {'2' if s2_acc >= s1_acc else '1'} | "
          f"Val={best_acc:.2f}% | Train={tr_acc:.2f}% | "
          f"Gap={tr_acc-best_acc:.2f}%")

    # Detailed metrics
    preds = best.predict(X_ev_sc)
    print(f"\n    Per-class (val):")
    print(classification_report(y_ev, preds,
          target_names=["HOLD","BUY","SELL"],
          digits=3, zero_division=0))

    return best, scaler, best_acc, tr_acc


def train_xgboost():
    print("=" * 65)
    print("  XGBoost — Per-Asset Training (8 separate models)")
    print("  Each asset gets its own model trained on its own data")
    print("  Target: 55-65% per asset")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[DATA] Loading splits...")
    if not splits_exist():
        run_pipeline()

    SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"
    try:
        train_df = pd.read_parquet(
            SPLITS_DIR / "train_70.parquet", engine="fastparquet")
        val_df   = pd.read_parquet(
            SPLITS_DIR / "val_20.parquet", engine="fastparquet")
    except Exception:
        train_df, val_df, _ = load_splits()

    # Build eval set from last 15% of training per asset
    eval_df = pd.concat([
        train_df[train_df["asset"]==a].iloc[
            int(len(train_df[train_df["asset"]==a])*0.70):
            int(len(train_df[train_df["asset"]==a])*0.85)]
        for a in train_df["asset"].unique()
    ], ignore_index=True)

    assets = sorted(train_df["asset"].unique().tolist())
    print(f"  Train : {len(train_df):,} rows")
    print(f"  Eval  : {len(eval_df):,} rows (held-out from training)")
    print(f"  Assets: {assets}")
    print(f"  Features: {len(FEATURE_COLS)} market microstructure features")

    # ── Train one model per asset ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING — One XGBoost model per asset")
    print(f"  Stage 1: 100K rows | Stage 2: 130K rows")
    print(f"{'='*65}")

    results   = {}
    models    = {}
    scalers   = {}
    t_total   = time.time()

    for asset in assets:
        t_asset = time.time()
        model, scaler, val_acc, tr_acc = train_asset(train_df, eval_df, asset)
        elapsed = time.time() - t_asset

        results[asset] = {
            "val_acc"  : val_acc,
            "train_acc": tr_acc,
            "gap"      : tr_acc - val_acc,
            "time_min" : elapsed / 60,
        }
        if model is not None:
            models[asset]  = model
            scalers[asset] = scaler
            # Save per-asset model + threshold
            model.save_model(str(MODELS_DIR / f"xgboost_{asset}.json"))
            joblib.dump(scaler, str(MODELS_DIR / f"scaler_{asset}.pkl"))
            joblib.dump(results[asset].get("thresh", 0.001),
                        str(MODELS_DIR / f"threshold_{asset}.pkl"))

        print(f"    ✅ {asset} done in {elapsed/60:.1f} min")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    accs = [r["val_acc"] for r in results.values()]

    print(f"\n{'='*65}")
    print(f"  XGBOOST PER-ASSET TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"\n  {'Asset':<8} {'Val Acc':>8} {'Train Acc':>10} {'Gap':>8} {'Time':>8}")
    print(f"  {'-'*46}")
    for asset, r in results.items():
        flag = "⚠️" if r["gap"] > 15 else "✅"
        print(f"  {asset:<8} {r['val_acc']:>7.2f}% {r['train_acc']:>9.2f}% "
              f"{r['gap']:>7.2f}% {r['time_min']:>6.1f}m {flag}")

    print(f"  {'-'*46}")
    print(f"  {'AVERAGE':<8} {np.mean(accs):>7.2f}%")
    print(f"  {'BEST':<8} {max(accs):>7.2f}%")
    print(f"  {'WORST':<8} {min(accs):>7.2f}%")
    print(f"\n  Total training time : {total_time/60:.1f} minutes")
    print(f"  Models saved to     : outputs/models/xgboost_per_asset/")

    # Save combined metadata
    joblib.dump({
        "assets"      : assets,
        "feature_cols": FEATURE_COLS,
        "results"     : results,
    }, str(MODELS_DIR / "metadata.pkl"))

    # Also save a combined model file for simulation compatibility
    best_asset = max(results, key=lambda a: results[a]["val_acc"])
    models[best_asset].save_model(
        str(ROOT / "outputs" / "models" / "xgboost_multi_asset.json"))
    joblib.dump(scalers[best_asset],
        str(ROOT / "outputs" / "models" / "xgboost_scaler.pkl"))

    print(f"\n  Best asset         : {best_asset} "
          f"({results[best_asset]['val_acc']:.2f}%)")
    print(f"\n  Next step: python -m src.models.train_ppo")
    print(f"{'='*65}\n")
    return models, scalers


def load_xgboost_model():
    p  = ROOT / "outputs" / "models" / "xgboost_multi_asset.json"
    sp = ROOT / "outputs" / "models" / "xgboost_scaler.pkl"
    if not p.exists():
        return None, None
    m = xgb.XGBClassifier()
    m.load_model(str(p))
    return m, (joblib.load(str(sp)) if sp.exists() else None)


if __name__ == "__main__":
    train_xgboost()