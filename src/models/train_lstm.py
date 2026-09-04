"""
LSTM Trainer — Per-Asset Training
===================================
Trains one LSTM model per asset (8 models total).
Target: 65-75% accuracy per asset

Key improvements:
  1. Per-asset training — each model learns one asset's patterns
  2. Longer sequences (60 timesteps)
  3. Bigger model with attention (256×3 layers)
  4. Temporal features only (no raw prices)
  5. Lower learning rate with warm restart
  6. Adaptive per-asset threshold
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

from src.data.data_pipeline import load_splits, splits_exist, run_pipeline
from src.data.feature_builder import get_observation_features

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 60
HORIZON  = 10
MODELS_DIR = ROOT / "outputs" / "models" / "lstm_per_asset"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Use temporal features only — features that change meaningfully over time
_ALL_FEATURES = get_observation_features()
_TEMPORAL = {
    'f_returns', 'f_log_return', 'f_ofi', 'f_imbalance',
    'f_depth_imbalance', 'f_pressure', 'f_buy_ratio', 'f_sell_ratio',
    'f_return_5', 'f_return_10', 'f_return_20', 'f_return_60',
    'f_accel', 'f_jerk', 'f_vol_5', 'f_vol_10', 'f_vol_20', 'f_vol_60',
    'f_zscore_20', 'f_zscore_60', 'f_dev_vwap', 'f_bollinger_pct',
    'f_effective_spread', 'f_price_impact', 'f_bid_ask_bounce',
    'f_trade_sign', 'f_kyle_lambda', 'f_size_ratio', 'f_trend_strength',
    'f_vol_regime', 'f_spread_regime', 'f_liquidity_score',
    'f_prev_return', 'f_prev_spread', 'f_prev_imbalance', 'f_time_of_day',
    'f_spread_pct', 'f_volatility',
}
FEATURE_COLS = [f for f in _ALL_FEATURES if f in _TEMPORAL]
N_FEATURES   = len(FEATURE_COLS)


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMWithAttention(nn.Module):
    """LSTM with temporal attention — focuses on important timesteps."""

    def __init__(self, n_features, hidden=256, n_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 128)
        self.fc2  = nn.Linear(128, 64)
        self.fc3  = nn.Linear(64, 3)
        self.act  = nn.GELU()

    def forward(self, x):
        o, _  = self.lstm(x)                       # (B, T, hidden)
        w     = torch.softmax(self.attn(o), dim=1) # (B, T, 1)
        ctx   = (o * w).sum(dim=1)                 # (B, hidden)
        ctx   = self.bn(ctx)
        ctx   = self.drop(self.act(self.fc1(ctx)))
        ctx   = self.drop(self.act(self.fc2(ctx)))
        return self.fc3(ctx)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AssetDataset(Dataset):
    """Sequence dataset for a single asset."""

    def __init__(self, df, asset, rows, thresh, scaler, label):
        adf = df[df["asset"] == asset].head(rows).copy().reset_index(drop=True)

        # Generate labels
        fr  = adf["mid_price"].shift(-HORIZON).sub(
              adf["mid_price"]).div(adf["mid_price"] + 1e-9)
        lbs = pd.Series(0, index=adf.index)
        lbs[fr >  thresh] = 1
        lbs[fr < -thresh] = 2

        # Build features
        feat = adf[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0).values
        feat = scaler.transform(feat).astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

        # Build sequences
        Xs, ys = [], []
        for i in range(SEQ_LEN, len(feat) - HORIZON):
            Xs.append(feat[i-SEQ_LEN:i])
            ys.append(lbs.iloc[i])

        self.X = np.array(Xs, dtype=np.float32)
        self.y = np.array(ys, dtype=np.int64)

        h = (self.y==0).mean()*100
        b = (self.y==1).mean()*100
        s = (self.y==2).mean()*100
        print(f"    {label}: {len(self.X):,} seqs | "
              f"HOLD={h:.1f}% BUY={b:.1f}% SELL={s:.1f}%")

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, crit, opt=None):
    model.train() if opt else model.eval()
    tot_loss = tot_correct = tot_n = 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            loss = crit(out, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss    += loss.item()
            tot_correct += (out.argmax(1) == y).sum().item()
            tot_n       += len(y)
    return tot_loss / max(len(loader), 1), tot_correct / max(tot_n, 1) * 100


def train_asset_lstm(train_df, eval_df, asset):
    """Train a two-stage LSTM for one asset."""
    print(f"\n  ══ {asset} ══════════════════════════════════════")

    # Compute threshold and scaler from training data
    adf_tr = train_df[train_df["asset"] == asset].head(100_000).copy()
    fr_tr  = adf_tr["mid_price"].shift(-HORIZON).sub(
             adf_tr["mid_price"]).div(adf_tr["mid_price"] + 1e-9)
    thresh = max(fr_tr.abs().quantile(0.40), 1e-8)
    print(f"    Threshold: {thresh*100:.4f}% | "
          f"Features: {N_FEATURES} temporal signals")

    scaler = StandardScaler()
    scaler.fit(adf_tr[FEATURE_COLS].replace([np.inf,-np.inf],0).fillna(0))

    # ── Stage 1 ───────────────────────────────────────────────────────────
    print(f"    Stage 1: 50K rows, 10 epochs")
    tr_ds1 = AssetDataset(train_df, asset, 50_000, thresh, scaler, "Train S1")
    ev_ds1 = AssetDataset(eval_df,  asset, 10_000, thresh, scaler, "Eval S1 ")

    classes = np.unique(tr_ds1.y)
    cw      = compute_class_weight("balanced", classes=classes, y=tr_ds1.y)
    weights = torch.zeros(3).to(DEVICE)
    for i, c in enumerate(classes):
        weights[c] = cw[i]
    crit = nn.CrossEntropyLoss(weight=weights)

    tr_ld1 = DataLoader(tr_ds1, 256, True,  num_workers=0, pin_memory=False)
    ev_ld1 = DataLoader(ev_ds1, 256, False, num_workers=0, pin_memory=False)

    model = LSTMWithAttention(N_FEATURES).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)

    best_acc  = 0.0
    best_wts  = None
    patience  = 3
    p_counter = 0
    t0        = time.time()

    for ep in range(1, 11):
        tr_loss, tr_acc = train_epoch(model, tr_ld1, crit, opt)
        _,       vl_acc = train_epoch(model, ev_ld1, crit)
        sch.step()
        gap  = tr_acc - vl_acc
        flag = "⚠️" if gap > 15 else ""
        print(f"    [S1] Ep {ep:>2}/10 | "
              f"Train={tr_acc:.1f}% Val={vl_acc:.1f}% "
              f"Gap={gap:.1f}% {flag} | "
              f"{(time.time()-t0)/60:.1f}m")
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_wts = {k: v.clone() for k, v in model.state_dict().items()}
            p_counter = 0
        else:
            p_counter += 1
            if p_counter >= patience:
                print(f"    Early stop (best={best_acc:.2f}%)")
                break

    model.load_state_dict(best_wts)
    s1_acc = best_acc
    print(f"    ✅ Stage 1: {s1_acc:.2f}% in {(time.time()-t0)/60:.1f}m")

    # ── Stage 2 ───────────────────────────────────────────────────────────
    print(f"    Stage 2: 80K rows, 15 epochs")
    tr_ds2 = AssetDataset(train_df, asset, 80_000, thresh, scaler, "Train S2")
    ev_ds2 = AssetDataset(eval_df,  asset, 15_000, thresh, scaler, "Eval S2 ")

    cw2 = compute_class_weight("balanced",
                               classes=np.unique(tr_ds2.y), y=tr_ds2.y)
    weights2 = torch.FloatTensor(cw2).to(DEVICE) if len(cw2)==3 else weights
    crit2 = nn.CrossEntropyLoss(weight=weights2)

    tr_ld2 = DataLoader(tr_ds2, 256, True,  num_workers=0)
    ev_ld2 = DataLoader(ev_ds2, 256, False, num_workers=0)

    opt2 = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-7)

    best_acc2  = s1_acc
    best_wts2  = best_wts
    p_counter2 = 0
    patience2  = 4
    t0         = time.time()

    for ep in range(1, 16):
        tr_loss, tr_acc = train_epoch(model, tr_ld2, crit2, opt2)
        _,       vl_acc = train_epoch(model, ev_ld2, crit2)
        sch2.step()
        gap  = tr_acc - vl_acc
        flag = "⚠️" if gap > 15 else ""
        print(f"    [S2] Ep {ep:>2}/15 | "
              f"Train={tr_acc:.1f}% Val={vl_acc:.1f}% "
              f"Gap={gap:.1f}% {flag} | "
              f"{(time.time()-t0)/60:.1f}m")
        if vl_acc > best_acc2:
            best_acc2 = vl_acc
            best_wts2 = {k: v.clone() for k, v in model.state_dict().items()}
            p_counter2 = 0
        else:
            p_counter2 += 1
            if p_counter2 >= patience2:
                print(f"    Early stop (best={best_acc2:.2f}%)")
                break

    model.load_state_dict(best_wts2)
    s2_acc = best_acc2

    # Final metrics
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in ev_ld2:
            out = model(X.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y.numpy())

    print(f"\n    Per-class breakdown:")
    print(classification_report(all_true, all_preds,
          target_names=["HOLD","BUY","SELL"], digits=3, zero_division=0))

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features"      : N_FEATURES,
        "hidden"          : 256,
        "n_layers"        : 3,
        "seq_len"         : SEQ_LEN,
        "horizon"         : HORIZON,
        "threshold"       : thresh,
        "feature_cols"    : FEATURE_COLS,
        "s1_acc"          : s1_acc,
        "s2_acc"          : s2_acc,
    }, str(MODELS_DIR / f"lstm_{asset}.pt"))
    joblib.dump(scaler, str(MODELS_DIR / f"scaler_{asset}.pkl"))

    print(f"    ✅ {asset}: S1={s1_acc:.2f}% → S2={s2_acc:.2f}% "
          f"(+{s2_acc-s1_acc:.2f}%)")
    return s2_acc if s2_acc > s1_acc else s1_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def train_lstm():
    print("=" * 65)
    print("  LSTM — Per-Asset Training (8 models)")
    print(f"  Device: {DEVICE}")
    print(f"  Architecture: LSTM(256×3) + Attention")
    print(f"  Features: {N_FEATURES} temporal signals")
    print(f"  Sequence: {SEQ_LEN} timesteps → predict {HORIZON} ahead")
    print(f"  Target: 65-75% per asset")
    print("=" * 65)

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

    # Eval from last 15% of training (same regime)
    eval_df = pd.concat([
        train_df[train_df["asset"]==a].iloc[
            int(len(train_df[train_df["asset"]==a])*0.70):
            int(len(train_df[train_df["asset"]==a])*0.85)]
        for a in train_df["asset"].unique()
    ], ignore_index=True)

    assets = sorted(train_df["asset"].unique().tolist())
    print(f"\n  Train : {len(train_df):,} rows")
    print(f"  Eval  : {len(eval_df):,} rows (same regime)")
    print(f"  Assets: {assets}")

    results  = {}
    t_total  = time.time()

    for asset in assets:
        t_asset = time.time()
        acc     = train_asset_lstm(train_df, eval_df, asset)
        results[asset] = {
            "acc"     : acc,
            "time_min": (time.time() - t_asset) / 60,
        }

    # Summary
    accs = [r["acc"] for r in results.values()]
    print(f"\n{'='*65}")
    print(f"  LSTM PER-ASSET TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"\n  {'Asset':<8} {'Val Acc':>8} {'Time':>8}")
    print(f"  {'-'*28}")
    for asset, r in results.items():
        flag = "✅" if r["acc"] >= 55 else "⚠️"
        print(f"  {asset:<8} {r['acc']:>7.2f}% {r['time_min']:>6.1f}m {flag}")
    print(f"  {'-'*28}")
    print(f"  {'AVERAGE':<8} {np.mean(accs):>7.2f}%")
    print(f"  {'BEST':<8} {max(accs):>7.2f}%")
    print(f"  {'WORST':<8} {min(accs):>7.2f}%")
    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} minutes")
    print(f"  Models saved: outputs/models/lstm_per_asset/")

    # Save best as main model for simulation
    best_asset = max(results, key=lambda a: results[a]["acc"])
    import shutil
    best_src = MODELS_DIR / f"lstm_{best_asset}.pt"
    best_dst = ROOT / "outputs" / "models" / "lstm_multi_asset.pt"
    shutil.copy(str(best_src), str(best_dst))
    print(f"  Best: {best_asset} ({results[best_asset]['acc']:.2f}%)")
    print(f"\n  Next step: python -m src.models.train_transformer")
    print(f"{'='*65}\n")


def load_lstm_model():
    p = ROOT / "outputs" / "models" / "lstm_multi_asset.pt"
    if not p.exists():
        return None
    ckpt  = torch.load(str(p), map_location="cpu", weights_only=False)
    model = LSTMWithAttention(
        ckpt.get("n_features", N_FEATURES),
        ckpt.get("hidden", 256),
        ckpt.get("n_layers", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    train_lstm()