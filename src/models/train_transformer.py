"""
Transformer Trainer — Fast Two-Stage Curriculum
================================================
Stage 1: 5K rows/asset, 3 epochs   (~6 min)
Stage 2: 10K rows/asset, 8 epochs  (~15 min)
Total: ~21 min

Regularisation: dropout, label smoothing, L2 (AdamW)
Cross validation: 3-fold walk-forward
Metrics: Accuracy, F1, Precision, Recall
"""

from pathlib import Path
import sys
import math
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.data_pipeline import load_splits, splits_exist, run_pipeline
from src.data.feature_builder import get_observation_features

MODELS_DIR   = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN      = 30
HORIZON      = 5
THRESHOLD    = 0.001
FEATURE_COLS = get_observation_features()


def generate_labels(df, thresh=None):
    fr = df["mid_price"].shift(-HORIZON).sub(df["mid_price"]).div(
        df["mid_price"] + 1e-9)
    if thresh is None:
        thresh = max(fr.abs().quantile(0.40), 1e-8)
    labels = pd.Series(0, index=df.index)
    labels[fr >  thresh] = 1
    labels[fr < -thresh] = 2
    return labels, thresh


class TFDataset(Dataset):
    def __init__(self, df, rows_per_asset, label, thresh=None):
        from sklearn.preprocessing import StandardScaler
        subset = pd.concat(
            [df[df["asset"]==a].head(rows_per_asset) for a in df["asset"].unique()],
            ignore_index=True)
        all_X, all_y = [], []
        for asset in subset["asset"].unique():
            adf  = subset[subset["asset"]==asset].copy().reset_index(drop=True)
            if len(adf) < SEQ_LEN + HORIZON + 5:
                continue
            feat = adf[FEATURE_COLS].values.astype(np.float32)
            feat = StandardScaler().fit_transform(feat)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            lbs, thresh = generate_labels(adf, thresh)
            lbs  = lbs.values
            for i in range(SEQ_LEN, len(feat) - HORIZON):
                all_X.append(feat[i-SEQ_LEN:i])
                all_y.append(lbs[i])
        self.X = np.array(all_X, dtype=np.float32)
        self.y = np.array(all_y, dtype=np.int64)
        b = (self.y==1).mean()*100; s = (self.y==2).mean()*100
        print(f"  {label}: {len(self.X):,} seqs | "
              f"BUY={b:.1f}% SELL={s:.1f}% HOLD={(100-b-s):.1f}%")

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])


class PositionalEncoding(nn.Module):
    """Dynamic positional encoding — works with any sequence length."""
    def __init__(self, d, max_len=512, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.d    = d
        # Build large PE table, slice at runtime
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        # x: (B, T, d) — slice PE to match T regardless of saved size
        T = x.size(1)
        if T > self.pe.size(1):
            # Extend PE if needed
            extra = T - self.pe.size(1)
            pos = torch.arange(self.pe.size(1),
                               self.pe.size(1) + extra,
                               device=x.device).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, self.d, 2,
                            device=x.device).float() * (-math.log(10000)/self.d))
            new_pe = torch.zeros(1, extra, self.d, device=x.device)
            new_pe[0, :, 0::2] = torch.sin(pos * div)
            new_pe[0, :, 1::2] = torch.cos(pos * div)
            pe = torch.cat([self.pe.to(x.device), new_pe], dim=1)
        else:
            pe = self.pe.to(x.device)
        return self.drop(x + pe[:, :T, :])


class TFTrader(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, ff=128, drop=0.1):
        super().__init__()
        self.proj    = nn.Linear(len(FEATURE_COLS), d)
        self.pe      = PositionalEncoding(d, SEQ_LEN+5, drop)
        enc = nn.TransformerEncoderLayer(d, heads, ff, drop,
                                          batch_first=True, norm_first=True)
        self.tf      = nn.TransformerEncoder(enc, layers)
        self.norm    = nn.LayerNorm(d)
        self.drop    = nn.Dropout(drop)
        self.fc1     = nn.Linear(d, 32)
        self.fc2     = nn.Linear(32, 3)
        self.act     = nn.GELU()

    def forward(self, x):
        x = self.pe(self.proj(x))
        x = self.norm(self.tf(x)[:, -1, :])
        return self.fc2(self.act(self.fc1(self.drop(x))))


def run_epoch(model, loader, crit, opt=None):
    model.train() if opt else model.eval()
    ls = cor = tot = 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            loss = crit(out, y)
            if opt:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ls  += loss.item()
            cor += (out.argmax(1)==y).sum().item()
            tot += len(y)
    return ls/max(len(loader),1), cor/max(tot,1)*100


def get_metrics(model, loader, label=""):
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score,
                                  classification_report)
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(DEVICE)).argmax(1).cpu().numpy())
            labs.extend(y.numpy())
    acc  = accuracy_score(labs, preds)*100
    f1   = f1_score(labs, preds, average="macro", zero_division=0)*100
    prec = precision_score(labs, preds, average="macro", zero_division=0)*100
    rec  = recall_score(labs, preds, average="macro", zero_division=0)*100
    print(f"  {label}: Acc={acc:.2f}% F1={f1:.2f}% "
          f"Prec={prec:.2f}% Rec={rec:.2f}%")
    print(classification_report(labs, preds,
          target_names=["HOLD","BUY","SELL"], digits=3))
    return acc, f1


def train_stage(model, tr_ds, vl_ds, epochs, lr, label, patience=2):
    tr_l = DataLoader(tr_ds, 512, True,  num_workers=0)
    vl_l = DataLoader(vl_ds, 512, False, num_workers=0)
    lc   = np.bincount(tr_ds.y)
    wts  = torch.FloatTensor(1/(lc+1)/(1/(lc+1)).sum()).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=wts, label_smoothing=0.05)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
    best = 0.0; bst = None; p = 0; t0 = time.time()
    for ep in range(1, epochs+1):
        tl, ta = run_epoch(model, tr_l, crit, opt)
        vl, va = run_epoch(model, vl_l, crit)
        sch.step()
        gap  = ta - va
        flag = "  ⚠️" if gap > 10 else ""
        lr_n = opt.param_groups[0]["lr"]
        print(f"  [{label}] Ep {ep}/{epochs} | "
              f"Train={ta:.1f}% Val={va:.1f}% Gap={gap:.1f}%{flag} | "
              f"LR={lr_n:.1e} | {(time.time()-t0)/60:.1f}m")
        if va > best:
            best = va
            bst  = {k: v.clone() for k,v in model.state_dict().items()}
            p    = 0
        else:
            p += 1
            if p >= patience:
                print(f"  Early stop ep {ep} (best={best:.2f}%)")
                break
    if bst:
        model.load_state_dict(bst)
    return model, best


def walk_forward_cv(df, n_splits=3, rows=3_000, epochs=2):
    from sklearn.metrics import accuracy_score, f1_score
    print(f"\n  Walk-forward CV ({n_splits} folds)...")
    n = len(df); fs = n//(n_splits+1)
    accs = []
    for fold in range(n_splits):
        tr_df = df.iloc[:fs*(fold+1)]
        vl_df = df.iloc[fs*(fold+1):fs*(fold+2)]
        if len(tr_df) < rows*2 or len(vl_df) < 500:
            continue
        try:
            tr_ds = TFDataset(tr_df, rows,   f"  F{fold+1} tr")
            vl_ds = TFDataset(vl_df, rows//3, f"  F{fold+1} vl")
        except Exception:
            continue
        if len(tr_ds) < 50 or len(vl_ds) < 10:
            continue
        tr_l = DataLoader(tr_ds, 512, True)
        vl_l = DataLoader(vl_ds, 512, False)
        lc   = np.bincount(tr_ds.y)
        wts  = torch.FloatTensor(1/(lc+1)/(1/(lc+1)).sum()).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=wts, label_smoothing=0.05)
        m    = TFTrader().to(DEVICE)
        opt  = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        for _ in range(epochs):
            run_epoch(m, tr_l, crit, opt)
        preds, labs = [], []
        m.eval()
        with torch.no_grad():
            for X, y in vl_l:
                preds.extend(m(X.to(DEVICE)).argmax(1).cpu().numpy())
                labs.extend(y.numpy())
        acc = accuracy_score(labs, preds)*100
        f1  = f1_score(labs, preds, average="macro", zero_division=0)*100
        print(f"    Fold {fold+1}: acc={acc:.1f}% F1={f1:.1f}%")
        accs.append(acc)
    if accs:
        print(f"  CV: {np.mean(accs):.2f}%±{np.std(accs):.2f}%")
    return np.mean(accs) if accs else 0


def train_transformer():
    print("=" * 60)
    print(f"  Transformer — Two-Stage Curriculum | {DEVICE}")
    print("  Stage 1: 5K rows/asset, 3 epochs   (~6 min)")
    print("  Stage 2: 10K rows/asset, 8 epochs  (~15 min)")
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

    walk_forward_cv(train_df, n_splits=3, rows=3_000, epochs=2)

    # Build eval set from last 15% of training (same regime as train)
    eval_df = pd.concat([
        train_df[train_df["asset"]==a].iloc[
            int(len(train_df[train_df["asset"]==a])*0.70):
            int(len(train_df[train_df["asset"]==a])*0.85)]
        for a in train_df["asset"].unique()
    ], ignore_index=True)
    print(f"  Eval set: {len(eval_df):,} rows (same regime as training)")

    # Stage 1
    print("\n── STAGE 1 ──────────────────────────────────────────────")
    tr_ds_s1 = TFDataset(train_df, 20_000, "Train S1")
    vl_ds_s1 = TFDataset(eval_df,  5_000, "Eval S1 ")
    vl_l_s1  = DataLoader(vl_ds_s1, 512, False)

    model = TFTrader().to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params: {n_p:,} | d=64, heads=4, layers=2")
    print(f"  drop=0.1 | label_smooth=0.05 | L2=1e-4\n")

    t0 = time.time()
    model, s1_acc = train_stage(model, tr_ds_s1, vl_ds_s1,
                                3, 1e-4, "S1", patience=2)
    print(f"\n  ✅ Stage 1 done in {(time.time()-t0)/60:.1f} min | "
          f"best val={s1_acc:.2f}%")
    get_metrics(model, vl_l_s1, "Stage 1 Val")

    s1_path = MODELS_DIR / "transformer_stage1.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "s1_acc": s1_acc, "seq_len": SEQ_LEN,
                "input_size": len(FEATURE_COLS)}, str(s1_path))

    # Stage 2
    print("\n── STAGE 2 ──────────────────────────────────────────────")
    tr_ds_s2 = TFDataset(train_df, 30_000, "Train S2")
    vl_ds_s2 = TFDataset(eval_df,   7_000, "Eval S2 ")
    vl_l_s2  = DataLoader(vl_ds_s2, 512, False)

    ckpt = torch.load(str(s1_path), weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print("  Stage 1 weights loaded ✅\n")

    t0 = time.time()
    model, s2_acc = train_stage(model, tr_ds_s2, vl_ds_s2,
                                8, 3e-5, "S2", patience=3)
    print(f"\n  ✅ Stage 2 done in {(time.time()-t0)/60:.1f} min | "
          f"best val={s2_acc:.2f}%")
    print(f"  Improvement: {s2_acc-s1_acc:+.2f}%")
    get_metrics(model, vl_l_s2, "Final Val")

    final = MODELS_DIR / "transformer_multi_asset.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "input_size": len(FEATURE_COLS), "d_model": 64,
                "nhead": 4, "num_layers": 2, "dim_feedforward": 128,
                "n_classes": 3, "seq_len": SEQ_LEN,
                "horizon": HORIZON, "threshold": THRESHOLD,
                "s1_acc": s1_acc, "s2_acc": s2_acc}, str(final))
    print(f"\n  Model saved: {final}")
    print("=" * 60)
    return model


def load_transformer_model():
    path = MODELS_DIR / "transformer_multi_asset.pt"
    if not path.exists():
        return None
    ckpt  = torch.load(str(path), weights_only=False, map_location=DEVICE)
    model = TFTrader().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    train_transformer()