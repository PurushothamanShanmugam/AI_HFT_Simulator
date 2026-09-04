"""
HFT AI Simulator — Live Dashboard (Single File)
"""
from pathlib import Path
import sys, time, json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import random
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Alpha / Sentiment layer ────────────────────────────────────────────────────
try:
    from src.news.live_sentiment_provider import LiveSentimentProvider
    from src.alpha.alpha_signal import AlphaSignal
    SENTIMENT     = LiveSentimentProvider()
    ALPHA_SIGNAL  = AlphaSignal()
    ALPHA_ENABLED = True
except:
    ALPHA_ENABLED = False
    SENTIMENT     = None
    ALPHA_SIGNAL  = None

st.set_page_config(page_title="HFT AI Simulator", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@600;700&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.block-container{
    background-color:#060b18!important;color:#e2e8f0!important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{
    background-color:#0d1117!important;border-right:1px solid #1e3a5f!important;}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p{color:#00ffe7!important;}
[data-testid="stSidebar"] hr{border-color:#1e3a5f!important;}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] small{color:#b0c4de!important;}
[data-testid="stSlider"] label,[data-testid="stSlider"] p{color:#b0c4de!important;}
.stSlider>div>div>div>div{background-color:#00ffe7!important;}
[data-testid="stSidebar"] button[kind="primary"]{
    background:linear-gradient(135deg,#00ffe7,#0099aa)!important;
    color:#000000!important;border:none!important;font-weight:800!important;
    letter-spacing:1px!important;border-radius:5px!important;
    box-shadow:0 0 14px rgba(0,255,231,0.4)!important;}
[data-testid="stSidebar"] button[kind="secondary"]{
    background:transparent!important;color:#ff2d78!important;
    border:2px solid #ff2d78!important;font-weight:700!important;
    letter-spacing:1px!important;border-radius:5px!important;}
[data-testid="stSidebar"] button[kind="secondary"]:hover{
    background:#ff2d78!important;color:#ffffff!important;}
[data-testid="stAlert"]{background-color:#0d1f33!important;border-color:#1e3a5f!important;}
[data-testid="stAlert"] p{color:#b0c4de!important;}
[data-testid="stProgressBar"]>div>div{background-color:#00ffe7!important;}
.stCaption,[data-testid="stCaptionContainer"] p{color:#b0c4de!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#060b18;}
::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:3px;}
footer,#MainMenu{display:none!important;}
</style>""", unsafe_allow_html=True)

AGENT_COLORS = {
    "DQN":"#00ffe7","PPO":"#39ff14","LSTM":"#bf7fff",
    "XGBoost":"#ffaa00","Transformer":"#ff2d78",
}
ASSETS = ["BTC","ETH","ADA","AAPL","AMZN","GOOG","INTC","MSFT"]
INIT   = 100_000.0

for k,v in {
    "sim_data":              None,
    "sim_ready":             False,
    "rows":                  500,
    "live_mode":             False,
    "live_files":            [],
    "live_rows":             0,
    "show_live_uploader":    False,
    "live_upload_attempted": False,
}.items():
    if k not in st.session_state: st.session_state[k]=v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ HFT AI Simulator")
    st.markdown("---")

    # ── Mode indicator ────────────────────────────────────────────────────────
    if st.session_state.live_mode:
        st.markdown(
            '<div style="background:#0d2040;border:1px solid #ff9900;border-radius:6px;'
            'padding:7px 12px;margin-bottom:10px;text-align:center;">'
            '<span style="color:#ff9900;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.8rem;letter-spacing:1.5px;font-weight:700;">📡 LIVE DATA MODE</span>'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background:#0a1a0a;border:1px solid #1e3a5f;border-radius:6px;'
            'padding:7px 12px;margin-bottom:10px;text-align:center;">'
            '<span style="color:#00ffe7;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.8rem;letter-spacing:1.5px;">📊 DEFAULT MODE</span>'
            '</div>', unsafe_allow_html=True
        )

    rows = st.slider("Simulation ticks", 500, 5000, 1000, step=500)
    st.session_state.rows = rows
    st.markdown("---")

    # ── Default mode run ──────────────────────────────────────────────────────
    run_btn = st.button("▶ RUN SIMULATION", use_container_width=True, type="primary")

    st.markdown("---")

    # ── Live Data section ─────────────────────────────────────────────────────
    st.markdown("---")

    LIVE_DIR = ROOT / "data" / "live_data"
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Entry-point button — toggles the uploader open
    if st.button("Click Here for Live Data", use_container_width=True, type="secondary"):
        st.session_state.show_live_uploader = not st.session_state.get("show_live_uploader", False)
        st.session_state.live_upload_attempted = False

    live_btn = False

    if st.session_state.get("show_live_uploader", False):
        st.caption("Upload your tick data CSV to simulate on live data.")
        uploaded_files = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="live_csv_uploader",
        )

        if uploaded_files is not None:
            # User uploaded a file — save it to live_data/
            dest = LIVE_DIR / uploaded_files.name
            dest.write_bytes(uploaded_files.read())
            st.success(f"✅ **{uploaded_files.name}** saved to live_data/")
            live_btn = st.button(
                "▶ RUN LIVE SIMULATION",
                use_container_width=True,
                type="primary",
                key="live_run_btn",
            )
        else:
            # Uploader is open but nothing uploaded yet
            if st.session_state.get("live_upload_attempted", False):
                st.error(
                    "⚠️ No file uploaded. Execution will fall back to the previous default method."
                )
                live_btn = st.button(
                    "▶ RUN WITH RANDOM DATA",
                    use_container_width=True,
                    type="secondary",
                    key="live_run_random_btn",
                )
            else:
                # Show a "proceed anyway" option after a moment
                if st.button(
                    "Continue without uploading →",
                    use_container_width=True,
                    type="secondary",
                    key="live_skip_btn",
                ):
                    st.session_state.live_upload_attempted = True
                    st.rerun()

    # ── Reset ─────────────────────────────────────────────────────────────────
    reset_btn = st.button("↺ RESET", use_container_width=True, type="secondary")
    if reset_btn:
        st.session_state.sim_data              = None
        st.session_state.sim_ready             = False
        st.session_state.live_mode             = False
        st.session_state.live_rows             = 0
        st.session_state.live_files            = []
        st.session_state.show_live_uploader    = False
        st.session_state.live_upload_attempted = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Agents")
    for n,c in AGENT_COLORS.items():
        st.markdown(f'<span style="color:{c};font-size:15px;font-weight:700">● {n}</span>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Space=play · →/← step · F=fullscreen")

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    m = {}
    MD = ROOT/"outputs"/"models"
    try:
        from stable_baselines3 import DQN as _D
        m["DQN"] = _D.load(str(MD/"dqn_multi_asset.zip"))
    except: m["DQN"] = None
    try:
        from stable_baselines3 import PPO as _P
        m["PPO"] = _P.load(str(MD/"ppo_multi_asset.zip"))
    except: m["PPO"] = None
    try:
        from src.models.train_lstm import load_lstm_model
        m["LSTM"] = load_lstm_model()
    except: m["LSTM"] = None
    try:
        import xgboost as xgb
        mm = xgb.XGBClassifier()
        mm.load_model(str(MD/"xgboost_multi_asset.json"))
        m["XGBoost"] = mm
    except: m["XGBoost"] = None
    try:
        from src.models.train_transformer import load_transformer_model
        m["Transformer"] = load_transformer_model()
    except: m["Transformer"] = None
    return m

# Pre-load on startup
_ph = st.empty()
with _ph: _mdls = load_models()
_ph.empty()

# ── Synthetic LOB data generator ──────────────────────────────────────────────
def make_lob_data(n_ticks, seed=None):
    if seed is None:
        seed = int(time.time()*1000) % 9999999
    rng    = np.random.default_rng(seed)
    PRICES = {"BTC":50000.0,"ETH":2000.0,"ADA":1.2,"AAPL":180.0,
              "AMZN":180.0,"GOOG":140.0,"INTC":35.0,"MSFT":380.0}
    prices = dict(PRICES)
    regime_len = max(50, n_ticks//(len(ASSETS)*2))
    regimes    = {a:int(rng.choice([-1,1])) for a in ASSETS}
    reg_ticks  = {a:0 for a in ASSETS}
    rows=[]
    ao = list(ASSETS); rng.shuffle(ao)
    for i in range(n_ticks):
        asset = ao[i%len(ao)]
        p_now = prices[asset]
        reg_ticks[asset] += 1
        if reg_ticks[asset] >= regime_len:
            regimes[asset] = int(rng.choice([-1,1]))
            reg_ticks[asset] = 0
        reg = regimes[asset]
        sp  = p_now*rng.uniform(0.0003,0.0007)
        imb = float(np.clip(reg*0.60+rng.normal(0,0.15),-1,1))
        dep = float(np.clip(imb*0.80+rng.normal(0,0.08),-1,1))
        prs = float(np.clip(0.7*imb+0.3*dep,-1,1))
        directional = reg*p_now*abs(imb)*0.0010
        noise       = rng.normal(0,p_now*0.0003)
        prices[asset] = max(p_now*0.3, p_now+directional+noise)
        p   = prices[asset]
        ret = (p-p_now)/p_now
        vol = abs(ret)*10
        rows.append({
            "asset":asset,"mid_price":p,"spread":sp,
            "returns":ret,"imbalance":imb,"depth_imbalance":dep,
            "volatility":vol,"microstructure_pressure":prs,
            "order_flow_imbalance":imb,
            "f_returns":ret,"f_imbalance":imb,"f_volatility":vol,
            "f_spread":sp,"f_pressure":prs,"f_ofi":imb,
        })
    return pd.DataFrame(rows)

# ── Agent trader ───────────────────────────────────────────────────────────────
class Trader:
    def __init__(self, name, model):
        self.name       = name
        self.model      = model
        self.cash       = INIT
        self.inv        = {a:0 for a in ASSETS}
        self.trades     = 0
        self.wins       = 0
        self.pnl        = 0.0
        self.entry      = {}
        self.hold_ticks = {}
        self.consec     = {}
        self.buf        = []
        self._last_alpha  = 0.0
        self._last_regime = "SIDEWAYS"
        self._last_source = "RULE"

    def portfolio_value(self, pmap):
        return round(self.cash+sum(self.inv.get(a,0)*pmap.get(a,0) for a in ASSETS),2)

    def check_sl_tp(self, asset, price, spread):
        if self.inv.get(asset,0)==0: return False,0.0
        cost  = max(spread/2,price*0.0001)
        bid   = price-spread/2
        entry = self.entry.get(asset,price)
        pnl   = bid-entry-cost
        held  = self.hold_ticks.get(asset,0)

        # ── Asset-aware SL/TP ─────────────────────────────────────────
        # BTC/ETH are highly volatile — need wider SL and more hold time
        # AAPL/AMZN/GOOG/MSFT/INTC/ADA — tight SL is fine
        # ── Your strategy: tight 0.06% SL for all assets ─────────────
        # Only 1-2 agents enter, so tight SL = small loss per agent
        # Quick exit if wrong → capital freed for next signal
        sl = -(price * 0.0006)          # 0.06% of price
        tp =  max(cost*4, price*0.002)  # TP unchanged

        if pnl <= sl:             return True, round(pnl,4)
        if pnl >= tp and held>=3: return True, round(pnl,4)
        return False,0.0

    def execute(self, action, asset, price, spread):
        cost  = max(spread/2,price*0.0001)
        inv   = self.inv.get(asset,0)
        bid   = round(price-spread/2,5)
        ask   = round(price+spread/2,5)
        entry = self.entry.get(asset,price)
        needed = {"DQN":1,"PPO":1,"XGBoost":1,"LSTM":1,"Transformer":1}.get(self.name,2)
        if action=="BUY" and self.cash>=ask+cost and inv==0:
            self.consec[asset] = self.consec.get(asset,0)+1
            if self.consec[asset] < needed:
                return False,0.0,bid,ask,"-","-"
            self.consec[asset]=0
            self.cash -= ask+cost
            self.inv[asset]=1
            self.entry[asset]=ask
            self.hold_ticks[asset]=0
            self.trades+=1
            return True,0.0,bid,ask,"Market ask-side","OPEN"
        elif action=="SELL" and inv>0:
            held=self.hold_ticks.get(asset,0)
            if held<3: return False,0.0,bid,ask,"-","-"
            pnl=bid-entry-cost
            self.cash+=bid-cost
            self.inv[asset]=0
            self.pnl+=pnl
            self.consec[asset]=0
            self.hold_ticks[asset]=0
            if asset in self.entry: del self.entry[asset]
            self.trades+=1
            if pnl>0: self.wins+=1
            return True,round(pnl,4),bid,ask,"Market bid-side","Profit" if pnl>0 else "Loss"
        if action=="SELL" and inv==0: self.consec[asset]=0
        return False,0.0,bid,ask,"-","-"

    def get_strategy_reason(self, action, asset, prs, alpha, regime, source):
        src = {"ALPHA":"🔶 ALPHA","MODEL":"🤖 MODEL","RULE":"📐 RULE"}.get(source,source)
        reg = {"BULL":"🟢 BULL","BEAR":"🔴 BEAR","SIDEWAYS":"🟡 SIDE","HIGH_VOL":"⚠️ HV"}.get(regime,regime)
        desc = {
            "DQN"        :{"BUY":f"DQN Momentum prs={prs:.3f}","SELL":"DQN exit"},
            "PPO"        :{"BUY":"PPO Conservative","SELL":"PPO exit"},
            "XGBoost"    :{"BUY":f"XGB Trend imb={prs:.3f}","SELL":"XGB exit"},
            "LSTM"       :{"BUY":f"LSTM Scalper prs={prs:.3f}","SELL":"LSTM exit"},
            "Transformer":{"BUY":f"TRF Mkt-Maker prs={prs:.3f}","SELL":"TRF exit"},
        }.get(self.name,{}).get(action,f"{self.name} {action}")
        return f"{src} | {action} | {desc} | α={alpha:+.3f} | {reg} | {asset}"

    def _try_rl_model(self, obs, inv_a):
        """Try DQN/PPO model. Falls back to feature-based model if weights missing."""
        # obs = [f_sp, f_ret, f_vol, f_imb, f_prs, f_ofi, inv_n]
        try:
            if self.model:
                a,_ = self.model.predict(obs, deterministic=False)
                act = ["HOLD","BUY","SELL"][int(a)]
                if act=="BUY"  and inv_a>0:  act="HOLD"
                if act=="SELL" and inv_a==0: act="HOLD"
                return act
        except: pass
        # ── Feature-based fallback using exact training feature indices ──
        # obs = [spread, ret, vol, ofi, dep, prs, inv/6]
        f_ret = float(obs[1])   # returns
        f_ofi = float(obs[3])   # order_flow_imbalance
        f_prs = float(obs[5])   # microstructure_pressure
        # DQN: momentum-based
        if self.name == "DQN":
            score = 0.4*f_ret + 0.3*f_ofi + 0.3*f_prs
            if inv_a == 0 and score > 0.10: return "BUY"
            if inv_a >  0 and score < -0.10: return "SELL"
        # PPO: conservative
        elif self.name == "PPO":
            score = 0.35*f_ret + 0.35*f_ofi + 0.30*f_prs
            if inv_a == 0 and score > 0.15: return "BUY"
            if inv_a >  0 and score < -0.12: return "SELL"
        return "HOLD"

    def _try_xgb_model(self, row, inv_a):
        """XGBoost model — uses real parquet features."""
        # Skip on synthetic data (no real features)
        f_ret = float(row.get("f_returns", 0.0))
        f_imb = float(row.get("f_imbalance", 0.0))
        if f_ret == 0.0 and f_imb == 0.0:
            return "HOLD"  # synthetic data — skip
        if not self.model:
            # Feature-based fallback: uses same features as training
            raw_ofi = float(row.get("f_ofi", row.get("order_flow_imbalance", f_imb)))
            raw_dep = float(row.get("depth_imbalance", row.get("f_depth_imbalance", 0.0)))
            raw_prs = 0.5 * raw_ofi + 0.5 * raw_dep
            raw_ret = float(row.get("returns", f_ret))
            score = 0.35*raw_ret + 0.35*raw_ofi + 0.30*raw_prs
            if inv_a == 0 and score > 0.12: return "BUY"
            if inv_a >  0 and score < -0.12: return "SELL"
            return "HOLD"
        try:
            from src.data.feature_builder import get_observation_features
            fc   = get_observation_features()
            feat = np.nan_to_num(
                np.array([float(row.get(f,0.0)) for f in fc], dtype=np.float32
            ).reshape(1,-1))
            proba = self.model.predict_proba(feat)[0]
            pred  = int(self.model.predict(feat)[0])
            if proba.max() < 0.40: return "HOLD"
            act = ["HOLD","BUY","SELL"][pred]
            if act=="BUY"  and inv_a>0:  act="HOLD"
            if act=="SELL" and inv_a==0: act="HOLD"
            return act
        except: return "HOLD"

    def _try_seq_model(self, row, inv_a, seq_len, feature_set):
        """Try LSTM or Transformer with sequence buffer."""
        if not self.model: return "HOLD"
        try:
            import torch
            from src.data.feature_builder import get_observation_features
            fc   = get_observation_features()
            fidx = [i for i,f in enumerate(fc) if f in feature_set]
            allf = np.array([float(row.get(f,0.0)) for f in fc], dtype=np.float32)
            self.buf.append(allf[fidx])
            if len(self.buf)>seq_len: self.buf.pop(0)
            if len(self.buf)>=seq_len:
                x = torch.tensor(np.nan_to_num(np.array(self.buf))).unsqueeze(0)
                with torch.no_grad(): pred = self.model(x).argmax(1).item()
                act = ["HOLD","BUY","SELL"][pred]
                if act=="BUY"  and inv_a>0:  act="HOLD"
                if act=="SELL" and inv_a==0: act="HOLD"
                return act
        except: pass
        return "HOLD"

    def decide(self, row):
        sp    = float(row.get("spread",   0.01))
        ret   = float(row.get("returns",  0.0))
        vol   = float(row.get("volatility",0.0))
        imb   = float(row.get("imbalance", 0.0))
        dep   = float(row.get("depth_imbalance",0.0))
        prs   = float(row.get("microstructure_pressure",0.7*imb+0.3*dep))
        asset = str(row.get("asset","BTC"))
        inv_a = self.inv.get(asset,0)
        held  = self.hold_ticks.get(asset,999)
        inv_n = sum(abs(v) for v in self.inv.values())/6.0

        # ── Alpha + Sentiment ─────────────────────────────────────────────
        alpha=0.0; alpha_signal="HOLD"; regime="SIDEWAYS"
        try:
            if ALPHA_ENABLED:
                sent         = SENTIMENT.get(asset)
                news_score   = float(sent.get("sentiment_score",0.0))
                news_conf    = float(sent.get("confidence",0.0))
                ad           = ALPHA_SIGNAL.generate(row=row,
                                   sentiment_score=news_score,
                                   sentiment_confidence=news_conf)
                alpha        = float(ad.get("alpha",0.0))
                alpha_signal = str(ad.get("signal","HOLD"))
                regime       = str(ad.get("regime","SIDEWAYS"))
        except: pass

        self._last_alpha  = alpha
        self._last_regime = regime
        self._last_source = "RULE"

        # Regime filter
        bearish = regime in ("BEAR","HIGH_VOL","HIGH_VOL_BEAR","BEARISH")
        if regime=="HIGH_VOL" and inv_a==0: return "HOLD"

        # ── Spread guard: block entry on dangerously wide spreads ────────
        # BTC/ETH with spread > $8 = extreme volatility, skip entry
        # This prevents buying into a spike that will immediately SL
        MAX_SPREAD = {"BTC": 8.0, "ETH": 1.5, "AAPL": 0.5,
                      "AMZN": 0.5, "GOOG": 0.8, "MSFT": 0.3,
                      "INTC": 0.1, "ADA": 0.005}
        if inv_a == 0 and sp > MAX_SPREAD.get(asset, 999):
            return "HOLD"  # spread too wide — skip this entry

        # ── Build observation EXACTLY matching RealLOBTradingEnv._get_observation()
        # Training used 7 features: spread, returns, volatility,
        # order_flow_imbalance, depth_imbalance, microstructure_pressure, inventory/6
        # We must match this EXACTLY — wrong inputs = model always predicts HOLD

        # obs[0]: raw spread (NOT normalised — training used raw dollar spread)
        raw_spread = sp  # already read above as float(row.get("spread", 0.01))

        # obs[1]: returns (pct_change of mid_price — same as training)
        raw_ret = ret  # float(row.get("returns", 0.0))

        # obs[2]: volatility (rolling std of returns — same as training)
        raw_vol = vol  # float(row.get("volatility", 0.0))

        # obs[3]: order_flow_imbalance = (buys - sells) / (buys + sells)
        # Training computed this from actual buy/sell counts
        # Parquet has f_ofi or f_imbalance as best proxy
        raw_ofi = float(row.get("f_ofi",
                    row.get("order_flow_imbalance",
                    row.get("f_imbalance", imb))))
        raw_ofi = float(np.clip(raw_ofi, -1.0, 1.0))

        # obs[4]: depth_imbalance = (bid_depth - ask_depth) / total_depth
        raw_dep = float(row.get("depth_imbalance",
                    row.get("f_depth_imbalance",
                    dep)))
        raw_dep = float(np.clip(raw_dep, -1.0, 1.0))

        # obs[5]: microstructure_pressure = 0.5*ofi + 0.5*depth_imb
        raw_prs = 0.5 * raw_ofi + 0.5 * raw_dep

        # obs[6]: inventory / max_inventory (6) — SINGLE ASSET, not sum of all
        raw_inv = float(inv_a) / 6.0  # inv_a = inventory for THIS asset only

        obs_scaled = np.array([
            raw_spread,   # obs[0]: raw spread  (e.g. BTC=6.65, MSFT=0.01)
            raw_ret,      # obs[1]: returns
            raw_vol,      # obs[2]: volatility
            raw_ofi,      # obs[3]: order_flow_imbalance [-1, 1]
            raw_dep,      # obs[4]: depth_imbalance [-1, 1]
            raw_prs,      # obs[5]: microstructure_pressure
            raw_inv,      # obs[6]: inventory/6  (0.0 to 1.0)
        ], dtype=np.float32)

        # ── Try trained models ────────────────────────────────────────────
        LSTM_FEATS = {
            'f_returns','f_log_return','f_ofi','f_imbalance','f_depth_imbalance',
            'f_pressure','f_buy_ratio','f_sell_ratio','f_return_5','f_return_10',
            'f_return_20','f_return_60','f_accel','f_jerk','f_vol_5','f_vol_10',
            'f_vol_20','f_vol_60','f_zscore_20','f_zscore_60','f_dev_vwap',
            'f_bollinger_pct','f_effective_spread','f_price_impact','f_bid_ask_bounce',
            'f_trade_sign','f_kyle_lambda','f_size_ratio','f_trend_strength',
            'f_vol_regime','f_spread_regime','f_liquidity_score','f_prev_return',
            'f_prev_spread','f_prev_imbalance','f_time_of_day','f_spread_pct','f_volatility',
        }

        model_act = "HOLD"
        if   self.name in ("DQN","PPO"):   model_act = self._try_rl_model(obs_scaled, inv_a)
        elif self.name == "XGBoost":       model_act = self._try_xgb_model(row, inv_a)
        elif self.name == "LSTM":          model_act = self._try_seq_model(row, inv_a, 10, LSTM_FEATS)  # 10 ticks enough to predict
        elif self.name == "Transformer":   model_act = self._try_seq_model(row, inv_a, 5,  LSTM_FEATS)  # 5 ticks enough to predict

        if model_act != "HOLD":
            self._last_source = "MODEL"
            return model_act

        # ── Alpha signal as secondary (only when model says HOLD) ─────────
        # Alpha fires on strong conviction only — no rule fallback
        if alpha_signal=="BUY"  and alpha>0.35  and inv_a==0 and not bearish:
            self._last_source="ALPHA"; return "BUY"
        if alpha_signal=="SELL" and alpha<-0.35 and inv_a>0:
            self._last_source="ALPHA"; return "SELL"

        # ── No rule-based fallback — AI models or HOLD ────────────────────
        return "HOLD"

# ── Build simulation JSON ──────────────────────────────────────────────────────
def build_sim_json(n_ticks):
    pb = st.progress(0, text="⏳ Loading models...")
    models = load_models()
    pb.progress(25, text="⏳ Loading market data...")

    # ── Load real market data from 50-lakh dataset ──────────────────────
    df = None
    seed_now = int(time.time()*1000) % 9_999_999

    try:
        SDIR     = ROOT / "data" / "splits"
        rng_data = np.random.default_rng(seed_now)
        import fastparquet as _fp

        # Priority: use test split (unseen during training)
        # Fallback chain: test_10_new → test_10 → val_20 → train_70
        candidates = [
            SDIR / "test_10_new.parquet",
            SDIR / "test_10.parquet",
            SDIR / "val_20.parquet",
            SDIR / "train_70.parquet",
        ]
        parquet_path = next((p for p in candidates if p.exists()), None)

        if parquet_path is None:
            raise FileNotFoundError("No parquet splits found in data/splits/")

        pb.progress(28, text=f"⏳ Reading {parquet_path.name} ({parquet_path.stat().st_size//1_048_576} MB)...")
        full_df = _fp.ParquetFile(str(parquet_path)).to_pandas()
        total_rows = len(full_df)

        pb.progress(32, text=f"⏳ {total_rows:,} rows loaded · sampling {n_ticks} unseen ticks...")

        assets = list(full_df["asset"].unique())
        rng_data.shuffle(assets)   # different asset order every run

        rows_per_asset = max(200, n_ticks // max(len(assets), 1))

        frames = []
        for a in assets:
            adf = full_df[full_df["asset"] == a].reset_index(drop=True)
            if len(adf) < rows_per_asset:
                frames.append(adf)
                continue
            # Random window anywhere in the full asset history
            max_start = len(adf) - rows_per_asset
            start     = int(rng_data.integers(0, max_start + 1))
            frames.append(adf.iloc[start : start + rows_per_asset].copy())

        df = pd.concat(frames, ignore_index=True)

        # Interleave assets by shuffling rows so simulation sees mixed assets
        df = df.sample(frac=1, random_state=seed_now).reset_index(drop=True)
        # Trim to exactly n_ticks rows so progress bar is always correct
        df = df.head(n_ticks).reset_index(drop=True)

        # ── Normalise column names ─────────────────────────────────────
        col_map = {
            "f_returns"  : "returns",
            "f_volatility": "volatility",
            "f_imbalance": "imbalance",
            "f_spread"   : "spread",
            "f_ofi"      : "order_flow_imbalance",
            "f_pressure" : "microstructure_pressure",
            "f_depth_imbalance": "depth_imbalance",
        }
        for src, dst in col_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        # Ensure required columns exist
        for col in ["returns","volatility","imbalance","spread",
                    "mid_price","microstructure_pressure","depth_imbalance"]:
            if col not in df.columns:
                df[col] = 0.0

        # Ensure mid_price is valid
        if df["mid_price"].isna().all() or (df["mid_price"] == 0).all():
            if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
                df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0

        df = df.fillna(0.0).reset_index(drop=True)
        pb.progress(38, text=f"⏳ {len(df):,} ticks sampled from {parquet_path.name} (seed={seed_now})")

    except Exception as _data_err:
        df = None
        pb.progress(38, text=f"⚠️ Real data unavailable ({_data_err.__class__.__name__}) — using synthetic")

    if df is None or len(df) < 10:
        pb.progress(38, text="⏳ Generating synthetic market data...")
        df = make_lob_data(n_ticks, seed=seed_now)
    pb.progress(40, text=f"⏳ Running {len(df):,} ticks across 5 agents...")

    traders   = {n:Trader(n,models[n]) for n in AGENT_COLORS}
    price_map = {"BTC":50000.0,"ETH":2000.0,"ADA":1.2,"AAPL":180.0,
                 "AMZN":180.0,"GOOG":140.0,"INTC":35.0,"MSFT":380.0}
    # ── Your strategy: confident-first entry for volatile assets ──
    # State tracking across ticks
    asset_last_price   = {}   # {asset: price at previous tick}
    asset_first_buyer  = {}   # {asset: agent_name} — who entered first this signal
    asset_second_done  = {}   # {asset: bool} — has random 2nd agent entered yet
    VOLATILE_ASSETS    = {"BTC", "ETH"}
    ticks_js=[]; pfs_js=[]; trades_js=[]

    def safe(row,*cols,default=0.0):
        for c in cols:
            try:
                v=float(row[c])
                if v==v and abs(v)<1e15: return v
            except: pass
        return default

    for i in range(len(df)):
        try:
            row    = df.iloc[i]
            asset  = str(row.get("asset","BTC") if hasattr(row,"get") else row["asset"])
            price  = safe(row,"mid_price",default=100.0)
            spread = safe(row,"spread","f_spread",default=price*0.0002)
            ret    = safe(row,"returns","f_returns","f_return_5")
            imb    = safe(row,"imbalance","f_imbalance","order_flow_imbalance")
            vol    = safe(row,"volatility","f_volatility","f_vol_20")
            prs    = safe(row,"microstructure_pressure","f_pressure")
            if price<=0: price=100.0
            if spread<=0: spread=price*0.0002
            price_map[asset]=price
            row_dict={"asset":asset,"mid_price":price,"spread":spread,
                      "returns":ret,"imbalance":imb,"volatility":vol,
                      "microstructure_pressure":prs,"s":i,
                      "depth_imbalance":safe(row,"depth_imbalance","f_depth_imbalance")}
            for col in df.columns:
                if col not in row_dict:
                    try: row_dict[col]=float(row[col])
                    except: pass
            row=row_dict
            ticks_js.append({"s":i,"ast":asset,"p":round(price,4),
                "sp":round(spread,5),"r":round(ret,6),"imb":round(imb,4),"vol":round(vol,6)})
            pfs_js.append({n:t.portfolio_value(price_map) for n,t in traders.items()})
            # Track price movement vs previous tick for volatile assets
            asset_last_price[asset] = asset_last_price.get(asset, price)
        except: continue

        for name,trader in traders.items():
            for a in list(trader.hold_ticks.keys()):
                trader.hold_ticks[a]=trader.hold_ticks.get(a,0)+1

            # SL/TP
            sl_hit,sl_pnl=trader.check_sl_tp(asset,price,spread)
            if sl_hit:
                cost2=max(spread/2,price*0.0001)
                bid2=round(price-spread/2,5); ask2=round(price+spread/2,5)
                tag="TP" if sl_pnl>0 else "SL"
                trader.cash+=price-cost2; trader.inv[asset]=0
                trader.pnl+=sl_pnl; trader.hold_ticks[asset]=0; trader.consec[asset]=0
                if asset in trader.entry: del trader.entry[asset]
                trader.trades+=1
                if sl_pnl>0: trader.wins+=1
                res="Profit" if sl_pnl>0 else "Loss"
                trades_js.append({"s":i,"tid":name,"ast":asset,"a":"SELL",
                    "p":round(price,4),"bid":bid2,"ask":ask2,"spread":round(spread,5),
                    "cost":round(cost2,5),"slip":round(spread*0.05,6),"pnl":round(sl_pnl,4),
                    "result":res,"buyer":"Market","seller":name,
                    "reason":f"{tag} auto-exit | {name} on {asset} | regime={trader._last_regime}",
                    "lat":round(np.random.uniform(0.1,0.5),2),
                    "pv":round(trader.portfolio_value(price_map),2)})
                continue

            try: action=trader.decide(row)
            except: action="HOLD"

            # ── Trend-following strategy for BTC/ETH ──────────────────────
            #
            # ENTRY RULES:
            #   Tick N:   Signal fires → MOST CONFIDENT agent buys (highest alpha)
            #             All others HOLD
            #   Tick N+1: Price ROSE vs tick N →
            #             The holding agent SELLS for profit (locked)
            #             ONE new free agent BUYS at confirmed higher price
            #             (trend confirmed = fresh entry with same 0.06% SL)
            #   Tick N+1: Price FELL vs tick N →
            #             No new entries
            #             Existing positions stay OPEN (wait for recovery)
            #             SL exits automatically if loss > 0.06%
            #
            # This way: agents ride the uptrend one at a time, each taking
            # a quick profit and handing off to the next agent

            if asset in VOLATILE_ASSETS:
                prev_price  = asset_last_price.get(asset, price)
                price_rose  = price > prev_price   # strictly rose
                price_fell  = price < prev_price   # strictly fell
                all_agents  = list(traders.keys())

                # Find agents currently holding this asset
                holders     = [a for a in all_agents if traders[a].inv.get(asset,0) > 0]
                free_agents = [a for a in all_agents if traders[a].inv.get(asset,0) == 0]

                if action == "BUY":
                    if len(holders) == 0:
                        # ── Nobody holding — find MOST CONFIDENT agent ──────
                        # Most confident = highest alpha score this tick
                        # Build alpha scores for all agents
                        # (we use _last_alpha from the decide() call already done)
                        # Since we are inside the per-agent loop, we pick the agent
                        # with highest alpha by comparing with a shared tracker
                        my_alpha = abs(getattr(trader, "_last_alpha", 0.0))
                        best_alpha = asset_first_buyer.get(f"{asset}_best_alpha", -1)

                        if my_alpha > best_alpha:
                            # This agent is currently the most confident
                            asset_first_buyer[f"{asset}_best_alpha"] = my_alpha
                            asset_first_buyer[f"{asset}_best_name"]  = name
                            # Don't enter yet — wait until all agents have been
                            # evaluated this tick (handled below after loop)
                        action = "HOLD"  # will be overridden for winner below

                    elif price_fell:
                        # Price fell → no new entries, existing stay open
                        action = "HOLD"

                    elif price_rose and len(holders) == 1:
                        # Price rose + only 1 holder →
                        # That holder will SELL (handled in SELL block below)
                        # ONE free agent now enters at confirmed higher price
                        # Pick the free agent with highest alpha
                        my_alpha = abs(getattr(trader, "_last_alpha", 0.0))
                        best_free_alpha = asset_second_done.get(f"{asset}_best_alpha", -1)
                        if name in free_agents and my_alpha > best_free_alpha:
                            asset_second_done[f"{asset}_best_alpha"] = my_alpha
                            asset_second_done[f"{asset}_best_name"]  = name
                        action = "HOLD"  # winner set below

                    else:
                        # Already 2+ holders or no clear signal → HOLD
                        action = "HOLD"

                elif action == "SELL":
                    # Selling is fine — agents exit on their own signal
                    # No restriction on SELL
                    pass

            # ── After all agents evaluated: grant BUY to the winner ────────
            # For the most confident first-entry
            if asset in VOLATILE_ASSETS:
                winner_first = asset_first_buyer.get(f"{asset}_best_name")
                if winner_first == name and traders[name].inv.get(asset,0) == 0:
                    action = "BUY"   # this agent won the confidence contest
                    # Clear so next tick starts fresh
                    asset_first_buyer.pop(f"{asset}_best_alpha", None)
                    asset_first_buyer.pop(f"{asset}_best_name",  None)

                winner_second = asset_second_done.get(f"{asset}_best_name")
                if winner_second == name and traders[name].inv.get(asset,0) == 0:
                    action = "BUY"   # this agent enters on confirmed trend
                    asset_second_done.pop(f"{asset}_best_alpha", None)
                    asset_second_done.pop(f"{asset}_best_name",  None)

            # Update last price
            if asset in VOLATILE_ASSETS:
                asset_last_price[asset] = price

            # Reset all tracking when no one holds this asset
            if asset in VOLATILE_ASSETS:
                any_holding = any(traders[a].inv.get(asset,0)>0 for a in traders)
                if not any_holding:
                    for k in list(asset_first_buyer.keys()):
                        if k.startswith(asset): asset_first_buyer.pop(k, None)
                    for k in list(asset_second_done.keys()):
                        if k.startswith(asset): asset_second_done.pop(k, None)

            if action in ("BUY","SELL"):
                try: ok,pnl,bid,ask,counterparty,result=trader.execute(action,asset,price,spread)
                except: ok=False;pnl=bid=ask=0.0;counterparty="-";result="-"
                if ok:
                    reason=trader.get_strategy_reason(action,asset,prs,
                        getattr(trader,"_last_alpha",0.0),
                        getattr(trader,"_last_regime","UNKNOWN"),
                        getattr(trader,"_last_source","RULE"))
                    trades_js.append({"s":i,"tid":name,"ast":asset,"a":action,
                        "p":round(price,4),"bid":round(bid,5),"ask":round(ask,5),
                        "spread":round(spread,5),"cost":round(max(spread/2,price*0.0001),5),
                        "slip":round(spread*0.05,6),"pnl":round(pnl,4),"result":result,
                        "buyer":name if action=="BUY" else "Market",
                        "seller":name if action=="SELL" else "Market",
                        "reason":reason,
                        "lat":round(np.random.uniform(1,3) if name in("DQN","PPO")
                              else np.random.uniform(100,2000),1),
                        "pv":round(trader.portfolio_value(price_map),2)})

        if i%max(1,len(df)//10)==0:
            pct = min(95, 40 + int(i/max(len(df),1)*55))
            pb.progress(pct, text=f"⏳ Tick {i}/{len(df)} | Trades: {len(trades_js)}")

    # Force-close open positions
    for name,trader in traders.items():
        for asset,inv in list(trader.inv.items()):
            if inv>0:
                price2 = price_map.get(asset,100.0)
                spread2= price2*0.0004
                cost2  = max(spread2/2,price2*0.0001)
                bid2   = price2-spread2/2
                entry2 = trader.entry.get(asset,price2)
                pnl2   = bid2-entry2-cost2
                trader.cash+=bid2-cost2; trader.inv[asset]=0; trader.pnl+=pnl2
                trader.trades+=1
                if pnl2>0: trader.wins+=1
                res="Profit" if pnl2>0 else "Loss"
                trades_js.append({"s":len(ticks_js)-1,"tid":name,"ast":asset,"a":"SELL",
                    "p":round(price2,4),"bid":round(bid2,5),"ask":round(price2+spread2/2,5),
                    "spread":round(spread2,5),"cost":round(cost2,5),"slip":round(spread2*0.05,6),
                    "pnl":round(pnl2,4),"result":res,"buyer":"Market","seller":name,
                    "reason":f"📐 EOD-CLOSE | SELL | {name} forced close | α=+0.000 | 🔚 END | {asset}",
                    "lat":0.1,"pv":round(trader.portfolio_value(price_map),2)})

    pb.progress(98, text="⏳ Finalising...")
    st.caption(f"✅ {len(ticks_js)} ticks · {len(trades_js)} trades · {len(pfs_js)} snapshots")

    traders_js=[]
    for name,t in traders.items():
        pv   = t.portfolio_value(price_map)
        hist = [p[name] for p in pfs_js]
        rets = np.diff(hist)/np.maximum(np.array(hist[:-1]),1)
        sh   = float(np.mean(rets)/np.std(rets)*np.sqrt(252)) if np.std(rets)>1e-9 else 0.0
        peak = max(hist) if hist else INIT
        dd   = round((1-pv/peak)*100,2) if peak>0 else 0.0
        traders_js.append({"id":name,"color":AGENT_COLORS[name],
            "finalPv":round(pv,2),"pnl":round(pv-INIT,2),
            "nTrades":t.trades,"wins":t.wins,
            "winRate":round(t.wins/max(t.trades,1)*100,1),
            "sharpe":round(max(-99.0,min(99.0,sh)),3),"dd":dd,
            "lat":round(np.random.uniform(1,3) if name in("DQN","PPO")
                        else np.random.uniform(100,2000),1)})

    pb.progress(100,text=f"✅ Done! {len(ticks_js)} ticks · {len(trades_js)} trades")
    time.sleep(0.4); pb.empty()
    return json.dumps({"ticks":ticks_js,"portfolios":pfs_js,"trades":trades_js,
        "traders":traders_js,"meta":{"total_ticks":len(ticks_js),
        "total_trades":len(trades_js),"assets":len(ASSETS)},"initCash":INIT})

# ── Live simulation builder ────────────────────────────────────────────────────
def build_sim_json_live(n_ticks: int) -> str:
    """
    Live Data mode pipeline:
      1. Process all files in data/live_data/ → normalise → push to Kafka
      2. Pull ticks back from Kafka consumer
      3. Run the same simulation loop as build_sim_json() on the live DataFrame
    Falls back to synthetic data if Kafka is unreachable.
    """
    from kafka_consumer import kafka_available, consume_ticks
    from src.live.live_data_processor import process_live_folder

    pb = st.progress(0, text="📡 Initialising live data pipeline...")

    # ── Step 1: load models (same as default mode) ────────────────────────
    pb.progress(5, text="📡 Loading AI models...")
    models = load_models()

    # ── Step 2: process files → Kafka ─────────────────────────────────────
    pb.progress(15, text="📡 Processing uploaded files → Kafka...")
    df_live = None
    live_files_used = []
    live_rows_sent  = 0

    try:
        if not kafka_available():
            st.warning(
                "⚠️ Kafka broker not reachable at localhost:9092. "
                "Start it with: `docker compose --profile kafka up`\n\n"
                "Falling back to direct file processing (no Kafka)."
            )
            # Direct path: normalise files and use them without Kafka
            from src.live.live_data_processor import (
                process_live_folder as _plf,
                LIVE_DIR, _read_file, normalise,
            )
            files = sorted([
                f for f in LIVE_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in (".csv", ".json")
                and not f.name.startswith(".")
            ])
            if not files:
                raise FileNotFoundError("No files in data/live_data/")
            frames = []
            for f in files:
                raw = _read_file(f)
                frames.append(normalise(raw, source_name=f.stem))
                live_files_used.append(f.name)
            df_live = pd.concat(frames, ignore_index=True)
            live_rows_sent = len(df_live)
            pb.progress(40, text=f"📡 {live_rows_sent:,} ticks loaded from files (no Kafka)")

        else:
            # Full Kafka path
            pb.progress(20, text="📡 Streaming files to Kafka...")
            combined, sent = process_live_folder(verbose=False)
            live_rows_sent = sent

            pb.progress(35, text=f"📡 {sent:,} ticks sent to Kafka · Reading back...")

            # Pull from Kafka consumer
            df_live = consume_ticks(
                n=max(n_ticks, sent),
                topic="market-ticks",
                timeout_ms=15_000,
            )

            if df_live is None or len(df_live) < 10:
                st.warning("⚠️ Kafka consumer returned no data. Using directly processed files.")
                df_live = combined

            # Track which files were processed (they were moved to .processed/)
            from src.live.live_data_processor import LIVE_DIR
            done_dir = LIVE_DIR / ".processed"
            if done_dir.exists():
                live_files_used = [
                    f.name for f in done_dir.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]

            pb.progress(40, text=f"📡 {len(df_live):,} ticks received from Kafka")

    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        pb.empty()
        return ""
    except Exception as e:
        st.error(f"❌ Live pipeline failed: {e}")
        pb.empty()
        return ""

    # ── Step 3: trim and normalise column names (same as default mode) ────
    df_live = df_live.head(n_ticks).reset_index(drop=True)

    col_map = {
        "f_returns":           "returns",
        "f_volatility":        "volatility",
        "f_imbalance":         "imbalance",
        "f_spread":            "spread",
        "f_ofi":               "order_flow_imbalance",
        "f_pressure":          "microstructure_pressure",
        "f_depth_imbalance":   "depth_imbalance",
    }
    for src, dst in col_map.items():
        if src in df_live.columns and dst not in df_live.columns:
            df_live[dst] = df_live[src]

    for col in ["returns","volatility","imbalance","spread",
                "mid_price","microstructure_pressure","depth_imbalance"]:
        if col not in df_live.columns:
            df_live[col] = 0.0

    df_live = df_live.fillna(0.0).reset_index(drop=True)

    # ── Step 4: store live run stats in session ────────────────────────────
    st.session_state.live_rows  = live_rows_sent
    st.session_state.live_files = live_files_used

    pb.progress(45, text=f"📡 Running {len(df_live):,} live ticks across 5 agents...")

    # ── Step 5: run the exact same simulation loop ─────────────────────────
    # Reuse build_sim_json internals by patching the dataframe source.
    # We do this by calling _run_simulation_loop() which is the inner core.
    traders   = {n: Trader(n, models[n]) for n in AGENT_COLORS}
    price_map = {"BTC":50000.0,"ETH":2000.0,"ADA":1.2,"AAPL":180.0,
                 "AMZN":180.0,"GOOG":140.0,"INTC":35.0,"MSFT":380.0}

    asset_last_price  = {}
    asset_first_buyer = {}
    asset_second_done = {}
    VOLATILE_ASSETS   = {"BTC", "ETH"}
    ticks_js=[]; pfs_js=[]; trades_js=[]

    def safe(row, *cols, default=0.0):
        for c in cols:
            try:
                v = float(row[c])
                if v == v and abs(v) < 1e15: return v
            except: pass
        return default

    for i in range(len(df_live)):
        try:
            row    = df_live.iloc[i]
            asset  = str(row.get("asset","BTC") if hasattr(row,"get") else row["asset"])
            price  = safe(row,"mid_price",default=100.0)
            spread = safe(row,"spread","f_spread",default=price*0.0002)
            ret    = safe(row,"returns","f_returns","f_return_5")
            imb    = safe(row,"imbalance","f_imbalance","order_flow_imbalance")
            vol    = safe(row,"volatility","f_volatility","f_vol_20")
            prs    = safe(row,"microstructure_pressure","f_pressure")
            if price  <= 0: price  = 100.0
            if spread <= 0: spread = price * 0.0002
            price_map[asset] = price
            row_dict = {
                "asset":asset,"mid_price":price,"spread":spread,
                "returns":ret,"imbalance":imb,"volatility":vol,
                "microstructure_pressure":prs,"s":i,
                "depth_imbalance":safe(row,"depth_imbalance","f_depth_imbalance"),
            }
            for col in df_live.columns:
                if col not in row_dict:
                    try: row_dict[col] = float(row[col])
                    except: pass
            row = row_dict
            ticks_js.append({"s":i,"ast":asset,"p":round(price,4),
                "sp":round(spread,5),"r":round(ret,6),"imb":round(imb,4),"vol":round(vol,6)})
            pfs_js.append({n:t.portfolio_value(price_map) for n,t in traders.items()})
            asset_last_price[asset] = asset_last_price.get(asset, price)
        except: continue

        for name, trader in traders.items():
            for a in list(trader.hold_ticks.keys()):
                trader.hold_ticks[a] = trader.hold_ticks.get(a,0) + 1

            sl_hit, sl_pnl = trader.check_sl_tp(asset, price, spread)
            if sl_hit:
                cost2  = max(spread/2, price*0.0001)
                bid2   = round(price-spread/2,5); ask2 = round(price+spread/2,5)
                tag    = "TP" if sl_pnl > 0 else "SL"
                trader.cash += price - cost2; trader.inv[asset] = 0
                trader.pnl  += sl_pnl; trader.hold_ticks[asset] = 0
                trader.consec[asset] = 0
                if asset in trader.entry: del trader.entry[asset]
                trader.trades += 1
                if sl_pnl > 0: trader.wins += 1
                res = "Profit" if sl_pnl > 0 else "Loss"
                trades_js.append({"s":i,"tid":name,"ast":asset,"a":"SELL",
                    "p":round(price,4),"bid":bid2,"ask":ask2,"spread":round(spread,5),
                    "cost":round(cost2,5),"slip":round(spread*0.05,6),"pnl":round(sl_pnl,4),
                    "result":res,"buyer":"Market","seller":name,
                    "reason":f"{tag} auto-exit | {name} on {asset} [LIVE]",
                    "lat":round(np.random.uniform(0.1,0.5),2),
                    "pv":round(trader.portfolio_value(price_map),2)})
                continue

            try: action = trader.decide(row)
            except: action = "HOLD"

            if asset in VOLATILE_ASSETS:
                prev_price = asset_last_price.get(asset, price)
                price_rose = price > prev_price
                price_fell = price < prev_price
                all_agents = list(traders.keys())
                holders    = [a for a in all_agents if traders[a].inv.get(asset,0) > 0]
                free_agents= [a for a in all_agents if traders[a].inv.get(asset,0) == 0]

                if action == "BUY":
                    if len(holders) == 0:
                        my_alpha   = abs(getattr(trader,"_last_alpha",0.0))
                        best_alpha = asset_first_buyer.get(f"{asset}_best_alpha",-1)
                        if my_alpha > best_alpha:
                            asset_first_buyer[f"{asset}_best_alpha"] = my_alpha
                            asset_first_buyer[f"{asset}_best_name"]  = name
                        action = "HOLD"
                    elif price_fell:
                        action = "HOLD"
                    elif price_rose and len(holders) == 1:
                        my_alpha       = abs(getattr(trader,"_last_alpha",0.0))
                        best_free_alpha= asset_second_done.get(f"{asset}_best_alpha",-1)
                        if name in free_agents and my_alpha > best_free_alpha:
                            asset_second_done[f"{asset}_best_alpha"] = my_alpha
                            asset_second_done[f"{asset}_best_name"]  = name
                        action = "HOLD"
                    else:
                        action = "HOLD"

            if asset in VOLATILE_ASSETS:
                winner_first = asset_first_buyer.get(f"{asset}_best_name")
                if winner_first == name and traders[name].inv.get(asset,0) == 0:
                    action = "BUY"
                    asset_first_buyer.pop(f"{asset}_best_alpha",None)
                    asset_first_buyer.pop(f"{asset}_best_name", None)
                winner_second = asset_second_done.get(f"{asset}_best_name")
                if winner_second == name and traders[name].inv.get(asset,0) == 0:
                    action = "BUY"
                    asset_second_done.pop(f"{asset}_best_alpha",None)
                    asset_second_done.pop(f"{asset}_best_name", None)

            if asset in VOLATILE_ASSETS:
                asset_last_price[asset] = price
                any_holding = any(traders[a].inv.get(asset,0) > 0 for a in traders)
                if not any_holding:
                    for k in list(asset_first_buyer.keys()):
                        if k.startswith(asset): asset_first_buyer.pop(k,None)
                    for k in list(asset_second_done.keys()):
                        if k.startswith(asset): asset_second_done.pop(k,None)

            if action in ("BUY","SELL"):
                try: ok,pnl,bid,ask,counterparty,result = trader.execute(action,asset,price,spread)
                except: ok=False; pnl=bid=ask=0.0; counterparty="-"; result="-"
                if ok:
                    reason = trader.get_strategy_reason(
                        action, asset, prs,
                        getattr(trader,"_last_alpha",0.0),
                        getattr(trader,"_last_regime","UNKNOWN"),
                        getattr(trader,"_last_source","RULE"),
                    )
                    trades_js.append({"s":i,"tid":name,"ast":asset,"a":action,
                        "p":round(price,4),"bid":round(bid,5),"ask":round(ask,5),
                        "spread":round(spread,5),"cost":round(max(spread/2,price*0.0001),5),
                        "slip":round(spread*0.05,6),"pnl":round(pnl,4),"result":result,
                        "buyer":name if action=="BUY" else "Market",
                        "seller":name if action=="SELL" else "Market",
                        "reason":reason + " [LIVE]",
                        "lat":round(np.random.uniform(1,3) if name in ("DQN","PPO")
                              else np.random.uniform(100,2000),1),
                        "pv":round(trader.portfolio_value(price_map),2)})

        if i % max(1, len(df_live)//10) == 0:
            pct = min(95, 45 + int(i/max(len(df_live),1)*50))
            pb.progress(pct, text=f"📡 Tick {i}/{len(df_live)} | Trades: {len(trades_js)}")

    # Force-close open positions
    for name, trader in traders.items():
        for asset, inv in list(trader.inv.items()):
            if inv > 0:
                price2  = price_map.get(asset, 100.0)
                spread2 = price2 * 0.0004
                cost2   = max(spread2/2, price2*0.0001)
                bid2    = price2 - spread2/2
                entry2  = trader.entry.get(asset, price2)
                pnl2    = bid2 - entry2 - cost2
                trader.cash += bid2 - cost2; trader.inv[asset] = 0; trader.pnl += pnl2
                trader.trades += 1
                if pnl2 > 0: trader.wins += 1
                res = "Profit" if pnl2 > 0 else "Loss"
                trades_js.append({"s":len(ticks_js)-1,"tid":name,"ast":asset,"a":"SELL",
                    "p":round(price2,4),"bid":round(bid2,5),"ask":round(price2+spread2/2,5),
                    "spread":round(spread2,5),"cost":round(cost2,5),"slip":round(spread2*0.05,6),
                    "pnl":round(pnl2,4),"result":res,"buyer":"Market","seller":name,
                    "reason":f"📐 EOD-CLOSE | {name} forced close [LIVE]",
                    "lat":0.1,"pv":round(trader.portfolio_value(price_map),2)})

    pb.progress(98, text="📡 Finalising live simulation...")
    st.caption(
        f"📡 LIVE MODE · {len(ticks_js)} ticks · {len(trades_js)} trades · "
        f"{live_rows_sent:,} rows streamed via Kafka"
    )

    traders_js = []
    for name, t in traders.items():
        pv   = t.portfolio_value(price_map)
        hist = [p[name] for p in pfs_js]
        rets = np.diff(hist)/np.maximum(np.array(hist[:-1]),1)
        sh   = float(np.mean(rets)/np.std(rets)*np.sqrt(252)) if np.std(rets)>1e-9 else 0.0
        peak = max(hist) if hist else INIT
        dd   = round((1-pv/peak)*100,2) if peak > 0 else 0.0
        traders_js.append({"id":name,"color":AGENT_COLORS[name],
            "finalPv":round(pv,2),"pnl":round(pv-INIT,2),
            "nTrades":t.trades,"wins":t.wins,
            "winRate":round(t.wins/max(t.trades,1)*100,1),
            "sharpe":round(max(-99.0,min(99.0,sh)),3),"dd":dd,
            "lat":round(np.random.uniform(1,3) if name in ("DQN","PPO")
                        else np.random.uniform(100,2000),1)})

    pb.progress(100, text=f"✅ Live simulation done! {len(ticks_js)} ticks · {len(trades_js)} trades")
    time.sleep(0.4); pb.empty()

    return json.dumps({
        "ticks":ticks_js,"portfolios":pfs_js,"trades":trades_js,
        "traders":traders_js,
        "meta":{
            "total_ticks":   len(ticks_js),
            "total_trades":  len(trades_js),
            "assets":        len(ASSETS),
            "mode":          "LIVE",
            "kafka_rows":    live_rows_sent,
            "source_files":  live_files_used,
        },
        "initCash": INIT,
    })


# ── Run button ─────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.live_mode = False
    st.session_state.show_live_uploader = False
    sim_json = build_sim_json(st.session_state.rows)
    if sim_json:
        st.session_state.sim_data  = sim_json
        st.session_state.sim_ready = True
        st.rerun()

if live_btn:
    # Check if there are actual files in live_data/ to process
    queued = [
        f for f in LIVE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".csv", ".json")
        and not f.name.startswith(".")
    ] if LIVE_DIR.exists() else []

    if queued:
        # Files available — run live pipeline through Kafka
        st.session_state.live_mode = True
        sim_json = build_sim_json_live(st.session_state.rows)
    else:
        # No file uploaded — fall back to the previous default execution method
        st.session_state.live_mode = False
        st.warning(
            "⚠️ No file was uploaded. Running on the previous default execution method "
            "(parquet splits → synthetic data fallback)."
        )
        sim_json = build_sim_json(st.session_state.rows)

    if sim_json:
        st.session_state.sim_data  = sim_json
        st.session_state.sim_ready = True
        st.rerun()

# ── HTML frontend ──────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--void:#060b18;--panel:#0d1117;--border:#1e3a5f;--cyan:#00ffe7;--green:#39ff14;
  --pink:#ff2d78;--amber:#ffaa00;--text:#ffffff;--dim:#b0c4de;--mono:'Share Tech Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--void);color:var(--text);font-family:'Rajdhani',sans-serif;font-size:16px;overflow-x:hidden}
#app{min-height:100vh;padding-bottom:30px}
#fsBtn{position:fixed;top:10px;right:14px;z-index:9999;background:rgba(0,255,231,.08);
  border:1px solid var(--cyan);color:var(--cyan);font-family:var(--mono);font-size:.9rem;
  padding:6px 16px;border-radius:3px;cursor:pointer;letter-spacing:1px;transition:all .15s}
#fsBtn:hover{background:var(--cyan);color:#000}
.hdr{background:linear-gradient(90deg,#0a0f1e,#0d2040);border-bottom:1px solid var(--border);
  padding:14px 22px;display:flex;align-items:center;gap:14px}
.hdr h1{font-size:1.8rem;font-weight:700;color:var(--cyan);letter-spacing:2px}
.hdr-meta{font-family:var(--mono);font-size:.9rem;color:var(--dim);letter-spacing:1px}
.lb-dot{width:11px;height:11px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.badge{font-family:var(--mono);font-size:.8rem;padding:3px 9px;border-radius:2px;
  letter-spacing:1px;font-weight:700;background:rgba(0,255,231,.15);
  color:var(--cyan);border:1px solid var(--cyan)}
.tabs{display:flex;background:var(--panel);border-bottom:1px solid var(--border)}
.tab{font-family:var(--mono);font-size:.95rem;padding:10px 24px;cursor:pointer;
  color:var(--dim);border-bottom:3px solid transparent;transition:all .15s;letter-spacing:1px}
.tab:hover{color:var(--cyan)}
.tab.active{color:var(--cyan);border-bottom-color:var(--cyan);background:rgba(0,255,231,.04)}
.tab-content{display:none}.tab-content.active{display:block}
.ctrl{display:flex;align-items:center;gap:10px;padding:10px 22px;background:var(--panel);
  border-bottom:1px solid var(--border);flex-wrap:wrap}
.btn{font-family:var(--mono);font-size:.92rem;padding:7px 18px;border-radius:3px;
  cursor:pointer;letter-spacing:1px;text-transform:uppercase;transition:all .12s;border:1px solid}
.btn-cyan{border-color:var(--cyan);color:var(--cyan);background:rgba(0,255,231,.06)}
.btn-cyan:hover{background:var(--cyan);color:#000}
.btn-pink{border-color:var(--pink);color:var(--pink);background:rgba(255,45,120,.06)}
.btn-pink:hover{background:var(--pink);color:#fff}
.btn-amber{border-color:var(--amber);color:var(--amber);background:rgba(255,170,0,.06)}
.btn-amber:hover{background:var(--amber);color:#000}
.cl{font-family:var(--mono);font-size:.85rem;color:var(--dim);letter-spacing:1px}
.cv{font-family:var(--mono);font-size:.95rem;color:var(--cyan);min-width:32px}
input[type=range]{width:110px;accent-color:var(--cyan)}
.prog-wrap{display:flex;align-items:center;gap:8px;flex:1}
.prog-bar{flex:1;background:#0d2040;border:1px solid var(--border);border-radius:2px;height:9px;cursor:pointer}
.prog-fill{height:100%;background:var(--cyan);border-radius:2px;width:0%}
#tickLbl{font-family:var(--mono);font-size:.85rem;color:var(--dim);white-space:nowrap}
.main{padding:14px 18px;display:flex;flex-direction:column;gap:14px}
.lb-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:11px}
.lb-card{background:var(--panel);border-radius:9px;padding:16px;text-align:center;
  border:1px solid var(--border);transition:transform .1s}
.lb-card:hover{transform:translateY(-2px)}
.card-name{font-size:1.15rem;font-weight:700;letter-spacing:1px;margin-bottom:5px}
.card-pv{font-family:var(--mono);font-size:1.6rem;font-weight:700}
.card-pnl{font-family:var(--mono);font-size:.95rem;margin:3px 0}
.card-stats{font-family:var(--mono);font-size:.78rem;color:var(--dim);margin-top:7px;line-height:1.8}
.green{color:var(--green)}.red{color:var(--pink)}
.charts-row{display:grid;grid-template-columns:2fr 1fr;gap:14px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:9px;padding:16px}
.panel-title{font-family:var(--mono);font-size:.9rem;color:var(--cyan);letter-spacing:1.5px;
  text-transform:uppercase;margin-bottom:11px;border-bottom:1px solid var(--border);padding-bottom:7px}
.bottom-row{display:grid;grid-template-columns:3fr 2fr;gap:14px}
.bl-hdr{display:grid;grid-template-columns:100px 55px 60px 55px 90px 100px 80px;
  gap:4px;padding:7px 9px;border-bottom:2px solid rgba(0,255,231,.25);
  font-family:var(--mono);font-size:.78rem;color:#fff;letter-spacing:1.5px;text-transform:uppercase}
.bl-body{max-height:230px;overflow-y:auto}
.bl-row{display:grid;grid-template-columns:100px 55px 60px 55px 90px 100px 80px;
  gap:4px;padding:6px 9px;border-bottom:1px solid rgba(0,255,231,.06);font-family:var(--mono);font-size:.86rem}
.bl-row.nw{animation:flash .45s ease both}
@keyframes flash{0%{background:rgba(0,255,231,.13)}100%{background:transparent}}
.empty-bl{text-align:center;padding:30px;color:var(--dim);font-family:var(--mono);font-size:.86rem;letter-spacing:2px}
.risk-row{background:#0a0f1e;border-left:3px solid;padding:10px 14px;margin-bottom:9px;border-radius:4px}
.risk-name{font-size:1.05rem;font-weight:700;font-family:var(--mono);letter-spacing:1px}
.risk-bar-bg{background:var(--border);height:5px;border-radius:2px;margin:5px 0}
.risk-bar-fg{height:5px;border-radius:2px}
.risk-stats{font-family:var(--mono);font-size:.78rem;color:var(--dim)}
.pnl-row{display:flex;align-items:center;gap:9px;margin-bottom:8px}
.pnl-lbl{width:90px;font-family:var(--mono);font-size:.86rem;text-align:right}
.pnl-bg{flex:1;background:var(--border);border-radius:2px;height:20px;overflow:hidden}
.pnl-fg{height:20px;border-radius:2px;min-width:2px}
.pnl-val{font-family:var(--mono);font-size:.86rem;width:90px}
.sec-ttl{font-family:var(--mono);font-size:.88rem;color:var(--cyan);letter-spacing:2px;
  text-transform:uppercase;margin-bottom:9px;border-bottom:1px solid var(--border);padding-bottom:5px}
.report-wrap{padding:18px}
.report-section{background:var(--panel);border:1px solid var(--border);border-radius:9px;padding:18px;margin-bottom:16px}
.report-h1{font-family:var(--mono);font-size:1.1rem;color:var(--cyan);letter-spacing:2px;
  text-transform:uppercase;margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:8px}
.rtbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.86rem}
.rtbl th{background:#0d2040;color:var(--cyan);padding:8px 12px;text-align:left;font-size:.8rem;letter-spacing:1px;text-transform:uppercase}
.rtbl td{padding:7px 12px;border-bottom:1px solid rgba(0,255,231,.08);color:#fff}
.report-meta{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.meta-box{background:#0a0f1e;border:1px solid var(--border);border-radius:6px;padding:12px;text-align:center}
.meta-val{font-family:var(--mono);font-size:1.4rem;font-weight:700;color:var(--cyan)}
.meta-lbl{font-family:var(--mono);font-size:.75rem;color:var(--dim);margin-top:3px;letter-spacing:1px}
.dl-btn{font-family:var(--mono);font-size:.88rem;padding:8px 20px;
  background:rgba(0,255,231,.08);border:1px solid var(--cyan);color:var(--cyan);
  border-radius:3px;cursor:pointer;letter-spacing:1px;transition:all .15s;margin-top:10px}
.dl-btn:hover{background:var(--cyan);color:#000}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--void)}
::-webkit-scrollbar-thumb{background:rgba(0,255,231,.18);border-radius:2px}
</style></head><body>
<div id="app">
<button id="fsBtn" onclick="toggleFS()">⛶ &nbsp;FULL SCREEN</button>
<div class="hdr">
  <div class="lb-dot"></div><h1>⚡ HFT AI SIMULATOR</h1>
  <div class="badge">LIVE FEED</div>
  <span class="hdr-meta">5 AI MODELS · REAL STRATEGIES · HFT SIMULATION</span>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('sim',this)">📈 SIMULATION</div>
  <div class="tab" onclick="switchTab('report',this)">📄 REPORT</div>
</div>
<div id="tab-sim" class="tab-content active">
<div class="ctrl">
  <button class="btn btn-cyan" id="btnPlay" onclick="togglePlay()">▶ PLAY</button>
  <button class="btn btn-pink" onclick="doReset()">■ RESET</button>
  <button class="btn btn-amber" onclick="stepFwd()">▷ STEP</button>
  <span class="cl">SPEED</span>
  <input type="range" id="spd" min="1" max="10" value="3" oninput="onSpd(this.value)">
  <span class="cv" id="spdLbl">3×</span>
  <div class="prog-wrap">
    <span class="cl">TICK</span>
    <div class="prog-bar" id="progBar" onclick="onProgClick(event)">
      <div class="prog-fill" id="progFill"></div>
    </div>
    <span id="tickLbl">0 / 0</span>
  </div>
  <button class="btn btn-amber" onclick="downloadReport()" style="margin-left:auto">⬇ REPORT</button>
</div>
<div class="main">
  <div><div class="sec-ttl">🏆 Live Leaderboard</div><div class="lb-grid" id="lbGrid"></div></div>
  <div class="charts-row">
    <div class="panel"><div class="panel-title">📈 Portfolio Value — All Agents</div><canvas id="cPf" height="200"></canvas></div>
    <div class="panel"><div class="panel-title">💰 P&L Comparison</div><div id="pnlBars"></div></div>
  </div>
  <div class="bottom-row">
    <div class="panel">
      <div class="panel-title">📋 Live Trade Log</div>
      <div class="bl-hdr"><span>AGENT</span><span>TICK</span><span>ASSET</span><span>SIDE</span><span>P&L</span><span>PORTFOLIO</span><span>LATENCY</span></div>
      <div class="bl-body" id="blBody"><div class="empty-bl">▶ PRESS PLAY TO START</div></div>
    </div>
    <div class="panel"><div class="panel-title">📊 Risk Metrics</div><div id="riskPanel"></div></div>
  </div>
</div></div>
<div id="tab-report" class="tab-content">
  <div class="report-wrap" id="reportWrap">
    <div style="text-align:center;padding:40px;color:var(--dim);font-family:var(--mono)">CLICK PLAY AND LET SIMULATION RUN TO GENERATE REPORT</div>
  </div>
</div>
</div>
<script>
var SIM_DATA=__SIM_JSON__;
var TRADERS=SIM_DATA.traders,TICKS=SIM_DATA.ticks,PFS=SIM_DATA.portfolios,
    TRADES=SIM_DATA.trades,META=SIM_DATA.meta,INIT=SIM_DATA.initCash;
var MEDALS=['🥇','🥈','🥉','4️⃣','5️⃣'];
var tick=0,playing=false,timer=null,spd=3;
var SPD=[500,220,110,65,40,25,16,10,6,3];
var ctxPf=document.getElementById('cPf').getContext('2d');
var pfChart=new Chart(ctxPf,{type:'line',
  data:{labels:[],datasets:TRADERS.map(t=>({label:t.id,data:[],borderColor:t.color,
    backgroundColor:t.color+'18',borderWidth:2.5,pointRadius:0,tension:0.3}))},
  options:{animation:false,responsive:true,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#fff',font:{size:13},usePointStyle:true}},
      tooltip:{backgroundColor:'#1e3a5f',titleColor:'#fff',bodyColor:'#b0c4de',
        callbacks:{label:c=>' '+c.dataset.label+': $'+Math.round(c.parsed.y).toLocaleString()}}},
    scales:{x:{display:false},y:{grid:{color:'#1e3a5f'},
      ticks:{color:'#fff',font:{size:13},callback:v=>'$'+(v/1000).toFixed(1)+'K'}}}}});
function update(t){
  tick=t;var T=TICKS.length-1;
  var pf=PFS[Math.min(t,PFS.length-1)]||{};
  document.getElementById('progFill').style.width=(T>0?t/T*100:0)+'%';
  document.getElementById('tickLbl').textContent=t+' / '+T;
  var sorted=[...TRADERS].sort((a,b)=>(pf[b.id]||INIT)-(pf[a.id]||INIT));
  document.getElementById('lbGrid').innerHTML=sorted.map((tr,i)=>{
    var pv=pf[tr.id]||INIT,pnl=pv-INIT,pct=(pnl/INIT*100).toFixed(2);
    var pcls=pnl>=0?'green':'red',sign=pnl>=0?'+':'';
    return '<div class="lb-card" style="border-color:'+tr.color+'55">'+
      '<div class="card-name" style="color:'+tr.color+'">'+MEDALS[i]+' '+tr.id+'</div>'+
      '<div class="card-pv '+pcls+'">\$'+Math.round(pv).toLocaleString()+'</div>'+
      '<div class="card-pnl '+pcls+'">'+sign+'\$'+Math.abs(Math.round(pnl)).toLocaleString()+' ('+sign+pct+'%)</div>'+
      '<hr style="border-color:#1e3a5f;margin:7px 0">'+
      '<div class="card-stats">Win: <b style="color:#fff">'+tr.winRate+'%</b> Trades: <b style="color:#fff">'+tr.nTrades+'</b><br>'+
      'Sharpe: <b style="color:#fff">'+tr.sharpe+'</b> DD: <b style="color:#fff">'+tr.dd+'%</b><br>'+
      'Lat: <b style="color:#fff">'+tr.lat+'μs</b></div></div>';
  }).join('');
  var ws=Math.max(0,t-200+1);
  pfChart.data.labels=TICKS.slice(ws,t+1).map((_,i)=>i);
  TRADERS.forEach((tr,i)=>{if(pfChart.data.datasets[i])
    pfChart.data.datasets[i].data=PFS.slice(ws,t+1).map(p=>p[tr.id]||INIT);});
  pfChart.update('none');
  var maxA=Math.max(...TRADERS.map(tr=>Math.abs((pf[tr.id]||INIT)-INIT)),1);
  document.getElementById('pnlBars').innerHTML=sorted.map(tr=>{
    var pnl=(pf[tr.id]||INIT)-INIT,bw=Math.abs(pnl)/maxA*100;
    var col=pnl>=0?'var(--green)':'var(--pink)',sign=pnl>=0?'+':'';
    return '<div class="pnl-row"><span class="pnl-lbl" style="color:'+tr.color+'">'+tr.id+'</span>'+
      '<div class="pnl-bg"><div class="pnl-fg" style="width:'+bw+'%;background:'+col+'"></div></div>'+
      '<span class="pnl-val" style="color:'+col+'">'+sign+'\$'+Math.abs(Math.round(pnl)).toLocaleString()+'</span></div>';
  }).join('');
  document.getElementById('riskPanel').innerHTML=sorted.map(tr=>{
    var pv=pf[tr.id]||INIT,pnl=pv-INIT;
    var col=pnl>=0?'var(--green)':'var(--pink)',bar=Math.max(0,Math.min(100,50+pnl/INIT*100));
    var sign=pnl>=0?'+':'';
    return '<div class="risk-row" style="border-color:'+tr.color+'">'+
      '<div style="display:flex;justify-content:space-between">'+
      '<span class="risk-name" style="color:'+tr.color+'">'+tr.id+'</span>'+
      '<span style="font-family:var(--mono);font-size:.95rem;color:'+col+';font-weight:700">'+sign+'\$'+Math.abs(Math.round(pnl)).toLocaleString()+'</span></div>'+
      '<div class="risk-bar-bg"><div class="risk-bar-fg" style="width:'+bar+'%;background:'+tr.color+'"></div></div>'+
      '<div class="risk-stats">Sharpe <b style="color:#fff">'+tr.sharpe+'</b> | DD <b style="color:#fff">'+tr.dd+'%</b> | WR <b style="color:#fff">'+tr.winRate+'%</b> | Lat <b style="color:#fff">'+tr.lat+'μs</b></div></div>';
  }).join('');
  var trs=TRADES.filter(x=>x.s<=t).slice(-30).reverse();
  if(trs.length===0){document.getElementById('blBody').innerHTML='<div class="empty-bl">▶ PRESS PLAY TO START</div>';}
  else{document.getElementById('blBody').innerHTML=trs.map((x,i)=>{
    var tr=TRADERS.find(t=>t.id===x.tid)||{};
    var col=tr.color||'#fff',scls=x.a==='BUY'?'green':'red',pcls=x.pnl>=0?'green':'red';
    var sign=x.pnl>=0?'+':'',nw=i===0?' nw':'';
    return '<div class="bl-row'+nw+'">'+
      '<span style="color:'+col+';font-weight:700">'+x.tid+'</span>'+
      '<span>'+x.s+'</span><span>'+x.ast+'</span>'+
      '<span class="'+scls+'" style="font-weight:700">'+x.a+'</span>'+
      '<span class="'+pcls+'">'+sign+'\$'+Math.abs(x.pnl).toFixed(2)+'</span>'+
      '<span>\$'+Math.round(x.pv).toLocaleString()+'</span>'+
      '<span style="color:var(--dim)">'+x.lat+'μs</span></div>';
  }).join('');}
}
function togglePlay(){playing?pause():play();}
function play(){
  if(TICKS.length===0)return;
  if(tick>=TICKS.length-1)tick=0;
  playing=true;
  document.getElementById('btnPlay').textContent='⏸ PAUSE';
  document.getElementById('btnPlay').className='btn btn-pink';
  schedNext();
}
function pause(){
  playing=false;clearTimeout(timer);
  document.getElementById('btnPlay').textContent='▶ PLAY';
  document.getElementById('btnPlay').className='btn btn-cyan';
}
function schedNext(){
  if(!playing)return;
  timer=setTimeout(()=>{
    if(tick<TICKS.length-1){update(tick+1);schedNext();}
    else{pause();document.getElementById('btnPlay').textContent='✓ DONE';buildReport();}
  },SPD[Math.min(spd-1,9)]);
}
function doReset(){pause();tick=0;document.getElementById('btnPlay').textContent='▶ PLAY';document.getElementById('btnPlay').className='btn btn-cyan';update(0);}
function stepFwd(){if(tick<TICKS.length-1){pause();update(tick+1);}}
function onSpd(v){spd=parseInt(v);document.getElementById('spdLbl').textContent=v+'×';if(playing){clearTimeout(timer);schedNext();}}
function onProgClick(e){var r=document.getElementById('progBar').getBoundingClientRect();var f=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));update(Math.round(f*(TICKS.length-1)));}
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();togglePlay();}
  if(e.code==='ArrowRight'){pause();if(tick<TICKS.length-1)update(tick+1);}
  if(e.code==='ArrowLeft'){pause();if(tick>0)update(tick-1);}
  if(e.key==='f'||e.key==='F'){e.preventDefault();toggleFS();}
});
function toggleFS(){
  var btn=document.getElementById('fsBtn'),app=document.getElementById('app');
  if(!document.fullscreenElement){
    document.documentElement.requestFullscreen().then(()=>{btn.innerHTML='✕ EXIT';}).catch(()=>{
      app.style.cssText='position:fixed;top:0;left:0;width:100vw;height:100vh;overflow:auto;z-index:99999;background:var(--void)';
      btn.innerHTML='✕ EXIT';});
  }else{document.exitFullscreen().catch(()=>{});app.style.cssText='';btn.innerHTML='⛶ &nbsp;FULL SCREEN';}
}
document.addEventListener('fullscreenchange',()=>{
  if(!document.fullscreenElement){
    document.getElementById('fsBtn').innerHTML='⛶ &nbsp;FULL SCREEN';
    document.getElementById('app').style.cssText='';
  }
});
function switchTab(name,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');document.getElementById('tab-'+name).classList.add('active');
  if(name==='report')buildReport();
}
function buildReport(){
  var sorted=[...TRADERS].sort((a,b)=>b.finalPv-a.finalPv);
  var wins=TRADES.filter(x=>x.pnl>0).length,losses=TRADES.filter(x=>x.pnl<0).length;
  var ac={};TRADES.forEach(x=>{ac[x.ast]=(ac[x.ast]||0)+1;});
  document.getElementById('reportWrap').innerHTML=
    '<div class="report-section"><div class="report-h1">📄 SIMULATION REPORT</div>'+
    '<div class="report-meta">'+
    '<div class="meta-box"><div class="meta-val">'+TICKS.length+'</div><div class="meta-lbl">TICKS</div></div>'+
    '<div class="meta-box"><div class="meta-val">'+TRADES.length+'</div><div class="meta-lbl">TRADES</div></div>'+
    '<div class="meta-box"><div class="meta-val" style="color:var(--green)">'+sorted[0].id+'</div><div class="meta-lbl">BEST AGENT</div></div>'+
    '<div class="meta-box"><div class="meta-val" style="color:'+(sorted[0].pnl>=0?'var(--green)':'var(--pink)')+'">$'+Math.abs(Math.round(sorted[0].pnl)).toLocaleString()+'</div><div class="meta-lbl">BEST P&L</div></div>'+
    '</div><button class="dl-btn" onclick="downloadReport()">⬇ DOWNLOAD HTML REPORT</button></div>'+
    '<div class="report-section"><div class="report-h1">🏆 AGENT PERFORMANCE</div>'+
    '<table class="rtbl"><thead><tr><th>Rank</th><th>Agent</th><th>Final PV</th><th>P&L</th><th>Trades</th><th>Win Rate</th><th>Sharpe</th><th>Max DD</th><th>Latency</th></tr></thead><tbody>'+
    sorted.map((tr,i)=>{var pcls=tr.pnl>=0?'var(--green)':'var(--pink)';var sign=tr.pnl>=0?'+':'';
      return '<tr><td>'+MEDALS[i]+'</td><td style="color:'+tr.color+';font-weight:700">'+tr.id+'</td>'+
      '<td>$'+Math.round(tr.finalPv).toLocaleString()+'</td>'+
      '<td style="color:'+pcls+'">'+sign+'$'+Math.abs(Math.round(tr.pnl)).toLocaleString()+'</td>'+
      '<td>'+tr.nTrades+'</td><td>'+tr.winRate+'%</td><td>'+tr.sharpe+'</td><td>'+tr.dd+'%</td><td>'+tr.lat+'μs</td></tr>';
    }).join('')+'</tbody></table></div>'+
    '<div class="report-section"><div class="report-h1">📋 TRADE LOG (LAST 50)</div>'+
    '<table class="rtbl"><thead><tr><th>#</th><th>Tick</th><th>Agent</th><th>Asset</th><th>Side</th><th>Price</th><th>P&L</th><th>Portfolio</th><th>Source</th><th>Reason</th></tr></thead><tbody>'+
    TRADES.slice(-50).reverse().map((x,i)=>{
      var tr=TRADERS.find(t=>t.id===x.tid)||{};
      var pcls=x.pnl>=0?'var(--green)':'var(--pink)';var sign=x.pnl>=0?'+':'';var acls=x.a==='BUY'?'var(--green)':'var(--pink)';
      var src=x.reason&&x.reason.includes('MODEL')?'🤖':x.reason&&x.reason.includes('ALPHA')?'🔶':x.reason&&x.reason.includes('SL')?'🛑':x.reason&&x.reason.includes('TP')?'✅':'📐';
      return '<tr><td>'+(i+1)+'</td><td>'+x.s+'</td><td style="color:'+(tr.color||'#fff')+';font-weight:700">'+x.tid+'</td>'+
      '<td>'+x.ast+'</td><td style="color:'+acls+';font-weight:700">'+x.a+'</td>'+
      '<td>$'+x.p.toFixed(4)+'</td><td style="color:'+pcls+'">'+sign+'$'+Math.abs(x.pnl).toFixed(4)+'</td>'+
      '<td>$'+Math.round(x.pv).toLocaleString()+'</td>'+
      '<td style="font-size:1.2rem">'+src+'</td>'+
      '<td style="color:#b0c4de;font-size:.75rem">'+(x.reason||'')+'</td></tr>';
    }).join('')+'</tbody></table></div>';
}
function downloadReport(){
  var sorted=[...TRADERS].sort((a,b)=>b.finalPv-a.finalPv);
  var wins=TRADES.filter(x=>x.result==='Profit').length;
  var losses=TRADES.filter(x=>x.result==='Loss').length;
  var opens=TRADES.filter(x=>x.result==='OPEN').length;
  var ac={};TRADES.forEach(x=>{ac[x.ast]=(ac[x.ast]||0)+1;});
  var modelT=TRADES.filter(x=>x.reason&&x.reason.includes('MODEL')).length;
  var alphaT=TRADES.filter(x=>x.reason&&x.reason.includes('ALPHA')).length;
  var ruleT =TRADES.filter(x=>x.reason&&x.reason.includes('RULE')).length;
  var perfRows=sorted.map((tr,i)=>{var pcls=tr.pnl>=0?'profit':'loss';var sign=tr.pnl>=0?'+':'';
    return '<tr><td>'+(i+1)+'</td><td style="color:'+tr.color+';font-weight:bold">'+tr.id+'</td>'+
    '<td>AI Model</td><td>$'+Math.round(tr.finalPv-tr.pnl).toLocaleString()+'</td>'+
    '<td>$'+Math.round(tr.finalPv).toLocaleString()+'</td>'+
    '<td class="'+pcls+'">'+sign+'$'+Math.abs(Math.round(tr.pnl)).toLocaleString()+'</td>'+
    '<td>'+tr.nTrades+'</td><td>'+tr.wins+'</td><td>'+tr.winRate+'%</td>'+
    '<td>'+tr.sharpe+'</td><td>'+tr.dd+'%</td><td>'+tr.lat+'μs</td></tr>';
  }).join('');
  var tradeRows=TRADES.map((x,i)=>{
    var tr=TRADERS.find(t=>t.id===x.tid)||{};
    var pcls=x.result==='Profit'?'profit':x.result==='Loss'?'loss':'open';
    var acls=x.a==='BUY'?'buy':'sell';
    return '<tr data-action="'+x.a+'" data-trader="'+x.tid+'" data-asset="'+x.ast+'" data-result="'+x.result+'">'+
    '<td>'+(i+1)+'</td><td>'+x.s+'</td><td>'+x.ast+'</td>'+
    '<td style="color:'+(tr.color||'#00ffe7')+';font-weight:bold">'+x.tid+'</td>'+
    '<td class="'+acls+'">'+x.a+'</td><td>MARKET</td>'+
    '<td>'+(x.buyer||x.tid)+'</td><td>'+(x.seller||'Market')+'</td>'+
    '<td>Market liquidity</td><td>1</td>'+
    '<td>$'+(x.bid||x.p).toFixed(5)+'</td><td>$'+(x.ask||x.p).toFixed(5)+'</td>'+
    '<td>$'+x.p.toFixed(5)+'</td><td>$'+x.p.toFixed(5)+'</td>'+
    '<td>'+(x.spread||0).toFixed(5)+'</td><td>$'+(x.cost||0).toFixed(5)+'</td>'+
    '<td>'+x.lat+'μs</td><td>$'+(x.slip||0).toFixed(6)+'</td>'+
    '<td class="'+pcls+'">'+( x.pnl>=0?'+':'')+'$'+Math.abs(x.pnl).toFixed(4)+'</td>'+
    '<td class="'+pcls+'">'+x.result+'</td>'+
    '<td>$'+Math.round(x.pv).toLocaleString()+'</td>'+
    '<td>$'+(x.pv-100000).toFixed(2)+'</td>'+
    '<td>'+(x.reason||x.a+' on '+x.ast)+'</td></tr>';
  }).join('');
  var assetRows=Object.entries(ac).sort((a,b)=>b[1]-a[1])
    .map(e=>'<tr><td>'+e[0]+'</td><td>'+e[1]+'</td><td>'+(e[1]/TRADES.length*100).toFixed(1)+'%</td></tr>').join('');
  var html='<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>'+
  '<title>HFT AI Simulator Report</title>'+
  '<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">'+
  '<style>:root{--bg:#060b18;--card:#0d1117;--text:#e2e8f0;--muted:#b0c4de;--line:#1e3a5f;'+
  '--head:#00ffe7;--profit:#39ff14;--loss:#ff2d78;--open:#ffaa00;--buy:#39ff14;--sell:#ff2d78;}'+
  'body{font-family:"Share Tech Mono",monospace;background:var(--bg);color:var(--text);padding:26px;font-size:15px;}'+
  'h1{color:var(--head);font-size:24px;letter-spacing:2px;}h2{color:var(--head);margin-top:28px;font-size:18px;border-bottom:1px solid var(--line);padding-bottom:6px;}'+
  '.small{color:var(--muted);font-size:13px;}.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin:16px 0;}'+
  '.metric{background:var(--card);border:1px solid var(--line);padding:14px;border-radius:8px;text-align:center;}'+
  '.metric b{display:block;font-size:22px;margin-top:6px;color:var(--head);}'+
  '.source-box{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0;}'+
  '.src{background:var(--card);border:1px solid var(--line);padding:12px;border-radius:8px;text-align:center;}'+
  '.src b{display:block;font-size:28px;margin-top:4px;}'+
  '.controls{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--line);}'+
  'input,select{padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--text);font-size:14px;font-family:inherit;}'+
  'table{width:100%;border-collapse:collapse;background:var(--card);margin-bottom:30px;border-radius:6px;overflow:hidden;}'+
  'th{background:#0d2040;color:var(--head);padding:9px 10px;text-align:left;font-size:13px;letter-spacing:1px;}'+
  'td{border-bottom:1px solid var(--line);padding:7px 10px;font-size:13px;color:var(--text);}'+
  'tr:nth-child(even) td{background:rgba(0,255,231,.03);}tr:hover td{background:rgba(0,255,231,.07);}'+
  '.profit{color:var(--profit);font-weight:bold;}.loss{color:var(--loss);font-weight:bold;}'+
  '.open{color:var(--open);font-weight:bold;}.buy{color:var(--buy);font-weight:bold;}.sell{color:var(--sell);font-weight:bold;}</style></head><body>'+
  '<h1>⚡ HFT AI Simulator — Complete Trading Report</h1>'+
  '<div class="small">Generated: '+new Date().toLocaleString()+'</div>'+
  '<div class="grid">'+
  '<div class="metric">Total Agents<b>5</b></div>'+
  '<div class="metric">Total Trades<b>'+TRADES.length+'</b></div>'+
  '<div class="metric">Profit/Loss/Open<b>'+wins+' / '+losses+' / '+opens+'</b></div>'+
  '<div class="metric">Initial Capital<b>$'+INIT.toLocaleString()+'</b></div>'+
  '<div class="metric">Total Ticks<b>'+TICKS.length+'</b></div>'+
  '<div class="metric">Assets Traded<b>'+Object.keys(ac).length+'</b></div>'+
  '<div class="metric">Best Agent<b style="color:var(--profit)">'+sorted[0].id+'</b></div>'+
  '<div class="metric">Best P&L<b style="color:'+(sorted[0].pnl>=0?'var(--profit)':'var(--loss)')+'">'+(sorted[0].pnl>=0?'+':'')+'$'+Math.abs(Math.round(sorted[0].pnl)).toLocaleString()+'</b></div></div>'+
  '<h2>Decision Sources</h2>'+
  '<div class="source-box">'+
  '<div class="src">🤖 AI MODEL Trades<b style="color:#00ffe7">'+modelT+'</b><small>Trained DQN/PPO/LSTM/XGB/TRF</small></div>'+
  '<div class="src">🔶 ALPHA Trades<b style="color:#ffaa00">'+alphaT+'</b><small>Sentiment + AlphaSignal</small></div>'+
  '<div class="src">📐 RULE Trades<b style="color:#bf7fff">'+ruleT+'</b><small>Strategy fallback</small></div>'+
  '</div>'+
  '<h2>Agent Performance</h2>'+
  '<table><thead><tr><th>Rank</th><th>Agent</th><th>Strategy</th><th>Cash</th><th>Final PV</th><th>P&L</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>Sharpe</th><th>Max DD</th><th>Latency</th></tr></thead><tbody>'+perfRows+'</tbody></table>'+
  '<h2>Asset Distribution</h2>'+
  '<table><thead><tr><th>Asset</th><th>Trades</th><th>%</th></tr></thead><tbody>'+assetRows+'</tbody></table>'+
  '<h2>Complete Trade Log</h2>'+
  '<div class="controls"><input id="searchBox" placeholder="Search..." oninput="filterRows()"/>'+
  '<select id="actionFilter" onchange="filterRows()"><option value="ALL">All</option><option value="BUY">BUY</option><option value="SELL">SELL</option></select>'+
  '<select id="traderFilter" onchange="filterRows()"><option value="ALL">All agents</option>'+
  TRADERS.map(t=>'<option value="'+t.id+'">'+t.id+'</option>').join('')+
  '</select><select id="resultFilter" onchange="filterRows()"><option value="ALL">All</option><option value="Profit">Profit</option><option value="Loss">Loss</option><option value="OPEN">Open</option></select>'+
  '<span class="small" id="visibleCount"></span></div>'+
  '<table id="tradeTable"><thead><tr><th>#</th><th>Tick</th><th>Asset</th><th>Agent</th><th>Action</th><th>Type</th><th>Buyer</th><th>Seller</th><th>Counterparty</th><th>Qty</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Exec</th><th>Spread</th><th>Cost</th><th>Latency</th><th>Slippage</th><th>P&L</th><th>Result</th><th>Portfolio</th><th>Total P&L</th><th>Reason</th></tr></thead><tbody>'+tradeRows+'</tbody></table>'+
  '<script>function filterRows(){var s=document.getElementById("searchBox").value.toLowerCase(),a=document.getElementById("actionFilter").value,tr=document.getElementById("traderFilter").value,r=document.getElementById("resultFilter").value,rows=document.querySelectorAll("#tradeTable tbody tr"),v=0;rows.forEach(row=>{var show=(!s||row.textContent.toLowerCase().includes(s))&&(a==="ALL"||row.dataset.action===a)&&(tr==="ALL"||row.dataset.trader===tr)&&(r==="ALL"||row.dataset.result===r);row.style.display=show?"":"none";if(show)v++;});document.getElementById("visibleCount").textContent=v+" trades shown";}filterRows();<\/script></body></html>';
  var blob=new Blob([html],{type:'text/html'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');a.href=url;a.download='hft_simulation_report.html';a.click();URL.revokeObjectURL(url);
}
update(0);
</script></body></html>
"""

def build_html(sim_json):
    return HTML_TEMPLATE.replace("__SIM_JSON__", sim_json)

# ── Render ─────────────────────────────────────────────────────────────────────
if st.session_state.sim_ready and st.session_state.sim_data:
    # Mode banner above the simulation
    if st.session_state.live_mode:
        files_str = (
            ", ".join(st.session_state.live_files[:3])
            + ("..." if len(st.session_state.live_files) > 3 else "")
        ) if st.session_state.live_files else "uploaded files"
        st.markdown(
            f'<div style="background:#0d1a00;border:1px solid #ff9900;border-radius:7px;'
            f'padding:9px 16px;margin-bottom:10px;display:flex;align-items:center;gap:12px;">'
            f'<span style="color:#ff9900;font-size:1.1rem;">📡</span>'
            f'<span style="color:#ff9900;font-family:\'Share Tech Mono\',monospace;'
            f'font-size:.82rem;letter-spacing:1px;">'
            f'LIVE DATA MODE &nbsp;·&nbsp; {st.session_state.live_rows:,} ticks streamed via Kafka'
            f' &nbsp;·&nbsp; Source: {files_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#060b18;border:1px solid #1e3a5f;border-radius:7px;'
            'padding:9px 16px;margin-bottom:10px;">'
            '<span style="color:#00ffe7;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.82rem;letter-spacing:1px;">📊 DEFAULT MODE &nbsp;·&nbsp; '
            'Using parquet splits / synthetic data</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    components.html(build_html(st.session_state.sim_data), height=1200, scrolling=True)
else:
    st.markdown("""
    <div style="background:#060b18;text-align:center;padding:60px 20px;min-height:70vh">
      <div style="font-size:5rem">⚡</div>
      <h1 style="color:#00ffe7;font-size:2.8rem;letter-spacing:4px;margin:12px 0;font-family:'Share Tech Mono',monospace">
        HFT AI SIMULATOR
      </h1>
      <p style="color:#b0c4de;font-size:1rem;margin:0 auto 8px;letter-spacing:2px">
        5 AI MODELS · REAL STRATEGIES · LIVE HFT SIMULATION
      </p>
      <p style="color:#4b5563;font-size:.85rem;font-family:'Share Tech Mono',monospace;margin-bottom:30px">
        Models loaded · Click ▶ RUN SIMULATION to begin
      </p>
    </div>""", unsafe_allow_html=True)
    cols = st.columns(5)
    for i,(n,c) in enumerate(AGENT_COLORS.items()):
        with cols[i]:
            st.markdown(f'''<div style="background:#0d1117;border:1px solid {c}55;border-radius:9px;
                padding:20px 14px;text-align:center;box-shadow:0 0 20px {c}11">
              <div style="color:{c};font-size:1.3rem;font-weight:800;font-family:'Share Tech Mono',monospace;letter-spacing:2px">{n}</div>
              <div style="color:#b0c4de;font-size:0.85rem;margin-top:6px">AI Model Ready</div>
            </div>''', unsafe_allow_html=True)