import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Load all 3 days of Round 3 data (underlyings only; vouchers analyzed separately) ─
CSV_DIR = Path("/Users/jacobklugerman/Downloads/ROUND_3")

UNDERLYINGS = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")


def load_data(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for day in [0, 1, 2]:
        p = csv_dir / f"prices_round_3_day_{day}.csv"
        if p.exists():
            df = pd.read_csv(p, sep=";")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No Round 3 price CSVs found in {csv_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    df = df[df["mid_price"] > 0].reset_index(drop=True)
    return df


def sort_data(df: pd.DataFrame):
    df_hydrogel = df[df["product"] == "HYDROGEL_PACK"].copy()
    df_velvet = df[df["product"] == "VELVETFRUIT_EXTRACT"].copy()
    return df_hydrogel, df_velvet


df = load_data(CSV_DIR)
df_hydrogel, df_velvet = sort_data(df)

# L1–L3 prices & volumes (official `prices_round_*` CSV schema; same as orderbook_dashboard)
BID_P = ["bid_price_1", "bid_price_2", "bid_price_3"]
ASK_P = ["ask_price_1", "ask_price_2", "ask_price_3"]
BID_V = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
ASK_V = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]


def wall_mid(df: pd.DataFrame) -> pd.Series:
    """Deepest bid + highest ask across L1–L3, then midpoint (WallMid)."""
    bp = df.reindex(columns=BID_P)
    ap = df.reindex(columns=ASK_P)
    worst_bid = bp.min(axis=1, skipna=True)
    worst_ask = ap.max(axis=1, skipna=True)
    return ((worst_bid + worst_ask) / 2.0).astype(float)


def normalized_orderbook_imbalance(df: pd.DataFrame) -> pd.Series:
    """(bid_vol_sum − ask_vol_sum) / (bid_vol_sum + ask_vol_sum) in [−1, 1]."""
    bv = df.reindex(columns=BID_V).fillna(0).sum(axis=1)
    av = df.reindex(columns=ASK_V).fillna(0).abs().sum(axis=1)
    den = bv + av
    return ((bv - av) / den).where(den.abs() > 1e-9, 0.0).astype(float)


def add_tick_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["tick"] = np.arange(len(out), dtype=np.int64)
    return out


# Full book rows + tick for imbalance / correlation
df_hydrogel_book = add_tick_column(df_hydrogel)
df_velvet_book = add_tick_column(df_velvet)

# ── Smoothing functions ──────────────────────────────────────────────────

def smooth_R_mid_price(df: pd.DataFrame, window: int = 80) -> pd.Series:
    return df["mid_price"].rolling(window=window).mean()

def smooth_E_mid_price(df: pd.DataFrame, span: int = 80) -> pd.Series:
    return df["mid_price"].ewm(span=span).mean()

# ── CONFIG ────────────────────────────────────────────────────────────────
PRICE_COL     = "mid_price"
TIMESTAMP_COL = "tick"

PRODUCTS = {
    "HYDROGEL_PACK": df_hydrogel_book,
    "VELVETFRUIT_EXTRACT": df_velvet_book,
}

# Collect all figures, show at the end
all_figs = []

for product_name, df_prod in PRODUCTS.items():
    prices = df_prod[PRICE_COL].reset_index(drop=True)
    ticks = df_prod["tick"] if "tick" in df_prod.columns else pd.RangeIndex(len(df_prod))

    # ── 1. Basic stats ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {product_name}")
    print(f"{'='*60}")
    print(f"=== Basic Stats ===")
    print(f"  Rows         : {len(prices)}")
    print(f"  Price range  : {prices.min():.2f} -> {prices.max():.2f}")
    print(f"  Mean         : {prices.mean():.2f}")
    print(f"  Std dev      : {prices.std():.2f}")

    # ── 2. Autocorrelation (key test) ─────────────────────────────────
    print("\n=== Autocorrelation (returns) ===")
    returns = prices.diff().dropna()
    for lag in [1, 5, 10, 50]:
        ac = returns.autocorr(lag=lag)
        signal = "mean-reverting" if ac < -0.05 else ("trending" if ac > 0.05 else "~ random")
        print(f"  Lag {lag:>3}: {ac:+.4f}  {signal}")

    # ── 3. Hurst exponent ─────────────────────────────────────────────
    print("\n=== Hurst Exponent ===")
    lags = range(2, min(100, len(prices) // 4))
    tau  = [np.std(prices.diff(lag).dropna()) for lag in lags]
    H    = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    interp = "mean-reverting" if H < 0.45 else ("trending" if H > 0.55 else "random walk")
    print(f"  H = {H:.4f}  ->  {interp}")

    # ── 4. Deviation persistence ──────────────────────────────────────
    EWM_SPAN_DEV = 1050
    DEV_ROLL_WIN = 1050
    print(f"\n=== Deviation Persistence (vs EWM span={EWM_SPAN_DEV}) ===")
    ewm  = prices.ewm(span=EWM_SPAN_DEV).mean()
    dev  = prices - ewm
    dev_std = dev.rolling(window=DEV_ROLL_WIN, min_periods=DEV_ROLL_WIN // 4).std()
    dev_z = (dev / dev_std).replace([np.inf, -np.inf], np.nan)
    print(f"  Mean |deviation|   : {dev.abs().mean():.4f}")
    print(f"  Max  |deviation|   : {dev.abs().max():.4f}")
    print(f"  Dev std            : {dev.std():.4f}")
    print(f"  % ticks |z| > 1   : {(dev_z.abs() > 1.0).mean()*100:.1f}%")
    print(f"  % ticks |z| > 2   : {(dev_z.abs() > 2.0).mean()*100:.1f}%")

    # ── 4b. WallMid direction vs short EWMA of normalized imbalance ───
    IMB_EWM_SPAN = 20
    vol_cols_ok = all(c in df_prod.columns for c in BID_V + ASK_V)
    wall_cols_ok = all(c in df_prod.columns for c in BID_P + ASK_P)
    print("\n=== Correlation: Δ WallMid vs short EWMA(norm imbalance) ===")
    if not vol_cols_ok:
        print("  (skipped — CSV missing bid/ask volume columns)")
    elif not wall_cols_ok:
        print("  (skipped — CSV missing bid/ask price columns for WallMid)")
    else:
        wallmid = wall_mid(df_prod).reset_index(drop=True)
        imb_norm = normalized_orderbook_imbalance(df_prod)
        imb_short = imb_norm.ewm(span=IMB_EWM_SPAN, adjust=False).mean()
        # Price direction: tick-to-tick change in WallMid (not touch mid_price)
        ret = wallmid.diff()
        m = ret.notna() & imb_short.notna() & wallmid.notna()
        # ret[t] = wallmid[t]-wallmid[t-1]; imb_short.shift(1)[t] = EWMA at end of t-1 → predictive lag
        m_lag = ret.notna() & imb_short.shift(1).notna() & wallmid.notna()
        if int(m.sum()) < 30:
            print(f"  (skipped — only {int(m.sum())} overlapping valid rows)")
        else:
            r_cont, p_cont = stats.pearsonr(ret[m], imb_short[m])
            r_lag, p_lag = stats.pearsonr(ret[m_lag], imb_short.shift(1)[m_lag])
            r_fwd, p_fwd = stats.pearsonr(ret.shift(-1)[m], imb_short[m])
            rho_cont, _ = stats.spearmanr(ret[m], imb_short[m])
            rho_lag, _ = stats.spearmanr(ret[m_lag], imb_short.shift(1)[m_lag])
            print(f"  Imbalance EWMA span = {IMB_EWM_SPAN}")
            print(f"  Pearson  corr( ΔWallMid[t], EWMA_imb[t]   ) = {r_cont:+.4f}  (p={p_cont:.2e})")
            print(f"  Pearson  corr( ΔWallMid[t], EWMA_imb[t−1]) = {r_lag:+.4f}  (p={p_lag:.2e})  ← predictive")
            print(f"  Pearson  corr( ΔWallMid[t+1], EWMA_imb[t]) = {r_fwd:+.4f}  (p={p_fwd:.2e})")
            print(f"  Spearman corr( ΔWallMid, EWMA_imb ) (rank) = {rho_cont:+.4f}")
            print(f"  Spearman corr( ΔWallMid, EWMA_imb lag 1 )   = {rho_lag:+.4f}")

    # ── 5. EWM + Rolling Mean overlay ─────────────────────────────────
    smooth_r = smooth_R_mid_price(df_prod)
    smooth_e = smooth_E_mid_price(df_prod)

    fig_ewm, ax_ewm = plt.subplots(figsize=(20, 8))
    ax_ewm.plot(ticks, prices, lw=0.6, label="Mid Price (raw)")
    ax_ewm.plot(ticks, smooth_e, lw=1.2, label="EWM(80)")
    ax_ewm.plot(ticks, smooth_r, lw=1.2, label="Rolling Mean(80)")
    ax_ewm.legend()
    ax_ewm.set_title(f"Exponentially Weighted Moving Average (EWM) of Mid Price — {product_name}")
    ax_ewm.set_xlabel("Tick")
    ax_ewm.set_ylabel("Mid Price")
    ax_ewm.grid(True)
    all_figs.append(fig_ewm)

    # ── 6. Price vs EWM + Deviation z-score + Returns ─────────────────
    fig_dev, axes_dev = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes_dev[0].plot(ticks, prices, lw=0.6, label="Mid price")
    axes_dev[0].plot(ticks, ewm,    lw=1.5, label=f"EWM({EWM_SPAN_DEV})", alpha=0.85, color="darkorange")
    axes_dev[0].set_title(f"{product_name} — Price vs EWM")
    axes_dev[0].legend(loc="upper right")
    axes_dev[0].grid(True, alpha=0.3)

    # Scale-invariant z-score: (price − EWM) / rolling_std(dev)
    # Clip extreme values for display so thresholds stay readable.
    dev_z_plot = dev_z.clip(lower=-6, upper=6)
    axes_dev[1].plot(ticks, dev_z_plot, lw=0.6, color="steelblue")
    axes_dev[1].axhline( 0, color="black", lw=0.5)
    for thr, col in [(1, "gold"), (2, "red")]:
        axes_dev[1].axhline( thr, color=col, lw=0.8, linestyle="--", label=f"±{thr}")
        axes_dev[1].axhline(-thr, color=col, lw=0.8, linestyle="--")
    axes_dev[1].set_ylim(-6, 6)
    axes_dev[1].set_ylabel(f"z = dev / rolling_std({DEV_ROLL_WIN})")
    axes_dev[1].set_title(
        f"{product_name} — Z-score of deviation from EWM({EWM_SPAN_DEV})  "
        f"[raw |dev| max = {dev.abs().max():.1f}, dev std = {dev.std():.2f}]"
    )
    axes_dev[1].legend(loc="upper right")
    axes_dev[1].grid(True, alpha=0.3)

    axes_dev[2].plot(ticks.iloc[1:] if hasattr(ticks, "iloc") else ticks[1:],
                     returns.values, lw=0.4, color="gray")
    axes_dev[2].axhline(0, color="black", lw=0.4)
    axes_dev[2].set_title(f"{product_name} — Tick-to-tick returns (all ticks)")
    axes_dev[2].set_xlabel("Tick")
    axes_dev[2].grid(True, alpha=0.3)

    fig_dev.tight_layout()
    all_figs.append(fig_dev)

    # ── 7. Z-score calibration across spans ───────────────────────────
    SPANS = [5, 10, 20, 30, 50]

    fig_z, axes_z = plt.subplots(len(SPANS), 3, figsize=(15, 2.6 * len(SPANS)))
    fig_z.suptitle(f"Z-score calibration across EWM spans — {product_name}", fontsize=14)

    results = {}

    for i, span in enumerate(SPANS):
        ewm_span  = prices.ewm(span=span).mean()
        deviation = prices - ewm_span

        roll_std  = deviation.rolling(window=span * 2, min_periods=span).std()
        zscore    = deviation / roll_std

        z_clean = zscore.dropna()

        p = {
            "1%":  np.percentile(z_clean, 1),
            "5%":  np.percentile(z_clean, 5),
            "10%": np.percentile(z_clean, 10),
            "90%": np.percentile(z_clean, 90),
            "95%": np.percentile(z_clean, 95),
            "99%": np.percentile(z_clean, 99),
        }

        entry_freq = {
            "z>1":  (z_clean.abs() > 1.0).mean() * 100,
            "z>1.5":(z_clean.abs() > 1.5).mean() * 100,
            "z>2":  (z_clean.abs() > 2.0).mean() * 100,
            "z>2.5":(z_clean.abs() > 2.5).mean() * 100,
        }

        results[span] = {"percentiles": p, "entry_freq": entry_freq,
                         "std": roll_std.mean(), "zscore": z_clean}

        print(f"\n-- Span {span} -----------------------------------------------")
        print(f"  Mean rolling std (raw dev units) : {roll_std.mean():.4f}")
        print(f"  Percentiles of z-score:")
        for k, v in p.items():
            print(f"    {k:>4s}: {v:+.3f}")
        print(f"  Entry frequency (|z| exceeds threshold):")
        for k, v in entry_freq.items():
            print(f"    {k}: {v:.1f}% of ticks")

        # Plot 1: raw deviation
        axes_z[i, 0].plot(deviation.values, lw=0.5, color="steelblue")
        axes_z[i, 0].set_title(f"Span {span} — raw deviation")
        axes_z[i, 0].axhline(0, color="black", lw=0.5)

        # Plot 2: z-score over time
        axes_z[i, 1].plot(z_clean.values, lw=0.5, color="darkorange")
        for thresh, col in [(1, "gold"), (2, "red")]:
            axes_z[i, 1].axhline( thresh, color=col, lw=0.8, linestyle="--")
            axes_z[i, 1].axhline(-thresh, color=col, lw=0.8, linestyle="--")
        axes_z[i, 1].set_title(f"Span {span} — z-score (+/-1 gold, +/-2 red)")

        # Plot 3: distribution of z-scores vs normal
        axes_z[i, 2].hist(z_clean, bins=80, density=True,
                        color="steelblue", alpha=0.6, label="actual")
        xr = np.linspace(z_clean.min(), z_clean.max(), 300)
        axes_z[i, 2].plot(xr, stats.norm.pdf(xr), color="red",
                        lw=1.5, label="N(0,1)")
        axes_z[i, 2].set_title(f"Span {span} — z-score distribution")
        axes_z[i, 2].legend(fontsize=8)

    fig_z.tight_layout(rect=(0, 0, 1, 0.97))
    all_figs.append(fig_z)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n=== SUMMARY ({product_name}): entry frequency by span & threshold ===")
    print(f"{'Span':>6} | {'|z|>1':>8} | {'|z|>1.5':>8} | {'|z|>2':>8} | {'|z|>2.5':>8}")
    print("-" * 52)
    for span, r in results.items():
        ef = r["entry_freq"]
        print(f"{span:>6} | {ef['z>1']:>7.1f}% | {ef['z>1.5']:>7.1f}% |"
              f" {ef['z>2']:>7.1f}% | {ef['z>2.5']:>7.1f}%")

# ── Show all figures at once ─────────────────────────────────────────────
plt.show()
