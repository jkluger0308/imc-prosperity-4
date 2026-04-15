import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Load all 3 days of Round 1 data ──────────────────────────────────────
CSV_DIR = Path("PATH FOR DATA FOLDER "ROUND1" here!!! ")

def load_data(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for day in [-2, -1, 0]:
        p = csv_dir / f"prices_round_1_day_{day}.csv"
        if p.exists():
            df = pd.read_csv(p, sep=";")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    # Filter out rows with zero mid_price (missing book data)
    df = df[df["mid_price"] > 0].reset_index(drop=True)
    return df

def sort_data(df: pd.DataFrame):
    df_osmium = df[df["product"] == "ASH_COATED_OSMIUM"].copy()
    df_pepper = df[df["product"] == "INTARIAN_PEPPER_ROOT"].copy()
    return df_osmium, df_pepper

df = load_data(CSV_DIR)
df_osmium, df_pepper = sort_data(df)

# ── Mid price extraction ─────────────────────────────────────────────────

def mid_price(df: pd.DataFrame) -> pd.DataFrame:
    df_mid = df[["day", "timestamp", "mid_price"]].copy()
    # Create a unique sequential index across days
    df_mid["tick"] = range(len(df_mid))
    df_mid = df_mid.set_index("tick")
    return df_mid

df_osmium_mid = mid_price(df_osmium)
df_pepper_mid = mid_price(df_pepper)

# ── Smoothing functions ──────────────────────────────────────────────────

def smooth_R_mid_price(df: pd.DataFrame, window: int = 80) -> pd.Series:
    return df["mid_price"].rolling(window=window).mean()

def smooth_E_mid_price(df: pd.DataFrame, span: int = 80) -> pd.Series:
    return df["mid_price"].ewm(span=span).mean()

# ── CONFIG ────────────────────────────────────────────────────────────────
PRICE_COL     = "mid_price"
TIMESTAMP_COL = "tick"

PRODUCTS = {
    "ASH_COATED_OSMIUM": df_osmium_mid,
    "INTARIAN_PEPPER_ROOT": df_pepper_mid,
}

# Collect all figures, show at the end
all_figs = []

for product_name, df_prod in PRODUCTS.items():
    prices = df_prod[PRICE_COL]
    ticks  = df_prod.index  # tick number

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
    print("\n=== Deviation Persistence (vs EWM span=1050) ===")
    ewm  = prices.ewm(span=1050).mean()
    dev  = prices - ewm
    print(f"  Mean |deviation|   : {dev.abs().mean():.4f}")
    print(f"  Max  |deviation|   : {dev.abs().max():.4f}")
    print(f"  % ticks |dev| > 0.25 : {(dev.abs() > 0.25).mean()*100:.1f}%")
    print(f"  % ticks |dev| > 2.0  : {(dev.abs() > 2.0).mean()*100:.1f}%")

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

    # ── 6. Price vs EWM + Deviation + Returns ─────────────────────────
    fig_dev, axes_dev = plt.subplots(3, 1, figsize=(8, 6))

    axes_dev[0].plot(ticks, prices, lw=0.8, label="Mid price")
    axes_dev[0].plot(ticks, ewm,    lw=1.5, label="EWM(1050)", alpha=0.8)
    axes_dev[0].set_title(f"{product_name} — Price vs EWM")
    axes_dev[0].legend()

    axes_dev[1].plot(ticks, dev, lw=0.8, color="steelblue")
    axes_dev[1].axhline(0,    color="black", lw=0.5)
    axes_dev[1].axhline(2,    color="red",   lw=0.5, linestyle="--", label="+2")
    axes_dev[1].axhline(-2,   color="red",   lw=0.5, linestyle="--", label="-2")
    axes_dev[1].axhline(0.25, color="orange",lw=0.5, linestyle="--", label="+/-0.25")
    axes_dev[1].axhline(-0.25,color="orange",lw=0.5, linestyle="--")
    axes_dev[1].set_title(f"{product_name} — Deviation from EWM (signal)")
    axes_dev[1].legend()

    axes_dev[2].plot(returns.values[:500], lw=0.6, color="gray")
    axes_dev[2].set_title(f"{product_name} — Tick-to-tick returns (first 500 ticks)")

    fig_dev.tight_layout()
    all_figs.append(fig_dev)

    # ── 7. Z-score calibration across spans ───────────────────────────
    SPANS = [5, 10, 20, 30, 50]

    fig_z, axes_z = plt.subplots(len(SPANS), 3, figsize=(6, 2 * len(SPANS)))
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

    fig_z.tight_layout()
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
