"""
Voucher / VELVETFRUIT_EXTRACT options analysis (Round 3).

Vouchers are CALL options on VELVETFRUIT_EXTRACT.
  - LONG a call: pay premium, profit if S > K at expiry (cap = unlimited up, loss = premium).
  - SHORT a call: collect premium, profit if S <= K at expiry (cap = premium, loss = unlimited up).

This script builds five complementary views to find trades:
  1. Overview — underlying + every voucher mid over time.
  2. Extrinsic value per strike — how much each call costs above intrinsic = (S - K)+.
  3. Implied volatility per strike, time series (one curve per strike).
  4. Implied volatility smile — IV vs strike, one line per day, averaged across the day.
  5. Implied underlying via deep-ITM calls — S_implied = mid_K + K (works only for deep ITM).
     If implied S consistently differs from actual mid, that's a fair-value signal for the
     VELVETFRUIT_EXTRACT MM.
  6. No-arb diagnostics — monotonicity (calls must be monotonically non-increasing in K)
     and convexity (mid_K1 + mid_K3 >= 2 * mid_K2 for K1 < K2 < K3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

# ── Config ───────────────────────────────────────────────────────────────
CSV_DIR = Path("/Users/a16039/src/IMC/r3_code/ROUND_3")
DAYS = [0, 1, 2]
TICKS_PER_DAY = 1_000_000   # max timestamp + 100
DAY_FRACTION = 1.0          # one round = one "day" of TTE decay

# TTE at start of each historical day (per round-3 wiki)
DAY_TTE_START = {0: 8.0, 1: 7.0, 2: 6.0}
TIME_UNIT_DAYS_PER_YEAR = 252.0   # purely a scale; doesn't change relative IV shape

UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500,
}
DEEP_ITM_KEYS = ["VEV_4000", "VEV_4500"]
# How aggressively we down-sample for time-series plots (per day)
DOWNSAMPLE = 200


# ── Black-Scholes call ───────────────────────────────────────────────────
def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price: float, S: float, K: float, T: float) -> float | None:
    """Invert Black-Scholes for sigma. None if outside no-arb bounds."""
    intrinsic = max(S - K, 0.0)
    upper = S
    if not (intrinsic - 1e-6 <= price <= upper + 1e-6):
        return None
    if price <= intrinsic + 1e-6:
        return 1e-4   # essentially zero extrinsic
    f = lambda sig: bs_call(S, K, T, sig) - price
    try:
        return brentq(f, 1e-4, 5.0, maxiter=80)
    except Exception:
        return None


# ── Loaders ──────────────────────────────────────────────────────────────
def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = CSV_DIR / f"prices_round_3_day_{d}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, sep=";"))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["mid_price"] > 0].reset_index(drop=True)
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    df["global_ts"] = (df["day"] - DAYS[0]) * TICKS_PER_DAY + df["timestamp"]
    df["tte_days"] = df["day"].map(DAY_TTE_START) - df["timestamp"] / TICKS_PER_DAY * DAY_FRACTION
    df["T"] = df["tte_days"] / TIME_UNIT_DAYS_PER_YEAR
    return df


def pivot_mids(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: index=global_ts, cols=product, values=mid_price (+ T column)."""
    wide = df.pivot_table(index="global_ts", columns="product", values="mid_price", aggfunc="first")
    T = df.groupby("global_ts")["T"].first()
    wide["T"] = T
    return wide.dropna(subset=[UNDERLYING]).sort_index()


# ── Plot helpers ─────────────────────────────────────────────────────────
def fig_overview(wide: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    ax1.plot(wide.index, wide[UNDERLYING], lw=0.5, color="black", label=UNDERLYING)
    ax1.set_ylabel("Underlying mid")
    ax1.set_title("Overview — underlying and voucher mid prices")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    cmap = plt.get_cmap("turbo")
    keys = list(STRIKES.keys())
    for i, k in enumerate(keys):
        if k in wide:
            ax2.plot(wide.index, wide[k], lw=0.4, color=cmap(i / max(1, len(keys) - 1)),
                     label=f"{k} (K={STRIKES[k]})")
    ax2.set_yscale("log")
    ax2.set_ylabel("Voucher mid (log)")
    ax2.set_xlabel("Global tick (days laid end-to-end)")
    ax2.legend(loc="upper right", ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    for d in DAYS[1:]:
        ax1.axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax2.axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)

    fig.tight_layout()
    return fig


def fig_extrinsic(wide: pd.DataFrame):
    """For each strike, plot mid (solid) and intrinsic (dashed) — gap is extrinsic value."""
    keys = [k for k in STRIKES if k in wide]
    fig, axes = plt.subplots(len(keys), 1, figsize=(14, 1.6 * len(keys)), sharex=True)
    S = wide[UNDERLYING]
    cmap = plt.get_cmap("turbo")
    for i, k in enumerate(keys):
        K = STRIKES[k]
        ax = axes[i]
        intrinsic = (S - K).clip(lower=0)
        ax.plot(wide.index, wide[k], lw=0.6, color=cmap(i / max(1, len(keys) - 1)), label="mid")
        ax.plot(wide.index, intrinsic, lw=0.6, color="black", ls="--", label="intrinsic = max(S-K, 0)")
        ax.set_ylabel(f"{k}\nK={K}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Global tick")
    fig.suptitle("Voucher mid vs intrinsic — gap is extrinsic (time + vol value)", y=1.0)
    fig.tight_layout()
    return fig


def fig_iv_timeseries(wide: pd.DataFrame):
    """Compute IV at each tick for each strike, plot as a time series."""
    keys = [k for k in STRIKES if k in wide]
    # downsample for speed
    idx = wide.index[::DOWNSAMPLE]
    sub = wide.loc[idx]
    iv_data = {}
    for k in keys:
        K = STRIKES[k]
        ivs = []
        for ts, row in sub.iterrows():
            S = row[UNDERLYING]; T = row["T"]; price = row[k]
            if pd.isna(S) or pd.isna(T) or pd.isna(price):
                ivs.append(np.nan); continue
            iv = implied_vol(price, S, K, T)
            ivs.append(iv if iv is not None else np.nan)
        iv_data[k] = ivs

    fig, ax = plt.subplots(figsize=(20, 6))
    cmap = plt.get_cmap("turbo")
    for i, k in enumerate(keys):
        ax.plot(sub.index, iv_data[k], lw=0.7,
                color=cmap(i / max(1, len(keys) - 1)),
                label=f"{k} (K={STRIKES[k]})")
    ax.set_xlabel("Global tick")
    ax.set_ylabel(f"Implied vol (annualized, T scale = 1/{int(TIME_UNIT_DAYS_PER_YEAR)})")
    ax.set_title("Implied volatility over time, per strike")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    for d in DAYS[1:]:
        ax.axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, iv_data, sub


def fig_iv_smile(wide: pd.DataFrame, iv_data: dict, sub: pd.DataFrame):
    """Avg IV vs strike, one line per day."""
    sub = sub.copy()
    sub["day_idx"] = (sub.index // TICKS_PER_DAY).astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    for di, day in enumerate(DAYS):
        mask = (sub["day_idx"] == (day - DAYS[0]))
        if not mask.any():
            continue
        avg_S = sub.loc[mask, UNDERLYING].mean()
        Ks, ivs_avg = [], []
        for k, vals in iv_data.items():
            arr = np.array(vals)[mask.values]
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            Ks.append(STRIKES[k]); ivs_avg.append(arr.mean())
        ax.plot(Ks, ivs_avg, "o-", color=cmap(di / max(1, len(DAYS) - 1)),
                label=f"Day {day} (TTE={DAY_TTE_START[day]}d, avg S={avg_S:.1f})")
    ax.axvline(sub[UNDERLYING].mean(), color="black", lw=0.7, ls=":", label=f"avg S = {sub[UNDERLYING].mean():.0f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Average IV")
    ax.set_title("Implied volatility smile — average IV vs strike, by day")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_implied_underlying(wide: pd.DataFrame):
    """Implied S from deep-ITM calls vs actual S. S_impl = mid_K + K."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    S = wide[UNDERLYING]
    ax1.plot(wide.index, S, lw=0.5, color="black", label=f"actual {UNDERLYING}")
    diffs = {}
    for k in DEEP_ITM_KEYS:
        if k not in wide:
            continue
        K = STRIKES[k]
        S_impl = wide[k] + K
        ax1.plot(wide.index, S_impl, lw=0.5, alpha=0.7, label=f"S implied from {k} (mid + {K})")
        diffs[k] = S_impl - S
    ax1.set_ylabel("Underlying mid")
    ax1.set_title("Actual S vs implied S from deep-ITM calls")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for k, d in diffs.items():
        ax2.plot(wide.index, d, lw=0.5, label=f"{k} implied − actual")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Implied − actual S")
    ax2.set_xlabel("Global tick")
    ax2.set_title("Diff: positive → vouchers pricing higher S than spot (sell voucher / buy underlying)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for d in DAYS[1:]:
        ax1.axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax2.axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, diffs


def fig_noarb(wide: pd.DataFrame):
    """Monotonicity (mid should be non-increasing in K) + convexity butterfly check."""
    keys = [k for k in STRIKES if k in wide]
    keys_sorted = sorted(keys, key=lambda k: STRIKES[k])
    Ks = [STRIKES[k] for k in keys_sorted]

    # monotonicity violations: mid(K_i) < mid(K_{i+1})
    mono_viol = pd.Series(0, index=wide.index, dtype=float)
    for i in range(len(keys_sorted) - 1):
        a, b = keys_sorted[i], keys_sorted[i + 1]
        diff = wide[b] - wide[a]            # should be <= 0
        mono_viol += (diff > 0).astype(int) * diff.clip(lower=0)

    # convexity: butterfly C(K1) + C(K3) - 2*C(K2) should be >= 0 for K1<K2<K3 evenly spaced
    butt_viol = pd.Series(0.0, index=wide.index)
    for i in range(len(keys_sorted) - 2):
        a, b, c = keys_sorted[i], keys_sorted[i + 1], keys_sorted[i + 2]
        Ka, Kb, Kc = STRIKES[a], STRIKES[b], STRIKES[c]
        # weights so the spread is symmetric around Kb
        w_a = (Kc - Kb) / (Kc - Ka)
        w_c = (Kb - Ka) / (Kc - Ka)
        butt = w_a * wide[a] + w_c * wide[c] - wide[b]   # should be >= 0
        butt_viol += (-butt).clip(lower=0)               # accumulate magnitude of violation

    fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
    axes[0].plot(wide.index, mono_viol, lw=0.5, color="crimson")
    axes[0].set_title("No-arb: monotonicity violation magnitude (sum over adjacent strike pairs)")
    axes[0].set_ylabel("Σ max(mid_high - mid_low, 0)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(wide.index, butt_viol, lw=0.5, color="darkorange")
    axes[1].set_title("No-arb: butterfly (convexity) violation magnitude")
    axes[1].set_ylabel("Σ max(2*Cb - (wa*Ca + wc*Cc), 0)")
    axes[1].set_xlabel("Global tick")
    axes[1].grid(True, alpha=0.3)

    for d in DAYS[1:]:
        axes[0].axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)
        axes[1].axvline((d - DAYS[0]) * TICKS_PER_DAY, color="gray", lw=0.5, ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, mono_viol, butt_viol


# ── Driver ───────────────────────────────────────────────────────────────
def main():
    print("Loading round 3 data…")
    df = load_prices()
    wide = pivot_mids(df)
    print(f"  ticks: {len(wide)}")
    print(f"  underlying mid range: {wide[UNDERLYING].min():.1f} -> {wide[UNDERLYING].max():.1f}")
    print(f"  TTE range (days): {wide['T'].min()*TIME_UNIT_DAYS_PER_YEAR:.3f} -> {wide['T'].max()*TIME_UNIT_DAYS_PER_YEAR:.3f}")

    print("\nVoucher mid stats:")
    for k, K in STRIKES.items():
        if k in wide:
            mid = wide[k]
            avg_intrinsic = (wide[UNDERLYING] - K).clip(lower=0).mean()
            print(f"  {k:<10} K={K}  mid {mid.mean():>9.2f} (range {mid.min():>7.2f} → {mid.max():>7.2f})  avg intrinsic {avg_intrinsic:>7.2f}  avg extrinsic {mid.mean()-avg_intrinsic:>6.2f}")

    figs = []
    print("\n[1/6] overview…");                 figs.append(fig_overview(wide))
    print("[2/6] intrinsic vs market…");        figs.append(fig_extrinsic(wide))
    print("[3/6] IV time series…");             fig3, iv_data, sub = fig_iv_timeseries(wide); figs.append(fig3)
    print("[4/6] IV smile…");                   figs.append(fig_iv_smile(wide, iv_data, sub))
    print("[5/6] implied underlying…");         fig5, diffs = fig_implied_underlying(wide); figs.append(fig5)
    print("[6/6] no-arb diagnostics…");         fig6, mono, butt = fig_noarb(wide); figs.append(fig6)

    print("\n=== Implied-vs-actual S summary (deep ITM) ===")
    for k, d in diffs.items():
        print(f"  {k}: mean diff {d.mean():+.3f}  std {d.std():.3f}  |diff|>0.5 ticks: {(d.abs()>0.5).mean()*100:.1f}%")

    print("\n=== No-arb summary ===")
    print(f"  Ticks with monotonicity violation: {(mono > 0).sum()} / {len(mono)} ({(mono>0).mean()*100:.2f}%)")
    print(f"  Ticks with butterfly  violation: {(butt > 0).sum()} / {len(butt)} ({(butt>0).mean()*100:.2f}%)")

    plt.show()


if __name__ == "__main__":
    main()
