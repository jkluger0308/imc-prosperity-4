"""
Standalone Black–Scholes helpers for IMC Prosperity-style submission (no local imports).

European call, risk-free rate r = 0 (SeaShells numéraire). T in years; sigma annualized.
Rename to imc_methods.py if you need ``import imc_methods`` (hyphens are invalid in imports).
"""
from __future__ import annotations

import math
from typing import Optional


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def blackscholes(T: float, sigma: float, K: float, S: float) -> float:
    """
    Fair price of a European call (r = 0).

    T: time to expiry in years
    sigma: annualized volatility (std of log returns)
    K: strike
    S: spot / underlying
    """
    T, sigma, K, S = float(T), float(sigma), float(K), float(S)
    if S <= 0 or K <= 0:
        return max(S - K, 0.0)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / st
    d2 = d1 - st
    return S * _ncdf(d1) - K * _ncdf(d2)


def impliedvol(
    T: float,
    K: float,
    S: float,
    call_price: float,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    xtol: float = 1e-8,
) -> Optional[float]:
    """
    Implied sigma such that blackscholes(T, sigma, K, S) == call_price (r = 0).

    Returns None if no bracketed root on [lo, hi] (e.g. price at/below intrinsic or >= S).
    """
    T, K, S = float(T), float(K), float(S)
    C = float(call_price)
    intrinsic = max(S - K, 0.0)
    if T <= 0 or C <= intrinsic + 1e-12 or C >= S - 1e-12:
        return None

    def res(sig: float) -> float:
        return blackscholes(T, sig, K, S) - C

    a, b = float(lo), float(hi)
    fa, fb = res(a), res(b)
    if fa * fb > 0:
        return None
    for _ in range(100):
        m = 0.5 * (a + b)
        fm = res(m)
        if abs(fm) < xtol * max(1.0, abs(C)) or (b - a) < xtol:
            return float(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))
