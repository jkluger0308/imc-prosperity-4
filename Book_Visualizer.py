#!/usr/bin/env python3
"""
IMC Prosperity order-book dashboard: parse backtester / front-test logs and explore
microstructure (book, trades, PnL, position, logs) interactively.

Run:  python orderbook_dashboard.py [--root PATH] [--port 8050]
See ORDERBOOK_DASHBOARD_GUIDE.md for how to read the visualization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Optional Dash (fail fast with install hint)
# -----------------------------------------------------------------------------
try:
    from dash import Dash, Input, Output, State, dcc, html, callback_context, no_update
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install dashboard dependencies: pip install -r requirements-dashboard.txt\n" + str(e)
    ) from e

# -----------------------------------------------------------------------------
# Paths & cache
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOTS = [SCRIPT_DIR / "backtests", Path.home() / "Downloads"]
CACHE_DIR = SCRIPT_DIR / ".cache_dashboard"
GUIDE_PATH = SCRIPT_DIR / "ORDERBOOK_DASHBOARD_GUIDE.md"
# Bump when parsed book/trade metrics change so stale pickles are not reused.
PARSED_CACHE_VERSION = "5"


def _load_guide_markdown() -> str:
    if GUIDE_PATH.is_file():
        return GUIDE_PATH.read_text(encoding="utf-8", errors="replace")
    return "_Write-up not found: `ORDERBOOK_DASHBOARD_GUIDE.md`._"


def _relayout_x_range(rel: Optional[dict]) -> Optional[tuple[float, float]]:
    if not rel:
        return None
    if rel.get("xaxis.autorange") is True:
        return None
    if "xaxis.range" in rel:
        r = rel["xaxis.range"]
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            try:
                return float(r[0]), float(r[1])
            except (TypeError, ValueError):
                pass
    try:
        x0 = rel.get("xaxis.range[0]")
        x1 = rel.get("xaxis.range[1]")
        if x0 is not None and x1 is not None:
            return float(x0), float(x1)
    except (TypeError, ValueError):
        pass
    return None


def _file_fingerprint(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_load(key: str) -> Optional[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.pkl"
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(key: str, obj: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.pkl"
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------------------------
# JSON helpers (Prosperity trade history often has trailing commas)
# -----------------------------------------------------------------------------
def _strip_json_trailing_commas(s: str) -> str:
    s = re.sub(r",(\s*})", r"\1", s)
    s = re.sub(r",(\s*])", r"\1", s)
    return s


def _parse_json_loose(s: str) -> Any:
    s = _strip_json_trailing_commas(s.strip())
    return json.loads(s)


def _iter_json_objects(block: str) -> list[dict]:
    """Parse concatenated top-level JSON objects { ... }{ ... }."""
    out: list[dict] = []
    i, n = 0, len(block)
    while i < n:
        if block[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        while i < n:
            c = block[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    chunk = block[start:i]
                    try:
                        out.append(_parse_json_loose(chunk))
                    except json.JSONDecodeError:
                        pass
                    break
            i += 1
        else:
            break
    return out


# -----------------------------------------------------------------------------
# Log parsing: activities CSV (shared string format)
# -----------------------------------------------------------------------------
ACTIVITIES_HEADER = (
    "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;"
    "bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;"
    "ask_price_3;ask_volume_3;mid_price;profit_and_loss"
)


def _parse_activities_csv(text: str) -> pd.DataFrame:
    text = text.strip()
    if not text:
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    # Drop duplicate header inside body
    if lines[0].startswith("day;timestamp;product"):
        body = "\n".join(lines)
    else:
        body = ACTIVITIES_HEADER + "\n" + "\n".join(lines)
    from io import StringIO

    df = pd.read_csv(StringIO(body), sep=";")
    for c in df.columns:
        if c not in ("product",):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _extract_between(text: str, start: str, end: Optional[str]) -> Optional[str]:
    i = text.find(start)
    if i < 0:
        return None
    i += len(start)
    if end:
        j = text.find(end, i)
        if j < 0:
            return text[i:].strip()
        return text[i:j].strip()
    return text[i:].strip()


def parse_backtester_log(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")

    sandbox_block = _extract_between(raw, "Sandbox logs:", "\nActivities log:")
    logs_rows: list[dict] = []
    if sandbox_block:
        for obj in _iter_json_objects(sandbox_block):
            ts = obj.get("timestamp")
            sb = obj.get("sandboxLog") or ""
            lm = obj.get("lambdaLog") or ""
            if sb:
                logs_rows.append({"timestamp": ts, "source": "sandbox", "message": str(sb)})
            if lm:
                logs_rows.append({"timestamp": ts, "source": "lambda", "message": str(lm)})

    act_text = _extract_between(raw, "Activities log:", "\nTrade History:")
    if act_text is None:
        act_text = _extract_between(raw, "Activities log:", None)
        if act_text and "Trade History:" in act_text:
            act_text = act_text.split("Trade History:")[0].strip()
    book_df = _parse_activities_csv(act_text or "")

    trades_df = pd.DataFrame()
    th = raw.find("Trade History:")
    if th >= 0:
        rest = raw[th + len("Trade History:") :].strip()
        # slice from first [ to last ]
        a = rest.find("[")
        b = rest.rfind("]")
        if a >= 0 and b > a:
            try:
                arr = _parse_json_loose(rest[a : b + 1])
                trades_df = pd.DataFrame(arr)
            except Exception:
                trades_df = pd.DataFrame()

    logs_df = pd.DataFrame(logs_rows) if logs_rows else pd.DataFrame(columns=["timestamp", "source", "message"])
    return {
        "book_df": book_df,
        "trades_df": trades_df,
        "logs_df": logs_df,
        "positions_df": pd.DataFrame(),
        "graphlog_df": pd.DataFrame(),
        "submission_meta": {},
        "source": "backtester_log",
    }


def _empty_parse_result(source: str) -> dict[str, Any]:
    return {
        "book_df": pd.DataFrame(),
        "trades_df": pd.DataFrame(),
        "logs_df": pd.DataFrame(),
        "positions_df": pd.DataFrame(),
        "graphlog_df": pd.DataFrame(),
        "submission_meta": {},
        "source": source,
    }


def _parse_graph_log_csv(text: str) -> pd.DataFrame:
    """Front-test `graphLog` field: semicolon CSV, typically `timestamp;value`."""
    text = text.strip()
    if not text:
        return pd.DataFrame()
    from io import StringIO

    df = pd.read_csv(StringIO(text), sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) < 2:
        return pd.DataFrame()
    c0, c1 = df.columns[0], df.columns[1]
    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(df[c0], errors="coerce"),
            "value": pd.to_numeric(df[c1], errors="coerce"),
        }
    )
    return out.dropna(subset=["timestamp", "value"])


def _coerce_trades_frame(val: Any) -> pd.DataFrame:
    """Build a trades DataFrame from list of dicts or dict[symbol] -> list of dicts."""
    if val is None or val == [] or val == {}:
        return pd.DataFrame()
    if isinstance(val, list):
        if not val or not all(isinstance(x, dict) for x in val):
            return pd.DataFrame()
        return pd.DataFrame(val)
    if isinstance(val, dict):
        rows: list[dict] = []
        for sym, arr in val.items():
            if not isinstance(arr, list):
                continue
            for tr in arr:
                if not isinstance(tr, dict):
                    continue
                r = dict(tr)
                if "product" not in r and "symbol" not in r:
                    r["symbol"] = sym
                rows.append(r)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    return pd.DataFrame()


def parse_fronttest_dict(root: dict[str, Any], source_tag: str = "fronttest_json") -> dict[str, Any]:
    act = root.get("activitiesLog") or root.get("activities_log")
    book_df = _parse_activities_csv(act) if isinstance(act, str) else pd.DataFrame()

    trades_df = pd.DataFrame()
    for key in (
        "tradeHistory",
        "trade_history",
        "trades",
        "ownTrades",
        "own_trades",
        "marketTrades",
        "market_trades",
    ):
        if key not in root or not root[key]:
            continue
        try:
            trades_df = _coerce_trades_frame(root[key])
        except Exception:
            trades_df = pd.DataFrame()
        if not trades_df.empty:
            break

    logs_rows: list[dict[str, Any]] = []
    for key in ("sandboxLogs", "sandbox_logs", "sandboxLog", "logs"):
        if key not in root or not root[key]:
            continue
        val = root[key]
        if not isinstance(val, list):
            continue
        for item in val:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp", 0)
            sb = item.get("sandboxLog") or item.get("message") or ""
            lm = item.get("lambdaLog") or ""
            if sb:
                logs_rows.append({"timestamp": ts, "source": "sandbox", "message": str(sb)})
            if lm:
                logs_rows.append({"timestamp": ts, "source": "lambda", "message": str(lm)})
        if logs_rows:
            break
    logs_df = pd.DataFrame(logs_rows) if logs_rows else pd.DataFrame(columns=["timestamp", "source", "message"])

    positions_rows: list[dict[str, Any]] = []
    pos_raw = root.get("positions")
    if isinstance(pos_raw, list):
        for item in pos_raw:
            if not isinstance(item, dict):
                continue
            sym = item.get("symbol") or item.get("product")
            q = item.get("quantity", item.get("qty"))
            positions_rows.append(
                {
                    "product": str(sym) if sym is not None else "",
                    "quantity": pd.to_numeric(q, errors="coerce"),
                }
            )
    positions_df = pd.DataFrame(positions_rows) if positions_rows else pd.DataFrame()

    graphlog_df = pd.DataFrame()
    gl = root.get("graphLog") or root.get("graph_log")
    if isinstance(gl, str) and gl.strip():
        try:
            graphlog_df = _parse_graph_log_csv(gl)
        except Exception:
            graphlog_df = pd.DataFrame()

    submission_meta: dict[str, Any] = {}
    for mk in ("profit", "round", "status", "submissionId", "submission_id"):
        if mk in root and root[mk] is not None:
            submission_meta[mk] = root[mk]

    return {
        "book_df": book_df,
        "trades_df": trades_df,
        "logs_df": logs_df,
        "positions_df": positions_df,
        "graphlog_df": graphlog_df,
        "submission_meta": submission_meta,
        "source": source_tag,
    }


def parse_fronttest_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        root = _parse_json_loose(raw)
    except json.JSONDecodeError:
        return _empty_parse_result("json_error")
    if not isinstance(root, dict):
        return _empty_parse_result("json_error")
    return parse_fronttest_dict(root, "fronttest_json")


def parse_log_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    suf = path.suffix.lower()
    if suf == ".json":
        return parse_fronttest_json(path)
    if suf == ".log":
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if raw.startswith("{"):
            try:
                root = _parse_json_loose(raw)
                if isinstance(root, dict) and (root.get("activitiesLog") or root.get("activities_log")):
                    return parse_fronttest_dict(root, "fronttest_json_log")
            except json.JSONDecodeError:
                pass
    return parse_backtester_log(path)


# -----------------------------------------------------------------------------
# Metrics: wallmid, overlays, normalization
# -----------------------------------------------------------------------------
BID_P = ["bid_price_1", "bid_price_2", "bid_price_3"]
ASK_P = ["ask_price_1", "ask_price_2", "ask_price_3"]
BID_V = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
ASK_V = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]


def add_book_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Wall mid: center of the full displayed book (L1–L3). Deepest bid = lowest
    # non-null bid among levels; deepest ask = highest non-null ask (touch uses L1 only).
    bp = out.reindex(columns=BID_P)
    ap = out.reindex(columns=ASK_P)
    worst_bid = bp.min(axis=1, skipna=True)
    worst_ask = ap.max(axis=1, skipna=True)
    out["wallmid"] = (worst_bid + worst_ask) / 2.0
    out["best_bid"] = bp.iloc[:, 0]
    out["best_ask"] = ap.iloc[:, 0]
    out["mid_touch"] = (out["best_bid"] + out["best_ask"]) / 2.0
    bv = out.reindex(columns=BID_V).fillna(0).sum(axis=1)
    av = out.reindex(columns=ASK_V).fillna(0).abs().sum(axis=1)
    out["imbalance"] = bv - av
    out["spread"] = out["best_ask"] - out["best_bid"]
    return out


def normalize_price_series(y: pd.Series, base: pd.Series, mode: str) -> pd.Series:
    if mode == "none" or base is None:
        return y
    aligned = base.reindex(y.index)
    if mode == "subtract":
        return y - aligned
    if mode == "zscore":
        d = y - aligned
        s = d.std()
        if s and s > 1e-12:
            return d / s
        return d * 0.0
    return y


# -----------------------------------------------------------------------------
# Trade classification & position
# -----------------------------------------------------------------------------
def _side_is_submission(s: Any) -> bool:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return False
    t = str(s).strip().upper()
    return t == "SUBMISSION"


def _price_touch_tol(bb: float, ba: float, price: float) -> float:
    """Numeric tolerance for integer Prosperity prices (avoid float noise)."""
    spread = float(ba) - float(bb)
    return max(1e-9, min(0.51, 0.05 * spread if spread > 0 else 0.51))


def _infer_maker_taker(
    is_own: bool,
    buyer_sub: bool,
    seller_sub: bool,
    price: float,
    bb: Any,
    ba: Any,
) -> str:
    """
    Maker ≈ passive / resting fill at L1; taker ≈ aggressive (cross or lift/hit).

    Own (SUBMISSION): passive bid lifted (buy at bid) or passive ask lifted (sell at ask) → M;
    buy at/through ask, sell at/through bid, or inside spread → T.

    Others: print strictly inside the touch → T (aggressive / price improvement);
    print at or through best bid or ask → M (matches at touch in many tapes where both sides are passive).
    """
    if pd.isna(bb) or pd.isna(ba):
        return "U"
    try:
        bb_f = float(bb)
        ba_f = float(ba)
    except (TypeError, ValueError):
        return "U"
    if ba_f < bb_f:
        return "U"
    tol = _price_touch_tol(bb_f, ba_f, price)

    if is_own:
        if buyer_sub:
            if price <= bb_f + tol:
                return "M"
            return "T"
        if seller_sub:
            if price >= ba_f - tol:
                return "M"
            return "T"
        return "U"

    if bb_f + tol < price < ba_f - tol:
        return "T"
    return "M"


def classify_trades(trades_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    t = trades_df.copy()
    t.columns = [str(c).strip() for c in t.columns]
    sym = t.get("symbol", t.get("product", pd.Series([""] * len(t))))
    t["product"] = sym.astype(str)

    buyer = t.get("buyer", pd.Series([None] * len(t)))
    seller = t.get("seller", pd.Series([None] * len(t)))
    buy_sub = buyer.map(_side_is_submission)
    sell_sub = seller.map(_side_is_submission)
    t["is_own"] = buy_sub | sell_sub

    t["_ts"] = pd.to_numeric(t["timestamp"], errors="coerce")
    t["_price"] = pd.to_numeric(t.get("price", 0), errors="coerce")

    if book_df.empty:
        t["role"] = "U"
    else:
        b = book_df.copy()
        b["_ts"] = pd.to_numeric(b["timestamp"], errors="coerce")
        b["product"] = b["product"].astype(str)
        book_side = b[["_ts", "product", "bid_price_1", "ask_price_1"]].dropna(subset=["_ts"])
        book_side = book_side.sort_values(["product", "_ts"])
        book_side = book_side.rename(columns={"bid_price_1": "_bb", "ask_price_1": "_ba"})

        parts: list[pd.DataFrame] = []
        for prod, tt in t.groupby("product", sort=False):
            tt = tt.sort_values("_ts").reset_index(drop=True)
            bk = book_side[book_side["product"] == prod][["_ts", "_bb", "_ba"]].drop_duplicates("_ts", keep="last")
            if bk.empty:
                tt["role"] = "U"
                parts.append(tt)
                continue
            try:
                mg = pd.merge_asof(tt, bk, left_on="_ts", right_on="_ts", direction="backward")
            except ValueError:
                tt["role"] = "U"
                parts.append(tt)
                continue
            roles: list[str] = []
            for i in range(len(mg)):
                row = mg.iloc[i]
                p = row["_price"]
                if pd.isna(row["_ts"]) or pd.isna(p):
                    roles.append("U")
                    continue
                is_own = bool(row["is_own"])
                bs = _side_is_submission(row.get("buyer"))
                ss = _side_is_submission(row.get("seller"))
                roles.append(_infer_maker_taker(is_own, bs, ss, float(p), row["_bb"], row["_ba"]))
            mg["role"] = roles
            parts.append(mg)

        merged = pd.concat(parts, ignore_index=True)
        merged = merged.drop(columns=["_ts", "_price"], errors="ignore")
        t = merged

    # size bucket by product quantiles (own trades included for quantiles)
    if "quantity" in t.columns:
        t["qty_abs"] = pd.to_numeric(t["quantity"], errors="coerce").fillna(0.0).abs()
    else:
        t["qty_abs"] = 0.0
    t["size_bucket"] = "S"
    for prod, g in t.groupby("product"):
        q75 = g["qty_abs"].quantile(0.75)
        if pd.notna(q75) and q75 > 0:
            t.loc[g.index, "size_bucket"] = (g["qty_abs"] > q75).map({True: "B", False: "S"})
    return t


def position_from_own_trades(trades_df: pd.DataFrame, product: str) -> pd.DataFrame:
    if trades_df.empty or "timestamp" not in trades_df.columns:
        return pd.DataFrame(columns=["timestamp", "position"])
    t = trades_df.copy()
    if "product" not in t.columns and "symbol" in t.columns:
        t["product"] = t["symbol"]
    if "product" not in t.columns:
        return pd.DataFrame(columns=["timestamp", "position"])
    t = t[t["product"].astype(str) == str(product)].sort_values("timestamp")
    pos = 0.0
    rows: list[tuple[int, float]] = []
    buyer = t.get("buyer", pd.Series([""] * len(t)))
    seller = t.get("seller", pd.Series([""] * len(t)))
    qty = pd.to_numeric(t.get("quantity", pd.Series([0] * len(t))), errors="coerce").fillna(0.0).abs()
    ts_col = pd.to_numeric(t["timestamp"], errors="coerce")
    for i in range(len(t)):
        q = float(qty.iloc[i])
        if _side_is_submission(buyer.iloc[i]):
            pos += q
        elif _side_is_submission(seller.iloc[i]):
            pos -= q
        ts_i = ts_col.iloc[i]
        if pd.notna(ts_i):
            rows.append((int(ts_i), pos))
    return pd.DataFrame(rows, columns=["timestamp", "position"])


# -----------------------------------------------------------------------------
# Downsampling
# -----------------------------------------------------------------------------
def thin_series(df: pd.DataFrame, ts_col: str, max_points: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


# -----------------------------------------------------------------------------
# Build Plotly figures
# -----------------------------------------------------------------------------
def build_main_figure(
    book: pd.DataFrame,
    trades: pd.DataFrame,
    product: str,
    levels: list[int],
    norm_mode: str,
    norm_base: str,
    trade_filters: dict[str, bool],
    qty_min: Optional[float],
    qty_max: Optional[float],
    max_points: int,
    x_range: Optional[tuple[float, float]] = None,
    overlay_metrics: Optional[list[str]] = None,
    size_buckets: Optional[list[str]] = None,
) -> go.Figure:
    b = book[book["product"].astype(str) == product].copy()
    if b.empty:
        fig = go.Figure()
        fig.update_layout(title="No book data for product", template="plotly_dark")
        return fig

    if x_range:
        lo, hi = x_range
        b = b[(b["timestamp"] >= lo) & (b["timestamp"] <= hi)]

    b = thin_series(b.sort_values("timestamp"), "timestamp", max_points)
    b = add_book_metrics(b)

    base_col = norm_base if norm_base in b.columns else "wallmid"
    base_s = b[base_col] if base_col in b.columns else pd.Series(0.0, index=b.index)

    fig = go.Figure()

    for lvl in levels:
        if 1 <= lvl <= 3:
            bp, ap = f"bid_price_{lvl}", f"ask_price_{lvl}"
            if bp in b.columns:
                yb = normalize_price_series(b[bp], base_s, norm_mode)
                fig.add_trace(
                    go.Scattergl(
                        x=b["timestamp"],
                        y=yb,
                        mode="markers",
                        name=f"Bid L{lvl}",
                        marker=dict(size=6, color="royalblue", opacity=0.65),
                    )
                )
            if ap in b.columns:
                ya = normalize_price_series(b[ap], base_s, norm_mode)
                fig.add_trace(
                    go.Scattergl(
                        x=b["timestamp"],
                        y=ya,
                        mode="markers",
                        name=f"Ask L{lvl}",
                        marker=dict(size=6, color="tomato", opacity=0.65),
                    )
                )

    # overlays
    if "wallmid" in b.columns:
        yw = normalize_price_series(b["wallmid"], base_s, norm_mode)
        fig.add_trace(
            go.Scattergl(
                x=b["timestamp"],
                y=yw,
                mode="lines",
                name="WallMid",
                line=dict(color="white", width=1, dash="dot"),
            )
        )
    if "mid_touch" in b.columns:
        ym = normalize_price_series(b["mid_touch"], base_s, norm_mode)
        fig.add_trace(
            go.Scattergl(
                x=b["timestamp"],
                y=ym,
                mode="lines",
                name="Touch mid",
                line=dict(color="cyan", width=1),
            )
        )

    om = overlay_metrics or []
    has_spread = "spread" in om and "spread" in b.columns
    has_imb = "imbalance" in om and "imbalance" in b.columns
    use_y2 = has_spread or has_imb
    if has_spread:
        fig.add_trace(
            go.Scattergl(
                x=b["timestamp"],
                y=b["spread"],
                mode="lines",
                name="Spread",
                line=dict(color="lime", width=1),
                yaxis="y2",
            )
        )
    if has_imb:
        fig.add_trace(
            go.Scattergl(
                x=b["timestamp"],
                y=b["imbalance"],
                mode="lines",
                name="Imbalance (bid−ask vol)",
                line=dict(color="magenta", width=1),
                yaxis="y2",
            )
        )

    if not trades.empty and "product" in trades.columns and "timestamp" in trades.columns and "price" in trades.columns:
        tr = trades[trades["product"].astype(str) == product].copy()
        tr["timestamp"] = pd.to_numeric(tr["timestamp"], errors="coerce")
        tr = tr[tr["timestamp"].notna()]
        if x_range:
            lo, hi = x_range
            tr = tr[(tr["timestamp"] >= lo) & (tr["timestamp"] <= hi)]
        if "qty_abs" in tr.columns:
            qa = tr["qty_abs"].abs()
        elif "quantity" in tr.columns:
            qa = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0.0).abs()
        else:
            qa = pd.Series(0.0, index=tr.index)
        if qty_min is not None:
            tr = tr[qa >= qty_min]
        if qty_max is not None:
            tr = tr[qa <= qty_max]

        def _show_trade_row(row: pd.Series) -> bool:
            own = bool(row.get("is_own", False))
            r = str(row.get("role", "U"))
            if own:
                return bool(trade_filters.get("F", True))
            if r == "M":
                return bool(trade_filters.get("M", True))
            if r == "T":
                return bool(trade_filters.get("T", True))
            return bool(trade_filters.get("U", True))

        if "role" in tr.columns:
            tr = tr[tr.apply(_show_trade_row, axis=1)]

        sb = size_buckets if size_buckets is not None else ["S", "B"]
        if sb and "size_bucket" in tr.columns:
            tr = tr[tr["size_bucket"].astype(str).isin(sb)]

        tr = thin_series(tr.sort_values("timestamp"), "timestamp", max(400, min(8000, max_points // 2)))

        # Map trade price y: need normalization base at timestamp — merge_asof
        if not b.empty and base_col in b.columns:
            base_df = b[["timestamp", base_col]].dropna().sort_values("timestamp")
            tr = tr.sort_values("timestamp")
            merged = pd.merge_asof(
                tr,
                base_df.rename(columns={base_col: "_base"}),
                on="timestamp",
                direction="nearest",
            )
        else:
            merged = tr.copy()
            merged["_base"] = 0.0

        price = pd.to_numeric(merged["price"], errors="coerce")
        if norm_mode == "none":
            merged["yp"] = price
        elif norm_mode == "subtract":
            merged["yp"] = price - merged["_base"].fillna(0)
        else:
            d = price - merged["_base"].fillna(0)
            s = float(d.std()) if len(d) > 1 else 1.0
            merged["yp"] = d / s if s > 1e-12 else d * 0.0

        for col in ("price", "quantity", "buyer", "seller"):
            if col not in merged.columns:
                merged[col] = float("nan") if col in ("price", "quantity") else ""

        if "is_own" not in merged.columns:
            merged["is_own"] = False

        trace_specs: list[tuple[bool, str, str, str, str]] = [
            (True, "M", "x", "yellow", "Own maker (SUBMISSION)"),
            (True, "T", "triangle-up", "yellow", "Own taker (SUBMISSION)"),
            (False, "M", "square", "lightgreen", "Maker (others)"),
            (False, "T", "triangle-up", "orange", "Taker (others)"),
            (False, "U", "diamond", "gray", "Unknown book"),
        ]
        for own_want, role, marker, color, tname in trace_specs:
            sub = merged[(merged["is_own"] == own_want) & (merged["role"] == role)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=sub["timestamp"],
                    y=sub["yp"],
                    mode="markers",
                    name=tname,
                    marker=dict(size=10, symbol=marker, color=color, line=dict(width=0.5, color="black")),
                    customdata=sub[["price", "quantity", "buyer", "seller"]].values,
                    hovertemplate="price=%{customdata[0]} qty=%{customdata[1]} buyer=%{customdata[2]} seller=%{customdata[3]}<extra></extra>",
                )
            )

    ytitle = "Price" if norm_mode == "none" else f"Price ({norm_mode} vs {norm_base})"
    layout: dict[str, Any] = dict(
        template="plotly_dark",
        height=520,
        title=f"Order book + trades — {product}",
        xaxis_title="Timestamp",
        yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        dragmode="zoom",
    )
    if use_y2:
        layout["yaxis2"] = dict(
            title="Spread / imbalance",
            overlaying="y",
            side="right",
            showgrid=False,
        )
    fig.update_layout(**layout)
    if x_range:
        lo, hi = x_range
        fig.update_xaxes(range=[lo, hi])
    return fig


def build_pnl_figure(book: pd.DataFrame, product: str) -> go.Figure:
    b = book[book["product"].astype(str) == product]
    fig = go.Figure()
    if b.empty or "profit_and_loss" not in b.columns:
        fig.update_layout(title="PnL (no book rows for product)", template="plotly_dark", height=220)
        return fig
    fig.add_trace(
        go.Scatter(
            x=b["timestamp"],
            y=b["profit_and_loss"],
            mode="lines",
            name="PnL",
            line=dict(color="gold"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"PnL — {product} (activities log)",
    )
    return fig


def build_total_pnl_graphlog_figure(graphlog_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if graphlog_df.empty or "timestamp" not in graphlog_df.columns or "value" not in graphlog_df.columns:
        fig.update_layout(
            title="Total PnL (no `graphLog` in file)",
            template="plotly_dark",
            height=200,
            margin=dict(l=40, r=20, t=45, b=30),
        )
        return fig
    gl = graphlog_df.sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=gl["timestamp"],
            y=gl["value"],
            mode="lines",
            name="Total PnL",
            line=dict(color="aquamarine", width=1.5),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=40, r=20, t=45, b=30),
        title="Total PnL (`graphLog` — full submission, all products)",
    )
    return fig


def build_position_figure(
    trades: pd.DataFrame,
    product: str,
    book: pd.DataFrame,
    positions_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    pos_df = position_from_own_trades(trades, product)
    fig = go.Figure()
    if not pos_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pos_df["timestamp"],
                y=pos_df["position"],
                mode="lines+markers",
                name="Position",
                line=dict(color="violet"),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=220,
            margin=dict(l=40, r=20, t=40, b=40),
            title="Position (from SUBMISSION trades)",
        )
        return fig

    if positions_df is not None and not positions_df.empty and "product" in positions_df.columns:
        sub = positions_df[positions_df["product"].astype(str) == str(product)]
        if not sub.empty:
            qraw = sub.iloc[0].get("quantity")
            if pd.notna(qraw):
                q = float(qraw)
                b = (
                    book[book["product"].astype(str) == str(product)]
                    if not book.empty and "product" in book.columns
                    else pd.DataFrame()
                )
                if not b.empty and "timestamp" in b.columns:
                    ts = pd.to_numeric(b["timestamp"], errors="coerce").dropna()
                    if not ts.empty:
                        t0, t1 = int(ts.min()), int(ts.max())
                        fig.add_trace(
                            go.Scatter(
                                x=[t0, t1],
                                y=[q, q],
                                mode="lines",
                                name="Final position",
                                line=dict(color="violet", width=2),
                            )
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            height=220,
                            margin=dict(l=40, r=20, t=40, b=40),
                            title=f"Position (final `positions` export): {q:g}",
                        )
                        return fig
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[q],
                        mode="markers",
                        marker=dict(size=14, color="violet"),
                        name="Final",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=220,
                    margin=dict(l=40, r=20, t=40, b=40),
                    title=f"Position (final `positions` export, no book for product): {q:g}",
                )
                return fig

    fig.update_layout(title="Position (no trades or `positions` entry)", template="plotly_dark", height=220)
    return fig


def _format_submission_meta(meta: dict[str, Any]) -> str:
    if not meta:
        return ""
    parts: list[str] = []
    if meta.get("profit") is not None:
        parts.append(f"**Profit:** {meta['profit']}")
    if meta.get("status"):
        parts.append(f"**Status:** {meta['status']}")
    if meta.get("round") is not None:
        parts.append(f"**Round:** {meta['round']}")
    sid = meta.get("submissionId") or meta.get("submission_id")
    if sid:
        parts.append(f"**Submission:** `{sid}`")
    return " · ".join(parts) if parts else ""


def logs_at_hover(logs_df: pd.DataFrame, ts: Optional[float], window: int = 500) -> str:
    if logs_df.empty or ts is None:
        return "_No log lines in range._"
    if "timestamp" not in logs_df.columns:
        return "_No timestamp column in logs._"
    t = float(ts)
    tsn = pd.to_numeric(logs_df["timestamp"], errors="coerce")
    sub = logs_df[tsn.notna() & (tsn >= t - window) & (tsn <= t + window)]
    if sub.empty:
        return f"_No logs within ±{window} of {t}._"
    lines = []
    sub = sub.assign(_tsn=pd.to_numeric(sub["timestamp"], errors="coerce"))
    for _, r in sub.sort_values("_tsn").iterrows():
        ts_show = int(r["_tsn"]) if pd.notna(r["_tsn"]) else r["timestamp"]
        lines.append(f"**{ts_show}** [{r.get('source','')}] {r.get('message','')}")
    return "\n\n".join(lines[:80])


# -----------------------------------------------------------------------------
# Discover log files
# -----------------------------------------------------------------------------
def _optional_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _looks_like_prosperity_json(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:8192]
    except OSError:
        return False
    return "activitiesLog" in head or "activities_log" in head or "day;timestamp;product" in head


def _looks_like_prosperity_log(path: Path) -> bool:
    """Keep backtester text logs and JSON submission `.log` files; drop unrelated logs."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:32768]
    except OSError:
        return False
    h = head.strip()
    if h.startswith("{"):
        return "activitiesLog" in head or "activities_log" in head
    return "Activities log:" in head or "Sandbox logs:" in head


def discover_log_files(roots: list[Path]) -> list[dict]:
    opts: list[dict] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for pat in ("**/*.log", "**/*.json"):
            for p in root.glob(pat):
                if ".cache" in str(p):
                    continue
                # skip huge non-prosperity logs
                if p.stat().st_size > 80 * 1024 * 1024:
                    continue
                if p.suffix.lower() == ".json" and not _looks_like_prosperity_json(p):
                    continue
                if p.suffix.lower() == ".log" and not _looks_like_prosperity_log(p):
                    continue
                opts.append({"label": str(p.relative_to(root)) + f" @ {root.name}", "value": str(p)})
    # de-dupe by value
    seen = set()
    out = []
    for o in opts:
        if o["value"] not in seen:
            seen.add(o["value"])
            out.append(o)
    return sorted(out, key=lambda x: x["label"])


def load_parsed(path_str: str, use_cache: bool = True) -> dict:
    path = Path(path_str)
    if not path.is_file():
        e = _empty_parse_result("missing")
        return e
    fp = _file_fingerprint(path)
    key = f"parsed_{PARSED_CACHE_VERSION}_{fp}"
    if use_cache:
        hit = _cache_load(key)
        if hit is not None:
            return hit
    data = parse_log_file(path)
    if not data["book_df"].empty:
        data["book_df"] = add_book_metrics(data["book_df"])
    if not data["trades_df"].empty:
        data["trades_df"] = classify_trades(data["trades_df"], data["book_df"])
    if not data["logs_df"].empty and "timestamp" in data["logs_df"].columns:
        data["logs_df"] = data["logs_df"].copy()
        data["logs_df"]["timestamp"] = pd.to_numeric(data["logs_df"]["timestamp"], errors="coerce")
    data["file_path"] = str(path)
    if use_cache:
        _cache_save(key, data)
    return data


# -----------------------------------------------------------------------------
# Dash application
# -----------------------------------------------------------------------------
def create_app(roots: list[Path]) -> Dash:
    file_options = discover_log_files(roots)
    default_file = file_options[0]["value"] if file_options else ""

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "IMC Order Book Dashboard"

    app.layout = html.Div(
        [
            html.H3("IMC Prosperity — Order book dashboard"),
            dcc.Markdown(id="submission-meta", style={"marginBottom": "10px", "fontSize": "14px"}),
            html.Div(
                [
                    html.Label("Log / JSON file"),
                    dcc.Dropdown(id="file-dd", options=file_options, value=default_file, clearable=False, style={"minWidth": "420px"}),
                    html.Label("Product"),
                    dcc.Dropdown(id="product-dd", clearable=False),
                    html.Label("Normalization"),
                    dcc.Dropdown(
                        id="norm-mode",
                        options=[
                            {"label": "None (raw price)", "value": "none"},
                            {"label": "Subtract baseline", "value": "subtract"},
                            {"label": "Z-score vs baseline", "value": "zscore"},
                        ],
                        value="none",
                        clearable=False,
                    ),
                    html.Label("Baseline column"),
                    dcc.Dropdown(
                        id="norm-base",
                        options=[
                            {"label": "wallmid", "value": "wallmid"},
                            {"label": "mid_touch", "value": "mid_touch"},
                            {"label": "mid_price (CSV)", "value": "mid_price"},
                        ],
                        value="wallmid",
                        clearable=False,
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "alignItems": "flex-end", "marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Book levels"),
                    dcc.Checklist(
                        id="levels-cl",
                        options=[{"label": f"L{i}", "value": i} for i in (1, 2, 3)],
                        value=[1, 2, 3],
                        inline=True,
                    ),
                    html.Label("Trade roles"),
                    dcc.Checklist(
                        id="roles-cl",
                        options=[
                            {"label": "Maker (inferred)", "value": "M"},
                            {"label": "Taker (inferred)", "value": "T"},
                            {"label": "Own (SUBMISSION)", "value": "F"},
                            {"label": "Unknown", "value": "U"},
                        ],
                        value=["M", "T", "F", "U"],
                        inline=True,
                    ),
                    html.Label("Qty min"),
                    dcc.Input(id="qty-min", type="number", placeholder="min", style={"width": "80px"}),
                    html.Label("Qty max"),
                    dcc.Input(id="qty-max", type="number", placeholder="max", style={"width": "80px"}),
                    html.Label("Max points (downsample)"),
                    dcc.Slider(
                        id="max-points",
                        min=2000,
                        max=50000,
                        step=1000,
                        value=15000,
                        marks={2000: "2k", 15000: "15k", 50000: "50k"},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Label("Metric overlays (right axis)"),
                    dcc.Checklist(
                        id="overlay-cl",
                        options=[
                            {"label": "Spread", "value": "spread"},
                            {"label": "Imbalance", "value": "imbalance"},
                        ],
                        value=[],
                        inline=True,
                    ),
                    html.Label("Size buckets"),
                    dcc.Checklist(
                        id="size-bucket-cl",
                        options=[
                            {"label": "Small (S)", "value": "S"},
                            {"label": "Big (B)", "value": "B"},
                        ],
                        value=["S", "B"],
                        inline=True,
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "alignItems": "center", "marginBottom": "12px"},
            ),
            dcc.Store(id="parsed-store"),
            dcc.Store(id="viewport-store", data=None),
            dcc.Graph(id="main-graph", style={"height": "540px"}),
            dcc.Graph(id="total-pnl-graph", style={"height": "200px", "maxWidth": "1400px"}),
            html.Div(
                [
                    dcc.Graph(id="pnl-graph", style={"width": "33%", "display": "inline-block"}),
                    dcc.Graph(id="pos-graph", style={"width": "33%", "display": "inline-block"}),
                    dcc.Markdown(id="log-md", style={"width": "33%", "display": "inline-block", "verticalAlign": "top", "maxHeight": "260px", "overflowY": "scroll", "backgroundColor": "#111", "padding": "8px"}),
                ]
            ),
            html.Details(
                [
                    html.Summary("Help — how to read this dashboard"),
                    dcc.Markdown(_load_guide_markdown(), style={"maxHeight": "420px", "overflowY": "auto"}),
                ],
                style={"marginTop": "16px"},
            ),
        ],
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "12px", "fontFamily": "system-ui"},
    )

    @app.callback(
        Output("parsed-store", "data"),
        Input("file-dd", "value"),
    )
    def load_file(path_str):
        if not path_str:
            return {}
        load_parsed(path_str, use_cache=True)
        return {"path": path_str}

    @app.callback(
        Output("viewport-store", "data"),
        Input("main-graph", "relayoutData"),
        Input("file-dd", "value"),
        State("viewport-store", "data"),
    )
    def sync_viewport(relayout, _path, prev_vp):
        trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        if trig.startswith("file-dd"):
            return None
        if not relayout:
            return no_update
        if relayout.get("xaxis.autorange") is True:
            return None
        xr = _relayout_x_range(relayout)
        if xr is None:
            return no_update
        if prev_vp and prev_vp.get("x0") == xr[0] and prev_vp.get("x1") == xr[1]:
            return no_update
        return {"x0": xr[0], "x1": xr[1]}

    @app.callback(
        Output("product-dd", "options"),
        Output("product-dd", "value"),
        Input("parsed-store", "data"),
    )
    def set_products(store):
        if not store or not store.get("path"):
            return [], None
        d = load_parsed(store["path"], use_cache=True)
        df = d.get("book_df", pd.DataFrame())
        pos = d.get("positions_df", pd.DataFrame())
        prod_set: set[str] = set()
        if not df.empty and "product" in df.columns:
            prod_set |= set(df["product"].astype(str).unique())
        if not pos.empty and "product" in pos.columns:
            prod_set |= {p for p in pos["product"].astype(str).unique() if p and str(p) != "nan"}
        if not prod_set:
            return [{"label": "NONE", "value": "NONE"}], "NONE"
        prods = sorted(prod_set)
        opts = [{"label": p, "value": p} for p in prods]
        return opts, prods[0]

    @app.callback(
        Output("main-graph", "figure"),
        Output("pnl-graph", "figure"),
        Output("pos-graph", "figure"),
        Output("log-md", "children"),
        Output("total-pnl-graph", "figure"),
        Output("submission-meta", "children"),
        Input("parsed-store", "data"),
        Input("product-dd", "value"),
        Input("norm-mode", "value"),
        Input("norm-base", "value"),
        Input("levels-cl", "value"),
        Input("roles-cl", "value"),
        Input("qty-min", "value"),
        Input("qty-max", "value"),
        Input("max-points", "value"),
        Input("overlay-cl", "value"),
        Input("size-bucket-cl", "value"),
        Input("main-graph", "hoverData"),
        State("viewport-store", "data"),
    )
    def update_plots(
        store,
        product,
        norm_mode,
        norm_base,
        levels,
        roles,
        qty_min,
        qty_max,
        max_pts,
        overlay_vals,
        size_buckets,
        hover,
        viewport,
    ):
        empty = go.Figure()
        empty.update_layout(template="plotly_dark")
        if not store or not store.get("path"):
            z = build_total_pnl_graphlog_figure(pd.DataFrame())
            return empty, empty, empty, "_Select a file._", z, ""
        d = load_parsed(store["path"], use_cache=True)
        meta_md = _format_submission_meta(d.get("submission_meta") or {})
        graphlog_df = d.get("graphlog_df", pd.DataFrame())
        tot_pnl = build_total_pnl_graphlog_figure(graphlog_df if isinstance(graphlog_df, pd.DataFrame) else pd.DataFrame())
        if not product or product == "NONE":
            return empty, empty, empty, "_Select a product._", tot_pnl, meta_md
        book = d["book_df"]
        trades = d.get("trades_df", pd.DataFrame())
        logs = d.get("logs_df", pd.DataFrame())
        positions_df = d.get("positions_df", pd.DataFrame())

        lv = [int(x) for x in (levels or [1, 2, 3])]
        rf = {c: c in (roles or []) for c in ("M", "T", "F", "U")}
        qmin = _optional_float(qty_min)
        qmax = _optional_float(qty_max)
        mp = int(max_pts) if max_pts else 15000

        x_range = None
        if viewport and "x0" in viewport and "x1" in viewport:
            x_range = (float(viewport["x0"]), float(viewport["x1"]))

        overlays = [str(x) for x in (overlay_vals or [])]
        buckets = [str(x) for x in (size_buckets or ["S", "B"])]
        main = build_main_figure(
            book,
            trades,
            product,
            lv,
            norm_mode or "none",
            norm_base or "wallmid",
            rf,
            qmin,
            qmax,
            mp,
            x_range=x_range,
            overlay_metrics=overlays,
            size_buckets=buckets if buckets else None,
        )
        main.update_layout(uirevision=f"{store.get('path')}:{product}")
        pnl = build_pnl_figure(book, product)
        pos = build_position_figure(
            trades,
            product,
            book,
            positions_df if isinstance(positions_df, pd.DataFrame) else pd.DataFrame(),
        )

        ts = None
        if hover and "points" in hover and hover["points"]:
            ts = hover["points"][0].get("x")
        log_text = logs_at_hover(logs, ts)
        return main, pnl, pos, log_text, tot_pnl, meta_md

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="IMC Prosperity order book dashboard")
    ap.add_argument("--root", action="append", help="Root folder to scan for .log/.json (repeatable)")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    roots = [Path(p) for p in args.root] if args.root else DEFAULT_ROOTS
    app = create_app(roots)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
