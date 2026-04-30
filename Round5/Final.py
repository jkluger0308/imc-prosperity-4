from typing import Dict, List, Optional, Tuple
from datamodel import Order, TradingState
import json

BASE_LIMIT = 10
# EWMA span ≈ 5: alpha = 2 / (span + 1)
EWMA_ALPHA = 2.0 / (5 + 1)

# Proven consistent winners across all 3 days — full size
HIGH_CONVICTION = {
    "PEBBLES_S",
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_PISTACHIO",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "ROBOT_DISHES",
    "SLEEP_POD_SUEDE",  # +14k, +6k, -6k — 2/3 good
    "PANEL_1X4",  # +16k, +4k, +9k — consistent
    "MICROCHIP_OVAL",  # +4.6k, +5k, +1k — steady
    "UV_VISOR_AMBER",  # +8k, -0.6k, +11k
    "TRANSLATOR_VOID_BLUE",  # +4k, +1.8k, +10.5k
}

# Volatile but not hopeless — trade at half size
LOW_CONVICTION = {
    "ROBOT_MOPPING",
    "SLEEP_POD_LAMB_WOOL",
    "PANEL_1X2",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "PEBBLES_M",
    "TRANSLATOR_SPACE_GRAY",  # +9.6k, -0.5k, -14.8k — too swingy
    "TRANSLATOR_GRAPHITE_MIST",  # -4.7k, -0.8k, +4.2k
    "UV_VISOR_MAGENTA",  # +3.3k, -8.9k, -2.9k
    "MICROCHIP_RECTANGLE",  # +12.7k, +5.5k, -13.6k — very swingy
}


def get_limit(product: str) -> int:
    if product in HIGH_CONVICTION:
        return 10
    if product in LOW_CONVICTION:
        return 4
    return 8  # default — don't fully cut anything


def best_bid_ask(order_depth) -> Tuple[Optional[int], Optional[int]]:
    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return best_bid, best_ask


def _load_ewma_pnl(prev: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Restore EWMA and cumulative PnL maps from traderData JSON."""
    if not prev:
        return {}, {}
    ew = prev.get("ewma")
    pn = prev.get("pnl")
    if isinstance(ew, dict) and isinstance(pn, dict):
        return (
            {k: float(v) for k, v in ew.items() if isinstance(v, (int, float))},
            {k: float(v) for k, v in pn.items() if isinstance(v, (int, float))},
        )
    # Legacy: flat product -> ewma float only
    if isinstance(prev, dict):
        legacy_ew = {k: float(v) for k, v in prev.items() if isinstance(v, (int, float))}
        return legacy_ew, {}
    return {}, {}


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        prev: Dict = {}
        if state.traderData:
            try:
                loaded = json.loads(state.traderData)
                prev = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                prev = {}

        ewma_map, pnl_map = _load_ewma_pnl(prev)

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            result[product] = orders

            pnl_before = float(pnl_map.get(product, 0.0))
            pnl_after = pnl_before
            for trade in state.own_trades.get(product, []):
                pnl_after += (
                    trade.price * abs(trade.quantity)
                    if trade.seller == "SUBMISSION"
                    else -trade.price * abs(trade.quantity)
                )
            pnl_map[product] = pnl_after

            best_bid, best_ask = best_bid_ask(order_depth)
            prev_ewma = ewma_map.get(product)
            prev_f = float(prev_ewma) if prev_ewma is not None else None

            if best_bid is not None and best_ask is not None:
                worst_bid = min(order_depth.buy_orders)
                worst_ask = max(order_depth.sell_orders)
                wall_mid = (worst_ask + worst_bid) / 2
                if prev_f is None:
                    ew_new = wall_mid
                else:
                    ew_new = EWMA_ALPHA * wall_mid + (1 - EWMA_ALPHA) * prev_f
                ewma_map[product] = float(ew_new)

            ewma_val = ewma_map.get(product)
            if ewma_val is not None:
                ewma_val = float(ewma_val)

            position = state.position.get(product, 0)
            limit = get_limit(product)

            # Tick PnL vs prior cumulative (scale floor avoids div-by-zero).
            tick_pnl = pnl_after - pnl_before
            scale = max(abs(pnl_before), 1.0)
            # tick_pnl / scale is dimensionless; compared directly to STOP_RATIO (not "int 1–100 = %").
            # Fire when tick_pnl < STOP_RATIO * scale. E.g. STOP_RATIO=-0.01 → loss > 1% of scale; -1 → > scale; -100 → > 100*scale.
            STOP_RATIO = -10
            stop_triggered = tick_pnl / scale < STOP_RATIO
            if stop_triggered and position != 0:
                flatten_px: Optional[int] = None
                if position > 0 and best_bid is not None:
                    flatten_px = best_bid
                elif position < 0 and best_ask is not None:
                    flatten_px = best_ask
                elif ewma_val is not None:
                    flatten_px = int(round(ewma_val))
                if flatten_px is not None:
                    orders.append(Order(product, flatten_px, -position))
                continue

            bid_price = None
            ask_price = None
            if best_bid is None or best_ask is None:
                if best_bid is None and best_ask is not None:
                    bid_price = best_ask - 2
                    ask_price = best_ask
                elif best_ask is None and best_bid is not None:
                    ask_price = best_bid + 2
                    bid_price = best_bid
                elif ewma_val is not None:
                    z = int(round(ewma_val))
                    ask_price = z + 1
                    bid_price = z - 1
                else:
                    continue
            else:
                s = best_ask - best_bid
                if s <= 1:
                    continue

            buy_capacity = max(0, limit - position)
            sell_capacity = max(0, limit + position)

            skew = position / limit
            if skew > 0.5:
                buy_capacity = max(0, buy_capacity // 2)
            elif skew < -0.5:
                sell_capacity = max(0, sell_capacity // 2)

            if bid_price is None and ask_price is None:
                bid_price = best_bid + 1
                ask_price = best_ask - 1

            if bid_price >= ask_price:
                continue

            if buy_capacity > 0:
                orders.append(Order(product, bid_price, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(product, ask_price, -sell_capacity))

        trader_data = json.dumps(
            {"ewma": ewma_map, "pnl": pnl_map},
            separators=(",", ":"),
        )
        return result, conversions, trader_data
