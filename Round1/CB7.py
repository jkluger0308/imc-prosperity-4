from datamodel import Order, TradingState
import json

# CB7: CB3 with aggressive inventory flattening.
# Changes from CB3:
#   - FLAT_THRESHOLD 50 -> 25 (start unwinding much earlier)
#   - FLAT_TARGET 40 -> 0 (unwind all the way to flat, not just to 40)
#   - Flat price: fair-1 for sells, fair+1 for buys (1 tick negative edge
#     to ensure the order actually fills, not just sits at fair)

FLAT_THRESHOLD = 25
FLAT_TARGET    = 0
BASE           = 24
EMA_ALPHA      = 0.05


def updatepos(pos: int, vol: int, maxpos: int = 80):
    new_pos = pos + vol
    new_pos = max(-maxpos, min(maxpos, new_pos))
    buy_room  = max(0, maxpos - new_pos)
    sell_room = max(0, maxpos + new_pos)
    return new_pos, buy_room, sell_room


class Trader:

    def run(self, state: TradingState):
        result = {}

        prev = {}
        if state.traderData:
            try:
                prev = json.loads(state.traderData)
            except:
                prev = {}

        new_state = {}

        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue

            pos     = state.position.get(prod, 0)
            orders  = []

            if prod == "INTARIAN_PEPPER_ROOT":
                if pos < 80 and od.sell_orders:
                    ba_p = min(od.sell_orders)
                    orders.append(Order(prod, ba_p, 80 - pos))

            elif prod == "ASH_COATED_OSMIUM":
                buy_room  = 80 - pos
                sell_room = 80 + pos

                cm_bid = sum(float(vol * price) for price, vol in od.buy_orders.items()) / sum(float(vol) for price, vol in od.buy_orders.items())
                cm_ask = sum(float(abs(vol) * price) for price, vol in od.sell_orders.items()) / sum(float(abs(vol)) for price, vol in od.sell_orders.items())
                vwap = (cm_bid + cm_ask) / 2

                prev_ema = prev.get("ema_fair")
                if prev_ema is not None:
                    fairprice = EMA_ALPHA * vwap + (1 - EMA_ALPHA) * prev_ema
                else:
                    fairprice = vwap
                new_state["ema_fair"] = fairprice

                initial_bids = list(od.buy_orders.items())
                initial_asks = list(od.sell_orders.items())

                pbid_below_fair = [p for p, v in initial_bids if p < fairprice]
                pask_above_fair = [p for p, v in initial_asks if p > fairprice]

                # ── Layer 1: take mispriced orders ────────────────────
                for price, vol in initial_bids:
                    if price > fairprice + 1 and sell_room > 0:
                        size = min(vol, sell_room)
                        orders.append(Order(prod, price, -size))
                        pos, buy_room, sell_room = updatepos(pos, -size)
                    elif price > fairprice and sell_room > 0 and pos > 0:
                        size = min(vol, sell_room)
                        orders.append(Order(prod, price, -size))
                        pos, buy_room, sell_room = updatepos(pos, -size)

                for price, vol in initial_asks:
                    if price < fairprice - 1 and buy_room > 0:
                        size = min(-vol, buy_room)
                        orders.append(Order(prod, price, size))
                        pos, buy_room, sell_room = updatepos(pos, size)
                    elif price < fairprice and buy_room > 0 and pos < 0:
                        size = min(-vol, buy_room)
                        orders.append(Order(prod, price, size))
                        pos, buy_room, sell_room = updatepos(pos, size)

                # ── Layer 2: passive quotes with inventory size skew ──
                if pbid_below_fair and pask_above_fair:
                    bb = max(pbid_below_fair)
                    ba = min(pask_above_fair)

                    if ba - 1 > fairprice and bb + 1 < fairprice:
                        I = pos / 80
                        bid_size = int(BASE * (1 - I))
                        ask_size = int(BASE * (1 + I))

                        bid_size = max(0, min(bid_size, buy_room))
                        ask_size = max(0, min(ask_size, sell_room))

                        if ask_size > 0:
                            orders.append(Order(prod, ba - 1, -ask_size))
                            pos, buy_room, sell_room = updatepos(pos, -ask_size)

                        if bid_size > 0:
                            orders.append(Order(prod, bb + 1, bid_size))
                            pos, buy_room, sell_room = updatepos(pos, bid_size)

                # ── Layer 3: aggressive inventory flattening ──────────
                if pos >= FLAT_THRESHOLD and sell_room > 0:
                    flat_price = int(fairprice) - 1  # 1 tick below fair to ensure fill
                    flat_size  = min(pos - FLAT_TARGET, sell_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, -flat_size))
                        pos, buy_room, sell_room = updatepos(pos, -flat_size)

                elif pos <= -FLAT_THRESHOLD and buy_room > 0:
                    flat_price = int(fairprice) + 1 + (1 if fairprice != int(fairprice) else 0)
                    flat_size  = min(-pos - FLAT_TARGET, buy_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, flat_size))
                        pos, buy_room, sell_room = updatepos(pos, flat_size)

            result[prod] = orders

        trader_data = json.dumps(new_state)
        return result, 0, trader_data
