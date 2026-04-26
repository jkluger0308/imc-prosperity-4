from datamodel import Order, TradingState
import json

FLAT_THRESHOLD_SOFT = 5
FLAT_TARGET_SOFT = 0
BASE = 24
OSM_SPANF = 5
OSM_SPANS = 4 * OSM_SPANF
OSMIUM_EDGE = 0.5
ROOM = 50

MM_PRODUCTS = {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}

def updatepos(pos: int, vol: int, maxpos: int = ROOM):
    new_pos = pos + vol
    new_pos = max(-maxpos, min(maxpos, new_pos))
    buy_room = max(0, maxpos - new_pos)
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
            if prod not in MM_PRODUCTS:
                result[prod] = []
                continue

            cm_bid = (
                sum(vol * price for price, vol in od.buy_orders.items()) /
                sum(vol for _, vol in od.buy_orders.items())
            ) if od.buy_orders else prev.get(f"{prod}_ema_fast")

            cm_ask = (
                sum(abs(vol) * price for price, vol in od.sell_orders.items()) /
                sum(abs(vol) for _, vol in od.sell_orders.items())
            ) if od.sell_orders else prev.get(f"{prod}_ema_fast")

            VWAP = (cm_bid + cm_ask) / 2 if cm_bid and cm_ask else None

            pos = state.position.get(prod, 0)
            sell_room = ROOM + pos
            buy_room = ROOM - pos
            orders = []

            # EMA trend filter
            if VWAP is not None:
                alpha_f = 2 / (OSM_SPANF + 1)
                prev_ema_f = prev.get(f"{prod}_ema_fast")
                ema_f = alpha_f * VWAP + (1 - alpha_f) * prev_ema_f if prev_ema_f else VWAP
                new_state[f"{prod}_ema_fast"] = ema_f

                alpha_s = 2 / (OSM_SPANS + 1)
                prev_ema_s = prev.get(f"{prod}_ema_slow")
                ema_s = alpha_s * VWAP + (1 - alpha_s) * prev_ema_s if prev_ema_s else VWAP
                new_state[f"{prod}_ema_slow"] = ema_s

                trending_up = ema_f > ema_s + OSMIUM_EDGE
                trending_down = ema_f + OSMIUM_EDGE < ema_s
            else:
                trending_up = trending_down = False

            fairprice = VWAP
            ba = min(od.sell_orders) if od.sell_orders else None
            bb = max(od.buy_orders) if od.buy_orders else None

            # Layer 2: passive quotes with inventory skew
            if fairprice is not None:
                I = pos / ROOM
                bid_size = max(0, min(int(BASE * (1 - I)), buy_room))
                ask_size = max(0, min(int(BASE * (1 + I)), sell_room))

                if not ba and bb:
                    orders.append(Order(prod, round(fairprice + 2), -ask_size))
                if ba and ba - 1 > fairprice and ask_size > 0:
                    orders.append(Order(prod, ba - 1, -ask_size))
                if not bb and ba:
                    orders.append(Order(prod, round(fairprice - 1), bid_size))
                if bb and bb + 1 < fairprice and bid_size > 0:
                    orders.append(Order(prod, bb + 1, bid_size))
                if not ba and not bb:
                    orders.append(Order(prod, round(fairprice - 1), bid_size))
                    orders.append(Order(prod, round(fairprice + 1), -ask_size))

            # Layer 3: inventory neutralization
            if fairprice:
                if pos >= FLAT_THRESHOLD_SOFT and sell_room > 0:
                    flat_size = min(pos - FLAT_TARGET_SOFT, sell_room)
                    if flat_size > 0:
                        orders.append(Order(prod, int(fairprice), -flat_size))
                elif pos <= -FLAT_THRESHOLD_SOFT and buy_room > 0:
                    flat_price = int(fairprice) + (1 if fairprice != int(fairprice) else 0)
                    flat_size = min(-pos - FLAT_TARGET_SOFT, buy_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, flat_size))

            result[prod] = orders

        trader_data = json.dumps(new_state)
        return result, 0, trader_data