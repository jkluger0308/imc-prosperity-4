from datamodel import Order, TradingState
import json

FLAT_THRESHOLD_HARD = 15
FLAT_TARGET_HARD = 10
FLAT_THRESHOLD_SOFT = FLAT_TARGET_HARD
FLAT_TARGET_SOFT    = 0
BASE           = 24  
EMA_ALPHA = 0.05
FAST_SPAN      = 10
SLOW_SPAN      = FAST_SPAN * 4
OSM_SPANF = 5
OSM_SPANS = 4*OSM_SPANF
PEPPER_EDGE = 2
OSMIUM_EDGE = 0.5


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

            cm_bid = sum([float(vol*price) for price, vol in od.buy_orders.items()])/sum([float(vol) for price, vol in od.buy_orders.items()])
            cm_ask = sum([float(abs(vol)*price) for price, vol in od.sell_orders.items()])/sum([float(abs(vol)) for price, vol in od.sell_orders.items()])
            VWAP = (cm_bid + cm_ask) / 2
            fairprice = VWAP
            
            pos = state.position.get(prod, 0)
            sell_room = 80 + pos
            buy_room = 80 - pos
            orders = []

            if prod == "INTARIAN_PEPPER_ROOT":
                wb  = min(od.buy_orders)
                wa  = max(od.sell_orders)
                wallmid = (wb + wa) / 2

                ba = min(od.sell_orders)
                bb = max(od.buy_orders)

                ewm_fast = prev.get("pepper_ewm_fast")
                ewm_slow = prev.get("pepper_ewm_slow")
                pepper_entry_live = prev.get("pepper_entry_live", None)

                alpha_f = 2 / (FAST_SPAN + 1)
                alpha_s = 2 / (SLOW_SPAN + 1)

                if ewm_fast is None:
                    ewm_fast = wallmid
                    ewm_slow = wallmid
                else:
                    ewm_fast = alpha_f * wallmid + (1 - alpha_f) * ewm_fast
                    ewm_slow = alpha_s * wallmid + (1 - alpha_s) * ewm_slow

                new_state["pepper_ewm_fast"] = ewm_fast
                new_state["pepper_ewm_slow"] = ewm_slow

                bullish = (
                    ewm_fast >= ewm_slow
                )
                bearish = (
                    ewm_slow > ewm_fast + PEPPER_EDGE
                )

                if bullish:
                    pepper_entry_live = True
                elif bearish:
                    pepper_entry_live = False

                new_state["pepper_entry_live"] = pepper_entry_live

                if pepper_entry_live:
                    orders.append(Order(prod, ba, 80 - pos))
                    print(f"Order({prod}, {ba}, {80 - pos})")
                elif pepper_entry_live == False:
                    orders.append(Order(prod, ba - 1, -min(20, (80 + pos))))
                    print(f"Order({prod}, {ba - 1}, {-min(20, (80 + pos))})")

            elif prod == "ASH_COATED_OSMIUM":

                prev_ema = prev.get("ema_fair")
                if prev_ema is not None:
                    meanfair = EMA_ALPHA * VWAP + (1 - EMA_ALPHA) * prev_ema
                else:
                    meanfair = VWAP
                new_state["ema_fair"] = meanfair

                prev_ema_f = prev.get("ema_osmium_fast")
                alpha_f = 2/(OSM_SPANF + 1)
                if prev_ema_f is not None:
                    ema_f = alpha_f*VWAP + (1 - alpha_f)*prev_ema_f
                else:
                    ema_f = VWAP
                new_state["ema_osmium_fast"] = ema_f

                prev_ema_s = prev.get("ema_osmium_slow")
                alpha_s = 2/(OSM_SPANS + 1)
                if prev_ema_s is not None:
                    ema_s = alpha_s*VWAP + (1 - alpha_s)*prev_ema_s
                else:
                    ema_s = VWAP
                new_state["ema_osmium_slow"] = ema_s

                trending_up = ema_f > ema_s + OSMIUM_EDGE
                trending_down = ema_f + OSMIUM_EDGE < ema_s


                initial_bids = list(od.buy_orders.items())
                initial_asks = list(od.sell_orders.items())

                pbid_below_fair = sorted([p for p, v in initial_bids if p < meanfair], reverse=True)
                pask_above_fair = sorted([p for p, v in initial_asks if p > meanfair])
                ba = min(pask_above_fair) if pask_above_fair else None
                bb = max(pbid_below_fair) if pbid_below_fair else None

                # ── Layer 1: take mispriced orders ────────────────────
                for price, vol in initial_bids:
                    # if price >= meanfair + 1 and sell_room > 0:
                    #     size = min(vol, sell_room)
                    #     orders.append(Order(prod, price, -size))
                    #     print(f"Order({prod}, {price}, {-size})")
                    #     pos, buy_room, sell_room = updatepos(pos, -size)
                    if price > meanfair and sell_room > 0 and pos > 0:
                        size = min(vol, sell_room)
                        orders.append(Order(prod, price, -size))
                        print(f"Order({prod}, {price}, {-size})")
                        pos, buy_room, sell_room = updatepos(pos, -size)

                for price, vol in initial_asks:
                    # if price <= meanfair - 1 and buy_room > 0:
                    #     size = min(-vol, buy_room)
                    #     orders.append(Order(prod, price, size))
                    #     print(f"Order({prod}, {price}, {size})")
                    #     pos, buy_room, sell_room = updatepos(pos, size)
                    if price < meanfair and buy_room > 0 and pos < 0:
                        size = min(-vol, buy_room)
                        orders.append(Order(prod, price, size))
                        print(f"Order({prod}, {price}, {size})")
                        pos, buy_room, sell_room = updatepos(pos, size)

                #── Layer 2: passive quotes with inventory size skew ──
                if meanfair:
                    I = pos / 80
                    bid_size = int(BASE * (1 - I))
                    ask_size = int(BASE * (1 + I))
                    bid_size = max(0, min(bid_size, buy_room))
                    ask_size = max(0, min(ask_size, sell_room))


                    if ba and ba - 1 > fairprice and ask_size > 0:
                        sellprice = ba - 1
                        if trending_down and ba - 2 > fairprice:
                            sellprice = ba - 2
                        orders.append(Order(prod, sellprice, -ask_size))
                        print(f"Order({prod}, {sellprice}, {-ask_size})")

                    if bb and bb + 1 < fairprice and bid_size > 0:
                        buyprice = bb + 1
                        if trending_up and bb + 2 < fairprice:
                            buyprice = bb + 2
                        orders.append(Order(prod, buyprice, bid_size))
                        print(f"Order({prod}, {buyprice}, {bid_size})")

                #── Layer 3: zero-edge inventory neutralization ──────-

                # if pos >= FLAT_THRESHOLD_HARD and sell_room > 0:
                #     # ``ba`` can be None when no ask is above ``meanfair``; fall back to best ask + 1.
                #     flat_price = (bb) if bb is not None else round(fairprice)
                #     flat_size  = min(pos - FLAT_TARGET_HARD, sell_room)
                #     orders.append(Order(prod, flat_price, -flat_size))
                #     print(f"Order({prod}, {flat_price}, {-flat_size})")

                if pos >= FLAT_THRESHOLD_SOFT and sell_room > 0:
                    flat_price = int(fairprice)
                    flat_size  = min(pos - FLAT_TARGET_SOFT, sell_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, -flat_size))
                        print(f"Order({prod}, {flat_price}, {-flat_size})")
                
                # elif pos <= -FLAT_THRESHOLD_HARD and buy_room > 0:
                #     flat_price = (ba) if ba is not None else round(fairprice)
                #     flat_size  = min(-pos - FLAT_TARGET_HARD, buy_room)
                #     if flat_size > 0:
                #         orders.append(Order(prod, flat_price, flat_size))
                #         print(f"Order({prod}, {flat_price}, {flat_size})")

                elif pos <= -FLAT_THRESHOLD_SOFT and buy_room > 0:
                    flat_price = int(fairprice) + (1 if fairprice != int(fairprice) else 0)
                    flat_size  = min(-pos - FLAT_TARGET_SOFT, buy_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, flat_size))
                        print(f"Order({prod}, {flat_price}, {flat_size})")
                print(f"OSMPOS: {pos}")

            result[prod] = orders
        trader_data = json.dumps(new_state)
        return result, 0, trader_data
