from datamodel import Order, TradingState

# R1G: R1F with BASE doubled (12 -> 24).
# Hypothesis: current fills are saturating at ~10 units/side, meaning
# market orders often arrive bigger than our quoted size. Doubling BASE
# should capture the overflow and roughly scale PnL with fill volume.
# Layer 3 flattening (unchanged) handles the faster inventory growth.

FLAT_THRESHOLD = 35
FLAT_TARGET    = 15
BASE           = 24   # was 12 in R1B/R1F
MULT = 1


def updatepos(pos: int, vol: int, maxpos: int = 80):
    new_pos = pos + vol
    new_pos = max(-maxpos, min(maxpos, new_pos))
    buy_room  = max(0, maxpos - new_pos)
    sell_room = max(0, maxpos + new_pos)
    return new_pos, buy_room, sell_room


class Trader:

    def run(self, state: TradingState):
        try:
            parts = state.traderData.split("|")
            osm_ewm = float(parts[0])
            osm_ewm2 = float(parts[1])
        except:
            osm_ewm = None
            osm_ewm2 = None

        result = {}

        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue

            #wb      = min(od.buy_orders)
            #wa      = max(od.sell_orders)
            #wallmid = (wb + wa) / 2
            pos     = state.position.get(prod, 0)
            orders  = []

            if prod == "INTARIAN_PEPPER_ROOT":
                if pos < 80 and od.sell_orders:
                    ba_p = min(od.sell_orders)
                    orders.append(Order(prod, ba_p, 80 - pos))
                    print(f"Order({prod}, {ba_p}, {80 - pos})")

            elif prod == "ASH_COATED_OSMIUM":
                buy_room  = 80 - pos
                sell_room = 80 + pos

                cm_bid = sum([float(vol*price) for price, vol in od.buy_orders.items()])/sum([float(vol) for price, vol in od.buy_orders.items()])
                cm_ask = sum([float(abs(vol)*price) for price, vol in od.sell_orders.items()])/sum([float(abs(vol)) for price, vol in od.sell_orders.items()])
                VWAP = (cm_bid + cm_ask) / 2
                fairprice = VWAP

                initial_bids = list(od.buy_orders.items())
                initial_asks = list(od.sell_orders.items())

                bids_below_fair = [(p, v) for p, v in initial_bids if p < fairprice]
                asks_above_fair = [(p, v) for p,v in initial_asks if p > fairprice]

                pbid_below_fair = [p for p, v in bids_below_fair]
                pask_above_fair = [p for p, v in asks_above_fair]

                consumed_bid_prices = set()
                consumed_ask_prices = set()

                bb = max(pbid_below_fair)
                ba = min(pask_above_fair)

                #pointmid = (ba + bb) / 2
                pointmid = VWAP

                alpha = 2 / (5 + 1) # Manual exponentially weighted mean calculation provides lower latency than using Pandas.
                if osm_ewm is None: 
                    osm_ewm = pointmid
                    osm_ewm2 = pointmid * pointmid
                else:
                    osm_ewm = alpha * pointmid + (1 - alpha) * osm_ewm
                    osm_ewm2 = alpha * (pointmid * pointmid) + (1 - alpha) * osm_ewm2

                var = osm_ewm2 - osm_ewm * osm_ewm
                std = var ** 0.5 if var > 0 else 0.001

                # ── Layer 1: take mispriced orders ────────────────────
                for price, vol in initial_bids:
                    if price > fairprice and sell_room > 0:
                        size = min(vol, sell_room)
                        orders.append(Order(prod, price, -size))
                        print(f"Order({prod}, {price}, {-size})")
                        pos, buy_room, sell_room = updatepos(pos, -size)

                for price, vol in initial_asks:
                    if price < fairprice and buy_room > 0:
                        size = min(-vol, buy_room)
                        orders.append(Order(prod, price, size))
                        print(f"Order({prod}, {price}, {size})")
                        pos, buy_room, sell_room = updatepos(pos, size)
                
                for price, vol in bids_below_fair:
                    deviation = abs(price - fairprice)
                    if deviation <= MULT * abs(std):
                        size = min(vol, sell_room)
                        orders.append(Order(prod, price, -size))
                        print(f"Order({prod}, {price}, {-size})")
                        pos, buy_room, sell_room = updatepos(pos, -size)
                        consumed_bid_prices.add(price)

                
                for price, vol in asks_above_fair:
                    deviation = abs(price - fairprice)
                    if deviation <= MULT * abs(std):
                        size = min(-vol, buy_room)
                        orders.append(Order(prod, price, size))
                        print(f"Order({prod}, {price}, {size})")
                        pos, buy_room, sell_room = updatepos(pos, size)
                        consumed_ask_prices.add(price)
                
                

                # ── Layer 2: passive quotes with inventory size skew ──
                pbid_below_fair = [p for p, v in bids_below_fair if p not in consumed_bid_prices]
                pask_above_fair = [p for p, v in asks_above_fair if p not in consumed_ask_prices]

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
                            print(f"Order({prod}, {ba-1}, {-ask_size})")

                        if bid_size > 0:
                            orders.append(Order(prod, bb + 1, bid_size))
                            pos, buy_room, sell_room = updatepos(pos, bid_size)
                            print(f"Order({prod}, {bb+1}, {bid_size})")

                # ── Layer 3: zero-edge inventory neutralization ───────
                if pos >= FLAT_THRESHOLD and sell_room > 0:
                    flat_price = int(fairprice)
                    flat_size = min(pos - FLAT_TARGET, sell_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, -flat_size))
                        print(f"Order({prod}, {flat_price}, {-flat_size})")
                        pos, buy_room, sell_room = updatepos(pos, -flat_size)

                elif pos <= -FLAT_THRESHOLD and buy_room > 0:
                    flat_price = int(fairprice)
                    flat_size  = min(-pos - FLAT_TARGET, buy_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, flat_size))
                        print(f"Order({prod}, {flat_price}, {flat_size})")
                        pos, buy_room, sell_room = updatepos(pos, flat_size)

            result[prod] = orders
        trader_data = f"{osm_ewm}|{osm_ewm2}" if osm_ewm is not None else ""
        return result, 0, trader_data
