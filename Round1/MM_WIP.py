from datamodel import Order, TradingState

FLAT_THRESHOLD = 30
FLAT_TARGET    = 0
BASE           = 24   # was 12 in R1B/R1F


def updatepos(pos: int, vol: int, maxpos: int = 80):
    new_pos = pos + vol
    new_pos = max(-maxpos, min(maxpos, new_pos))
    buy_room  = max(0, maxpos - new_pos)
    sell_room = max(0, maxpos + new_pos)
    return new_pos, buy_room, sell_room


class Trader:

    def run(self, state: TradingState):
        result = {}

        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue
            
            

            wb      = min(od.buy_orders)
            wa      = max(od.sell_orders)
            wallmid = (wb + wa) / 2

            cm_bid = sum([float(vol*price) for price, vol in od.buy_orders.items()])/sum([float(vol) for price, vol in od.buy_orders.items()])
            cm_ask = sum([float(abs(vol)*price) for price, vol in od.sell_orders.items()])/sum([float(abs(vol)) for price, vol in od.sell_orders.items()])
            VWAP = (cm_bid + cm_ask) / 2
            
            fairprice = VWAP
            pos     = state.position.get(prod, 0)
            orders  = []

            if prod == "INTARIAN_PEPPER_ROOT":
                if pos < 80 and od.sell_orders:
                    ba_p = min(od.sell_orders)
                    bb_p = max(od.buy_orders)
                    #buyprice = round(fairprice + 1)
                    buyprice = ba_p
                    orders.append(Order(prod, buyprice, 80 - pos))
                    print(f"Order({prod}, {ba_p}, {80 - pos})")

            elif prod == "ASH_COATED_OSMIUM":
                buy_room  = 80 - pos
                sell_room = 80 + pos


                initial_bids = list(od.buy_orders.items())
                initial_asks = list(od.sell_orders.items())

                pbid_below_fair = sorted([p for p, v in initial_bids if p < fairprice], reverse=True)
                pask_above_fair = sorted([p for p, v in initial_asks if p > fairprice])
                ba = min([x for x in pask_above_fair if abs(x - fairprice) > 2])
                bb = max([x for x in pbid_below_fair if abs(x - fairprice) > 2])

                # ── Layer 1: take mispriced orders ────────────────────
                for price, vol in initial_bids:
                    if price > fairprice and sell_room > 0:
                        orders.append(Order(prod, price, -min(vol, sell_room)))
                        pos, buy_room, sell_room = updatepos(pos, -min(vol, sell_room))
                        orders.append(Order(prod, bb + 2, min(buy_room, 20)))
                        print(f"Order({prod}, {price}, {-min(vol, sell_room)})")
                        print(f"Order({prod}, {bb + 2}, {min(20, buy_room)})")
                        pos, buy_room, sell_room = updatepos(pos, min(20, buy_room))

                for price, vol in initial_asks:
                    if price < fairprice and buy_room > 0:
                        orders.append(Order(prod, price, min(-vol, buy_room)))
                        pos, buy_room, sell_room = updatepos(pos, min(-vol, buy_room))
                        orders.append(Order(prod, ba - 2, -min(20, sell_room)))
                        pos, buy_room, sell_room = updatepos(pos, -min(20, sell_room))
                        print(f"Order({prod}, {price}, {min(-vol, buy_room)})")
                        print(f"Order({prod}, {ba - 2}, {-min(20, sell_room)})")

                

                #── Layer 2: passive quotes with inventory size skew ──
                if fairprice:
                    I = pos / 80
                    bid_size = int(BASE * (1 - I))
                    ask_size = int(BASE * (1 + I))
                    bid_size = max(0, min(bid_size, buy_room))
                    ask_size = max(0, min(ask_size, sell_room))


                    if ba - 1 > fairprice and ask_size > 0:
                        # if abs(od.sell_orders[ba]) == 1:
                        #     ask_size = 1
                        orders.append(Order(prod, ba - 1, -ask_size))
                        # pos, buy_room, sell_room = updatepos(pos, -ask_size)
                        print(f"Order({prod}, {ba-1}, {-ask_size})")

                    if bb + 1 < fairprice and bid_size > 0:
                        # if abs(od.buy_orders[bb]) == 1:
                        #      bid_size = 1
                        orders.append(Order(prod, bb + 1, bid_size))
                        # pos, buy_room, sell_room = updatepos(pos, bid_size)
                        print(f"Order({prod}, {bb+1}, {bid_size})")

                #── Layer 3: zero-edge inventory neutralization ───────
                if pos >= FLAT_THRESHOLD and sell_room > 0:
                    flat_price = int(fairprice)
                    flat_size  = min(pos - FLAT_TARGET, sell_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, -flat_size))
                        print(f"Order({prod}, {flat_price}, {-flat_size})")
                        pos, buy_room, sell_room = updatepos(pos, -flat_size)

                elif pos <= -FLAT_THRESHOLD and buy_room > 0:
                    flat_price = int(fairprice) + (1 if fairprice != int(fairprice) else 0)
                    flat_size  = min(-pos - FLAT_TARGET, buy_room)
                    if flat_size > 0:
                        orders.append(Order(prod, flat_price, flat_size))
                        print(f"Order({prod}, {flat_price}, {flat_size})")
                        pos, buy_room, sell_room = updatepos(pos, flat_size)

            result[prod] = orders

        return result, 0, ""
