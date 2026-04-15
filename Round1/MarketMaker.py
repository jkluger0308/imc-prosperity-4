from datamodel import Order, TradingState
# Current pnl: 3000 (with no parameter tuning)

def updatepos(pos:int, vol:int,maxpos: int = 80):
    new_pos = min(pos + vol, 80) if vol > 0 else max(pos + vol, -80)
    buy_room = max(0, maxpos - new_pos)
    sell_room = max(0, maxpos + new_pos)
    return new_pos, buy_room, sell_room

class Trader:

    def run(self, state: TradingState):
        result = {}
        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue
            
            cm_bid = sum([float(vol*price) for price, vol in od.buy_orders.items()])/sum([float(vol) for price, vol in od.buy_orders.items()])
            cm_ask = sum([float(abs(vol)*price) for price, vol in od.sell_orders.items()])/sum([float(abs(vol)) for price, vol in od.sell_orders.items()])
            # wb = min(od. buy_orders)
            # wa = max(od.sell_orders)
            wallmid = (cm_bid + cm_ask) / 2
            pos = state.position.get(prod, 0)
            orders = []
            
            if prod == "INTARIAN_PEPPER_ROOT":
                pos = state.position.get(prod, 0)
                if not od.buy_orders or not od.sell_orders:
                    result[prod] = []
                    continue
                if pos < 80 and od.sell_orders:
                    ba = min(od.sell_orders)
                    orders.append(Order(prod, ba, 80 - pos))
                    print(f"Order({prod}, {ba}, {80 - pos})")

            elif prod == "ASH_COATED_OSMIUM":
                if not od.buy_orders or not od.sell_orders:
                    result[prod] = []
                    continue

                buy_room = 80 - pos
                sell_room = 80 + pos
                fairprice = wallmid

                initial_bids = od.buy_orders.items()
                initial_asks = od.sell_orders.items()

                bids_below_fair = (
                    (price, vol) for price, vol in od.buy_orders.items() if price < fairprice
                )
                asks_above_fair = (
                    (price, vol) for price, vol in od.sell_orders.items() if price > fairprice
                )
                pbid_below_fair = [x[0] for x in bids_below_fair]
                pask_above_fair = [x[0] for x in asks_above_fair]

                for price, vol in initial_bids:
                    if price > fairprice and sell_room > 0:
                        orders.append(Order(prod, price, -min(vol, sell_room)))
                        print(f"Order({prod}, {price}, {-min(vol, sell_room)})")
                        pos, buy_room, sell_room = updatepos(pos, -min(vol, sell_room))
                for price, vol in initial_asks:
                    if price < fairprice and buy_room > 0:
                        orders.append(Order(prod, price, min(-vol, buy_room)))
                        print(f"Order({prod}, {price}, {min(-vol, buy_room)})")
                        pos, buy_room, sell_room = updatepos(pos, min(-vol, buy_room))


                bb = max(pbid_below_fair)
                ba = min(pask_above_fair)
                
                Spread = ba - bb
                if Spread >= 2:
                    if sell_room > 0:
                        orders.append(Order(prod, ba - 1, -min(35, sell_room)))
                        print(f"Order({prod}, {ba - 1}, {-min(35, sell_room)})")
                    if buy_room > 0:
                        orders.append(Order(prod, bb + 1, min(35, buy_room)))
                        print(f"Order({prod}, {bb + 1}, {min(35, buy_room)})")
                if pos > 0:
                    orders.append(Order(prod, bb, -pos))
                if pos < 0:
                    orders.append(Order(prod, ba, -pos))
                
            result[prod] = orders
        return result, 0, ""
