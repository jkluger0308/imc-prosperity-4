from datamodel import Order, TradingState
# Current pnl: 3000 (with no parameter tuning)

def updatepos(pos:int, vol:int,maxpos: int = 80):
    new_pos = min(pos + vol, 80) if vol < pos else max(pos + vol, -80)
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

            wb = min(od. buy_orders)
            wa = max(od.sell_orders)
            wallmid = (wb + wa) / 2
            pos = state.position.get(prod, 0)
            orders = []
            trades = state.own_trades.get(prod, [])
            
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
                sb = sorted(set(pbid_below_fair), reverse = True)
                sa = sorted(set(pask_above_fair))

                bb = sb[0] if sb else None
                ba = sa[0] if sa else None

                bb_next = sb[1] if len(sb) > 1 else None
                ba_next = sa[1] if len(sa) > 1 else None


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
                
                if ba - bb >= 2 and ba - round(wallmid) > 6 and bb - round(wallmid) < -6:
                    if sell_room > 0:
                        orders.append(Order(prod, ba - 1, -min(35, sell_room)))
                        print(f"Order({prod}, {ba - 1}, {-min(35, sell_room)})")
                    if buy_room > 0:
                        orders.append(Order(prod, bb + 1, min(35, buy_room)))
                        print(f"Order({prod}, {bb + 1}, {min(35, buy_room)})")
                else:
                    if ba - round(wallmid) <= 6:
                        if buy_room > 0:
                            orders.append(Order(prod, ba, int(od.sell_orders[ba])))
                            if ba_next:
                                orders.append(Order(prod, ba_next - 1, -min(35, sell_room)))
                    if bb - round(wallmid) >= -6:
                        orders.append(Order(prod, bb, int(od.buy_orders[bb])))
                        if bb_next:
                            orders.append(Order(prod, bb_next + 1, min(35, buy_room)))
                
            result[prod] = orders
        return result, 0, ""
