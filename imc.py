from datamodel import TradingState, Order
class Trader:
    def run(self, state: TradingState):
        result = {}
        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue
            bb = max(od.buy_orders)
            ba = min(od.sell_orders)
            pos = state.position.get(prod, 0)
            orders = []
            if ba - bb >= 2:
                buy_room = 50 - pos
                sell_room = 50 + pos
                if buy_room > 0:
                    orders.append(Order(prod, bb + 1, min(15, buy_room)))
                if sell_room > 0:
                    orders.append(Order(prod, ba - 1, -min(15, sell_room)))
            result[prod] = orders
        return result, 0, ""