import pandas as pd
import math
from datamodel import Order, TradingState
mid_prices = []
class Trader:
    def run(self, state: TradingState):
        result = {}
        Span = 2500
        for prod, od in state.order_depths.items():
            if prod != 'TOMATOES':
                result[prod] = []
                continue
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue
            bb, ba = max(od.buy_orders), min(od.sell_orders)
            mid = (bb + ba) / 2
            mid_prices.append((mid, state.timestamp))
            mid_prices.sort(key = lambda x: x[1])
            mid_prices_s = pd.Series([x[0] for x in mid_prices])
            ewm = mid_prices_s.ewm(span=Span).mean()
            pos = state.position.get(prod, 0)
            orders = []
            spread = abs(bb-ba)
            if mid - ewm.iloc[len(ewm) - 1] <= -1:
                buy_room = 80 - pos
                sell_room = 80 + pos
                if spread >= 2:
                    orders.append(Order(prod, ba - 1, -min(20, sell_room)))
                if mid - ewm.iloc[len(ewm) - 1] <= -2:
                    if buy_room > 0:
                        orders.append(Order(prod, bb+1, min(60, buy_room)))
                else:
                    if buy_room > 0:
                        orders.append(Order(prod, bb+1, min(30, buy_room)))
            if mid - ewm.iloc[len(ewm) - 1] >= 1:
                sell_room = 80 + pos
                buy_room = 80 - pos
                if spread >= 2:
                    orders.append(Order(prod, bb+1, min(20, buy_room)))
                if mid - ewm.iloc[len(ewm) - 1] >= 2:
                    if sell_room > 0:
                        orders.append(Order(prod, ba-1, -min(60, sell_room)))
                else:
                    if sell_room > 0:
                        orders.append(Order(prod, ba-1, -min(30, sell_room)))
            result[prod] = orders
        return result, 0, ""    
