import pandas as pd
import math
from datamodel import Order, TradingState
mid_prices = []
# WE BROKE THE 2518 BARRIER
# key strat: adjust prices that you place orders at when expecting price to go up or down
class Trader:
    def run(self, state: TradingState):
        result = {}

        Metric = 1050
        for prod, od in state.order_depths.items():
            if prod == 'EMERALDS':
                if not od.buy_orders or not od.sell_orders:
                    result[prod] = []
                    continue
                orders = []
                bb, ba = max(od.buy_orders), min(od.sell_orders)
                pos = state.position.get(prod, 0)
                if ba - bb >= 2:
                    buy_room = 80 - pos 
                    sell_room = 80 + pos 
                    if buy_room > 0:
                        orders.append(Order(prod, bb+1, min(15, buy_room)))
                    if sell_room > 0:
                        orders.append(Order(prod, ba - 1, -min(15, sell_room)))
            elif prod == 'TOMATOES':
             if not od.buy_orders or not od.sell_orders:
                 result[prod] = []
                 continue
             bb, ba = max(od.buy_orders), min(od.sell_orders)
             mid = (bb + ba) / 2
             mid_prices.append((mid, state.timestamp))
             mid_prices.sort(key = lambda x: x[1])
             mid_prices_s = pd.Series([x[0] for x in mid_prices])
             ewm = mid_prices_s.ewm(span=Metric).mean()
             pos = state.position.get(prod, 0)
             orders = []
             spread = ba-bb
             if mid - ewm.iloc[len(ewm) - 1] <= -0.5:
                 buy_room = 80 - pos
                 sell_room = 80 + pos
                 if spread >= 2 and mid - ewm.iloc[len(ewm) - 1] > -2:
                     orders.append(Order(prod, ba-1, -min(15, sell_room)))
                 if mid - ewm.iloc[len(ewm) - 1] <= -2:
                     if buy_room > 0:
                         orders.append(Order(prod, bb+2, min(40, buy_room)))
                 else:
                    if buy_room > 0:
                        orders.append(Order(prod, bb+1, min(40, buy_room)))
             if mid - ewm.iloc[len(ewm) - 1] >= 0.5:
                 sell_room = 80 + pos
                 buy_room = 80 - pos
                 if spread >= 2 and mid - ewm.iloc[len(ewm) - 1] < 2:
                     orders.append(Order(prod, bb+1, min(15, buy_room)))
                 if mid - ewm.iloc[len(ewm) - 1] >= 2:
                     if sell_room > 0:
                         orders.append(Order(prod, ba-2, -min(40, sell_room)))
                 else:
                     if sell_room > 0:
                         orders.append(Order(prod, ba-1, -min(40, sell_room)))
            result[prod] = orders
        return result, 0, ""
