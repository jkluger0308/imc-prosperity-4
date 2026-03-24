import pandas as pd
import pandas as pd
import math
from datamodel import Order, TradingState
mid_prices = []
class Trader:
    def run(self, state: TradingState):
        result = {}
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
             orders = []

             cm_bid = sum([float(vol*price) for price, vol in od.buy_orders.items()])/sum([float(vol) for price, vol in od.buy_orders.items()])
             cm_ask = sum([float(vol*price) for price, vol in od.sell_orders.items()])/sum([float(vol) for price, vol in od.sell_orders.items()])
             mid = (cm_bid + cm_ask) / 2

             mid_prices.append((mid, state.timestamp))
             mid_prices.sort(key = lambda x: x[1])
             mid_prices_s = pd.Series([x[0] for x in mid_prices])
             
             s = mid_prices_s.ewm(span = 1000).mean()
             f = mid_prices_s.ewm(span=250).mean()
             delta = [mid_prices[i][0]-mid_prices[i-1][0] for i in range(1, len(mid_prices))]
            
             N = min(15, len(delta))
             avg_g = pd.Series([max(delta[len(delta)-1-i], 0) for i in range(0, N)]).mean()
             avg_l = pd.Series([max(-delta[len(delta)-1-i], 0) for i in range(0, N)]).mean() if pd.Series([max(-delta[len(delta)-1-i], 0) for i in range(0, N)]).mean() > 0 else 1
            
             RS = avg_g/avg_l
             RSI = 100 - (100/(1+RS))
             pos = state.position.get(prod, 0)
             buy_room = 80 - pos
             sell_room = 80 + pos
             if f.iloc[-1] > s.iloc[-1]:
                if RSI < 70:
                    orders.append(Order(prod, bb + 2, min(30, buy_room)))
                    orders.append(Order(prod, ba - 1, -min(10, sell_room)))
                else:
                    orders.append(Order(prod, ba - 1, -min(30, sell_room)))
             elif f.iloc[-1] < s.iloc[-1]:
                if RSI > 30:
                    orders.append(Order(prod, ba - 2, -min(30, sell_room)))
                    orders.append(Order(prod, bb + 1, min(10, buy_room)))
                else:
                    orders.append(Order(prod, bb + 1, -min(30, sell_room)))
             else:
                if buy_room > 0:
                    orders.append(Order(prod, bb+1, min(15, buy_room)))
                if sell_room > 0:
                    orders.append(Order(prod, ba - 1, -min(15, sell_room)))
            result[prod] = orders
        return result, 0, ""
