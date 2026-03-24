import pandas as pd
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
                bb, ba = max(od.buy_orders), min(od.sell_orders)
                mid = (bb+ba)/2
                orders = []
                pos = state.position.get(prod, 0)
                mid_prices.append((mid, state.timestamp))
                mid_prices.sort(key= lambda x: x[1])
                mid_prices_s = pd.Series([x[0] for x in mid_prices])
                s = mid_prices_s.rolling(window = 100).mean()   
                f = mid_prices_s.ewm(span = 20).mean()
                P = min(15, len(mid_prices) - 1)
                delta = [mid_prices[i][0]-mid_prices[i-1][0] for i in range(1, len(mid_prices))]
                N = min(15, len(delta))
                avg_g = pd.Series([max(delta[len(delta)-1-i], 0) for i in range(0, N)]).mean()
                avg_l = pd.Series([max(-delta[len(delta)-1-i], 0) for i in range(0, N)]).mean() if pd.Series([max(-delta[len(delta)-1-i], 0) for i in range(0, N)]).mean() > 0 else 1
                RS = avg_g/avg_l
                RSI = 100 - (100/(1+RS))
                if f.iloc[-1] > s.iloc[-1]:
                    buy_room = 80 - pos
                    sell_room = 80 + pos
                    if RSI <= 70:
                        if ba - bb >= 2:
                            orders.append(Order(prod, ba - 1, -min(20, sell_room)))
                        orders.append(Order(prod, bb+1, min(40, buy_room)))
                    else:
                        if ba - bb >= 2:
                            orders.append(Order(prod, bb + 1, min(20, buy_room)))
                        orders.append(Order(prod, ba - 1, -min(40, sell_room)))
                if f.iloc[-1] < s.iloc[-1]:
                    buy_room = 80 - pos
                    sell_room = 80 + pos
                    if RSI >= 30:    
                        if ba - bb >= 2:
                            orders.append(Order(prod, bb + 1, min(20, buy_room)))
                        orders.append(Order(prod, ba - 1, -min(40, sell_room)))
                    else:
                        if ba - bb >= 2:
                            orders.append(Order(prod, ba - 1, -min(20, sell_room)))
                        orders.append(Order(prod, ba - 1, -min(40, buy_room)))
            result[prod] = orders
        return result, 0, ""
