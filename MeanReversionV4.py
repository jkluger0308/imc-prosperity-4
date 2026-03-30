from datamodel import Order, TradingState
import ast
# Current pnl (with parameter tuning): 3024
class Trader:
    def run(self, state: TradingState):
        try:
            parts = state.traderData.split("|")
            tom_ewm = float(parts[0])
            tom_ewm2 = float(parts[1])
            pnl = float(parts[2])
        except:
            tom_ewm = None
            tom_ewm2 = None
            pnl = None

        result = {}

        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue

            bb = max(od.buy_orders)
            ba = min(od.sell_orders)
            mid = (bb + ba) / 2
            pos = state.position.get(prod, 0)
            orders = []
            trades = state.own_trades.get(prod, [])
            
            # PnL calculation. This returns the algorithm's PnL up to the tick right before the current one (inclusive).
            if pnl == None:
                pnl = 0
            oldpnl = pnl
            for trade in trades:
                pnl += trade.price * abs(trade.quantity) if trade.seller == "SUBMISSION" else -trade.price * abs(trade.quantity)
            
            if prod == "EMERALDS":
                if ba - bb >= 2:
                    if 80 - pos > 0:
                        orders.append(Order(prod, bb + 1, min(80, 80 - pos)))
                    if 80 + pos > 0:
                        orders.append(Order(prod, ba - 1, -min(80, 80 + pos)))

            elif prod == "TOMATOES":
                if pnl is not None and oldpnl != 0:
                    if (pnl - oldpnl) / oldpnl < -0.01: # Stop loss parameter, doesn't trade this tick if pnl decreased by more than 1% over the last two ticks.
                        result[prod] = []
                        continue
                
                alpha = 2 / (20 + 1) # Manual exponentially weighted mean calculation provides lower latency than using Pandas.
                if tom_ewm is None: 
                    tom_ewm = mid
                    tom_ewm2 = mid * mid
                else:
                    tom_ewm = alpha * mid + (1 - alpha) * tom_ewm
                    tom_ewm2 = alpha * (mid * mid) + (1 - alpha) * tom_ewm2

                var = tom_ewm2 - tom_ewm * tom_ewm
                std = var ** 0.5 if var > 0 else 0.001
                z = (mid - tom_ewm) / std

                max_pos = 80
                buy_room = max_pos - pos
                sell_room = max_pos + pos
                
                if z < -2:
                    # Strong conviction BUY — cross spread, full room
                    if buy_room > 0:
                        orders.append(Order(prod, ba, round(0.5*buy_room)))

                elif z > 2:
                    # Strong conviction SELL — cross spread, full room 
                    if sell_room > 0:
                        orders.append(Order(prod, bb, -round(0.5*sell_room)))

                else:
                    # |z| < 2: Aggressive MM with more buying power 
                    mm_sz = 35
                    if ba - bb >= 2:
                            if buy_room > 0:
                                orders.append(Order(prod, bb + 1, min(mm_sz, buy_room)))
                            if sell_room > 0:
                                orders.append(Order(prod, ba - 1, -min(mm_sz, sell_room)))
            result[prod] = orders
        trader_data = f"{tom_ewm}|{tom_ewm2}|{pnl}" if tom_ewm is not None else ""
        return result, 0, trader_data
