from datamodel import Order, TradingState
# Current pnl: 3254 (with parameter tuning)
# This is a trend following mean reversion
class Trader:
    def run(self, state: TradingState):
        try:
            parts = state.traderData.split("|")
            tom_ewm = float(parts[0])
            tom_ewm2 = float(parts[1])
            pnl = float(parts[2])
            counter = float(parts[3])
            gain = float(parts[4])
            loss = float(parts[5])
            oldmid = float(parts[6])
            tom_ewm_s = float(parts[7])
            tom_ewm_f = float(parts[8])

        except:
            tom_ewm = None
            tom_ewm2 = None
            pnl = None
            counter = None
            gain = None
            loss = None
            oldmid = None
            tom_ewm_s = None
            tom_ewm_f = None
        result = {}

        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[prod] = []
                continue

            bb = max(od.buy_orders)
            wb = min(od.buy_orders)
            ba = min(od.sell_orders)
            wa = max(od.sell_orders)
            mid = (bb + ba) / 2
            wallmid = (wb + wa)/2
            pos = state.position.get(prod, 0)
            max_pos = 80
            buy_room = max_pos - pos
            sell_room = max_pos + pos
            orders = []
            trades = state.own_trades.get(prod, [])
            
            # PnL calculation. This returns the algorithm's PnL up to the tick right before the current one (inclusive).
            if pnl == None:
                pnl = 0
            oldpnl = pnl
            for trade in trades:
                pnl += trade.price * abs(trade.quantity) if trade.seller == "SUBMISSION" else -trade.price * abs(trade.quantity)

            if prod == "EMERALDS":
                mm_sz = 35
                buyq = min(mm_sz, buy_room) 
                sellq = -min(mm_sz, sell_room) 
                buyprice = bb + 1
                sellprice = ba - 1
                if bb + 1 < wallmid and buy_room > 0:
                    orders.append(Order(prod, buyprice, buyq))
                if ba - 1 > wallmid and sell_room > 0:
                    orders.append(Order(prod, sellprice, sellq))
                else:
                    if pos > 0:
                        orders.append(Order(prod, sellprice, -pos))
                    elif pos < 0:
                        orders.append(Order(prod, buyprice, -pos))

            elif prod == "TOMATOES":
                if counter == None:
                    counter = 1
                else:
                    counter += 1
                
                if oldmid == None:
                    oldmid = wallmid
                
                # STOP LOSS
                if pnl is not None and oldpnl != 0:
                    if (pnl - oldpnl) / oldpnl <= -0.01: # Stop loss parameter, doesn't trade this tick if pnl decreased by more than 1% over the last two ticks.
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
                
                # RSI ---------------------------------------------------------------------------
                delta = wallmid - oldmid
                if gain == None:
                    gain = max(0, delta)
                else:
                    gain = (max(0, delta) + gain * (counter - 1))/counter
                if loss == None:
                    loss = abs(min(0, delta))
                else:
                    loss = (abs(min(0, delta)) + loss*(counter - 1))/counter

                RS = gain/loss if loss != 0 else None
                RSI = 100 - (100/(1+RS)) if RS else 50


                # Momentum -----------------------------------------------------------------------
                alpha_s = 2 / (60 + 1)
                alpha_f = 2 / (10 + 1)
                # slow EWM
                if tom_ewm_s == None:
                    tom_ewm_s = wallmid
                else:
                    tom_ewm_s = alpha_s * wallmid + (1-alpha_s) * tom_ewm_s
                # fast EWM
                if tom_ewm_f == None:
                    tom_ewm_f = wallmid
                else:
                    tom_ewm_f = alpha_f * wallmid + (1-alpha_f) * tom_ewm_f
                
                trendingup = tom_ewm_s < tom_ewm_f and RSI < 70
                trendingdown = tom_ewm_s > tom_ewm_f and RSI > 30

                zmin = -2.1
                zmax = 2.1

                if trendingup: # filter out bad counter-trend trades
                    zmax = 2.2
                if trendingdown:
                    zmin = -2.2

                # Buy signals ------------------------------------------------

                if z < zmin:
                    # Strong conviction BUY — cross spread, full room
                    if buy_room > 0:
                        orders.append(Order(prod, ba, min(40, buy_room)))

                # Sell signals -----------------------------------------------

                elif z > zmax:
                    # Strong conviction SELL — cross spread, full room 
                    if sell_room > 0:
                        orders.append(Order(prod, bb, -min(40, sell_room)))


                else:
                    # |z| < 2: Aggressive MM with more buying power 
                    mm_sz = 35
                    buyq = min(mm_sz, buy_room) 
                    sellq = -min(mm_sz, sell_room) 
                    buyprice = bb + 1
                    sellprice = ba - 1
                    if bb + 1 < wallmid and buy_room > 0:
                        orders.append(Order(prod, buyprice, buyq))
                    if ba - 1 > wallmid and sell_room > 0:
                        orders.append(Order(prod, sellprice, sellq))
                    else:
                        if pos > 0:
                            orders.append(Order(prod, sellprice, -pos))
                        elif pos < 0:
                            orders.append(Order(prod, buyprice, -pos))
                oldmid = wallmid
            result[prod] = orders
        trader_data = f"{tom_ewm}|{tom_ewm2}|{pnl}|{counter}|{gain}|{loss}|{oldmid}|{tom_ewm_s}|{tom_ewm_f}" if tom_ewm is not None else "" # add |{counter}|{gain}|{loss}|{oldmid}|{tom_ewm_s}|{tom_ewm_f} if uncommenting out
        return result, 0, trader_data
