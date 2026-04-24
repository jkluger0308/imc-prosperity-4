from datamodel import Order, TradingState
import json

# This also needs to be fitted for VelvetFruit extract as it uses different params

FLAT_THRESHOLD_SOFT = 15
FLAT_TARGET_SOFT    = 0
BASE           = 24  
GEL_SPANF = 5
GEL_SPANS = 10*GEL_SPANF
GEL_EDGE = 0.5

RSI_PERIOD = 14

def wilder_rsi_update(price: float, st: dict, period: int = RSI_PERIOD):
    """
    Stateful Wilder RSI update (tick-by-tick).
    Returns: (rsi_or_none, updated_state_dict)
    """
    p = max(2, int(period))
    prev_price = st.get("prev_price")
    avg_gain = st.get("avg_gain")
    avg_loss = st.get("avg_loss")
    warm_gains = st.get("warm_gains", [])
    warm_losses = st.get("warm_losses", [])

    if prev_price is None:
        st["prev_price"] = float(price)
        st["avg_gain"] = avg_gain
        st["avg_loss"] = avg_loss
        st["warm_gains"] = warm_gains
        st["warm_losses"] = warm_losses
        return None, st

    delta = float(price) - float(prev_price)
    gain = delta if delta > 0 else 0.0
    loss = -delta if delta < 0 else 0.0

    # Warmup with simple means for first `p` deltas
    if avg_gain is None or avg_loss is None:
        warm_gains = (warm_gains + [gain])[-p:]
        warm_losses = (warm_losses + [loss])[-p:]
        if len(warm_gains) < p:
            st["prev_price"] = float(price)
            st["avg_gain"] = None
            st["avg_loss"] = None
            st["warm_gains"] = warm_gains
            st["warm_losses"] = warm_losses
            return None, st
        avg_gain = sum(warm_gains) / p
        avg_loss = sum(warm_losses) / p
    else:
        # Wilder RMA update
        avg_gain = ((avg_gain * (p - 1)) + gain) / p
        avg_loss = ((avg_loss * (p - 1)) + loss) / p

    # Match dashboard flat handling
    if avg_gain <= 0.0 and avg_loss <= 0.0:
        rsi = 50.0
    elif avg_loss <= 0.0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = max(0.0, min(100.0, rsi))

    st["prev_price"] = float(price)
    st["avg_gain"] = avg_gain
    st["avg_loss"] = avg_loss
    st["warm_gains"] = warm_gains
    st["warm_losses"] = warm_losses
    return rsi, st




def updatepos(pos: int, vol: int, maxpos: int = 200):
    new_pos = pos + vol
    new_pos = max(-maxpos, min(maxpos, new_pos))
    buy_room  = max(0, maxpos - new_pos)
    sell_room = max(0, maxpos + new_pos)
    return new_pos, buy_room, sell_room


class Trader:

    def run(self, state: TradingState):
        result = {}

        prev = {}
        if state.traderData:
            try:
                prev = json.loads(state.traderData)
            except:
                prev = {}

        new_state = {}

        for prod, od in state.order_depths.items():
            
            ba = min(od.sell_orders) if od.sell_orders else None
            bb = max(od.buy_orders) if od.buy_orders else None
            touch_mid = (bb + ba) / 2 if bb and ba else ba if ba and (not bb) else bb if bb and (not ba) else None
            ema_fast_key = f"ema_{prod}_fast"
            ema_slow_key = f"ema_{prod}_slow"
            ema_fb = prev.get(ema_fast_key) or prev.get(ema_slow_key)

            cm_bid = sum([float(vol*price) for price, vol in od.buy_orders.items()])/sum([float(vol) for price, vol in od.buy_orders.items()]) if od.buy_orders else ema_fb
            cm_ask = sum([float(abs(vol)*price) for price, vol in od.sell_orders.items()])/sum([float(abs(vol)) for price, vol in od.sell_orders.items()]) if od.sell_orders else ema_fb
            VWAP = (cm_bid + cm_ask) / 2 if cm_bid and cm_ask else None
            if VWAP is not None:
                fairprice = VWAP
            elif touch_mid is not None:
                fairprice = touch_mid
            else:
                fairprice = 10000
            
            pos = state.position.get(prod, 0)
            sell_room = 200 + pos
            buy_room = 200 - pos
            orders = []

            if prod =="HYDROGEL_PACK":
                
                rsi_key = f"rsi_{prod}"
                rsi_st = prev.get(rsi_key, {})

                rsi_val, rsi_st = wilder_rsi_update(touch_mid, rsi_st, RSI_PERIOD)
                new_state[rsi_key] = rsi_st
                if rsi_val is not None:
                    reverseup =rsi_val < 30
                    reversedown = rsi_val > 70
                else:
                    reverseup = False
                    reversedown = False
                
                if fairprice is not None:
                    prev_ema_f = prev.get(ema_fast_key)
                    alpha_f = 2/(GEL_SPANF + 1)
                    if prev_ema_f is not None:
                        ema_f = alpha_f*fairprice + (1 - alpha_f)*prev_ema_f
                    else:
                        ema_f = fairprice
                    new_state[ema_fast_key] = ema_f

                    prev_ema_s = prev.get(ema_slow_key)
                    alpha_s = 2/(GEL_SPANS + 1)
                    if prev_ema_s is not None:
                        ema_s = alpha_s*fairprice + (1 - alpha_s)*prev_ema_s
                    else:
                        ema_s = fairprice
                    new_state[ema_slow_key] = ema_s

                    trending_up = ema_f > ema_s + GEL_EDGE
                    trending_down = ema_f + GEL_EDGE < ema_s

               # Market making parameters 
                I = pos / 200
                bid_size = int(BASE * (1 - I))
                ask_size = int(BASE * (1 + I))
                bid_size = max(0, min(bid_size, buy_room))
                ask_size = max(0, min(ask_size, sell_room))

                #——— Contingency in case orderbook is partially empty
                if (not ba) or (not bb):
                    if (not bb) and ba:
                            orders.append(Order(prod, ba, bid_size))
                    elif (not ba) and bb:
                            orders.append(Order(prod, bb, -ask_size))
                    elif (not ba) and (not bb):
                            orders.append(Order(prod, round(fairprice - 1), bid_size))
                            orders.append(Order(prod, round(fairprice + 1), -ask_size))

                #── Passive quotes with inventory size skew ──
                elif fairprice is not None:

                    if ba and ba - 1 > fairprice and ask_size > 0:
                        sellprice = ba - 1
                        if trending_up:
                            if reversedown:
                                sellprice = ba - 2
                            else:
                                sellprice = ba
                        elif trending_down and ba - 2 > fairprice:
                            sellprice = ba - 2
                            if reverseup:
                                sellprice = ba
                        orders.append(Order(prod, sellprice, -ask_size))
                        print(f"Order({prod}, {sellprice}, {-ask_size})")

                    if bb and bb + 1 < fairprice and bid_size > 0:
                        buyprice = round(fairprice - 1)
                        if trending_up:
                            if reversedown:
                                buyprice = bb
                            elif bb + 2 < fairprice:
                                buyprice = bb + 2
                        elif trending_down:
                            if reverseup:
                                buyprice = bb + 2
                            else:
                                buyprice = bb
                        orders.append(Order(prod, buyprice, bid_size))
                        print(f"Order({prod}, {buyprice}, {bid_size})")
                        
                #── Layer 3: zero-edge inventory neutralization ──────-
                if VWAP is not None or touch_mid is not None:
                    if pos >= FLAT_THRESHOLD_SOFT and sell_room > 0:
                        flat_price = int(fairprice)
                        flat_size  = min(pos - FLAT_TARGET_SOFT, sell_room)
                        if flat_size > 0:
                            orders.append(Order(prod, flat_price, -flat_size))
                            print(f"Order({prod}, {flat_price}, {-flat_size})")

                    elif pos <= -FLAT_THRESHOLD_SOFT and buy_room > 0:
                        flat_price = int(fairprice) + (1 if fairprice != int(fairprice) else 0)
                        flat_size  = min(-pos - FLAT_TARGET_SOFT, buy_room)
                        if flat_size > 0:
                            orders.append(Order(prod, flat_price, flat_size))
                            print(f"Order({prod}, {flat_price}, {flat_size})")
                print(f"GELPOS: {pos}")

            result[prod] = orders
        trader_data = json.dumps(new_state)
        return result, 0, trader_data
