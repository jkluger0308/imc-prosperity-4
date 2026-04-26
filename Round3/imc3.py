from datamodel import Order, TradingState
import json

# MM Constants (Reverted to stable Change #13 values)
BASE = 25
ROOM = 50
FLAT_THRESHOLD_SOFT = 5
FLAT_TARGET_SOFT = 0

# Voucher Constants
VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500
}
VOUCHER_LIMIT = 300
VOUCHER_ITM_THRESHOLD = 200 
VOUCHER_EDGE = 0

class Trader:
    def run(self, state: TradingState):
        result = {}
        
        # 1. Calculate Fair Price for Velvetfruit (Underlying)
        ve_fair = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            ve_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if ve_od.buy_orders and ve_od.sell_orders:
                ve_fair = (max(ve_od.buy_orders) + min(ve_od.sell_orders)) / 2

        for prod, od in state.order_depths.items():
            orders = []
            pos = state.position.get(prod, 0)

            # --- VOUCHER ARBITRAGE (Aggressive Taking) ---
            if prod in VOUCHER_STRIKES:
                strike = VOUCHER_STRIKES[prod]
                if ve_fair is None or ve_fair < strike + VOUCHER_ITM_THRESHOLD:
                    result[prod] = []
                    continue

                fair_voucher = ve_fair - strike
                buy_room = VOUCHER_LIMIT - pos
                sell_room = VOUCHER_LIMIT + pos

                # Sweep the Sell side (to Buy)
                for price, vol in sorted(od.sell_orders.items()):
                    if price < fair_voucher - VOUCHER_EDGE and buy_room > 0:
                        take_vol = min(abs(vol), buy_room)
                        orders.append(Order(prod, price, take_vol))
                        buy_room -= take_vol

                # Sweep the Buy side (to Sell)
                for price, vol in sorted(od.buy_orders.items(), reverse=True):
                    if price > fair_voucher + VOUCHER_EDGE and sell_room > 0:
                        take_vol = min(abs(vol), sell_room)
                        orders.append(Order(prod, price, -take_vol))
                        sell_room -= take_vol

                result[prod] = orders
                continue

            # --- MARKET MAKING (Hydrogel & Velvetfruit) ---
            if prod in {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}:
                # Robust Fair Value calculation (Entire book average)
                cm_bid = sum(p * v for p, v in od.buy_orders.items()) / sum(od.buy_orders.values()) if od.buy_orders else None
                cm_ask = sum(p * abs(v) for p, v in od.sell_orders.items()) / sum(abs(v) for v in od.sell_orders.values()) if od.sell_orders else None
                fairprice = (cm_bid + cm_ask) / 2 if cm_bid and cm_ask else None

                if fairprice is None:
                    result[prod] = []
                    continue

                sell_room = ROOM + pos
                buy_room = ROOM - pos
                
                # Inventory Skew logic
                I = pos / ROOM
                bid_size = max(0, min(int(BASE * (1 - I)), buy_room))
                ask_size = max(0, min(int(BASE * (1 + I)), sell_room))

                # Passive Quotes
                ba = min(od.sell_orders) if od.sell_orders else None
                bb = max(od.buy_orders) if od.buy_orders else None

                if ba and ba - 1 > fairprice and ask_size > 0:
                    orders.append(Order(prod, ba - 1, -ask_size))
                elif not ba and bb:
                    orders.append(Order(prod, round(fairprice + 2), -ask_size))
                
                if bb and bb + 1 < fairprice and bid_size > 0:
                    orders.append(Order(prod, bb + 1, bid_size))
                elif not bb and ba:
                    orders.append(Order(prod, round(fairprice - 1), bid_size))

                # Inventory Neutralization
                if pos >= FLAT_THRESHOLD_SOFT:
                    orders.append(Order(prod, int(fairprice), -min(pos - FLAT_TARGET_SOFT, sell_room)))
                elif pos <= -FLAT_THRESHOLD_SOFT:
                    orders.append(Order(prod, int(fairprice) + (1 if fairprice % 1 != 0 else 0), min(-pos - FLAT_TARGET_SOFT, buy_room)))

                result[prod] = orders

        return result, 0, ""
