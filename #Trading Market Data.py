from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    def run(self, state: TradingState):
        result = {}
        # Max limits for Round 1
        LIMITS = {"EMERALDS": 75, "TOMATOES": 75}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            current_pos = state.position.get(product, 0)
            limit = LIMITS.get(product, 75)
            
            bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])

            if not bids or not asks: continue

            # --- 3-2-1 IMBALANCE (The 'Lean' Generator) ---
            def get_v(order_list, idx):
                return abs(order_list[idx][1]) if len(order_list) > idx else 0

            w_bid = (get_v(bids, 0) * 3) + (get_v(bids, 1) * 2) + (get_v(bids, 2) * 1)
            w_ask = (get_v(asks, 0) * 3) + (get_v(asks, 1) * 2) + (get_v(asks, 2) * 1)
            imbalance = (w_bid - w_ask) / (w_bid + w_ask) if (w_bid + w_ask) > 0 else 0

            best_bid = bids[0][0]
            best_ask = asks[0][0]

            # --- THE FULL PNL MARKET MAKING ENGINE ---
            
            # We determine our 'Fair Price' as the midpoint
            # Then we shift it based on your 3-2-1 imbalance
            shift = 1 if imbalance > 0.1 else (-1 if imbalance < -0.1 else 0)
            
            # AGGRESSIVE PRICING:
            # We want to be 1 tick inside the spread to 'capture' the trade
            buy_price = best_bid + 1 + shift
            sell_price = best_ask - 1 + shift
            
            # Ensure we don't accidentally buy higher than we sell
            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            # Maximize Reward: Quote the entire remaining limit
            # This ensures every time someone hits us, we take as much of their money as possible
            
            # Place the Buy Side
            if current_pos < limit:
                buy_qty = limit - current_pos
                orders.append(Order(product, int(buy_price), int(buy_qty)))
                
            # Place the Sell Side
            if current_pos > -limit:
                sell_qty = -limit - current_pos
                orders.append(Order(product, int(sell_price), int(sell_qty)))

            result[product] = orders
            
        return result, 0, ""