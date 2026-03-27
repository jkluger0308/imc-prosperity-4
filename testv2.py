from datamodel import Order, TradingState


class Trader:
    def run(self, state: TradingState):
        try:
            parts = state.traderData.split(",")
            tom_ewm = float(parts[0])
            tom_ewm2 = float(parts[1])
        except:
            tom_ewm = None
            tom_ewm2 = None

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

            if prod == "EMERALDS":
                if 80 - pos > 0:
                    orders.append(Order(prod, bb + 1, min(80, 80 - pos)))
                if 80 + pos > 0:
                    orders.append(Order(prod, ba - 1, -min(80, 80 + pos)))

            elif prod == "TOMATOES":
                alpha = 2 / (20 + 1)

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

                if z < -2.0:
                    if buy_room > 0:
                        orders.append(Order(prod, ba, buy_room))

                elif z > 2.0:
                    if sell_room > 0:
                        orders.append(Order(prod, bb, -sell_room))

                else:
                    mm_sz = 50
                    if buy_room > 0:
                        orders.append(Order(prod, bb + 1, min(mm_sz, buy_room)))
                    if sell_room > 0:
                        orders.append(Order(prod, ba - 1, -min(mm_sz, sell_room)))

            result[prod] = orders

        trader_data = f"{tom_ewm},{tom_ewm2}" if tom_ewm is not None else ""
        return result, 0, trader_data