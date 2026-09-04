from collections import defaultdict, deque

from src.simulator.order import Order, OrderSide, OrderType


class OrderBook:
    def __init__(self):
        self.bids = defaultdict(deque)
        self.asks = defaultdict(deque)
        self.trades = []

    def add_order(self, order: Order):
        if order.order_type == OrderType.MARKET:
            return self._match_market_order(order)

        if order.order_type == OrderType.LIMIT:
            return self._add_limit_order(order)

        raise ValueError(f"Unsupported order type: {order.order_type}")

    def _add_limit_order(self, order: Order):
        if order.price is None:
            raise ValueError("Limit order must have a price")

        if order.side == OrderSide.BUY:
            self.bids[order.price].append(order)
        else:
            self.asks[order.price].append(order)

        return []

    def _match_market_order(self, order: Order):
        trades = []
        remaining_quantity = order.quantity

        if order.side == OrderSide.BUY:
            book_side = self.asks
            prices = sorted(book_side.keys())
        else:
            book_side = self.bids
            prices = sorted(book_side.keys(), reverse=True)

        for price in prices:
            queue = book_side[price]

            while queue and remaining_quantity > 0:
                resting_order = queue[0]
                traded_quantity = min(remaining_quantity, resting_order.quantity)

                trade = {
                    "incoming_order_id": order.order_id,
                    "resting_order_id": resting_order.order_id,
                    "price": price,
                    "quantity": traded_quantity,
                    "buyer": (
                        order.agent_id
                        if order.side == OrderSide.BUY
                        else resting_order.agent_id
                    ),
                    "seller": (
                        resting_order.agent_id
                        if order.side == OrderSide.BUY
                        else order.agent_id
                    ),
                    "timestamp": order.timestamp,
                }

                trades.append(trade)
                self.trades.append(trade)

                remaining_quantity -= traded_quantity
                resting_order.quantity -= traded_quantity

                if resting_order.quantity == 0:
                    queue.popleft()

            if not queue:
                del book_side[price]

            if remaining_quantity == 0:
                break

        return trades

    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

    def spread(self):
        if self.best_bid() is None or self.best_ask() is None:
            return None
        return self.best_ask() - self.best_bid()

    def summary(self):
        return {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "spread": self.spread(),
            "total_trades": len(self.trades),
        }