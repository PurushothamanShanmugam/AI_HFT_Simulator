"""
Multi-Match Trader-to-Trader Execution Engine
=============================================

Purpose:
- Collect BUY and SELL orders from multiple traders.
- Match ALL possible buyers and sellers for the same asset and tick.
- Use price-time-latency priority.
- If direct match is not possible, remaining orders can later fall back to market liquidity.

This engine supports:
- Multiple buyers
- Multiple sellers
- Partial fills
- Direct trader-to-trader execution
- Matching efficiency calculation
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import itertools

_order_id_counter = itertools.count(1)


@dataclass
class Order:
    trader_id: str
    trader_label: str
    trader_emoji: str
    asset: str
    side: str  # BUY or SELL
    quantity: int
    limit_price: float
    tick: int
    latency_ms: float
    reason: str
    order_id: Optional[int] = None

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = next(_order_id_counter)

        self.side = self.side.upper()


@dataclass
class MatchedTrade:
    tick: int
    asset: str

    buyer_id: str
    seller_id: str
    buyer_label: str
    seller_label: str
    buyer_emoji: str
    seller_emoji: str

    quantity: int
    execution_price: float
    bid_price: float
    ask_price: float
    spread: float
    latency_ms: float

    buyer_reason: str
    seller_reason: str
    match_type: str = "DIRECT"


class MultiMatchExecutionEngine:
    """
    Price-time-latency priority matching engine.

    BUY priority:
        1. Highest limit price
        2. Earliest tick
        3. Lowest latency

    SELL priority:
        1. Lowest limit price
        2. Earliest tick
        3. Lowest latency
    """

    def __init__(self):
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.direct_matches: List[MatchedTrade] = []
        self.market_fallback_orders: List[Order] = []

    def submit_order(self, order: Order):
        if order.quantity <= 0:
            return

        if order.side == "BUY":
            self.bids.append(order)
        elif order.side == "SELL":
            self.asks.append(order)

    def submit_orders(self, orders: List[Order]):
        for order in orders:
            self.submit_order(order)

    def match_tick_asset(self, tick: int, asset: str) -> List[MatchedTrade]:
        """
        Match all possible BUY and SELL orders for one asset at one tick.
        """

        matched = []

        asset_bids = [
            o
            for o in self.bids
            if o.asset == asset and o.tick == tick and o.quantity > 0
        ]

        asset_asks = [
            o
            for o in self.asks
            if o.asset == asset and o.tick == tick and o.quantity > 0
        ]

        asset_bids.sort(key=lambda o: (-o.limit_price, o.tick, o.latency_ms))
        asset_asks.sort(key=lambda o: (o.limit_price, o.tick, o.latency_ms))

        while asset_bids and asset_asks:
            best_bid = asset_bids[0]
            best_ask = asset_asks[0]

            if best_bid.limit_price < best_ask.limit_price:
                break

            qty = min(best_bid.quantity, best_ask.quantity)

            execution_price = round(
                (best_bid.limit_price + best_ask.limit_price) / 2, 5
            )

            spread = round(best_bid.limit_price - best_ask.limit_price, 6)

            latency_ms = round((best_bid.latency_ms + best_ask.latency_ms) / 2, 2)

            trade = MatchedTrade(
                tick=tick,
                asset=asset,
                buyer_id=best_bid.trader_id,
                seller_id=best_ask.trader_id,
                buyer_label=best_bid.trader_label,
                seller_label=best_ask.trader_label,
                buyer_emoji=best_bid.trader_emoji,
                seller_emoji=best_ask.trader_emoji,
                quantity=qty,
                execution_price=execution_price,
                bid_price=round(best_bid.limit_price, 5),
                ask_price=round(best_ask.limit_price, 5),
                spread=spread,
                latency_ms=latency_ms,
                buyer_reason=best_bid.reason,
                seller_reason=best_ask.reason,
                match_type="DIRECT",
            )

            matched.append(trade)
            self.direct_matches.append(trade)

            best_bid.quantity -= qty
            best_ask.quantity -= qty

            if best_bid.quantity <= 0:
                self.bids.remove(best_bid)
                asset_bids.pop(0)

            if best_ask.quantity <= 0:
                self.asks.remove(best_ask)
                asset_asks.pop(0)

        return matched

    def match_all_for_tick(self, tick: int) -> List[MatchedTrade]:
        """
        Match all assets for the given tick.
        """

        assets = set(
            [o.asset for o in self.bids if o.tick == tick]
            + [o.asset for o in self.asks if o.tick == tick]
        )

        all_matches = []

        for asset in assets:
            all_matches.extend(self.match_tick_asset(tick, asset))

        return all_matches

    def get_unmatched_orders_for_tick(self, tick: int) -> List[Order]:
        """
        Remaining unmatched orders for this tick.
        These can be executed using market fallback logic.
        """

        unmatched = [
            o for o in self.bids + self.asks if o.tick == tick and o.quantity > 0
        ]

        return unmatched

    def clear_tick_orders(self, tick: int):
        """
        Remove leftover orders for one tick after fallback execution.
        """

        self.bids = [o for o in self.bids if o.tick != tick]
        self.asks = [o for o in self.asks if o.tick != tick]

    def matching_summary(self, total_trade_rows: int) -> Dict[str, float]:
        direct_rows = len(self.direct_matches)
        fallback_rows = max(total_trade_rows - direct_rows, 0)

        efficiency = 0.0
        if total_trade_rows > 0:
            efficiency = round((direct_rows / total_trade_rows) * 100, 2)

        return {
            "direct_match_rows": direct_rows,
            "market_fallback_rows": fallback_rows,
            "matching_efficiency_pct": efficiency,
        }

    def book_snapshot(self, asset: Optional[str] = None) -> Dict[str, List[Dict]]:
        if asset:
            bids = [o for o in self.bids if o.asset == asset]
            asks = [o for o in self.asks if o.asset == asset]
        else:
            bids = self.bids
            asks = self.asks

        return {
            "bids": [
                {
                    "order_id": o.order_id,
                    "trader": o.trader_id,
                    "asset": o.asset,
                    "price": round(o.limit_price, 5),
                    "qty": o.quantity,
                    "tick": o.tick,
                    "latency_ms": o.latency_ms,
                }
                for o in sorted(
                    bids, key=lambda x: (-x.limit_price, x.tick, x.latency_ms)
                )
            ],
            "asks": [
                {
                    "order_id": o.order_id,
                    "trader": o.trader_id,
                    "asset": o.asset,
                    "price": round(o.limit_price, 5),
                    "qty": o.quantity,
                    "tick": o.tick,
                    "latency_ms": o.latency_ms,
                }
                for o in sorted(
                    asks, key=lambda x: (x.limit_price, x.tick, x.latency_ms)
                )
            ],
        }