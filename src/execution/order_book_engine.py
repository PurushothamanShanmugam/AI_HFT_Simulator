"""
Final Exchange-Grade Execution Engine for HFT Nexus
===================================================

File:
    src/execution/order_book_engine.py

Purpose:
    This is the final execution engine for your HFT simulator.

What it improves:
    1. Persistent exchange-style order book
    2. Price-time-latency priority
    3. Multi-match execution
    4. Partial fills
    5. Liquidity-provider quotes
    6. Price tolerance matching
    7. Reduced market fallback dependency
    8. Matching efficiency metrics
    9. Clean buyer/seller trade records for dashboard and report

Important:
    This file is designed to replace your existing:
        src/execution/order_book_engine.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import itertools
import random

_order_id_counter = itertools.count(1)


@dataclass
class LimitOrder:
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
    is_liquidity_provider: bool = False

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = next(_order_id_counter)

        self.side = str(self.side).upper().strip()
        self.asset = str(self.asset).upper().strip()
        self.quantity = int(max(self.quantity, 0))
        self.limit_price = float(self.limit_price)
        self.latency_ms = float(max(self.latency_ms, 0.0))


@dataclass
class BookTrade:
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
    mid_price: float
    spread: float
    transaction_cost: float
    latency_ms: float
    slippage: float

    buyer_reason: str
    seller_reason: str

    match_type: str = "DIRECT"
    match_id: str = ""
    counterparty: str = ""


class FinalExecutionEngine:
    """
    Final exchange-grade execution engine.

    BUY book priority:
        1. Highest limit price
        2. Real traders before liquidity provider
        3. Earliest tick
        4. Lowest latency

    SELL book priority:
        1. Lowest limit price
        2. Real traders before liquidity provider
        3. Earliest tick
        4. Lowest latency

    Why liquidity-provider orders are included:
        In real markets, market makers continuously post both bid and ask quotes.
        Without this, your matching efficiency remains low because agents rarely
        produce opposite-side orders at the same time.
    """

    def __init__(
        self,
        price_tolerance_bps: float = 8.0,
        liquidity_provider_enabled: bool = True,
        lp_quantity: int = 2,
        max_order_age: int = 25,
    ):
        self.bids: Dict[str, List[LimitOrder]] = {}
        self.asks: Dict[str, List[LimitOrder]] = {}

        self.direct_trades: List[BookTrade] = []
        self.fallback_orders: List[LimitOrder] = []

        self.total_submitted_orders = 0
        self.total_lp_orders = 0

        self.price_tolerance_bps = float(price_tolerance_bps)
        self.liquidity_provider_enabled = bool(liquidity_provider_enabled)
        self.lp_quantity = int(max(lp_quantity, 1))
        self.max_order_age = int(max_order_age)

    # ---------------------------------------------------------------------
    # Basic book operations
    # ---------------------------------------------------------------------
    def submit_order(self, order: LimitOrder) -> None:
        if order.quantity <= 0:
            return

        self.total_submitted_orders += 1

        self.bids.setdefault(order.asset, [])
        self.asks.setdefault(order.asset, [])

        if order.side == "BUY":
            self.bids[order.asset].append(order)
        elif order.side == "SELL":
            self.asks[order.asset].append(order)

    def submit_orders(self, orders: List[LimitOrder]) -> None:
        for order in orders:
            self.submit_order(order)

    def sort_books(self, asset: str) -> None:
        asset = str(asset).upper().strip()
        self.bids.setdefault(asset, [])
        self.asks.setdefault(asset, [])

        self.bids[asset].sort(
            key=lambda o: (
                -o.limit_price,
                1 if o.is_liquidity_provider else 0,
                o.tick,
                o.latency_ms,
                o.order_id,
            )
        )

        self.asks[asset].sort(
            key=lambda o: (
                o.limit_price,
                1 if o.is_liquidity_provider else 0,
                o.tick,
                o.latency_ms,
                o.order_id,
            )
        )

    # ---------------------------------------------------------------------
    # Liquidity provider
    # ---------------------------------------------------------------------
    def add_liquidity_provider_quotes(
        self,
        asset: str,
        mid_price: float,
        spread: float,
        tick: int,
    ) -> None:
        """
        Add realistic passive bid and ask orders.

        This is not fake fallback. It behaves like a standing market maker:
        - LP bid can be hit by sellers.
        - LP ask can be lifted by buyers.
        - LP improves direct matching without immediately forcing execution.
        """
        if not self.liquidity_provider_enabled:
            return

        asset = str(asset).upper().strip()
        mid_price = float(mid_price)
        spread = max(float(spread), 0.00001)

        half_spread = spread / 2.0

        # Slightly inside the public spread to increase crossing probability.
        lp_bid = mid_price - half_spread * random.uniform(0.25, 0.75)
        lp_ask = mid_price + half_spread * random.uniform(0.25, 0.75)

        lp_bid_order = LimitOrder(
            trader_id="LP-BID",
            trader_label="Synthetic Liquidity Provider",
            trader_emoji="🏛️",
            asset=asset,
            side="BUY",
            quantity=self.lp_quantity,
            limit_price=round(lp_bid, 5),
            tick=tick,
            latency_ms=random.uniform(0.2, 1.2),
            reason="Liquidity provider posts passive bid liquidity.",
            is_liquidity_provider=True,
        )

        lp_ask_order = LimitOrder(
            trader_id="LP-ASK",
            trader_label="Synthetic Liquidity Provider",
            trader_emoji="🏛️",
            asset=asset,
            side="SELL",
            quantity=self.lp_quantity,
            limit_price=round(lp_ask, 5),
            tick=tick,
            latency_ms=random.uniform(0.2, 1.2),
            reason="Liquidity provider posts passive ask liquidity.",
            is_liquidity_provider=True,
        )

        self.total_lp_orders += 2
        self.submit_orders([lp_bid_order, lp_ask_order])

    # ---------------------------------------------------------------------
    # Matching logic
    # ---------------------------------------------------------------------
    def _can_cross(
        self, best_bid: LimitOrder, best_ask: LimitOrder, mid_price: float
    ) -> bool:
        """
        Normal crossing:
            bid >= ask

        Tolerance crossing:
            allow near-crossing when prices are very close.
            This avoids unrealistic non-matches caused by tiny price rounding gaps.
        """
        if best_bid.limit_price >= best_ask.limit_price:
            return True

        tolerance = float(mid_price) * (self.price_tolerance_bps / 10_000.0)
        gap = best_ask.limit_price - best_bid.limit_price

        return gap <= tolerance

    def _execution_price(
        self,
        best_bid: LimitOrder,
        best_ask: LimitOrder,
        mid_price: float,
    ) -> float:
        """
        Price formation rule:
        - If bid crosses ask, use midpoint between bid and ask.
        - If tolerance-crossed, use midpoint but keep it close to market mid.
        """
        px = (best_bid.limit_price + best_ask.limit_price) / 2.0

        # Prevent broken prices from drifting too far from selected market mid.
        max_deviation = max(abs(float(mid_price)) * 0.02, 0.00001)
        px = min(max(px, mid_price - max_deviation), mid_price + max_deviation)

        return round(px, 5)

    def match_asset(
        self,
        asset: str,
        tick: int,
        mid_price: float,
        public_spread: float,
    ) -> List[BookTrade]:
        asset = str(asset).upper().strip()
        self.sort_books(asset)

        trades: List[BookTrade] = []
        public_spread = max(float(public_spread), 0.00001)
        mid_price = float(mid_price)

        while self.bids.get(asset) and self.asks.get(asset):
            best_bid = self.bids[asset][0]
            best_ask = self.asks[asset][0]

            if not self._can_cross(best_bid, best_ask, mid_price):
                break

            qty = min(best_bid.quantity, best_ask.quantity)

            execution_price = self._execution_price(best_bid, best_ask, mid_price)
            bid_price = round(best_bid.limit_price, 5)
            ask_price = round(best_ask.limit_price, 5)

            raw_spread = abs(best_bid.limit_price - best_ask.limit_price)
            trade_spread = round(raw_spread, 6)

            latency_ms = round((best_bid.latency_ms + best_ask.latency_ms) / 2.0, 2)
            transaction_cost = round(max(trade_spread / 4.0, public_spread / 8.0), 6)
            slippage = round(max(public_spread * 0.005, trade_spread * 0.005), 6)

            match_id = f"OB_{tick}_{asset}_{best_bid.order_id}_{best_ask.order_id}"

            if best_bid.is_liquidity_provider or best_ask.is_liquidity_provider:
                match_type = "DIRECT_LIQUIDITY_PROVIDER"
            else:
                match_type = "DIRECT_TRADER_TO_TRADER"

            trade = BookTrade(
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
                bid_price=bid_price,
                ask_price=ask_price,
                mid_price=round(mid_price, 5),
                spread=trade_spread,
                transaction_cost=transaction_cost,
                latency_ms=latency_ms,
                slippage=slippage,
                buyer_reason=best_bid.reason,
                seller_reason=best_ask.reason,
                match_type=match_type,
                match_id=match_id,
                counterparty=f"Order-book direct match: {best_bid.trader_id} ↔ {best_ask.trader_id}",
            )

            trades.append(trade)
            self.direct_trades.append(trade)

            best_bid.quantity -= qty
            best_ask.quantity -= qty

            if best_bid.quantity <= 0:
                self.bids[asset].pop(0)

            if best_ask.quantity <= 0:
                self.asks[asset].pop(0)

        return trades

    def match_all(
        self,
        tick: int,
        market_context: Dict[str, Dict[str, float]],
    ) -> List[BookTrade]:
        """
        market_context example:
        {
            "BTC": {"mid_price": 56000.0, "spread": 0.5},
            "ETH": {"mid_price": 1970.0, "spread": 0.02},
        }
        """
        all_assets = set(list(self.bids.keys()) + list(self.asks.keys()))
        all_trades: List[BookTrade] = []

        for asset in all_assets:
            ctx = market_context.get(asset, {})
            mid_price = float(ctx.get("mid_price", 1.0))
            spread = float(ctx.get("spread", 0.0001))

            all_trades.extend(
                self.match_asset(
                    asset=asset,
                    tick=tick,
                    mid_price=mid_price,
                    public_spread=spread,
                )
            )

        return all_trades

    # ---------------------------------------------------------------------
    # Unmatched orders and cleanup
    # ---------------------------------------------------------------------
    def get_unmatched_orders_for_tick(self, tick: int) -> List[LimitOrder]:
        unmatched: List[LimitOrder] = []

        for orders in self.bids.values():
            unmatched.extend(
                [
                    o
                    for o in orders
                    if o.tick == tick and o.quantity > 0 and not o.is_liquidity_provider
                ]
            )

        for orders in self.asks.values():
            unmatched.extend(
                [
                    o
                    for o in orders
                    if o.tick == tick and o.quantity > 0 and not o.is_liquidity_provider
                ]
            )

        return unmatched

    def remove_orders_for_tick(self, tick: int) -> None:
        """
        Remove only current-tick real trader orders after fallback.
        LP and older resting orders can remain until stale cancellation.
        """
        for asset in list(self.bids.keys()):
            self.bids[asset] = [
                o
                for o in self.bids[asset]
                if not (o.tick == tick and not o.is_liquidity_provider)
            ]

        for asset in list(self.asks.keys()):
            self.asks[asset] = [
                o
                for o in self.asks[asset]
                if not (o.tick == tick and not o.is_liquidity_provider)
            ]

    def cancel_stale_orders(self, current_tick: int) -> None:
        for asset in list(self.bids.keys()):
            self.bids[asset] = [
                o
                for o in self.bids[asset]
                if current_tick - o.tick <= self.max_order_age
            ]

        for asset in list(self.asks.keys()):
            self.asks[asset] = [
                o
                for o in self.asks[asset]
                if current_tick - o.tick <= self.max_order_age
            ]

    # ---------------------------------------------------------------------
    # Dashboard/report helpers
    # ---------------------------------------------------------------------
    def book_snapshot(self, asset: Optional[str] = None, depth: int = 5) -> Dict:
        if asset:
            assets = [str(asset).upper().strip()]
        else:
            assets = list(set(list(self.bids.keys()) + list(self.asks.keys())))

        snapshot = {}

        for a in assets:
            self.sort_books(a)

            snapshot[a] = {
                "bids": [
                    {
                        "order_id": o.order_id,
                        "trader": o.trader_id,
                        "price": round(o.limit_price, 5),
                        "qty": o.quantity,
                        "tick": o.tick,
                        "latency_ms": round(o.latency_ms, 2),
                        "lp": o.is_liquidity_provider,
                    }
                    for o in self.bids.get(a, [])[:depth]
                ],
                "asks": [
                    {
                        "order_id": o.order_id,
                        "trader": o.trader_id,
                        "price": round(o.limit_price, 5),
                        "qty": o.quantity,
                        "tick": o.tick,
                        "latency_ms": round(o.latency_ms, 2),
                        "lp": o.is_liquidity_provider,
                    }
                    for o in self.asks.get(a, [])[:depth]
                ],
            }

        return snapshot

    def metrics(self, total_trade_rows: int) -> Dict:
        direct_rows = len(self.direct_trades)
        fallback_rows = max(int(total_trade_rows) - direct_rows, 0)

        efficiency = 0.0
        if total_trade_rows > 0:
            efficiency = round((direct_rows / total_trade_rows) * 100.0, 2)

        trader_direct = sum(
            1 for t in self.direct_trades if t.match_type == "DIRECT_TRADER_TO_TRADER"
        )

        lp_direct = sum(
            1 for t in self.direct_trades if t.match_type == "DIRECT_LIQUIDITY_PROVIDER"
        )

        return {
            "submitted_orders": self.total_submitted_orders,
            "liquidity_provider_orders": self.total_lp_orders,
            "direct_match_rows": direct_rows,
            "direct_trader_to_trader_rows": trader_direct,
            "direct_liquidity_provider_rows": lp_direct,
            "market_fallback_rows": fallback_rows,
            "matching_efficiency_pct": efficiency,
        }


# Backward-compatible names
RealOrderBookEngine = FinalExecutionEngine