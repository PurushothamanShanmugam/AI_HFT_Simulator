from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass
class Order:
    order_id: int
    agent_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float | None = None
    timestamp: int = 0