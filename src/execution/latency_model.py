"""
Latency Model (microsecond-scale, HFT-realistic)
==================================================
The previous version of this file modeled latency in whole milliseconds
with uniform random jitter - fine as a placeholder, but not representative
of how real trading infrastructure actually behaves. Two things were off:

  1. Units: real HFT operates in microseconds (us), often nanoseconds at
     the very top end (colocated, FPGA-based systems). Milliseconds are
     1000x too coarse to represent genuine tick-to-trade latency.
  2. Jitter shape: real network/processing latency is not uniformly
     distributed - it has a long right tail (usually modeled as
     lognormal), because most packets/messages arrive close to the
     typical time, but occasional congestion, garbage collection pauses,
     or OS scheduling delays produce rare, much larger spikes. A uniform
     distribution hides that entirely.

This version fixes both, and separates latency into the four components
that actually make up the round trip in real exchange connectivity:

  - tick_to_trade_us      : time from receiving a market data update to
                             your strategy deciding on an action
  - order_entry_us        : time for your order to reach the exchange
                             gateway over the network
  - exchange_processing_us: time for the exchange's own matching engine
                             to process and acknowledge the order
  - market_data_us        : time for the resulting price update to
                             propagate back out to all participants

Two built-in profiles represent the realistic ends of the spectrum:

  COLOCATED - server physically racked inside the exchange's data
      center, connected via direct cross-connect. This is what real
      institutional HFT firms pay for. Total round trip: roughly
      10-50 microseconds.

  RETAIL_API - a normal cloud server or home connection talking to an
      exchange over the public internet via a broker's API, which is
      the honest scope of most academic/retail trading simulators
      (including the previous millisecond-based version of this file).
      Total round trip: roughly 5,000-50,000 microseconds (5-50 ms).

Important honesty note: no software model can replicate the actual
infrastructure real HFT firms use - colocation, custom network hardware,
kernel-bypass networking, FPGAs. What this class *can* do is represent
the realistic timing envelope of each approach, which is genuinely useful
for understanding how latency affects fill quality and strategy design,
even though it can't reproduce the physical hardware itself.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class LatencyProfile(Enum):
    COLOCATED = "colocated"
    CROSS_CONNECT = "cross_connect"   # near-colo, e.g. same-metro cloud region
    RETAIL_API = "retail_api"


# (mean_us, sigma) per component, per profile. sigma is the lognormal
# shape parameter, not a standard deviation in microseconds - it controls
# how heavy the right tail is. Small sigma = tight and predictable
# (colocated, dedicated hardware). Larger sigma = more occasional spikes
# (shared infrastructure, public internet).
_PROFILE_PARAMS = {
    LatencyProfile.COLOCATED: {
        "tick_to_trade_us":       (3.0, 0.25),
        "order_entry_us":         (2.0, 0.20),
        "exchange_processing_us": (4.0, 0.20),
        "market_data_us":         (2.0, 0.20),
    },
    LatencyProfile.CROSS_CONNECT: {
        "tick_to_trade_us":       (150.0, 0.35),
        "order_entry_us":         (300.0, 0.35),
        "exchange_processing_us": (100.0, 0.30),
        "market_data_us":         (200.0, 0.35),
    },
    LatencyProfile.RETAIL_API: {
        "tick_to_trade_us":       (2000.0, 0.6),
        "order_entry_us":         (8000.0, 0.7),
        "exchange_processing_us": (1500.0, 0.5),
        "market_data_us":         (6000.0, 0.7),
    },
}


@dataclass
class LatencyBreakdown:
    tick_to_trade_us: float
    order_entry_us: float
    exchange_processing_us: float
    market_data_us: float

    @property
    def total_us(self) -> float:
        return (
            self.tick_to_trade_us
            + self.order_entry_us
            + self.exchange_processing_us
            + self.market_data_us
        )

    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0


class LatencyModel:
    """
    Drop-in replacement for the previous millisecond-based LatencyModel.
    Timestamps throughout this class are in microseconds - if the rest of
    your pipeline works in milliseconds or seconds, convert at the
    boundary (helpers below) rather than mixing units silently.

    Parameters
    ----------
    profile : LatencyProfile
        Which realistic latency envelope to simulate. Defaults to
        RETAIL_API, matching the honest scope of most of this project's
        components (they operate on historical/replayed data over
        standard infrastructure, not colocated hardware).
    seed : int
        For reproducible backtests. Set to None for non-deterministic
        jitter (useful once you're doing repeated Monte Carlo-style runs
        rather than a single reproducible backtest).
    """

    def __init__(self, profile: LatencyProfile = LatencyProfile.RETAIL_API,
                 seed: int = 42):
        self.profile = profile
        self._rng = np.random.RandomState(seed)
        self._params = _PROFILE_PARAMS[profile]

    def _sample_component(self, name: str) -> float:
        mean_us, sigma = self._params[name]
        # Lognormal parameterized so its mean matches mean_us, not its
        # median - this keeps the "typical" latency close to the
        # profile's stated value while still allowing the realistic
        # long right tail on occasional slow ticks.
        mu = np.log(mean_us) - (sigma ** 2) / 2
        return float(self._rng.lognormal(mean=mu, sigma=sigma))

    def sample_breakdown(self) -> LatencyBreakdown:
        return LatencyBreakdown(
            tick_to_trade_us=self._sample_component("tick_to_trade_us"),
            order_entry_us=self._sample_component("order_entry_us"),
            exchange_processing_us=self._sample_component("exchange_processing_us"),
            market_data_us=self._sample_component("market_data_us"),
        )

    def total_latency_us(self) -> float:
        return self.sample_breakdown().total_us

    def total_latency_ms(self) -> float:
        return self.total_latency_us() / 1000.0

    # ------------------------------------------------------------------
    # Backward-compatible interface (previous version returned whole
    # milliseconds as an int and exposed total_latency()/apply_latency())
    # ------------------------------------------------------------------
    def total_latency(self) -> int:
        """Kept for compatibility - returns whole milliseconds, rounded."""
        return round(self.total_latency_ms())

    def apply_latency(self, timestamp_us: float) -> float:
        """
        Applies a sampled latency to a microsecond timestamp. If your
        timestamps are in milliseconds instead, convert first:
            apply_latency(timestamp_ms * 1000) / 1000
        """
        return timestamp_us + self.total_latency_us()


if __name__ == "__main__":
    # Quick sanity check comparing profiles side by side.
    print(f"{'Profile':<15} {'Mean total (us)':>16} {'Mean total (ms)':>16}")
    for profile in LatencyProfile:
        model = LatencyModel(profile=profile, seed=1)
        samples = [model.total_latency_us() for _ in range(2000)]
        mean_us = sum(samples) / len(samples)
        print(f"{profile.value:<15} {mean_us:>16.2f} {mean_us/1000:>16.4f}")