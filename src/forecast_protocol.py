"""Rules for the leakage-safe electricity-price walk-forward backtest.

The target is the DE/LU day-ahead price for delivery day ``D``.  A forecast is
made on ``D-1`` before the day-ahead auction closes.  Keeping these rules in a
small, dependency-free module lets feature construction and tests use exactly
the same information boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd


BERLIN_TZ = "Europe/Berlin"


@dataclass(frozen=True)
class PriceWalkForwardProtocol:
    """Point-in-time rules for the frozen-model price evaluation.

    ``training_end_exclusive`` is deliberately also the first walk-forward
    delivery day.  We use the conservative physical-data rule until source
    publication delays are captured as historical snapshots: at D-1 before the
    auction, only completed actual days through D-2 may be used.
    """

    training_end_exclusive: str = "2025-10-01"
    timezone: str = BERLIN_TZ
    decision_time: time = time(11, 30)

    def target_start(self, target_date: str | pd.Timestamp) -> pd.Timestamp:
        """Return the local midnight at the start of delivery day D."""
        target = pd.Timestamp(target_date)
        if target.tz is None:
            return target.tz_localize(self.timezone).normalize()
        return target.tz_convert(self.timezone).normalize()

    def as_of(self, target_date: str | pd.Timestamp) -> pd.Timestamp:
        """Return the D-1 decision timestamp, before the day-ahead auction."""
        target = self.target_start(target_date)
        return (target - pd.Timedelta(days=1)).replace(
            hour=self.decision_time.hour,
            minute=self.decision_time.minute,
        )

    def price_known_until_exclusive(self, target_date: str | pd.Timestamp) -> pd.Timestamp:
        """All day-ahead prices strictly before D are known at the decision time.

        In particular, the complete price curve for D-1 was auctioned on D-2.
        """
        return self.target_start(target_date)

    def physical_actual_known_until_exclusive(self, target_date: str | pd.Timestamp) -> pd.Timestamp:
        """Conservative actual-data boundary: completed days through D-2 only."""
        return self.target_start(target_date) - pd.Timedelta(days=1)

    def is_evaluation_target(self, target_date: str | pd.Timestamp) -> bool:
        """Whether D belongs to the frozen-model walk-forward evaluation."""
        cutoff = self.target_start(self.training_end_exclusive)
        return self.target_start(target_date) >= cutoff


PRICE_WALK_FORWARD_PROTOCOL = PriceWalkForwardProtocol()
