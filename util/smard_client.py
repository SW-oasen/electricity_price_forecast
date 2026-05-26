"""
smard_client.py — generic client for the SMARD chart_data REST API.

No region or filter defaults — all domain values must be supplied by the caller
(typically from src/config.py for this project).

Usage example:
    from util.smard_client import SmardClient
    from config import SMARD_FILTER_NETZLAST, SMARD_REGION, SMARD_BASE, SMARD_HEADERS

    client = SmardClient(
        filter_id=SMARD_FILTER_NETZLAST,
        region=SMARD_REGION,
        base_url=SMARD_BASE,
        headers=SMARD_HEADERS,
    )
    df = client.fetch('2025-01-01', '2025-01-31')
    # columns: ['time', 'load_MWh']  — rename in the caller as needed

API reference: documents/smard_api.md
"""

from __future__ import annotations

import time

import pandas as pd
import requests


class SmardClient:
    """
    Fetch time-series data from the unofficial SMARD chart_data API.

    The SMARD API serves data in weekly buckets.  This client:
    1. Fetches the index of available weekly timestamps for the given filter.
    2. Identifies which buckets overlap the requested date range.
    3. Downloads each bucket and clips the result to the exact range.

    Parameters
    ----------
    filter_id : int
        SMARD filter ID (e.g. 410 = Netzlast actual, 411 = Netzlast forecast).
    region : str
        SMARD region code (e.g. 'DE', 'AT', '50Hertz').
    base_url : str
        Root URL of the SMARD chart_data endpoint.
    headers : dict | None
        HTTP headers to send with every request (User-Agent etc.).
    resolution : str
        Time resolution: 'hour' (default) or 'quarterhour'.
    sleep : float
        Seconds to sleep between bucket requests (be polite to the server).
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        filter_id: int,
        region: str,
        base_url: str,
        headers: dict | None = None,
        resolution: str = 'hour',
        sleep: float = 0.3,
        timeout: int = 15,
    ) -> None:
        self.filter_id  = filter_id
        self.region     = region
        self.base_url   = base_url.rstrip('/')
        self.headers    = headers or {}
        self.resolution = resolution
        self.sleep      = sleep
        self.timeout    = timeout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_index(self) -> list[int]:
        """Return the list of available weekly bucket timestamps (Unix ms)."""
        url = (
            f"{self.base_url}/{self.filter_id}/{self.region}"
            f"/index_{self.resolution}.json"
        )
        r = requests.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["timestamps"]

    def _fetch_week(self, timestamp_ms: int) -> list:
        """Fetch the raw series [[ts_ms, value], ...] for one weekly bucket."""
        url = (
            f"{self.base_url}/{self.filter_id}/{self.region}"
            f"/{self.filter_id}_{self.region}_{self.resolution}_{timestamp_ms}.json"
        )
        r = requests.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("series", [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for the given date range and return a tidy DataFrame.

        Parameters
        ----------
        start_date : str  'YYYY-MM-DD' — inclusive start (Europe/Berlin local time).
        end_date   : str  'YYYY-MM-DD' — inclusive end.

        Returns
        -------
        pd.DataFrame with columns:
            time      — tz-aware datetime64[s] (Europe/Berlin)
            load_MWh  — numeric value from the API (rename in the caller as needed)
        """
        start_dt = pd.to_datetime(start_date).tz_localize("Europe/Berlin")
        end_dt   = pd.to_datetime(end_date).tz_localize("Europe/Berlin")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms   = int(end_dt.timestamp() * 1000) + 86_400_000 - 1   # include full end day

        all_timestamps = self._get_index()

        week_ms  = 7 * 24 * 3600 * 1000
        relevant = [
            ts for ts in all_timestamps
            if ts <= end_ms and ts + week_ms >= start_ms
        ]

        if not relevant:
            return pd.DataFrame(columns=["time", "load_MWh"])

        rows: list = []
        for ts in relevant:
            rows.extend(self._fetch_week(ts))
            time.sleep(self.sleep)

        df = pd.DataFrame(rows, columns=["ts_ms", "load_MWh"])
        df = df.dropna(subset=["load_MWh"])
        df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)]

        df["time"] = (
            pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            .dt.tz_convert("Europe/Berlin")
            .dt.as_unit("s")
        )
        df = (
            df[["time", "load_MWh"]]
            .sort_values("time")
            .reset_index(drop=True)
        )
        return df
