"""
telemetry/spot_collector.py
───────────────────────────
Polls AWS EC2 and CloudWatch APIs every N seconds to collect spot price
signals, interruption risk metrics, and instance health data.
Writes structured FeatureVectors to DynamoDB.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import boto3
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SpotMarketSnapshot:
    """Raw spot market data for one instance type in one AZ."""
    instance_type: str
    az: str
    timestamp: datetime
    price: float              # $/hr current
    price_5m_avg: float       # 5-minute rolling avg
    price_30m_avg: float      # 30-minute rolling avg
    price_60m_avg: float      # 60-minute rolling avg
    pct_above_baseline: float # % above 24h baseline
    interruption_rate: str    # low | medium | high
    interruption_rate_num: float  # 0.05 | 0.15 | 0.35


@dataclass
class InstanceHealthSnapshot:
    """CloudWatch metrics for a running instance."""
    instance_id: str
    instance_type: str
    az: str
    timestamp: datetime
    cpu_utilization: float    # 0-100%
    network_in_mbps: float
    network_out_mbps: float
    disk_read_ops: float
    disk_write_ops: float


INTERRUPTION_RATE_MAP = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.35,
}


class SpotCollector:
    """
    Continuously collects spot price and interruption signals from AWS.

    Usage:
        collector = SpotCollector(config)
        await collector.start()          # runs forever
        snapshot = collector.get_latest("c6i.xlarge", "us-east-1a")
    """

    def __init__(self, config: dict):
        self.config = config
        self.region = config["aws"]["region"]
        self.poll_interval = config["aws"]["polling_interval_sec"]

        self.ec2 = boto3.client("ec2", region_name=self.region)
        self.cw = boto3.client("cloudwatch", region_name=self.region)

        # instance_type+az → list of (timestamp, price) for rolling avgs
        self._price_history: Dict[str, List[tuple]] = {}
        # Latest snapshot per instance_type+az
        self._latest: Dict[str, SpotMarketSnapshot] = {}

        # Load instance catalog
        with open("config/instances.yaml") as f:
            catalog_raw = yaml.safe_load(f)
        self._catalog = {i["type"]: i for i in catalog_raw["instances"]}

    # ─────────────────────────── Public API ──────────────────────────────────

    def get_latest(self, instance_type: str, az: str) -> Optional[SpotMarketSnapshot]:
        key = f"{instance_type}::{az}"
        return self._latest.get(key)

    def get_all_latest(self) -> List[SpotMarketSnapshot]:
        return list(self._latest.values())

    # ────────────────────────── Main Poll Loop ────────────────────────────────

    async def start(self):
        """Run the collection loop forever."""
        logger.info(f"SpotCollector starting — polling every {self.poll_interval}s")
        while True:
            try:
                await self._collect_all()
            except Exception as e:
                logger.error(f"SpotCollector error: {e}", exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def _collect_all(self):
        loop = asyncio.get_event_loop()
        # Run blocking boto3 call in thread pool
        await loop.run_in_executor(None, self._fetch_spot_prices)
        logger.debug(f"Collected spot prices for {len(self._latest)} type/AZ pairs")

    def _fetch_spot_prices(self):
        """Fetch current spot prices for all catalog instance types."""
        instance_types = list(self._catalog.keys())

        response = self.ec2.describe_spot_price_history(
            InstanceTypes=instance_types,
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(timezone.utc) - timedelta(hours=2),
        )

        for entry in response.get("SpotPriceHistory", []):
            itype = entry["InstanceType"]
            az = entry["AvailabilityZone"]
            price = float(entry["SpotPrice"])
            ts = entry["Timestamp"]

            if itype not in self._catalog:
                continue

            key = f"{itype}::{az}"
            if key not in self._price_history:
                self._price_history[key] = []

            self._price_history[key].append((ts, price))
            # Keep only last 2 hours of data
            cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
            self._price_history[key] = [
                (t, p) for (t, p) in self._price_history[key] if t > cutoff
            ]

            # Compute rolling averages
            prices = [p for (_, p) in self._price_history[key]]
            now = datetime.now(timezone.utc)

            def avg_window(minutes: int) -> float:
                cutoff_t = now - timedelta(minutes=minutes)
                window = [p for (t, p) in self._price_history[key] if t > cutoff_t]
                return float(np.mean(window)) if window else price

            p5 = avg_window(5)
            p30 = avg_window(30)
            p60 = avg_window(60)

            # Baseline = 24h avg fetched from longer history (approximated by 60m here)
            baseline = p60 if p60 > 0 else price
            pct_above = ((price - baseline) / baseline) * 100 if baseline > 0 else 0.0

            intr_rate_str = self._catalog[itype].get("interruption_rate", "medium")
            intr_rate_num = INTERRUPTION_RATE_MAP.get(intr_rate_str, 0.15)

            snapshot = SpotMarketSnapshot(
                instance_type=itype,
                az=az,
                timestamp=now,
                price=price,
                price_5m_avg=p5,
                price_30m_avg=p30,
                price_60m_avg=p60,
                pct_above_baseline=pct_above,
                interruption_rate=intr_rate_str,
                interruption_rate_num=intr_rate_num,
            )
            self._latest[key] = snapshot

    # ────────────────────────── Termination Warning ───────────────────────────

    @staticmethod
    def check_termination_notice(metadata_url: str = "http://169.254.169.254") -> bool:
        """
        Poll the EC2 instance metadata endpoint for a spot termination notice.
        Returns True if a 2-minute warning has been issued.
        Must be called FROM the spot instance itself.
        """
        import urllib.request
        try:
            url = f"{metadata_url}/latest/meta-data/spot/termination-time"
            with urllib.request.urlopen(url, timeout=1) as resp:
                # A 200 response means termination notice is active
                return resp.status == 200
        except Exception:
            return False

    # ────────────────────────── Historical Fetch ──────────────────────────────

    def fetch_historical_prices(
        self,
        instance_type: str,
        az: str,
        days: int = 90,
    ) -> List[tuple]:
        """
        Fetch up to `days` days of spot price history for training data.
        Returns list of (datetime, price_float) tuples, oldest first.
        """
        all_records = []
        start = datetime.now(timezone.utc) - timedelta(days=days)
        next_token = None

        while True:
            kwargs = dict(
                InstanceTypes=[instance_type],
                AvailabilityZone=az,
                ProductDescriptions=["Linux/UNIX"],
                StartTime=start,
            )
            if next_token:
                kwargs["NextToken"] = next_token

            response = self.ec2.describe_spot_price_history(**kwargs)
            for entry in response.get("SpotPriceHistory", []):
                all_records.append((entry["Timestamp"], float(entry["SpotPrice"])))

            next_token = response.get("NextToken")
            if not next_token:
                break

        # Sort chronologically (AWS returns newest first)
        all_records.sort(key=lambda x: x[0])
        return all_records
