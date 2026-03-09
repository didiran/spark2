"""
Kafka consumer simulator for the ML Training Pipeline.

Generates realistic streaming data events that simulate Kafka topic
consumption. Supports configurable data generation for credit card
fraud detection, IoT sensor data, and financial transaction scenarios.

This module enables local development and testing without requiring
a running Kafka cluster.
"""

import hashlib
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class KafkaConsumerSimulator:
    """
    Simulated Kafka consumer that generates realistic streaming data.

    Produces configurable batches of synthetic events mimicking
    Kafka topic consumption patterns with controlled throughput,
    data distributions, and domain-specific schemas.

    Attributes:
        topic: Simulated Kafka topic name.
        batch_size: Number of events per batch.
        schema: Data schema definition for event generation.
    """

    FRAUD_SCHEMA = {
        "transaction_id": "uuid",
        "timestamp": "timestamp",
        "user_id": "entity_id",
        "amount": {"type": "float", "min": 0.50, "max": 15000.0, "distribution": "lognormal"},
        "merchant_category": {
            "type": "categorical",
            "values": [
                "grocery", "gas_station", "restaurant", "online_retail",
                "travel", "entertainment", "healthcare", "utilities",
                "education", "electronics",
            ],
        },
        "merchant_id": {"type": "entity", "prefix": "MERCH", "count": 500},
        "card_type": {
            "type": "categorical",
            "values": ["visa", "mastercard", "amex", "discover"],
        },
        "transaction_type": {
            "type": "categorical",
            "values": ["purchase", "refund", "cash_advance", "balance_transfer"],
            "weights": [0.85, 0.08, 0.04, 0.03],
        },
        "is_international": {"type": "boolean", "true_probability": 0.12},
        "is_online": {"type": "boolean", "true_probability": 0.45},
        "distance_from_home": {"type": "float", "min": 0.0, "max": 5000.0, "distribution": "exponential"},
        "time_since_last_transaction": {"type": "float", "min": 0.0, "max": 720.0, "distribution": "exponential"},
        "daily_transaction_count": {"type": "int", "min": 1, "max": 50},
        "avg_transaction_amount_7d": {"type": "float", "min": 10.0, "max": 5000.0},
        "is_fraud": {"type": "label", "positive_rate": 0.02},
    }

    IOT_SCHEMA = {
        "reading_id": "uuid",
        "timestamp": "timestamp",
        "sensor_id": {"type": "entity", "prefix": "SENSOR", "count": 200},
        "temperature": {"type": "float", "min": -20.0, "max": 80.0, "distribution": "normal", "mean": 25.0, "std": 10.0},
        "humidity": {"type": "float", "min": 0.0, "max": 100.0, "distribution": "normal", "mean": 55.0, "std": 15.0},
        "pressure": {"type": "float", "min": 950.0, "max": 1050.0, "distribution": "normal", "mean": 1013.0, "std": 10.0},
        "vibration": {"type": "float", "min": 0.0, "max": 100.0, "distribution": "exponential"},
        "power_consumption": {"type": "float", "min": 0.0, "max": 500.0},
        "is_anomaly": {"type": "label", "positive_rate": 0.05},
    }

    FINANCIAL_SCHEMA = {
        "trade_id": "uuid",
        "timestamp": "timestamp",
        "account_id": {"type": "entity", "prefix": "ACC", "count": 1000},
        "instrument": {
            "type": "categorical",
            "values": ["equity", "bond", "option", "future", "forex", "commodity"],
        },
        "side": {"type": "categorical", "values": ["buy", "sell"], "weights": [0.52, 0.48]},
        "quantity": {"type": "int", "min": 1, "max": 10000},
        "price": {"type": "float", "min": 0.01, "max": 5000.0, "distribution": "lognormal"},
        "market_volatility": {"type": "float", "min": 0.0, "max": 1.0},
        "spread": {"type": "float", "min": 0.001, "max": 0.1},
        "is_suspicious": {"type": "label", "positive_rate": 0.01},
    }

    SCHEMAS = {
        "fraud_detection": FRAUD_SCHEMA,
        "iot_monitoring": IOT_SCHEMA,
        "financial_risk": FINANCIAL_SCHEMA,
    }

    def __init__(
        self,
        topic: str = "ml-training-data",
        batch_size: int = 1000,
        schema_name: str = "fraud_detection",
        custom_schema: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        num_entities: int = 1000,
    ):
        """
        Initialize the Kafka consumer simulator.

        Args:
            topic: Simulated Kafka topic name.
            batch_size: Number of events per generated batch.
            schema_name: Predefined schema to use.
            custom_schema: Optional custom schema definition.
            seed: Random seed for reproducibility.
            num_entities: Number of unique entity IDs to generate.
        """
        self.topic = topic
        self.batch_size = batch_size
        self.schema_name = schema_name
        self.schema = custom_schema or self.SCHEMAS.get(schema_name, self.FRAUD_SCHEMA)
        self.num_entities = num_entities
        self._event_count = 0
        self._batch_count = 0
        self._start_time = datetime.utcnow()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._entity_ids = [f"user_{i:06d}" for i in range(num_entities)]

        logger.info(
            f"KafkaConsumerSimulator initialized | topic={topic} "
            f"| batch_size={batch_size} | schema={schema_name} "
            f"| entities={num_entities}"
        )

    def generate_batch(
        self,
        batch_size: Optional[int] = None,
        start_time: Optional[datetime] = None,
        time_span_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Generate a batch of simulated streaming events.

        Args:
            batch_size: Override default batch size.
            start_time: Starting timestamp for the batch.
            time_span_minutes: Time range for event timestamps.

        Returns:
            DataFrame containing the generated batch of events.
        """
        size = batch_size or self.batch_size
        base_time = start_time or datetime.utcnow() - timedelta(minutes=time_span_minutes)

        records = []
        for _ in range(size):
            record = self._generate_record(base_time, time_span_minutes)
            records.append(record)

        df = pd.DataFrame(records)
        self._event_count += size
        self._batch_count += 1

        logger.info(
            f"Batch generated | topic={self.topic} | size={size} "
            f"| total_events={self._event_count} | batch_num={self._batch_count}"
        )
        return df

    def generate_stream(
        self,
        num_batches: int = 10,
        delay_seconds: float = 0.0,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generate a continuous stream of event batches.

        Yields batches with optional delay to simulate real-time
        consumption patterns.

        Args:
            num_batches: Number of batches to generate.
            delay_seconds: Delay between batches in seconds.
            batch_size: Override default batch size.

        Yields:
            DataFrame for each generated batch.
        """
        for i in range(num_batches):
            batch = self.generate_batch(batch_size=batch_size)
            yield batch

            if delay_seconds > 0 and i < num_batches - 1:
                time.sleep(delay_seconds)

    def generate_with_drift(
        self,
        num_batches: int = 10,
        drift_factor: float = 0.05,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generate batches with simulated data drift.

        Gradually shifts numeric distributions to simulate concept
        drift in the data stream.

        Args:
            num_batches: Number of batches to produce.
            drift_factor: Rate of distribution shift per batch.
            batch_size: Override default batch size.

        Yields:
            DataFrame with progressively drifted distributions.
        """
        for batch_idx in range(num_batches):
            batch = self.generate_batch(batch_size=batch_size)

            drift_amount = drift_factor * (batch_idx + 1)
            numeric_cols = batch.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ("is_fraud", "is_anomaly", "is_suspicious", "label"):
                    noise = np.random.normal(
                        drift_amount * batch[col].std(),
                        batch[col].std() * drift_factor,
                        size=len(batch),
                    )
                    batch[col] = batch[col] + noise

            logger.info(
                f"Drift batch generated | batch={batch_idx + 1}/{num_batches} "
                f"| drift_amount={drift_amount:.4f}"
            )
            yield batch

    def generate_imbalanced_batch(
        self,
        positive_rate: float = 0.02,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate a batch with a controlled class imbalance ratio.

        Args:
            positive_rate: Fraction of positive (fraud/anomaly) samples.
            batch_size: Override default batch size.

        Returns:
            DataFrame with the specified class distribution.
        """
        size = batch_size or self.batch_size
        n_positive = max(1, int(size * positive_rate))
        n_negative = size - n_positive

        positive_records = []
        negative_records = []

        label_field = self._find_label_field()

        for _ in range(size * 3):
            record = self._generate_record(datetime.utcnow() - timedelta(hours=1), 60)
            if label_field and record.get(label_field, 0) == 1:
                if len(positive_records) < n_positive:
                    positive_records.append(record)
            else:
                if len(negative_records) < n_negative:
                    negative_records.append(record)

            if len(positive_records) >= n_positive and len(negative_records) >= n_negative:
                break

        while len(positive_records) < n_positive:
            record = self._generate_record(datetime.utcnow() - timedelta(hours=1), 60)
            if label_field:
                record[label_field] = 1
            positive_records.append(record)

        while len(negative_records) < n_negative:
            record = self._generate_record(datetime.utcnow() - timedelta(hours=1), 60)
            if label_field:
                record[label_field] = 0
            negative_records.append(record)

        all_records = positive_records + negative_records
        random.shuffle(all_records)

        df = pd.DataFrame(all_records)
        self._event_count += len(df)
        self._batch_count += 1

        if label_field:
            actual_rate = df[label_field].mean()
            logger.info(
                f"Imbalanced batch | positive_rate={actual_rate:.4f} "
                f"| target_rate={positive_rate} | size={len(df)}"
            )

        return df

    def _generate_record(
        self,
        base_time: datetime,
        time_span_minutes: int,
    ) -> Dict[str, Any]:
        """Generate a single event record from the schema definition."""
        record: Dict[str, Any] = {}
        offset = random.uniform(0, time_span_minutes * 60)
        event_time = base_time + timedelta(seconds=offset)

        for field_name, field_spec in self.schema.items():
            if isinstance(field_spec, str):
                if field_spec == "uuid":
                    record[field_name] = str(uuid.uuid4())
                elif field_spec == "timestamp":
                    record[field_name] = event_time
                elif field_spec == "entity_id":
                    record[field_name] = random.choice(self._entity_ids)
            elif isinstance(field_spec, dict):
                record[field_name] = self._generate_field_value(field_spec)
            else:
                record[field_name] = None

        return record

    def _generate_field_value(self, spec: Dict[str, Any]) -> Any:
        """Generate a single field value from its specification."""
        field_type = spec.get("type", "float")

        if field_type == "float":
            return self._generate_float(spec)
        elif field_type == "int":
            return random.randint(spec.get("min", 0), spec.get("max", 100))
        elif field_type == "categorical":
            values = spec.get("values", [])
            weights = spec.get("weights")
            if weights:
                return random.choices(values, weights=weights, k=1)[0]
            return random.choice(values)
        elif field_type == "boolean":
            return int(random.random() < spec.get("true_probability", 0.5))
        elif field_type == "entity":
            prefix = spec.get("prefix", "ENT")
            count = spec.get("count", 100)
            return f"{prefix}_{random.randint(1, count):06d}"
        elif field_type == "label":
            positive_rate = spec.get("positive_rate", 0.5)
            return int(random.random() < positive_rate)
        else:
            return None

    def _generate_float(self, spec: Dict[str, Any]) -> float:
        """Generate a float value with the specified distribution."""
        distribution = spec.get("distribution", "uniform")
        min_val = spec.get("min", 0.0)
        max_val = spec.get("max", 1.0)

        if distribution == "uniform":
            value = random.uniform(min_val, max_val)
        elif distribution == "normal":
            mean = spec.get("mean", (min_val + max_val) / 2)
            std = spec.get("std", (max_val - min_val) / 6)
            value = np.random.normal(mean, std)
            value = max(min_val, min(max_val, value))
        elif distribution == "lognormal":
            mu = np.log((min_val + max_val) / 4)
            sigma = 1.0
            value = np.random.lognormal(mu, sigma)
            value = max(min_val, min(max_val, value))
        elif distribution == "exponential":
            scale = (max_val - min_val) / 3
            value = min_val + np.random.exponential(scale)
            value = max(min_val, min(max_val, value))
        else:
            value = random.uniform(min_val, max_val)

        return round(float(value), 4)

    def _find_label_field(self) -> Optional[str]:
        """Find the label field in the schema definition."""
        for field_name, field_spec in self.schema.items():
            if isinstance(field_spec, dict) and field_spec.get("type") == "label":
                return field_name
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Return consumption statistics for the simulator."""
        elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        throughput = self._event_count / max(elapsed, 1)

        return {
            "topic": self.topic,
            "schema": self.schema_name,
            "total_events": self._event_count,
            "total_batches": self._batch_count,
            "elapsed_seconds": round(elapsed, 2),
            "events_per_second": round(throughput, 2),
            "batch_size": self.batch_size,
        }

    def reset(self) -> None:
        """Reset event counters and timing."""
        self._event_count = 0
        self._batch_count = 0
        self._start_time = datetime.utcnow()
        logger.info("KafkaConsumerSimulator reset")
