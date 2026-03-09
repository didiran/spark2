"""
Tests for KafkaConsumerSimulator.

Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import pytest

from src.ingestion.kafka_consumer_simulator import KafkaConsumerSimulator


class TestKafkaConsumerSimulator:
    """Tests for the Kafka consumer simulator."""

    def test_generate_batch_returns_dataframe(self):
        simulator = KafkaConsumerSimulator(
            topic="test-topic",
            batch_size=50,
            schema_name="fraud_detection",
            seed=42,
        )
        batch = simulator.generate_batch()
        assert isinstance(batch, pd.DataFrame)
        assert len(batch) == 50

    def test_generate_batch_has_expected_columns(self):
        simulator = KafkaConsumerSimulator(
            topic="test-topic",
            batch_size=30,
            schema_name="fraud_detection",
            seed=42,
        )
        batch = simulator.generate_batch()
        required_columns = {"transaction_id", "amount", "is_fraud", "timestamp"}
        assert required_columns.issubset(set(batch.columns))

    def test_generate_stream_yields_correct_batches(self):
        simulator = KafkaConsumerSimulator(
            topic="test-topic",
            batch_size=25,
            schema_name="fraud_detection",
            seed=42,
        )
        batches = list(simulator.generate_stream(num_batches=4, delay_seconds=0.0))
        assert len(batches) == 4
        for batch in batches:
            assert len(batch) == 25

    def test_statistics_tracking(self):
        simulator = KafkaConsumerSimulator(
            topic="stats-topic",
            batch_size=20,
            schema_name="fraud_detection",
            seed=42,
        )
        list(simulator.generate_stream(num_batches=3, delay_seconds=0.0))
        stats = simulator.get_statistics()
        assert stats["total_events"] == 60
        assert stats["total_batches"] == 3

    def test_iot_schema(self):
        simulator = KafkaConsumerSimulator(
            topic="iot-topic",
            batch_size=10,
            schema_name="iot_sensor",
            seed=42,
        )
        batch = simulator.generate_batch()
        assert "sensor_id" in batch.columns or "device_id" in batch.columns
        assert len(batch) == 10

    def test_deterministic_with_seed(self):
        sim1 = KafkaConsumerSimulator(topic="t", batch_size=20, schema_name="fraud_detection", seed=123)
        sim2 = KafkaConsumerSimulator(topic="t", batch_size=20, schema_name="fraud_detection", seed=123)
        b1 = sim1.generate_batch()
        b2 = sim2.generate_batch()
        pd.testing.assert_frame_equal(b1, b2)
