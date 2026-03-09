"""
Shared fixtures for the Spark-Kafka ML Training Pipeline test suite.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_fraud_df():
    """Create a sample fraud-detection DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:05d}" for i in range(n)],
            "user_id": [f"user_{np.random.randint(0, 50):03d}" for _ in range(n)],
            "merchant_id": [f"merch_{np.random.randint(0, 30):03d}" for _ in range(n)],
            "amount": np.random.lognormal(mean=3.5, sigma=1.2, size=n).round(2),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "merchant_category": np.random.choice(
                ["retail", "food", "travel", "entertainment", "online"], size=n
            ),
            "card_type": np.random.choice(["visa", "mastercard", "amex", "discover"], size=n),
            "transaction_type": np.random.choice(["purchase", "refund", "transfer"], size=n, p=[0.8, 0.1, 0.1]),
            "is_international": np.random.choice([0, 1], size=n, p=[0.85, 0.15]).astype(int),
            "is_online": np.random.choice([0, 1], size=n, p=[0.6, 0.4]).astype(int),
            "distance_from_home": np.random.exponential(scale=20.0, size=n).round(2),
            "time_since_last_transaction": np.random.exponential(scale=120.0, size=n).round(2),
            "avg_transaction_amount_7d": np.random.lognormal(mean=3.0, sigma=0.8, size=n).round(2),
            "daily_transaction_count": np.random.poisson(lam=5, size=n).astype(float),
            "is_fraud": np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
        }
    )


@pytest.fixture
def numeric_df():
    """Create a simple numeric DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(10, 2, n),
            "feature_b": np.random.normal(50, 10, n),
            "feature_c": np.random.uniform(0, 1, n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for file-based tests."""
    return tmp_path
