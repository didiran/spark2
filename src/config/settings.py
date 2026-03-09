"""
Configuration management for the Spark-Kafka ML Training Pipeline.

Pydantic-based settings classes for all pipeline components with
YAML file loading and environment variable override support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SparkConfig:
    """Apache Spark session and runtime configuration."""

    app_name: str = "spark-kafka-ml-pipeline"
    master: str = "local[*]"
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 2
    num_executors: int = 2
    shuffle_partitions: int = 200
    default_parallelism: int = 8
    adaptive_enabled: bool = True
    delta_extensions: bool = True
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    extra_configs: Dict[str, str] = field(default_factory=dict)

    def to_spark_conf(self) -> Dict[str, str]:
        """Convert to Spark configuration dictionary."""
        conf = {
            "spark.app.name": self.app_name,
            "spark.master": self.master,
            "spark.driver.memory": self.driver_memory,
            "spark.executor.memory": self.executor_memory,
            "spark.executor.cores": str(self.executor_cores),
            "spark.executor.instances": str(self.num_executors),
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.default.parallelism": str(self.default_parallelism),
            "spark.sql.adaptive.enabled": str(self.adaptive_enabled).lower(),
            "spark.serializer": self.serializer,
        }
        if self.delta_extensions:
            conf.update({
                "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
                "spark.sql.catalog.spark_catalog": (
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog"
                ),
            })
        conf.update(self.extra_configs)
        return conf


@dataclass
class KafkaConfig:
    """Apache Kafka connection and consumer configuration."""

    bootstrap_servers: str = "localhost:9092"
    topics: List[str] = field(default_factory=lambda: ["ml-training-data"])
    group_id: str = "spark-ml-pipeline-consumer"
    auto_offset_reset: str = "earliest"
    max_offsets_per_trigger: int = 10000
    schema_registry_url: str = "http://localhost:8081"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    watermark_delay: str = "10 seconds"
    checkpoint_location: str = "/tmp/spark-checkpoints/kafka"
    starting_offsets: str = "earliest"
    fail_on_data_loss: bool = False
    extra_options: Dict[str, str] = field(default_factory=dict)

    def to_spark_options(self) -> Dict[str, str]:
        """Convert to Spark Structured Streaming readStream options."""
        options = {
            "kafka.bootstrap.servers": self.bootstrap_servers,
            "subscribe": ",".join(self.topics),
            "startingOffsets": self.starting_offsets,
            "maxOffsetsPerTrigger": str(self.max_offsets_per_trigger),
            "failOnDataLoss": str(self.fail_on_data_loss).lower(),
            "kafka.security.protocol": self.security_protocol,
        }
        if self.sasl_mechanism:
            options["kafka.sasl.mechanism"] = self.sasl_mechanism
        if self.sasl_username and self.sasl_password:
            jaas_config = (
                f'org.apache.kafka.common.security.plain.PlainLoginModule required '
                f'username="{self.sasl_username}" password="{self.sasl_password}";'
            )
            options["kafka.sasl.jaas.config"] = jaas_config
        options.update(self.extra_options)
        return options


@dataclass
class StorageConfig:
    """Data storage configuration for Delta Lake, PostgreSQL, and MongoDB."""

    delta_base_path: str = "/data/delta"
    feature_store_path: str = "/data/delta/feature_store"
    raw_data_path: str = "/data/delta/raw"
    processed_data_path: str = "/data/delta/processed"
    model_artifacts_path: str = "/data/models"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "ml_pipeline"
    postgres_user: str = "pipeline_user"
    postgres_password: str = "pipeline_pass"

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "ml_pipeline"
    mongodb_collection: str = "pipeline_metadata"

    @property
    def postgres_jdbc_url(self) -> str:
        return (
            f"jdbc:postgresql://{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )

    @property
    def postgres_connection_props(self) -> Dict[str, str]:
        return {
            "user": self.postgres_user,
            "password": self.postgres_password,
            "driver": "org.postgresql.Driver",
        }


@dataclass
class MLflowConfig:
    """MLflow tracking server and model registry configuration."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "spark-ml-training"
    artifact_location: str = "/data/mlflow/artifacts"
    registry_uri: Optional[str] = None
    auto_log_enabled: bool = True
    log_models: bool = True
    log_input_examples: bool = True
    tags: Dict[str, str] = field(default_factory=lambda: {
        "project": "spark-kafka-ml-pipeline",
        "team": "ml-engineering",
    })


@dataclass
class TrainingConfig:
    """ML model training configuration."""

    algorithms: List[str] = field(default_factory=lambda: [
        "random_forest",
        "gradient_boosted_trees",
        "logistic_regression",
    ])
    target_column: str = "label"
    feature_columns: List[str] = field(default_factory=list)
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    cross_validation_folds: int = 5
    primary_metric: str = "f1"
    metric_threshold: float = 0.75
    max_iterations: int = 100
    early_stopping_rounds: int = 10
    hyperparameter_grids: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "random_forest": {
            "numTrees": [50, 100, 200],
            "maxDepth": [5, 10, 15],
            "minInstancesPerNode": [1, 5, 10],
        },
        "gradient_boosted_trees": {
            "maxIter": [50, 100],
            "maxDepth": [3, 5, 8],
            "stepSize": [0.01, 0.1, 0.3],
        },
        "logistic_regression": {
            "maxIter": [50, 100, 200],
            "regParam": [0.0, 0.01, 0.1],
            "elasticNetParam": [0.0, 0.5, 1.0],
        },
    })


@dataclass
class PipelineSettings:
    """Top-level pipeline configuration aggregating all component configs."""

    spark: SparkConfig = field(default_factory=SparkConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    pipeline_name: str = "spark-kafka-ml-training-pipeline"
    pipeline_version: str = "1.0.0"
    environment: str = "development"
    retry_max_attempts: int = 3
    retry_delay_seconds: int = 30
    batch_processing_interval: str = "1 hour"
    stream_trigger_interval: str = "30 seconds"

    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineSettings":
        """
        Load pipeline settings from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            PipelineSettings instance populated from file values.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        spark_cfg = SparkConfig(**raw.get("spark", {}))
        kafka_cfg = KafkaConfig(**raw.get("kafka", {}))
        storage_cfg = StorageConfig(**raw.get("storage", {}))
        mlflow_cfg = MLflowConfig(**raw.get("mlflow", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))

        pipeline_raw = raw.get("pipeline", {})
        return cls(
            spark=spark_cfg,
            kafka=kafka_cfg,
            storage=storage_cfg,
            mlflow=mlflow_cfg,
            training=training_cfg,
            pipeline_name=pipeline_raw.get("name", cls.pipeline_name),
            pipeline_version=pipeline_raw.get("version", cls.pipeline_version),
            environment=pipeline_raw.get("environment", cls.environment),
            retry_max_attempts=pipeline_raw.get("retry_max_attempts", cls.retry_max_attempts),
            retry_delay_seconds=pipeline_raw.get("retry_delay_seconds", cls.retry_delay_seconds),
            batch_processing_interval=pipeline_raw.get(
                "batch_processing_interval", cls.batch_processing_interval
            ),
            stream_trigger_interval=pipeline_raw.get(
                "stream_trigger_interval", cls.stream_trigger_interval
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize settings to a flat dictionary for logging."""
        import dataclasses
        result = {}
        for fld in dataclasses.fields(self):
            value = getattr(self, fld.name)
            if dataclasses.is_dataclass(value):
                for inner_fld in dataclasses.fields(value):
                    result[f"{fld.name}.{inner_fld.name}"] = getattr(value, inner_fld.name)
            else:
                result[fld.name] = value
        return result
