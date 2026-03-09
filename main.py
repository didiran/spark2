"""
Spark-Kafka ML Training Pipeline - Demo Application.

Demonstrates the full pipeline execution with a credit card fraud
detection use case. Runs entirely with pandas + scikit-learn +
concurrent.futures, requiring no Spark or Kafka infrastructure.

Usage:
    python main.py
    python main.py --samples 5000 --batches 3
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.evaluation.evaluator_standalone import StandalonePipelineEvaluator
from src.ingestion.data_validator_standalone import RuleSeverity, StandaloneDataValidator
from src.ingestion.kafka_consumer_simulator import KafkaConsumerSimulator
from src.monitoring.pipeline_monitor import PipelineMonitor
from src.orchestration.pipeline import MLPipelineOrchestrator
from src.processing.feature_engineering import FeatureEngineer
from src.processing.spark_processor import SparkProcessor
from src.store.feature_store import FeatureStore
from src.training.distributed_trainer_standalone import StandaloneDistributedTrainer
from src.training.model_selector_standalone import StandaloneModelSelector
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_STORE_PATH = "./demo_feature_store"
RESULTS_DIR = "./demo_results"


def run_fraud_detection_pipeline(
    num_samples: int = 5000,
    num_batches: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the full fraud detection ML pipeline.

    Demonstrates all pipeline stages: data ingestion, validation,
    processing, feature engineering, feature store management,
    distributed model training, evaluation, and model selection.

    Args:
        num_samples: Number of transaction samples to generate.
        num_batches: Number of streaming batches to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Pipeline results dictionary.
    """
    monitor = PipelineMonitor(pipeline_name="fraud-detection-pipeline")
    orchestrator = MLPipelineOrchestrator(
        pipeline_name="fraud-detection-pipeline",
        fail_fast=True,
    )

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Stage 1: Data Ingestion
    # ----------------------------------------------------------------
    def ingest_data(ctx: Dict[str, Any]) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("STAGE 1: DATA INGESTION (Kafka Consumer Simulation)")
        logger.info("=" * 70)

        simulator = KafkaConsumerSimulator(
            topic="fraud-transactions",
            batch_size=num_samples // num_batches,
            schema_name="fraud_detection",
            seed=seed,
            num_entities=500,
        )

        all_batches = []
        for batch in simulator.generate_stream(
            num_batches=num_batches,
            delay_seconds=0.0,
        ):
            all_batches.append(batch)

        df = pd.concat(all_batches, ignore_index=True)

        stats = simulator.get_statistics()
        logger.info(f"Ingestion complete | total_events={stats['total_events']}")

        monitor.record_data_volume("ingestion", len(df), len(df.columns))

        return df

    # ----------------------------------------------------------------
    # Stage 2: Data Validation
    # ----------------------------------------------------------------
    def validate_data(ctx: Dict[str, Any]) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("STAGE 2: DATA VALIDATION")
        logger.info("=" * 70)

        df = ctx["result_ingest"]

        validator = StandaloneDataValidator()
        validator.expect_row_count(min_count=100, severity=RuleSeverity.CRITICAL)
        validator.expect_column_not_null("transaction_id", severity=RuleSeverity.CRITICAL)
        validator.expect_column_not_null("amount", severity=RuleSeverity.CRITICAL)
        validator.expect_column_not_null("is_fraud", severity=RuleSeverity.CRITICAL)
        validator.expect_column_values_in_range("amount", min_value=0.0, max_value=50000.0)
        validator.expect_column_values_in_set(
            "card_type", ["visa", "mastercard", "amex", "discover"],
        )
        validator.detect_outliers_iqr("amount", max_outlier_fraction=0.15)

        report = validator.validate(df, fail_on_critical=False)

        logger.info(
            f"Validation | score={report.quality_score}% "
            f"| passed={report.passed_rules}/{report.total_rules} "
            f"| valid={report.is_valid}"
        )

        return df

    # ----------------------------------------------------------------
    # Stage 3: Data Processing
    # ----------------------------------------------------------------
    def process_data(ctx: Dict[str, Any]) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("STAGE 3: DATA PROCESSING (Spark Processor Simulation)")
        logger.info("=" * 70)

        df = ctx["result_validate"]
        processor = SparkProcessor()

        df = processor.clean_data(
            df,
            drop_duplicates=True,
            subset_for_duplicates=["transaction_id"],
        )

        df = processor.fill_nulls(df, strategy="median")

        df = processor.clip_outliers(
            df,
            columns=["amount", "distance_from_home", "time_since_last_transaction"],
            method="iqr",
            iqr_multiplier=3.0,
        )

        profile = processor.get_data_profile(df)
        logger.info(f"Data profile | rows={profile['shape']['rows']} | cols={profile['shape']['columns']}")

        monitor.record_data_volume("processing", len(df), len(df.columns))

        return df

    # ----------------------------------------------------------------
    # Stage 4: Feature Engineering
    # ----------------------------------------------------------------
    def engineer_features(ctx: Dict[str, Any]) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("STAGE 4: FEATURE ENGINEERING")
        logger.info("=" * 70)

        df = ctx["result_process"]
        engineer = FeatureEngineer()

        df = engineer.add_temporal_features(
            df,
            timestamp_column="timestamp",
            features=["hour", "day_of_week", "is_weekend"],
        )

        df = engineer.encode_categorical(
            df,
            columns=["merchant_category", "card_type", "transaction_type"],
            method="label",
        )

        df = engineer.add_interaction_features(
            df,
            column_pairs=[
                ("amount", "is_international"),
                ("amount", "is_online"),
                ("distance_from_home", "time_since_last_transaction"),
            ],
        )

        df = engineer.add_ratio_features(
            df,
            numerator_denominator_pairs=[
                ("amount", "avg_transaction_amount_7d"),
                ("amount", "daily_transaction_count"),
            ],
        )

        lineage = engineer.get_lineage()
        logger.info(f"Feature engineering complete | transformations={len(lineage)}")

        monitor.record_data_volume("feature_engineering", len(df), len(df.columns))

        return df

    # ----------------------------------------------------------------
    # Stage 5: Feature Store
    # ----------------------------------------------------------------
    def store_features(ctx: Dict[str, Any]) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("STAGE 5: FEATURE STORE")
        logger.info("=" * 70)

        df = ctx["result_features"]
        store = FeatureStore(base_path=FEATURE_STORE_PATH)

        store.register_feature_group(
            name="fraud_features",
            description="Credit card fraud detection features",
            entity_key="user_id",
            timestamp_column="timestamp",
            tags={"domain": "fraud", "version": "1.0"},
        )

        version_info = store.ingest_features(
            name="fraud_features",
            df=df,
            description="Initial feature set for fraud detection",
        )

        logger.info(
            f"Features stored | version={version_info['version']} "
            f"| rows={version_info['row_count']}"
        )

        groups = store.list_feature_groups()
        logger.info(f"Feature store | groups={len(groups)}")

        return df

    # ----------------------------------------------------------------
    # Stage 6: Model Training
    # ----------------------------------------------------------------
    def train_models(ctx: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("STAGE 6: DISTRIBUTED MODEL TRAINING")
        logger.info("=" * 70)

        df = ctx["result_store"]

        exclude_cols = [
            "transaction_id", "timestamp", "user_id", "merchant_id",
            "merchant_category", "card_type", "transaction_type",
        ]
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and c != "is_fraud"
            and df[c].dtype in (np.float64, np.int64, np.float32, np.int32, float, int)
        ]

        trainer = StandaloneDistributedTrainer(
            algorithms=["random_forest", "gradient_boosting", "logistic_regression"],
            target_column="is_fraud",
            primary_metric="f1",
            cv_folds=3,
            max_workers=3,
            seed=seed,
            task_type="classification",
        )

        X_train, y_train, X_val, y_val, X_test, y_test = trainer.split_data(
            df[feature_cols + ["is_fraud"]],
            target_column="is_fraud",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        results = trainer.train(
            X_train,
            y_train,
            feature_columns=feature_cols,
            use_grid_search=False,
        )

        for r in results:
            monitor.record_model_metric(r.algorithm, "f1", r.metrics.get("f1", 0))
            monitor.record_model_metric(r.algorithm, "accuracy", r.metrics.get("accuracy", 0))

        return {
            "training_results": results,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_cols,
        }

    # ----------------------------------------------------------------
    # Stage 7: Model Evaluation
    # ----------------------------------------------------------------
    def evaluate_models(ctx: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("STAGE 7: MODEL EVALUATION")
        logger.info("=" * 70)

        train_data = ctx["result_train"]
        results = train_data["training_results"]
        X_test = train_data["X_test"]
        y_test = train_data["y_test"]
        X_train = train_data["X_train"]
        y_train = train_data["y_train"]

        evaluator = StandalonePipelineEvaluator(task_type="classification")
        evaluator.add_quality_gate("min_accuracy", "accuracy", 0.70)
        evaluator.add_quality_gate("min_f1", "f1", 0.50)
        evaluator.add_quality_gate("min_precision", "precision", 0.50)

        evaluations = []
        for result in results:
            eval_result = evaluator.evaluate(
                model=result.model,
                X_test=X_test,
                y_test=y_test,
                model_name=result.algorithm,
                cv_folds=3,
                X_train=X_train,
                y_train=y_train,
            )
            evaluations.append(eval_result)

            logger.info(
                f"Evaluation | model={result.algorithm} "
                f"| gates_passed={eval_result.quality_gates_passed} "
                f"| metrics={eval_result.metrics}"
            )

        return {
            "evaluations": evaluations,
            "training_results": results,
            "X_val": train_data["X_val"],
            "y_val": train_data["y_val"],
        }

    # ----------------------------------------------------------------
    # Stage 8: Model Selection
    # ----------------------------------------------------------------
    def select_model(ctx: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("STAGE 8: MODEL SELECTION")
        logger.info("=" * 70)

        eval_data = ctx["result_evaluate"]
        results = eval_data["training_results"]
        X_val = eval_data["X_val"]
        y_val = eval_data["y_val"]

        selector = StandaloneModelSelector(
            primary_metric="f1",
            metric_threshold=0.50,
            higher_is_better=True,
        )

        report = selector.select_best(
            training_results=results,
            X_val=X_val,
            y_val=y_val,
        )

        summary = selector.generate_report_summary(report)

        logger.info(
            f"Model selected | winner={summary['winner_algorithm']} "
            f"| f1={summary['winner_metric']:.4f} "
            f"| threshold_met={summary['threshold_met']}"
        )

        summary_path = Path(RESULTS_DIR) / "selection_report.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return {
            "report": report,
            "summary": summary,
        }

    # ----------------------------------------------------------------
    # Register all stages
    # ----------------------------------------------------------------
    orchestrator.add_stage("ingest", ingest_data, "Kafka consumer simulation")
    orchestrator.add_stage("validate", validate_data, "Data validation", ["ingest"])
    orchestrator.add_stage("process", process_data, "Data processing", ["validate"])
    orchestrator.add_stage("features", engineer_features, "Feature engineering", ["process"])
    orchestrator.add_stage("store", store_features, "Feature store ingestion", ["features"])
    orchestrator.add_stage("train", train_models, "Distributed model training", ["store"])
    orchestrator.add_stage("evaluate", evaluate_models, "Model evaluation", ["train"])
    orchestrator.add_stage("select", select_model, "Model selection", ["evaluate"])

    # ----------------------------------------------------------------
    # Execute pipeline
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("SPARK-KAFKA ML TRAINING PIPELINE")
    logger.info("Credit Card Fraud Detection Demo")
    logger.info("=" * 70)

    pipeline_result = orchestrator.run()
    monitor.record_pipeline_run(pipeline_result)

    # ----------------------------------------------------------------
    # Generate reports
    # ----------------------------------------------------------------
    health = monitor.get_health_report()
    metrics_summary = monitor.get_metrics_summary()

    health_path = Path(RESULTS_DIR) / "health_report.json"
    with open(health_path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Status: {pipeline_result['status']}")
    logger.info(f"  Duration: {pipeline_result['duration_seconds']:.2f}s")
    logger.info(f"  Stages completed: {pipeline_result['completed_stages']}/{pipeline_result['total_stages']}")
    logger.info(f"  Pipeline health: {health['status']}")
    logger.info(f"  Success rate: {health['success_rate_pct']}%")
    logger.info(f"  Results saved to: {RESULTS_DIR}/")
    logger.info("=" * 70)

    return {
        "pipeline_result": pipeline_result,
        "health_report": health,
        "metrics_summary": metrics_summary,
    }


def main():
    """Entry point for the fraud detection pipeline demo."""
    parser = argparse.ArgumentParser(
        description="Spark-Kafka ML Training Pipeline - Fraud Detection Demo",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of transaction samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Number of streaming batches to simulate (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    try:
        result = run_fraud_detection_pipeline(
            num_samples=args.samples,
            num_batches=args.batches,
            seed=args.seed,
        )

        if result["pipeline_result"]["status"] == "completed":
            logger.info("Pipeline completed successfully.")
            sys.exit(0)
        else:
            logger.error("Pipeline completed with failures.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
