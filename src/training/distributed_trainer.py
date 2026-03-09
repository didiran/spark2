"""
Distributed model training engine using Spark ML.

Implements multi-algorithm training with cross-validation,
hyperparameter grid search, and pipeline-based preprocessing
for scalable ML workloads on Spark clusters.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.regression import (
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame

from src.config.settings import TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Container for a single model training result."""
    algorithm: str
    model: PipelineModel
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    training_time_seconds: float
    cross_validation_scores: List[float] = field(default_factory=list)


class DistributedTrainer:
    """
    Distributed ML model trainer using Spark ML pipelines.

    Supports multiple classification and regression algorithms
    with built-in cross-validation and hyperparameter grid search.

    Attributes:
        config: Training configuration settings.
        results: List of training results for each algorithm run.
    """

    CLASSIFICATION_ALGORITHMS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosted_trees": GBTClassifier,
        "logistic_regression": LogisticRegression,
    }

    REGRESSION_ALGORITHMS = {
        "random_forest_regressor": RandomForestRegressor,
        "gradient_boosted_trees_regressor": GBTRegressor,
        "linear_regression": LinearRegression,
    }

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results: List[TrainingResult] = []
        logger.info(
            f"DistributedTrainer initialized | algorithms={config.algorithms} "
            f"| target={config.target_column}"
        )

    def train(
        self,
        train_df: DataFrame,
        feature_col: str = "features",
        label_col: Optional[str] = None,
        task_type: str = "classification",
        preprocessing_stages: Optional[List] = None,
    ) -> List[TrainingResult]:
        """
        Train multiple models with cross-validation and grid search.

        Args:
            train_df: Training DataFrame with feature and label columns.
            feature_col: Name of the assembled feature vector column.
            label_col: Name of the label column.
            task_type: "classification" or "regression".
            preprocessing_stages: Optional ML pipeline preprocessing stages.

        Returns:
            List of TrainingResult for each trained algorithm.
        """
        label = label_col or self.config.target_column
        self.results = []

        algorithms = self._resolve_algorithms(task_type)
        evaluator = self._create_evaluator(task_type, label)

        for algo_name in self.config.algorithms:
            if algo_name not in algorithms:
                logger.warning(f"Unknown algorithm: {algo_name}, skipping")
                continue

            logger.info(f"Training {algo_name} | task={task_type}")
            start_time = time.time()

            try:
                result = self._train_single(
                    train_df=train_df,
                    algo_name=algo_name,
                    algo_class=algorithms[algo_name],
                    feature_col=feature_col,
                    label_col=label,
                    evaluator=evaluator,
                    preprocessing_stages=preprocessing_stages,
                )
                elapsed = time.time() - start_time
                result.training_time_seconds = elapsed
                self.results.append(result)

                logger.info(
                    f"Training complete | algo={algo_name} "
                    f"| {self.config.primary_metric}={result.metrics.get(self.config.primary_metric, 'N/A')} "
                    f"| time={elapsed:.1f}s"
                )
            except Exception as e:
                logger.error(f"Training failed | algo={algo_name} | error={e}")

        return self.results

    def _resolve_algorithms(self, task_type: str) -> Dict:
        """Map task type to available algorithm classes."""
        if task_type == "classification":
            return self.CLASSIFICATION_ALGORITHMS
        elif task_type == "regression":
            return self.REGRESSION_ALGORITHMS
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _create_evaluator(self, task_type: str, label_col: str):
        """Create the appropriate Spark ML evaluator."""
        if task_type == "classification":
            return MulticlassClassificationEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
                metricName=self.config.primary_metric,
            )
        else:
            from pyspark.ml.evaluation import RegressionEvaluator
            return RegressionEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
                metricName="rmse",
            )

    def _train_single(
        self,
        train_df: DataFrame,
        algo_name: str,
        algo_class,
        feature_col: str,
        label_col: str,
        evaluator,
        preprocessing_stages: Optional[List] = None,
    ) -> TrainingResult:
        """
        Train a single algorithm with cross-validation.

        Builds a Spark ML Pipeline, constructs a hyperparameter grid,
        runs k-fold cross-validation, and extracts the best model.
        """
        estimator = algo_class(
            featuresCol=feature_col,
            labelCol=label_col,
            seed=self.config.seed,
        )

        if hasattr(estimator, "setMaxIter"):
            estimator.setMaxIter(self.config.max_iterations)

        stages = list(preprocessing_stages or [])
        stages.append(estimator)
        pipeline = Pipeline(stages=stages)

        param_grid = self._build_param_grid(estimator, algo_name)

        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=self.config.cross_validation_folds,
            seed=self.config.seed,
            parallelism=2,
        )

        cv_model = cross_validator.fit(train_df)

        best_model = cv_model.bestModel
        cv_scores = list(cv_model.avgMetrics)
        best_params = self._extract_best_params(cv_model)

        metrics = self._compute_metrics(best_model, train_df, label_col)

        return TrainingResult(
            algorithm=algo_name,
            model=best_model,
            metrics=metrics,
            best_params=best_params,
            training_time_seconds=0.0,
            cross_validation_scores=cv_scores,
        )

    def _build_param_grid(self, estimator, algo_name: str) -> List:
        """
        Build a ParamGrid from the training configuration.

        Maps hyperparameter names to Spark ML Param objects for
        the given estimator.
        """
        grid_config = self.config.hyperparameter_grids.get(algo_name, {})

        if not grid_config:
            return ParamGridBuilder().build()

        builder = ParamGridBuilder()

        param_map = {p.name: p for p in estimator.params}

        for param_name, values in grid_config.items():
            if param_name in param_map:
                builder = builder.addGrid(param_map[param_name], values)
            else:
                logger.warning(
                    f"Parameter '{param_name}' not found for {algo_name}, skipping"
                )

        grid = builder.build()
        logger.info(
            f"Param grid built | algo={algo_name} | combinations={len(grid)}"
        )
        return grid

    def _extract_best_params(self, cv_model) -> Dict[str, Any]:
        """Extract the best hyperparameters from the cross-validation model."""
        best_params = {}
        try:
            best_model = cv_model.bestModel
            last_stage = best_model.stages[-1]
            params = last_stage.extractParamMap()
            for param, value in params.items():
                best_params[param.name] = value
        except Exception as e:
            logger.warning(f"Could not extract best params: {e}")
        return best_params

    def _compute_metrics(
        self,
        model: PipelineModel,
        df: DataFrame,
        label_col: str,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics on the given DataFrame.

        Evaluates accuracy, precision, recall, F1, and AUC for
        classification tasks.
        """
        predictions = model.transform(df)
        metrics = {}

        try:
            mc_evaluator = MulticlassClassificationEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
            )
            for metric_name in ["f1", "accuracy", "weightedPrecision", "weightedRecall"]:
                mc_evaluator.setMetricName(metric_name)
                metrics[metric_name] = round(mc_evaluator.evaluate(predictions), 6)

            bc_evaluator = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="rawPrediction",
            )
            bc_evaluator.setMetricName("areaUnderROC")
            metrics["areaUnderROC"] = round(bc_evaluator.evaluate(predictions), 6)
            bc_evaluator.setMetricName("areaUnderPR")
            metrics["areaUnderPR"] = round(bc_evaluator.evaluate(predictions), 6)

        except Exception as e:
            logger.warning(f"Metric computation partial failure: {e}")

        return metrics

    def split_data(
        self,
        df: DataFrame,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Full DataFrame to split.

        Returns:
            Tuple of (train_df, validation_df, test_df).
        """
        ratios = [
            self.config.train_ratio,
            self.config.validation_ratio,
            self.config.test_ratio,
        ]
        splits = df.randomSplit(ratios, seed=self.config.seed)
        logger.info(
            f"Data split | train={splits[0].count()} "
            f"| validation={splits[1].count()} | test={splits[2].count()}"
        )
        return splits[0], splits[1], splits[2]
