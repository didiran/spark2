"""
Distributed model training engine for the ML Training Pipeline.

Implements multi-algorithm training using concurrent.futures for
parallel execution, with cross-validation, hyperparameter grid
search, and comprehensive metrics computation using scikit-learn.
"""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Container for a single model training result."""
    algorithm: str
    model: Any
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    training_time_seconds: float
    cross_validation_scores: List[float] = field(default_factory=list)
    feature_importances: Optional[Dict[str, float]] = None


class StandaloneDistributedTrainer:
    """
    Distributed ML model trainer using concurrent.futures.

    Supports multiple classification and regression algorithms
    with built-in cross-validation, hyperparameter grid search,
    and parallel training via thread or process pools.

    Attributes:
        algorithms: List of algorithm names to train.
        target_column: Name of the target column.
        primary_metric: Primary metric for model comparison.
        cv_folds: Number of cross-validation folds.
        max_workers: Number of parallel training workers.
        results: List of completed training results.
    """

    CLASSIFICATION_ALGORITHMS = {
        "random_forest": {
            "class": RandomForestClassifier,
            "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10],
            },
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "default_params": {"n_estimators": 100, "random_state": 42},
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "logistic_regression": {
            "class": LogisticRegression,
            "default_params": {"max_iter": 1000, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        },
    }

    REGRESSION_ALGORITHMS = {
        "random_forest_regressor": {
            "class": RandomForestRegressor,
            "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
            },
        },
        "gradient_boosting_regressor": {
            "class": GradientBoostingRegressor,
            "default_params": {"n_estimators": 100, "random_state": 42},
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "ridge_regression": {
            "class": Ridge,
            "default_params": {"random_state": 42},
            "param_grid": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
            },
        },
    }

    def __init__(
        self,
        algorithms: Optional[List[str]] = None,
        target_column: str = "label",
        primary_metric: str = "f1",
        cv_folds: int = 5,
        max_workers: int = 3,
        seed: int = 42,
        task_type: str = "classification",
    ):
        """
        Initialize the distributed trainer.

        Args:
            algorithms: List of algorithm names to train.
            target_column: Name of the target column.
            primary_metric: Primary metric for comparison.
            cv_folds: Number of cross-validation folds.
            max_workers: Number of parallel workers.
            seed: Random seed for reproducibility.
            task_type: Task type (classification or regression).
        """
        self.target_column = target_column
        self.primary_metric = primary_metric
        self.cv_folds = cv_folds
        self.max_workers = max_workers
        self.seed = seed
        self.task_type = task_type
        self.results: List[TrainingResult] = []

        if task_type == "classification":
            available = self.CLASSIFICATION_ALGORITHMS
            default_algos = ["random_forest", "gradient_boosting", "logistic_regression"]
        else:
            available = self.REGRESSION_ALGORITHMS
            default_algos = ["random_forest_regressor", "gradient_boosting_regressor", "ridge_regression"]

        self.algorithms = algorithms or default_algos
        self._algo_registry = available

        logger.info(
            f"StandaloneDistributedTrainer initialized | algorithms={self.algorithms} "
            f"| task={task_type} | cv_folds={cv_folds} | workers={max_workers}"
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: Optional[List[str]] = None,
        use_grid_search: bool = True,
        custom_param_grids: Optional[Dict[str, Dict[str, List]]] = None,
    ) -> List[TrainingResult]:
        """
        Train multiple models in parallel with cross-validation.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training target Series.
            feature_columns: Specific feature columns to use.
            use_grid_search: Whether to perform grid search.
            custom_param_grids: Custom hyperparameter grids.

        Returns:
            List of TrainingResult for each trained algorithm.
        """
        if feature_columns:
            X_train = X_train[feature_columns]

        X_train = X_train.select_dtypes(include=[np.number]).copy()
        X_train = X_train.fillna(0)

        self.results = []

        logger.info(
            f"Starting parallel training | algorithms={len(self.algorithms)} "
            f"| features={X_train.shape[1]} | samples={X_train.shape[0]}"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for algo_name in self.algorithms:
                if algo_name not in self._algo_registry:
                    logger.warning(f"Unknown algorithm: {algo_name}, skipping")
                    continue

                param_grid = None
                if use_grid_search:
                    if custom_param_grids and algo_name in custom_param_grids:
                        param_grid = custom_param_grids[algo_name]
                    else:
                        param_grid = self._algo_registry[algo_name].get("param_grid")

                future = executor.submit(
                    self._train_single,
                    algo_name,
                    X_train.copy(),
                    y_train.copy(),
                    param_grid,
                    list(X_train.columns),
                )
                futures[future] = algo_name

            for future in as_completed(futures):
                algo_name = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    logger.info(
                        f"Training complete | algo={algo_name} "
                        f"| {self.primary_metric}={result.metrics.get(self.primary_metric, 'N/A'):.4f} "
                        f"| time={result.training_time_seconds:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"Training failed | algo={algo_name} | error={e}")

        logger.info(
            f"All training complete | successful={len(self.results)}/{len(self.algorithms)}"
        )
        return self.results

    def _train_single(
        self,
        algo_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]],
        feature_names: List[str],
    ) -> TrainingResult:
        """
        Train a single algorithm with optional grid search.

        Args:
            algo_name: Algorithm identifier.
            X_train: Training features.
            y_train: Training labels.
            param_grid: Hyperparameter grid for grid search.
            feature_names: Names of feature columns.

        Returns:
            TrainingResult with model, metrics, and metadata.
        """
        start_time = time.time()
        algo_config = self._algo_registry[algo_name]
        model_class = algo_config["class"]
        default_params = algo_config["default_params"]

        logger.info(f"Training {algo_name} | grid_search={param_grid is not None}")

        if param_grid:
            scoring = self._get_scoring_metric()
            grid_search = GridSearchCV(
                model_class(**default_params),
                param_grid,
                cv=min(self.cv_folds, len(y_train)),
                scoring=scoring,
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_scores = list(grid_search.cv_results_["mean_test_score"])
        else:
            best_model = model_class(**default_params)
            best_model.fit(X_train, y_train)
            best_params = default_params

            cv_scores_array = cross_val_score(
                model_class(**default_params),
                X_train,
                y_train,
                cv=min(self.cv_folds, len(y_train)),
                scoring=self._get_scoring_metric(),
            )
            cv_scores = cv_scores_array.tolist()

        y_pred = best_model.predict(X_train)
        metrics = self._compute_metrics(y_train, y_pred, best_model, X_train)

        feature_importances = None
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            feature_importances = {
                name: round(float(imp), 6)
                for name, imp in zip(feature_names, importances)
            }
            feature_importances = dict(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            )

        elapsed = time.time() - start_time

        return TrainingResult(
            algorithm=algo_name,
            model=best_model,
            metrics=metrics,
            best_params=best_params,
            training_time_seconds=round(elapsed, 2),
            cross_validation_scores=[round(s, 6) for s in cv_scores],
            feature_importances=feature_importances,
        )

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics: Dict[str, float] = {}

        if self.task_type == "classification":
            metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)
            metrics["f1"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
            metrics["precision"] = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
            metrics["recall"] = round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6)

            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)
                    if y_proba.shape[1] == 2:
                        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 6)
                    else:
                        metrics["auc_roc"] = round(
                            float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
                            6,
                        )
            except Exception:
                pass

        else:
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6)
            metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 6)
            metrics["r2"] = round(float(r2_score(y_true, y_pred)), 6)
            metrics["mse"] = round(float(mean_squared_error(y_true, y_pred)), 6)

        return metrics

    def _get_scoring_metric(self) -> str:
        """Map primary metric to scikit-learn scoring string."""
        metric_map = {
            "f1": "f1_weighted",
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "auc_roc": "roc_auc",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "mse": "neg_mean_squared_error",
        }
        return metric_map.get(self.primary_metric, "f1_weighted")

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Full DataFrame with features and target.
            target_column: Target column name.
            train_ratio: Training set fraction.
            val_ratio: Validation set fraction.
            test_ratio: Test set fraction.
            stratify: Whether to use stratified splitting.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        from sklearn.model_selection import train_test_split

        target = target_column or self.target_column
        y = df[target]
        X = df.drop(columns=[target])

        stratify_col = y if (stratify and self.task_type == "classification") else None

        test_val_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_val_ratio, random_state=self.seed, stratify=stratify_col,
        )

        if stratify and self.task_type == "classification":
            stratify_temp = y_temp
        else:
            stratify_temp = None

        relative_test = test_ratio / test_val_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test, random_state=self.seed, stratify=stratify_temp,
        )

        logger.info(
            f"Data split | train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
