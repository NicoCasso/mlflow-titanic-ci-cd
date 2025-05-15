import os
import warnings
import pickle
from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
import mlflow

import src.plot_funcs as pf
from src.utils import print_divider
from src.config import PROCESSED_TRAIN_PATH


def divide_by_sum(x: np.ndarray) -> np.ndarray:
    return x / x.sum()


def get_scores(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def log_plot(args: Any, plot_func: Callable, fp: str) -> None:
    if not isinstance(args, tuple):
        args = (args,)

    plot_func(*args, fp)
    mlflow.log_artifact(fp)
    os.remove(fp)
    print(f"Logged {fp}")


def train_model(
    X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], exp_path: str
) -> Tuple[str, str]:

    fold_params = params["fold"]
    model_params = params["model"]
    fit_params = params["fit"]

    try:
        mlflow.create_experiment(exp_path)
    except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException):
        print(f"The specified experiment ({exp_path}) already exists.")

    mlflow.set_experiment(exp_path)

    skf = StratifiedKFold(**fold_params)
    models: List[lgbm.LGBMClassifier] = []
    metrics: List[Dict[str, Any]] = []

    y_proba = np.zeros(len(X))
    y_pred = np.zeros(len(X))

    feature_importances_split = np.zeros(X.shape[1])
    feature_importances_gain = np.zeros(X.shape[1])

    scores: Dict[str, float] = defaultdict(int)

    with mlflow.start_run() as run:
        corr = pd.concat((X, y), axis=1).corr()
        log_plot(corr, pf.corr_matrix, "correlation_matrix.png")
        log_plot(y.value_counts(), pf.label_share, "label_share.png")

        for fold_no, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
            print_divider(f"Fold: {fold_no}")

            X_train, X_valid = X.iloc[idx_train, :], X.iloc[idx_valid, :]
            y_train, y_valid = y.iloc[idx_train], y.iloc[idx_valid]

            model = lgbm.LGBMClassifier(**model_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_names=["valid"],
                callbacks=[lgbm.early_stopping(stopping_rounds=10)],
            )

            metrics.append(
                {
                    "name": model.metric,
                    "values": model.evals_result_["valid"][model.metric],
                    "best_iteration": model.best_iteration_,
                }
            )
            models.append(model)

            feature_importances_split += (
                divide_by_sum(model.booster_.feature_importance("split")) / skf.n_splits
            )
            feature_importances_gain += (
                divide_by_sum(model.booster_.feature_importance("gain")) / skf.n_splits
            )

            y_valid_proba = model.predict_proba(
                X_valid, num_iteration=model.best_iteration_
            )[:, 1]
            y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            y_proba[idx_valid] = y_valid_proba
            y_pred[idx_valid] = y_valid_pred

            scores_valid = get_scores(y_valid, y_valid_pred)

            mlflow.log_metrics(
                {**scores_valid, "best_iteration": model.best_iteration_},
                step=fold_no,
            )

            print("\nScores")
            print(scores_valid)

            for k, v in scores_valid.items():
                scores[k] += v / skf.n_splits

        mlflow.log_params(
            {
                **fold_params,
                **model_params,
                **fit_params,
                "cv": skf.__class__.__name__,
                "model": model.__class__.__name__,
            }
        )

        print_divider("Saving plots")
        log_plot(scores, pf.scores, "scores.png")

        features = np.array(model.booster_.feature_name())
        log_plot(
            (features, feature_importances_split, "Feature Importance: split"),
            pf.feature_importance,
            "feature_importance_split.png",
        )
        log_plot(
            (features, feature_importances_gain, "Feature Importance: gain"),
            pf.feature_importance,
            "feature_importance_gain.png",
        )

        log_plot(metrics, pf.metric, "metric_history.png")

        cm = confusion_matrix(y, y_pred)
        log_plot(cm, pf.confusion_matrix, "confusion_matrix.png")

        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_pred)
        log_plot((fpr, tpr, roc_auc), pf.roc_curve, "roc_curve.png")

        pre, rec, _ = precision_recall_curve(y, y_proba)
        pr_auc = average_precision_score(y, y_pred)
        log_plot((pre, rec, pr_auc), pf.pr_curve, "pr_curve.png")

        models_path = "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)

        print(f"Logging model to MLflow: {models_path}")
        mlflow.log_artifact(models_path)
        mlflow.log_param("model_path", os.path.join(run.info.artifact_uri, models_path))
        os.remove(models_path)

    print(f"Logging model to MLflow: {run.info.experiment_id}, {run.info.run_uuid}")
    return run.info.experiment_id, run.info.run_uuid


def main() -> None:
    warnings.filterwarnings("ignore")

    print(os.listdir("data"))
    train = pd.read_pickle(PROCESSED_TRAIN_PATH)

    X = train.drop("Survived", axis=1)
    y = train["Survived"]

    SEED = 0
    params = {
        "model": {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 100000,
            "learning_rate": 0.05,
            "random_state": SEED,
            "n_jobs": -1,
        },
        "fit": {"early_stopping_rounds": 100, "verbose": 10},
        "fold": {"n_splits": 5, "shuffle": True, "random_state": SEED},
    }

    experiment_id, run_uuid = train_model(X, y, params, "titanic")
    print_divider("MLflow UI")
    print(
        f"Run URL: http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_uuid}"
    )
    os.system("mlflow ui")


if __name__ == "__main__":
    main()
