import os
import re
import numpy as np
import pandas as pd
import mlflow
from typing import List, Any, Optional
from src.utils import load_pickle
from src.config import DATA_DIR, PROCESSED_TEST_PATH


class EnsembleModel:
    def __init__(self, models: List[Any]) -> None:
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(X), self.models[0].n_classes_))
        for model in self.models:
            proba += model.predict_proba(X, num_iteration=model.best_iteration_)
        return proba / len(self.models)


def get_experiment_id_folder_name() -> str:
    experiment_id_folder_name = "0"
    mlruns_path = os.path.join(os.getcwd(), "mlruns")
    if os.path.exists(mlruns_path):
        for root, dirs, files in os.walk(mlruns_path):
            for dir_name in dirs:
                if dir_name in ["0", ".trash"]:
                    continue
                experiment_id_folder_name = dir_name
                break
            if experiment_id_folder_name != "0":
                break
    else:
        print(f"❌ Le dossier {mlruns_path} n'existe pas.")
    return experiment_id_folder_name


def main() -> None:
    EXPERIMENT_ID = get_experiment_id_folder_name()

    # mlflow.search_runs attend une liste de str (ou None)
    runs: Optional[pd.DataFrame] = mlflow.search_runs(
        experiment_ids=[EXPERIMENT_ID],
        order_by=["metrics.accuracy DESC", "attribute.start_time DESC"],
        max_results=10,
    )

    if runs is None or runs.empty:
        print("Aucun run trouvé pour cet experiment_id.")
        return

    print(runs)
    print(runs.columns)
    print(runs.head())

    raw_path: str = runs.loc[0, "params.model_path"]
    match = re.search(r"mlruns[\\/](.*)", raw_path)
    if not match:
        raise ValueError(f"Chemin invalide : {raw_path}")

    relative_path: str = match.group(1).replace("\\", "/")
    models_path: str = os.path.join("/app/mlruns", relative_path)
    models_path = os.path.normpath(models_path)
    print(f"Chemin vers le modèle : {models_path}")

    models = load_pickle(models_path)
    model = EnsembleModel(models)

    X_test: pd.DataFrame = pd.read_pickle(PROCESSED_TEST_PATH)

    proba: np.ndarray = model.predict_proba(X_test)[:, 1]

    fp: str = os.path.join(DATA_DIR, "predictions.csv")
    print(f"Fichier de sortie : {fp}")
    pd.DataFrame(proba, columns=["probazee"]).to_csv(fp, index=False)


if __name__ == "__main__":
    main()
