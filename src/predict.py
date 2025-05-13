import os
import re
import numpy as np
import pandas as pd
import mlflow
from src.utils import load_pickle
from src.config import DATA_DIR, PROCESSED_TEST_PATH


# ______________________________________________________________________________
#
# region EnsembleModel
# ______________________________________________________________________________
class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        proba = np.zeros((len(X), self.models[0].n_classes_))
        for model in self.models:
            proba += model.predict_proba(X, num_iteration=model.best_iteration_)
        return proba / len(self.models)

#______________________________________________________________________________
#
# region get_experiment_id_folder
#______________________________________________________________________________
def get_experiment_id_folder_name() :
    experiment_id_folder_name = '0'
    mlruns_path = os.path.join(os.getcwd(), "mlruns")
    if os.path.exists(mlruns_path):
        for root, dirs, files in os.walk(mlruns_path):
            for dir_name in dirs:
                if dir_name in ["0", ".trash"] :
                    continue

                experiment_id_folder_name = dir_name
                break
        
            if experiment_id_folder_name != '0' : 
                break
    else:
        print(f"❌ Le dossier {mlruns_path} n'existe pas.")

    return experiment_id_folder_name

# ______________________________________________________________________________
#
# region main
# ______________________________________________________________________________
def main():
    EXPERIMENT_ID = get_experiment_id_folder_name()
    runs = mlflow.search_runs(
        EXPERIMENT_ID,
        order_by=["metrics.accuracy DESC", "attribute.start_time DESC"],
        max_results=10,
    )
    print(runs)
    print(runs.columns)
    print(runs.head())

    # Traitement du chemin du modèle
    raw_path = runs.loc[0, "params.model_path"]
    # Ne garder que la partie après "mlruns/"
    match = re.search(r"mlruns[\\/](.*)", raw_path)
    if not match:
        raise ValueError(f"Chemin invalide : {raw_path}")

    relative_path = match.group(1).replace("\\", "/")
    models_path = os.path.join("/app/mlruns", relative_path)
    models_path = os.path.normpath(models_path)
    print(f"Chemin vers le modèle : {models_path}")

    # Charger les modèles depuis le fichier pickle
    models = load_pickle(models_path)
    model = EnsembleModel(models)

    # Charger les données de test
    X_test = pd.read_pickle(PROCESSED_TEST_PATH)

    # Faire des prédictions
    proba = model.predict_proba(X_test)[:, 1]

    # Sauvegarder les prédictions dans un fichier CSV
    fp = os.path.join(DATA_DIR, "predictions.csv")
    print(f"Fichier de sortie : {fp}")
    pd.DataFrame(proba, columns=["probazee"]).to_csv(fp, index=False)

if __name__ == "__main__":
    main()
