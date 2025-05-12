import os
import re
import numpy as np
import pandas as pd
import mlflow
from utils import load_pickle
from config import DATA_DIR, PROCESSED_TEST_PATH

#______________________________________________________________________________
#
# region EnsembleModel
#______________________________________________________________________________
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


#______________________________________________________________________________
#
# region main
#______________________________________________________________________________
def main():
    
    experiment_id = get_experiment_id_folder_name()
 
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=['metrics.accuracy DESC', 'attribute.start_time DESC'],
        max_results=10)
    
    if runs.empty:
        raise ValueError(f"Aucun run trouvé pour l'expérience ID={experiment_id}")

    print(runs)

    if 'params.model_path' not in runs.columns:
        raise KeyError("Le paramètre 'model_path' n'a pas été loggé dans les runs.")
    
    raw_path = runs.loc[0, 'params.model_path']
    parsed_path = re.sub('^file://', '', raw_path)
    models_path = os.path.normpath(parsed_path.lstrip('/'))

    models_path= '/' + models_path # sous Linux
    
    #/home/nicolascassonnet/Documents/WORK/mlflow-titanic-ci-cd/mlruns/961726105335844291/9f7a1ec644254408b2f6e1034cad193f/artifacts/models.pkl
    models = load_pickle(models_path)
    model = EnsembleModel(models)

    X_test = pd.read_pickle(PROCESSED_TEST_PATH)
    proba = model.predict_proba(X_test)[:, 1]
    fp = os.path.join(DATA_DIR, 'prediction.csv')
    pd.DataFrame(proba, columns=['proba']).to_csv(fp, index=False)


if __name__ == '__main__':
    main()
