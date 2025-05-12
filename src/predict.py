import os
import re
import numpy as np
import pandas as pd
import mlflow
from src.utils import load_pickle
from src.config import DATA_DIR, PROCESSED_TEST_PATH

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
# region main
#______________________________________________________________________________
def main():
    EXPERIMENT_ID = '343152224088962963'
    runs = mlflow.search_runs(EXPERIMENT_ID,
                              order_by=['metrics.accuracy DESC', 'attribute.start_time DESC'],
                              max_results=10)
    print(runs)
    print(runs.columns)
    print(runs.head())

    #models_path = re.sub('^file://', '', runs.loc[0, 'params.model_path'])
    raw_path = runs.loc[0, 'params.model_path']
    parsed_path = re.sub('^file://', '', raw_path)
    models_path = os.path.normpath(parsed_path.lstrip('/'))

    models = load_pickle(models_path)
    model = EnsembleModel(models)

    X_test = pd.read_pickle(PROCESSED_TEST_PATH)
    proba = model.predict_proba(X_test)[:, 1]
    fp = os.path.join(DATA_DIR, 'prediction.csv')
    pd.DataFrame(proba, columns=['probaz']).to_csv(fp, index=False)


if __name__ == '__main__':
    main()
