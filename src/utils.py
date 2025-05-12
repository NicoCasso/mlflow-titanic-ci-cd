import os
import pickle

#______________________________________________________________________________
#
# region print_divider
#______________________________________________________________________________
def print_divider(title):
    print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))
#______________________________________________________________________________
#
# region load_pickle
#______________________________________________________________________________
def load_pickle(file_path):
    if os.path.getsize(file_path) > 0:  # Vérifie que le fichier n'est pas vide
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Le fichier {file_path} est vide.")
