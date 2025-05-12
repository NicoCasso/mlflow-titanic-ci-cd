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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    if os.path.getsize(file_path) == 0:  # Vérifie que le fichier n'est pas vide
        raise EOFError(f"Le fichier {file_path} est vide.")
    
    return_value = None
    with open(file_path, 'rb') as f:
        return_value = pickle.load(f)

    return return_value
        
