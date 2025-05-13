import os
import pickle
import tempfile
from src.utils import print_divider, load_pickle

def test_print_divider(capsys):
    print_divider("TEST")
    captured = capsys.readouterr()
    assert "TEST" in captured.out
    assert "-" * 25 in captured.out

def test_load_pickle():
    # Crée un objet à sauvegarder temporairement
    data = {"a": 1, "b": 2}

    # Utilise un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(data, tmp)
        tmp_path = tmp.name

    # Recharge avec load_pickle
    loaded = load_pickle(tmp_path)

    # Supprime le fichier temporaire
    os.remove(tmp_path)

    # Vérifie que les données sont identiques
    assert loaded == data
