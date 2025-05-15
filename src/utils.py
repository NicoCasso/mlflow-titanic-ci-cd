import pickle
import matplotlib.pyplot as plt
from typing import Any


# ______________________________________________________________________________
#
# region print_divider
# ______________________________________________________________________________
def print_divider(title: str) -> None:
    print("\n{} {} {}\n".format("-" * 25, title, "-" * 25))


# ______________________________________________________________________________
#
# region load_pickle
# ______________________________________________________________________________
def load_pickle(fp: str) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)


def close_all_figures() -> None:
    """Ferme toutes les figures ouvertes de matplotlib"""
    plt.close("all")
