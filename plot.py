import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt

from plot_utils import PlayBack

# modify the plotter
matplotlib.rcParams["font.family"] = ["serif"]

# import the colors
from helpers import *


# region local helpers
def create_folder_pairs() -> dict[str, list[tuple[str, str]]]:
    base_pairs_list: dict[str, list[tuple[str, str]]] = {}

    cases = [
        "case_1",
        "case_2",
        "case_3",
        "case_4",
    ]

    grids = [
        "grid_1",
        "grid_2",
        "grid_3",
        # "grid_4",
    ]

    for case in cases:
        arr = []

        for grid in grids:
            folder = f"{grid}-{case}"
            entry = (
                os.path.join(os.getcwd(), folder, "params"),
                os.path.join(os.getcwd(), folder, "results"),
            )
            arr.append(entry)

        base_pairs_list.update({case: arr})

    return base_pairs_list


# endregion

# region initializing the pair

FOLDER_PAIRS_LIST: dict[str, list[tuple[str, str]]] = create_folder_pairs()

target = None

# endregion

# get the argument from the command line and plot the required diagram

if len(sys.argv[1:]) == 0:
    warnings.warn("No target plot provided. draw base case with <key=case_1>")
    target = "case_1"
else:
    target = sys.argv[1]

keys = [k for k in FOLDER_PAIRS_LIST.keys()]   
keys.append("all")

if target not in keys:
    raise ValueError(
        f"Invalid plot target. \n\t valid keys are <{keys}>"
    )
if target == "all":
    for k in FOLDER_PAIRS_LIST.keys():
        playback = PlayBack(fp=FOLDER_PAIRS_LIST[k], cns=None, grid=[3, 1], plot_data_range=-2, scaling=[1])
        playback.fig.suptitle(k)

else:
    playback = PlayBack(fp=FOLDER_PAIRS_LIST[target], cns=None, grid=[3, 1], plot_data_range=-2, scaling=[1])
    playback.fig.suptitle(target)

plt.tight_layout()
plt.show()

