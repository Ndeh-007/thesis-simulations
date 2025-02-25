# %%
import sys
import os
import warnings

import matplotlib.pyplot as plt

from plot_utils import PlayBack

# endregion
# %%
# region local helpers
plot_grid = [3, 1]

# endregion

# %%  initializing the pair
# region initializing the pair

FOLDER_PAIRS_LIST: dict[str, list[tuple[str, str]]] = {
    "default": [
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_sc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_sc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_ic",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_ic",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
    ],
    "flow_rate":[
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.1\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.1\data\results\f1p1_pc",
        ),
    ],
    "yield_stress":[
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_1.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_1.0\data\results\f1p1_pc",
        ),
    ],
    "density":[
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_1.0\data\params\f2p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_1.0\data\results\f2p1_pc",
        ),
    ],
    "standoff":[
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_2.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_2.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_2.1\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_2.1\data\results\f1p1_pc",
        ),
    ],
    "density_unstable":[
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_3.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_3.0\data\results\f1p1_pc",
        ),
    ],
    "gap_thickness":[
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_0.0\data\results\f1p1_pc",
        ),
        (
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_4.0\data\params\f1p1_pc",
            r"E:\ComplexFluids\PlayGround\thesis_simulations\case_4.0\data\results\f1p1_pc",
        ),
    ],
    "all": []
}

target = None

# endregion
# %%
# get the argument from the command line and plot the required diagram

if len(sys.argv[1:]) == 0:
    warnings.warn("No target plot provided. draw base case with <key=default>")
    target = "default"
else:
    target = sys.argv[1]

if target not in FOLDER_PAIRS_LIST.keys():
    raise ValueError(
        f"Invalid plot target. \n\t valid keys are <{FOLDER_PAIRS_LIST.keys()}>"
    )


PLOT_GRIDS = {
        "default": [3, 1],
        "standoff": [3, 1],
        "flow_rate": [2, 1],
        "yield_stress": [2, 1],
        "density": [2, 1],
        "density_unstable": [2, 1],
        "gap_thickness": [2, 1],
}

PLOT_OPTS = {
        "default": {
            "grid": [3, 1],
            # "title": "Base Case for Surface, Intermediate and Production Casing.",
            "title": "",
        },
        "standoff": {
            "grid": [3, 1],
            # "title": "Standoff - 80%, 50%, 25%",
            "title": "",
        },
        "flow_rate": {
            "grid": [2, 1],
            # "title": "Doubled Pump Rate",
            "title": "",
        },
        "yield_stress": {
            "grid": [2, 1],
            # "title": "Low Yield Stress of Lead Slurry. Smaller than spacer",
            "title": "",
        },
        "density": {
            "grid": [2, 1],
            # "title": "High Lead Slurry Density, close to Tail",
            "title": "",
        },
        "density_unstable": {
            "grid": [2, 1],
            # "title": "Unstable Density sequence for Production Casing",
            "title": "",
        },
        "gap_thickness": {
            "grid": [2, 1],
            # "title": "75% Production Casing Gap Thickness Increase",
            "title": "",
        },
}

# %%
def create_plots():
    if target != "all":
        p = PlayBack(fp=FOLDER_PAIRS_LIST[target], cns=None, grid=PLOT_OPTS[target]["grid"])
        p.fig.suptitle(PLOT_OPTS[target]['title'])

        plt.tight_layout()
        plt.show()
        return [p]
    
    # otherwise, plot everything
    playbacks = []

    for k, v in FOLDER_PAIRS_LIST.items():
        if k == "all":
            continue
        p = PlayBack(fp=v, cns=None, grid=PLOT_OPTS[k]['grid'])
        
        p.fig.suptitle(PLOT_OPTS[k]['title'])
        playbacks.append(p)

    plt.tight_layout()
    plt.show()

    return playbacks


# %% invoke - oouhhh scary

playbacks = create_plots()