# %%
import os

from typing import Any
from PIL import Image

# from IPython.display import HTML

import numpy as np
import pandas as pd
import matplotlib.colors as clr
import scipy.ndimage as snd
import h5py

import traceback
# %% [markdown]
# ### Helper data entries

COLORS = [
    "#400a06",  # MUD
    "#1261b0",  # SPACER
    "#292b2e",  # TAIL
    "#63686e",  # LEAD
    "#c2440e",  # DISPLACEMENT
    # "#c5dfaf",
    # "#02e5be",
    # "#dd1270",
    # "#e12bd9",
    # "#d6b3b6",
]

COLOR_NAMES = ["MUD", "SPACER", "LEAD", "TAIL", "DISPLACEMENT"]


# %%

BASE_PLOT_CONFIG = {
    "ticks": {
        "x_min": 0,
        "x_max": 1,  # the min and max data value on the x-axis
        "y_min": 0,
        "y_max": 1,  # the min and max data value on the y-axis
        "x_count": 3,
        "y_count": 20,  # number of ticks on the xy-axis
        "x_size": 10,
        "y_size": 10,  # data-set grid size
        "label_type": "number",
        # the return type of the labels. can either be array[string] or array[number]. doesn't work in custom mode
        "n_decimals": 2,  # number of decimals points for number type ticks
        "label_mode": "default",  # labels are calculated normally: default | custom
        "custom_labels": (np.arange(0, 3), np.arange(0, 20)),
        # x and y contain list of labels that must match their corresponding label count
        "reverse_x_labels": False,
        "reverse_y_labels": False,
        # reverses the order of the computed ticks. doesn't work in custom mode
    }
}


def extract_files_from_dir(directory: str):
    """
    reads all the files from a directory and returns a list of tuples
    :return: list of tuples, list[tuple[file_name, absolute_file_path]]
    """
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)

        if os.path.isfile(file_path):
            file_name = os.path.splitext(file)[0]
            files.append((file_name, os.path.abspath(file_path)))
    return files


def read_plot_results(
    folder: str, storage: dict[str, np.ndarray | int | float | list | dict]
) -> tuple[bool, str | Exception]:
    try:

        files = extract_files_from_dir(folder)

        for key, file_path in files:
            file = h5py.File(file_path, "r")
            data = file[key]
            # handle TTinFP such all entries go to strings from bytes
            if key == "TTinFP":
                data = np.array([data[i].decode() for i in range(data.shape[0])])
            storage.update({key: data})

        return True, ""

    except Exception as e:
        print(e)
        et = traceback.format_exc()
        return False, f"{et}"


def load_plot_datagrid(
    folder: str, storage: dict[str, np.ndarray | int | float | list | dict]
) -> tuple[bool, str | Exception]:
    try:
        data = {
            "geometry": pd.read_csv(os.path.join(folder, "geometry.txt")),
            "fluids": pd.read_csv(os.path.join(folder, "fluids.txt")),
            "pump_schedule": pd.read_csv(os.path.join(folder, "pumping.txt")),
            "num_params": pd.read_csv(os.path.join(folder, "numparams.txt")),
            "options": pd.read_csv(os.path.join(folder, "options.txt")),
        }

        geometry = data["geometry"].values
        fluids = data["fluids"].values
        num_params = data["num_params"].values

        # collect columns into dictionary
        storage.update(
            {
                "measured_depth": geometry[:, 0],
                "inner_diameter": geometry[:, 1],
                "outer_diameter": geometry[:, 2],
                "eccentricity": geometry[:, 3],
                "roughness": geometry[:, 4],
                "inclination": geometry[:, 5],
                "fluid_count": fluids.shape[0],
                "fluid_colors": COLORS[0 : fluids.shape[0] + 1],
                "fluids": fluids,
                "num_params": num_params,
            }
        )

        return True, ""
    except Exception as e:
        print(e)
        et = traceback.format_exc()
        return False, f"{et}"


def collect_plot_data(
    storage: dict[str, np.ndarray | int | float | list | dict]
) -> tuple[bool, str | Exception]:
    """
    Collects the plot data into the provided dictionary
    """

    try:
        # create variables
        DTubeLength = storage["DTubelength"]
        nsections = storage["varsave"][0]
        nz = int(storage["num_params"][0])
        nz_sections = storage["nz_sections"]
        dz_sections = storage["dz_sections"]
        TTinFP = storage["TTinFP"]
        nfluids = storage["varsave"][2]

        # collect fluids in order of the pump schedule
        f_arr = []

        hex_colors = storage["fluid_colors"]
        for color in hex_colors:
            r, g, b = clr.to_rgb(str(color))
            f_arr.append([r, g, b])

        # the fluid in well
        rgb_fluid = np.array(f_arr)

        regimecolors = np.zeros((7, 3))
        regimecolors[0, :] = [1, 0, 0]  # turbulent
        regimecolors[1, :] = [1, 0.4, 0.6]  # transitional
        regimecolors[2, :] = [1, 1, 0]  # laminar density stable
        regimecolors[3, :] = [0, 1, 0]  # laminar mixed
        regimecolors[4, :] = [0, 1, 1]  # stratified inertial
        regimecolors[5, :] = [0, 0, 1]  # stratified laminar
        regimecolors[6, :] = [0, 0, 0]  # static laminar - only in annulus

        dpdzfricmax = -1000.0
        dpdzfricmin = 1000.0

        dpdzfricsave = storage["dpdzfricsave"]
        nphi = int(storage["varsave"][1])
        Nout = storage["fluids"].shape[0]

        for j in range(Nout - 3):
            for k in range(nsections):
                dpdzfricmax = max(
                    dpdzfricmax,
                    np.max(
                        dpdzfricsave[j, k, 2 : 3 + nphi, 3 : 4 + int(nz_sections[k])]
                    ),
                )
                dpdzfricmin = min(
                    dpdzfricmin,
                    np.min(
                        dpdzfricsave[j, k, 2 : 3 + nphi, 3 : 4 + int(nz_sections[k])]
                    ),
                )

        dpdzmax = -1000.0
        dpdzmin = 1000.0

        dpdzsave = storage["dpdzsave"]
        for j in range(Nout - 3):
            for k in range(nsections):
                dpdzmax = max(
                    dpdzmax,
                    np.max(dpdzsave[j, k, 2 : 3 + nphi, 3 : 4 + int(nz_sections[k])]),
                )
                dpdzmin = min(
                    dpdzmin,
                    np.min(dpdzsave[j, k, 2 : 3 + nphi, 3 : 4 + int(nz_sections[k])]),
                )

        dpdzlevels = np.arange(dpdzmin, dpdzmax, (dpdzmax - dpdzmin) / 40)
        dpdzfriclevels = np.arange(
            dpdzfricmin, dpdzfricmax, (dpdzfricmax - dpdzfricmin) / 40
        )

        # construct colors array
        N = rgb_fluid.shape[0]
        channels = rgb_fluid.shape[1]
        _, _, _, m, n = storage["csave"].shape
        base_colors = np.zeros((N, m, n, channels))

        for i in range(N):
            for ch in range(channels):
                grid = np.ones((m, n)) * rgb_fluid[i, ch]
                base_colors[i, :, :, ch] = grid

        storage.update(
            {
                "base_colors": base_colors,
                "DTubeLength": DTubeLength,
                "nsections": nsections,
                "nz": nz,
                "nz_sections": nz_sections,
                "dz_sections": dz_sections,
                "TTinFP": TTinFP,
                "rgb_fluid": rgb_fluid,
                "dpdzfricmax": dpdzfricmax,
                "dpdzfricmin": dpdzfricmin,
                "dpdzmax": dpdzmax,
                "dpdzmin": dpdzmin,
                "dpdzlevels": dpdzlevels,
                "dpdzfriclevels": dpdzfriclevels,
                "regimecolors": regimecolors,
            }
        )

        return True, ""
    except Exception as e:
        print(e)
        et = traceback.format_exc()
        return False, f"{et}"


# togo: ignore ??
def collect_legend_data(storage: dict) -> tuple[bool, str | Exception]:
    # create the legend
    try:
        regime_items = []
        regimes = [
            "Turbulent",
            "Transitional",
            "Laminar Stable",
            "Laminar Mixed",
            "Stratified Inertial",
            "Stratified Laminar",
            "Static Laminar",
        ]
        regimecolors = storage["regimecolors"]
        for i in range(regimecolors.shape[0]):
            # r, g, b = (regimecolors[i, :] * 255).astype(np.int64)
            # color = QColor(r, g, b).name()
            r, g, b = regimecolors[i, :]
            color = clr.to_hex((r, g, b))
            regime_items.append((color, regimes[i]))

        opts = {"legend": {"colors": []}, "regimeLegend": {"colors": regime_items}}
        storage.update(opts)

        return True, ""
    except Exception as e:
        print(e)
        et = traceback.format_exc()
        return False, f"{et}"


def construct_ticks(options: dict[str, Any]) -> dict[str, np.ndarray | list]:
    """
    builds custom ticks for the graphs and resolve the ${axis}_ticks and ${axis}_ticks_labels
    for the graph. options defines how the ticks should be generated
    """
    # clone the base plot tick configurations
    opts = dict(BASE_PLOT_CONFIG["ticks"])

    # override the base configuration with new configurations
    if len(options.keys()) > 0:
        opts.update(options)

    # begin tick creation
    if opts["label_mode"] == "default":  # compute the ticks

        x_ticks = np.linspace(0, opts["x_size"] - 1, opts["x_count"], dtype=int)
        y_ticks = np.linspace(0, opts["y_size"] - 1, opts["y_count"], dtype=int)

        x_labels = np.round(
            np.linspace(opts["x_min"], opts["x_max"], x_ticks.shape[0]),
            opts["n_decimals"],
        )
        y_labels = np.round(
            np.linspace(opts["y_min"], opts["y_max"], y_ticks.shape[0]),
            opts["n_decimals"],
        )

        if opts["reverse_x_labels"]:
            x_labels = np.flip(x_labels, axis=0)

        if opts["reverse_y_labels"]:
            y_labels = np.flip(y_labels, axis=0)

        if opts["label_type"] == "number":
            x_labels = x_labels.astype(np.float64)
            y_labels = y_labels.astype(np.float64)

        if opts["label_type"] == "string":
            x_labels = x_labels.astype(str)
            y_labels = y_labels.astype(str)

        # collect and resolve the ticks
        ticks = {
            "x_ticks": x_ticks,
            "x_ticks_labels": x_labels,
            "y_ticks": y_ticks,
            "y_ticks_labels": y_labels,
        }

        return ticks

    if opts["label_mode"] == "custom":

        x_ticks = np.linspace(0, opts["x_size"] - 1, opts["x_count"], dtype=int)
        y_ticks = np.linspace(0, opts["y_size"] - 1, opts["y_count"], dtype=int)

        # collect and resolve the ticks
        ticks = {
            "x_ticks": x_ticks,
            "x_ticks_labels": opts["custom_labels"][0],
            "y_ticks": y_ticks,
            "y_ticks_labels": opts["custom_labels"][1],
        }

        return ticks


def scale_with_pillow(
    arr: np.ndarray, factor: int | tuple[int, int] = 1, multiplier: int = 1
) -> np.ndarray:
    """
    scales an image using pillow
    """

    f_x, f_y = 1, 1
    if isinstance(factor, tuple):
        f_x, f_y = factor

    if isinstance(factor, int):
        f_x, f_y = (factor, factor)

    # create the pillow image
    img = Image.fromarray(arr)

    # scale it
    size = multiplier * f_x * img.width, multiplier * f_y * img.height
    s_img = img.resize(size, Image.LANCZOS)

    # resolve it as a np-array
    return np.array(s_img)


def apply_gaussian_filter(
    arr: np.ndarray, dimensions: list[int], factor: int = 1, multiplier: int = 1
) -> np.ndarray:
    """
    applies a uniform gaussian filter to target slices of array

    dimensions: a list integers of values 0 and 1. 0 for dimensions to be omited during filtration.
    len(dimensions) == len(arr.shape)
    """
    r = factor * multiplier

    # create the filter kernel
    kernel = []
    for i in range(len(dimensions)):
        kernel.append(dimensions[i] * r)

    f_arr = snd.gaussian_filter(arr, sigma=kernel)
    return f_arr


# %%
# define the plotting of each result case
# plot on the concentration map


def process_concentration_map(
    storage: dict[str, Any], index: list[int] | int | str = -1, scaling: list[int] = [1, 1]
) -> dict[str, Any]:
    # read the concentration map file
    # read the supporting file structures from the params
    # create the image at the provided index
    # resolve the image, and the axis ticks

    nphi = int(storage["varsave"][1])
    base_colors: np.ndarray = storage["base_colors"]
    c_save: np.ndarray = storage["csave"]
    psi_save: np.ndarray = storage["psisave"]
    TTinFP: list[str] = storage["TTinFP"]
    nsections: np.ndarray = storage["nsections"]
    nz_sections: np.ndarray = storage["nz_sections"]
    N_out: int = storage["csave"].shape[0] - 1
    DTubeOD: np.ndarray = storage["DTubeOD"]
    DTubeLength: np.ndarray = storage["DTubelength"]

    timesteps: list[int] = [-1]
    if isinstance(index, int):
        timesteps = [index]
    elif isinstance(index, list):
        timesteps = index
    elif index == "all":
        timesteps = range(N_out)
    else:
        timesteps = [-1]

    ann_psi = []
    ann_psi_x = None
    ann_psi_y = None
    ann_imgs: list[np.ndarray] = []
    pipe_imgs: list[np.ndarray] = []

    for j in timesteps:
        for k in [1]:  # because we want only the annulus
            c_vals = c_save[j, :, k, :, :]

            c, _, n = c_vals.shape

            colors = base_colors

            # account for the concentration at different locations
            if TTinFP[k] == "NAnn":
                alphas = c_vals
                # create alpha channels
                alpha_sum = np.sum(alphas, axis=0)
                alpha_sum[alpha_sum == 0] = 1.0  # replace all zero-sums with 1

                # compute weighted rbg using alphas as weights
                weighted_rgb: np.ndarray = (
                    np.sum(colors * alphas[..., np.newaxis], axis=0)
                    / alpha_sum[..., np.newaxis]
                )
            else:
                # get the 1D line
                alphas = c_vals[:, 0, :].reshape((c, 1, n))
                alpha_sum = np.sum(alphas, axis=0)
                alpha_sum[alpha_sum == 0] = 1.0  # replace all zero-sums with 1

                # compute weighted rbg using alphas as weights
                weighted_rgb: np.ndarray = (
                    np.sum(colors * alphas[..., np.newaxis], axis=0)
                    / alpha_sum[..., np.newaxis]
                )

            weighted_rgb[weighted_rgb > 1] = 1.0
            weighted_rgb[weighted_rgb < 0] = 0.0

            # prepare for scaling
            filter_weighted_rbg = (weighted_rgb * 255).astype(np.uint8)

            # scale the image and smooth image
            # r = 2
            # scaled_weighted_rbg = scale_with_pillow(filter_weighted_rbg, scaling[0], r)
            # smoothed_weighted_rbg = apply_gaussian_filter(scaled_weighted_rbg, [1, 1, 0], 2, r)

            smoothed_weighted_rbg = filter_weighted_rbg

            if TTinFP[k] == "NAnn":  # inside the narrow annulus

                x_size, y_size, _ = smoothed_weighted_rbg.shape

                # we want a horizontal image
                # smoothed_weighted_rbg: np.ndarray = snd.rotate(smoothed_weighted_rbg, 0)
                # smoothed_weighted_rbg: np.ndarray = np.flip(smoothed_weighted_rbg, axis=1)

                # collect the fluid concentrations
                ann_imgs.append(smoothed_weighted_rbg)

                # collect streamlines values
                Psivals = psi_save[j, k, 1 : 2 + nphi, 3 : 4 + int(nz_sections[k][0])]

                x_size, y_size, _ = smoothed_weighted_rbg.shape
                # scale the psi values
                # zoom_factors = (x_size / Psivals.shape[1], y_size / Psivals.shape[0])
                # Psivals: np.ndarray = snd.zoom(Psivals, zoom_factors, order=1)  # scale using linear interpolations

                xr = np.linspace(0, y_size - 1, Psivals.shape[1])
                yr = np.linspace(0, x_size - 1, Psivals.shape[0])

                # yr = np.flip(yr)

                ann_psi_x = xr
                ann_psi_y = yr
                ann_psi.append(Psivals)
            else:
                # inside the pipe
                x_size, y_size, _ = smoothed_weighted_rbg.shape
                smoothed_weighted_rbg = snd.rotate(smoothed_weighted_rbg, -90)
                pipe_imgs.append(smoothed_weighted_rbg)

    ann_tick_config = {
        "x_size": ann_imgs[0].shape[1],
        "y_size": ann_imgs[0].shape[0],
        "y_min": 0,
        "y_max": 1,  # for half annulus
        "x_min": 0,
        "x_max": DTubeLength[0],  # measured depth
        "x_count": 10,
        "y_count": 3,
        "reverse_x_labels": True,
    }

    # pipe_tick_config = {
    #     "x_size": pipe_imgs[0].shape[1], "y_size":  pipe_imgs[0].shape[0],
    #     "x_min": 0, "x_max": DTubeOD[1], # diameter of the pipe
    #     "y_min": 0, "y_max": DTubeLength[1], # measured depth
    #     "x_count": 3, "y_count": 20,
    #     "reverse_y_labels": True,
    # }

    c_opts = {
        "annulus_images": ann_imgs,
        "annulus_streamlines": ann_psi,
        "annulus_streamlines_x": ann_psi_x,
        "annulus_streamlines_y": ann_psi_y,
        "pipe_images": pipe_imgs,
        "time_steps": N_out,
        "legend": storage["legend"],
        "ticks": {
            "annulus": construct_ticks(ann_tick_config),
            # "pipe": construct_ticks(pipe_tick_config)
        },
    }

    return c_opts


# %%
def curate_plot_data(folder_pairs: str, data_range: int | str | list[int] = -5, scaling:list[int]=[1,1]) -> dict[str, dict]:
    res = {}

    for pair in folder_pairs:
        p, r = pair
        i = data_range  

        # load the data from memory for the related folder
        store = {}

        state, error = read_plot_results(r, store)
        if not state:
            raise Exception(error)

        state, error = load_plot_datagrid(p, store)
        if not state:
            raise Exception(error)

        state, error = collect_plot_data(store)
        if not state:
            raise Exception(error)

        state, error = collect_legend_data(store)
        if not state:
            raise Exception(error)

        # by default we want the last concentration entry
        opts = process_concentration_map(store, i, scaling=scaling)

        res.update({p: opts})

    return res


# %%
