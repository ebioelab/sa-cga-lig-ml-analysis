#!/usr/bin/env python
# coding: utf-8

"""
Feature extraction workflow for voltammetric SA/CGA sensing data.

This script reads raw DPV or SWV CSV files, fits each voltammogram using a
parabolic baseline plus Gaussian peak functions, and saves the fitted peak
parameters for downstream ML analysis.
"""

import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize


# -------------------------------------------------------------------------
# User-defined configuration
# -------------------------------------------------------------------------
# Update these values when processing a different analyte, method, or dataset.
ANALYTE = "SA"
INPUT_FILES = [
    "Sensor1_DPV.csv",
    "Sensor2_DPV.csv",
    "Sensor3_DPV.csv",
    "Sensor4_DPV.csv",
]
OUTPUT_PICKLE = f"{ANALYTE}_DPV.p"


def get_ps(fn, dirn):
    """
    Read current values from a PalmSens/PSTrace-style JSON export.

    Parameters
    ----------
    fn : str
        File name to read.
    dirn : str
        Directory containing the file.

    Returns
    -------
    list or None
        Current values extracted from the first available curve. Returns None
        if the file cannot be read or parsed.

    Notes
    -----
    This helper is kept for compatibility with PalmSens JSON exports. The
    main workflow below uses CSV files through `getDatFiles`.
    """
    try:
        file_path = os.path.join(dirn, fn)
        with open(file_path, encoding="utf-16") as f:
            content = f.readlines()

        content = [x.strip() for x in content]
        dat = json.loads(content[0][:-1])

        for curve in dat["measurements"][0]["curves"]:
            y_values = curve["yaxisdataarray"]["datavalues"]
            current = [point["v"] for point in y_values]
            return current

    except Exception as exc:
        print(f"INSPECT PS FAILED: {fn} | {exc}")
        return None


def getDatFiles(dirn):
    """
    Load a CSV file containing multiple voltammetric traces.

    Parameters
    ----------
    dirn : str
        Path to the CSV file.

    Returns
    -------
    list
        A list of records in the format:
        [trace_name, current_array, potential_array].

    Notes
    -----
    The CSV format is expected to contain paired potential/current columns.
    Concentration or sample labels are read from the first row and incorporated
    into the generated trace names.
    """
    retdat = []

    first_line = np.genfromtxt(dirn, delimiter=",", max_rows=1, dtype=str)
    data = np.loadtxt(dirn, delimiter=",", skiprows=2).T

    base_name = os.path.splitext(os.path.basename(dirn))[0]

    # Each pair of columns is assumed to contain potential and current values.
    for column_i in range(0, len(first_line), 2):
        header_name = str(first_line[column_i]).strip()
        header_name = header_name.replace(" ", "_").replace("/", "_")

        trace_name = f"A_{base_name}_{ANALYTE}_{header_name}"
        print(trace_name)

        potential = data[column_i]
        current = data[column_i + 1]
        retdat.append([trace_name, current, potential])

    return retdat


def getprms(x_array, y_array_gauss, cens=None, bounds=None, feats=None):
    """
    Fit one voltammogram using a parabolic baseline and Gaussian peaks.

    Parameters
    ----------
    x_array : array-like
        Potential values.
    y_array_gauss : array-like
        Current values.
    cens : list, optional
        Initial peak-center locations used for Gaussian fitting.
    bounds : list, optional
        Lower and upper bounds for the fit parameters.
    feats : list, optional
        Initial parameter guesses.

    Returns
    -------
    tuple
        popt_gauss : np.ndarray
            Optimized fit parameters. The first three values describe the
            parabolic baseline, followed by repeating Gaussian parameters
            [amplitude, center, sigma].
        resid : np.ndarray
            Residual between the measured current and the fitted curve.
    """

    def _1gaussian(*args):
        """Model function: parabolic baseline plus multiple Gaussian peaks."""
        x = args[0]
        bas1 = args[1]
        mag1 = args[2]
        off1 = args[3]
        args = args[4:]

        # Parabolic baseline term.
        ret = bas1 + mag1 * (x - off1) ** 2

        # Add all Gaussian peak contributions.
        for i in range(len(args))[::3]:
            amplitude = args[i]
            center = args[i + 1]
            sigma = args[i + 2]
            ret += amplitude * (1 / (sigma * np.sqrt(2 * np.pi))) * (
                np.exp((-0.5) * (((x - center) / sigma) ** 2))
            )

        return ret

    if cens is None:
        # Default candidate peak centers over the potential range.
        cens = [0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

    if bounds is None:
        # Initial baseline estimates.
        bas1 = 30
        mag1 = 20
        off1 = 0.1

        # Initial Gaussian estimates used for each candidate peak.
        amp = 10
        sigma = 0.1

        bounds = [[0, 0, -0.2], [100, 30, 0.2]]
        feats = [bas1, mag1, off1]

        for center in cens:
            feats += [amp, center, sigma]
            bounds[0] += [0, center - 0.02, 0.02]
            bounds[1] += [np.inf, center + 0.02, 0.1]

    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(
        _1gaussian,
        x_array,
        y_array_gauss,
        p0=feats,
        bounds=bounds,
        maxfev=10000,
    )

    fitted_curve = _1gaussian(x_array, *popt_gauss)
    resid = y_array_gauss - fitted_curve

    # Plot the fit, residual, baseline, and individual Gaussian components.
    plt.figure()
    plt.plot(x_array, y_array_gauss, label="Measured signal")
    plt.plot(x_array, fitted_curve, "k--", label="Total fit")
    plt.plot(x_array, resid, label="Residual")

    plt.xlabel("Potential (relative)")
    plt.ylabel("Current (relative)")

    args = popt_gauss
    bas1 = args[0]
    mag1 = args[1]
    off1 = args[2]
    gaussian_args = args[3:]

    print([bas1, mag1, off1])
    plt.plot(x_array, bas1 + mag1 * (x_array - off1) ** 2, label="Baseline")

    for i in range(len(gaussian_args))[::3]:
        amplitude = gaussian_args[i]
        center = gaussian_args[i + 1]
        sigma = gaussian_args[i + 2]

        gaussian_component = amplitude * (1 / (sigma * np.sqrt(2 * np.pi))) * (
            np.exp((-0.5) * (((x_array - center) / sigma) ** 2))
        )
        plt.plot(x_array, gaussian_component)

        print(cens[int(i / 3)], gaussian_args[i : i + 3])

    plt.legend()
    plt.show()

    return popt_gauss, resid


def getAllParams(dat, cens=None, bounds=None, feats=None):
    """
    Fit all traces in a dataset and collect the extracted parameters.

    Parameters
    ----------
    dat : list
        Output from `getDatFiles`.
    cens, bounds, feats : optional
        Fitting settings passed to `getprms`.

    Returns
    -------
    list
        Each output record contains:
        [trace_name, fit_success, fit_parameters, residual].
    """
    retdat = []

    for i, dati in enumerate(dat):
        print(i, dati[:1])

        try:
            # Edge points are excluded to reduce fitting artifacts near the
            # beginning and end of the potential window.
            p, r = getprms(dati[2][10:-10], dati[1][10:-10], cens, bounds, feats)
            retdat.append(dati[:1] + [True, p, r])
            print("DONE")

        except KeyboardInterrupt:
            print("Interrupted")
            return retdat

        except Exception as exc:
            print(f"FAILED: {exc}")
            retdat.append(dati[:1] + [False, [], []])

    return retdat


def main():
    """
    Run feature extraction for the input files defined above and save results.
    """
    res = []

    # Process each sensor file and append all fitted parameters to one list.
    for file_name in INPUT_FILES:
        print(file_name)
        file_path = os.path.join(".", file_name)
        res += getAllParams(getDatFiles(dirn=file_path))

    # Save extracted features for downstream ML modeling.
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(res, f)

    print(f"Saved {len(res)} fitted traces to {OUTPUT_PICKLE}")
    return res


if __name__ == "__main__":
    res = main()
