from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from numpy._typing import NDArray
from scipy.interpolate import RegularGridInterpolator


def read_feh_table(feh, base="."):
    """
    Read the table CIE table from Sutherland & Dopita (1993) for specific
    metallicities (Fe/H) into an Astropy table.

    Fe/H can be [0.5, -0.0, -0.5, -1, -1.5, -2.0, -3.0]
    """
    # Create the full filename based on the input metallicity
    feh = int(feh * 10)
    if feh != 0.0:
        name = f"m{feh:+03d}.cie"
    else:
        name = f"m-{feh:02d}.cie"

    path = Path(base) / name

    # Read the data, load it into an astropy table
    data = np.loadtxt(path, skiprows=4)
    names = ("log(T)", "ne", "nH", "nt", "log(lambda net)", "log(lambda norm)", "log(U)", "log(taucool)", "P12", "rho24", "Ci", "mubar")
    units = "K", "cm^-3", "cm^-3", "cm^-3", "log( cm^-3 s^-1 )", "log( cm^3 s^-1 )", "log(erg  cm^-3)", "log( s cm^-3 )", "ergs cm^-3 * 10^12", "g cm^-3 * 10^24", "s^-1", "10^24"
    tbl = Table(data, names=names, units=units)

    return tbl


def create_interpolate_func(tables_location="."):
    fehs = np.flip(np.array([0.5, -0.0, -0.5, -1, -1.5, -2.0, -3.0]))
    T_low, T_up = (4.1, 8.5)
    Ls = []
    for feh in fehs:
        tbl = read_feh_table(feh, base=tables_location)
        T = tbl["log(T)"]
        T_ax = tbl["log(T)"][(T >= T_low) & (T <= T_up)]
        L = tbl["log(lambda net)"][(T >= T_low) & (T <= T_up)]
        Ls.append(L)

    L = np.stack(Ls)
    # T_ax will be the same for the entire table (i.e. all metallicities)
    grid = (fehs, T_ax)
    interp = RegularGridInterpolator(grid, L)
    return interp


class CoolingFunc:
    """
    Interpolation of the CIE Cooling Function from Sutherland & Dopita (1993).

    Usage:

    ```python3
    import numpy as np
    import matplotlib.pyplot as plt

    # Define interpolation ranges
    Ts = np.linspace(4.1, 8.5, 100)  # Temperature in log space
    fehs = np.arange(-3.0, 1.0, 0.5) # [Fe/H]

    lamb_cie = CoolingFunc() # Initialize
    Ls = lamb_cie(fehs, Ts)  # Actually interpolate,
    # Ls is a (F, T) grid
    # with: F and T the size of the [Fe/H] and temperature array respectively

    # Plot the lines for different metallicity
    fig, ax = plt.subplots(1, 1)
    for feh, L in zip(fehs, Ls):
        ax.plot(Ts, L, label=feh)
    ax.set_ylabel(r"$\\log\\, \\Lambda_{\\rm CIE}(T)$ [erg / cm3 / s]")
    ax.set_xlabel(r"$\\log\\, T$ [K]")
    ax.legend()
    plt.show()
    ```
    """

    def __init__(self, data_location="./SutherlandCoolingFunctions"):
        # Initialize the interpolator with data in the given folder
        self.interpolator = create_interpolate_func(data_location)

    def __call__(self, metallicity: NDArray, temperature: NDArray, method: str = "linear"):
        # Assume input to be two arrays, both in log space
        feh = np.asarray(metallicity)
        temp = np.asarray(temperature)

        # Create the grid of points
        F, T = np.meshgrid(feh, temp, indexing="ij")
        test_points = np.array([F.ravel(), T.ravel()]).T

        # Interpolate
        Ls = self.interpolator(test_points, method=method)

        # Reshape result into the original grid
        Ls.shape = F.shape

        return Ls


def main():
    # Define interpolation ranges
    Ts = np.linspace(4.1, 8.5, 100)  # Temperature in log space
    fehs = np.arange(-3.0, 1.0, 0.5)  # [Fe/H]

    lamb_cie = CoolingFunc()  # Initialize
    Ls = lamb_cie(fehs, Ts)  # Actually interpolate,
    # Ls is a (F, T) grid
    # with: F and T the size of the [Fe/H] and temperature array respectively

    # Plot the lines for different metallicity
    fig, ax = plt.subplots(1, 1)
    for feh, L in zip(fehs, Ls):
        ax.plot(Ts, L, label=feh)
    ax.set_ylabel(r"$\log\, \Lambda_{\rm CIE}(T)$ [erg / cm3 / s]")
    ax.set_xlabel(r"$\log\, T$ [K]")
    ax.set_xlim(4.1, 8.4)
    ax.legend()
    fig.show()


if __name__ == "__main__":
    main()
