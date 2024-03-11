import constants as cst
import numba
import numpy as np


@numba.jit(nopython=True)
def nfw_potential_gradient(r, r_s, rho_s):
    potential_gradient = np.empty_like(r)

    x = r / r_s
    approx = x < 0.005
    # Taylor approx for small x
    potential_gradient[approx] = -16 * np.pi * cst.G * rho_s * r_s * (-1 / 2 + (2 / 3) * x - (3 / 4) * x * x)
    potential_gradient[~approx] = -16 * np.pi * cst.G * rho_s * r_s * ((x / (1 + x) - np.log(1 + x)) / x**2)
    return potential_gradient


@numba.jit(nopython=True)
def nfw_potential_gradient_scalar(r, r_s, rho_s):
    """
    Only works for r being a scalar
    """
    x = r / r_s
    if x < 0.005:  # Taylor around x -> 0 (up to second order)
        potential_gradient = -16 * np.pi * cst.G * rho_s * r_s * (-1 / 2 + (2 / 3) * x - (3 / 4) * x * x)
    else:
        potential_gradient = -16 * np.pi * cst.G * rho_s * r_s * ((x / (1 + x) - np.log(1 + x)) / x**2)
    return potential_gradient
