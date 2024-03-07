from typing import Callable

import numba
import numpy as np


@numba.jit(nopython=True)
def find_segment(x: float, arr: np.ndarray) -> int:
    """
    Simple binary search returning the index i in the sorted array arr, where
    arr[i] <= x < arr[i+1]
    """
    # TODO: Will not halt, or error when x is outside the array ?
    # TODO: better describe the result
    left = 0
    right = arr.size - 1
    while True:
        if (right - left) <= 1:
            return left
        i = (left + right) // 2
        if x < arr[i]:
            right = i
        else:
            left = i


@numba.jit(nopython=True)
def linear_spline(x: float, x_dat: np.ndarray, y_dat: np.ndarray) -> float:
    """
    Linear interpolation, the formulas have been taken from the slides of
    lecture 3 related to linear interpolation.
    """
    # TODO: Make version which can input an array of x
    i = find_segment(x, x_dat)
    diff = x_dat[i] - x_dat[i + 1]
    y = y_dat[i] * ((x - x_dat[i + 1]) / diff) + y_dat[i + 1] * ((x - x_dat[i]) / -diff)
    return y


@numba.jit(nopython=True)
def bisect(func: Callable, a: float, b: float, x_tol: float, max_iter=50):
    """
    Implementation adapted from Numerical Methods 3rd Edition. p449

    Note: the number of steps needed should be:
        n = log2( |a - b| / x_tol )

    Requirements:
        - func type must be known (i.e. numba.jit it!)
        - func(a) and func(b) must have opposite signs
    """

    f = func(a)
    f_mid = func(b)

    if f * f_mid >= 0.0:
        raise ValueError("f(a) and f(b) must have different signs")

    # Orient the search such that f > 0
    if f < 0.0:
        dx = b - a
        rtb = a
    else:
        dx = a - b
        rtb = b

    for _ in range(max_iter):
        dx *= 0.5
        x_mid = rtb + dx
        f_mid = func(x_mid)
        if f_mid <= 0.0:
            rtb = x_mid
        if abs(dx) < x_tol or f_mid == 0.0:
            return rtb

    raise RuntimeError("Bisect failed to converge in max number of iterations")


def integrate_pressure_mass_rk45(max_iter: int = 100):
    def func(t, y):
        return y

    h = 

    y_n = np.array([P0, 0])  # Initial conditions
    r_n = 0
    y = [y_n]
    r = [r_n]

    for i in range(max_iter):
        k1 = h * func(r_n, y_n)
        k2 = h * func(r_n + h / 2, y_n + k1 / 2)
        k3 = h * func(r_n + h / 2, y_n + k2 / 2)
        k4 = h * func(r_n + h, y_n + k3)

        # Update rule
        y_n = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        r_n += h
        y.append(y_n)
        r.append(r_n)

    return np.array(r), np.vstack(y).T  # (r, (P, M))


# Only use for initial conditions ???
def integrate_pressure_rk34(A, eps, sig0, Mtot, P0, nfw_param, h):
    """
    'Constants' needed:
    - A, eps, sig0 -> from initial fit, but after ?
    - Mtot -> to have a stopping condition
    - P0 -> A starting point
    - M0 -> M(0) = 0
    - nfw_param -> for the potential gradient
    - h -> step in r
    """
    # solve using Runge-Kutta34
    gamma = 5 / 3
    r_s, rho_s = nfw_param

    def f(r, y):
        P, M = y  # Unpack
        g = -nfw_potential_gradient_scalar_alt(r, r_s=r_s, rho_s=rho_s)
        #         print(P, M, A, (M / A)**(1/eps) + sig0)
        frac = (P / ((M / A) ** (1 / eps) + sig0)) ** (1 / gamma)
        return np.array([g * frac, 4 * np.pi * r**2 * frac])

    y_n = np.array([P0, 0])  # Initial conditions
    r_n = 0
    y = [y_n]
    r = [r_n]
    print(Mtot)
    while y_n[1] < Mtot:
        #         print(r_n, *y_n)
        k1 = h * f(r_n, y_n)
        k2 = h * f(r_n + h / 2, y_n + k1 / 2)
        k3 = h * f(r_n + h / 2, y_n + k2 / 2)
        k4 = h * f(r_n + h, y_n + k3)

        # Update rule
        y_n = y_n + (k1 + 2 * k2 + 3 * k3 + k4) / 6
        r_n += h
        y.append(y_n)
        r.append(r_n)

    return np.array(r), np.vstack(y).T  # (r, (P, M))
