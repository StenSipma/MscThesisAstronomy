from itertools import count
from typing import Callable

import numba
import numpy as np

DEFAULT_DTYPE = np.double


def float_array(x, *args, dtype=DEFAULT_DTYPE, **kwargs):
    return np.array(x, *args, dtype=dtype, **kwargs)


def float_asarray(x, *args, dtype=DEFAULT_DTYPE, **kwargs):
    return np.asarray(x, *args, dtype=dtype, **kwargs)


@numba.jit(nopython=True)
def relative_tolerance(x, x_new):
    """
    Calculate the relative tolerance, given the current (x_new) and previous (x)
    iteration.
    """
    return np.abs((x_new - x) / x_new)


@numba.jit(nopython=True)
def find_root_secant(f, a, b, tol=1e-6, args=None):
    """
    Find the root x of f, such that f(x) = 0, using the Secant method.

    Specify the starting interval a, b. And the function f

    The formula for updating x is taken from Kiasulaas p.151

    NOTE: function must have a second argument with 'args'
    """
    # Initialize parameters
    x1 = a
    x2 = b
    x_new = 0

    while True:
        fx2 = f(x2, args)
        fx1 = f(x1, args)

        # Update x for the new iteration
        x_new = x2 - fx2 * ((x2 - x1) / (fx2 - fx1))

        # Calculate relative tolerance
        rtol = relative_tolerance(x2, x_new)

        # Stop if the relative tolerance is low enough
        if rtol < tol:
            break

        # Update x's for the next iteration
        x1, x2 = x2, x_new
    return x_new


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
def bisect(func: Callable, a: float, b: float, x_tol: float, max_iter=50, args=None):
    """
    Implementation adapted from Numerical Methods 3rd Edition. p449

    Note: the number of steps needed should be:
        n = log2( |a - b| / x_tol )

    Requirements:
        - func type must be known (i.e. numba.jit it!)
        - func(a) and func(b) must have opposite signs
    """

    f = func(a, args)
    f_mid = func(b, args)

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
        f_mid = func(x_mid, args)
        if f_mid <= 0.0:
            rtb = x_mid
        if abs(dx) < x_tol or f_mid == 0.0:
            return rtb

    raise RuntimeError("Bisect failed to converge in max number of iterations")
