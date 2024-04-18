from itertools import count
from typing import Callable

import numba
import numpy as np
import debug


@numba.jit(nopython=True, cache=True)
def relative_tolerance(x, x_new):
    """
    Calculate the relative tolerance, given the current (x_new) and previous (x)
    iteration.
    """
    return np.abs((x_new - x) / x_new)


@numba.jit(nopython=True, cache=True)
def find_root_secant(func, a, b, x_tol=1e-10, y_tol=1e-6, max_iter=50, args=None):
    """
    Find the root x of func, such that func(x) = 0, using the Secant method.

    Specify the starting interval a, b. And the function f

    Implementation taken from Press, ch9.

    Also added a y_tol, a condition which allows for stopping when y is close to
    0 at the center.

    NOTE: function must have a second argument with 'args'
    """
    # Initialize parameters
    x1 = a
    x2 = b
    x_new = 0
    fl = func(x1, args)
    f = func(x2, args)
    
    xl = x1
    rts = x2
    if abs(fl) < abs(f):
        rts = x1
        xl = x2
        fl, f = f, fl # SWAP

    for _ in range(max_iter):
        dx = (xl - rts) * f / (f - fl)
        xl = rts
        fl = f
        rts += dx
        f = func(rts, args)

        # Stop if the tolerances are low enough
        debug.log_debugv("SECANT", rts, f, "left", xl, fl)
        if abs(dx) < x_tol:
            debug.log_warn("SECANT: X tolerance reached")
            return rts
        if abs(f) < y_tol:
            debug.log_debug("SECANT: Y tolerance reached")
            return rts

    raise RuntimeError("Secant method failed to converge in max number of iterations")


@numba.jit(nopython=True, cache=True)
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



@numba.jit(nopython=True, cache=True)
def linear_spline(x: float, x_dat: np.ndarray, y_dat: np.ndarray) -> float:
    """
    Linear interpolation, the formulas have been taken from the slides of
    lecture 3 related to linear interpolation.
    """
    # TODO: Make version which can input an array of x
    if np.isnan(x):
        return np.nan
    i = find_segment(x, x_dat)
    diff = x_dat[i] - x_dat[i + 1]
    y = y_dat[i] * ((x - x_dat[i + 1]) / diff) + y_dat[i + 1] * ((x - x_dat[i]) / -diff)
    return y


    
@numba.jit(nopython=True, cache=True)
def bisect(func, a: float, b: float, x_tol:float, y_tol: float, max_iter=200, args=None):
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
        debug.log_debugv("BISECT:", dx, x_mid, f_mid)
        if abs(dx) < x_tol:
            debug.log_warn("BISECT: X tol reached")
            return rtb
        if abs(f_mid) < y_tol:
            debug.log_debug("BISECT: Y tol reached")
            return rtb

    raise RuntimeError("Bisect failed to converge in max number of iterations")
