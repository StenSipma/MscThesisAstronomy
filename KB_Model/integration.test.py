import matplotlib.pyplot as plt
import numba
import numpy as np
from integration import stepper_dopr_853

g = 9.81


@numba.jit(nopython=True)
def gravity_derivative(x, y):
    return np.array([y[1], -g])


def analytic(x, v0, x0):
    return -g / 2 * x**2 + v0 * x + x0


def main():
    v0 = 100.0
    x0 = 0.0
    lims = (0, 100)

    y0 = np.array([x0, v0])
    print("Starting....")
    xs, ys = stepper_dopr_853(x0=lims[0], x1=lims[1], h0=0.01, y0=y0, atol=1e-1, rtol=1e-1, derivs=gravity_derivative)

    x_interval = np.linspace(*lims, 100)
    y_ref = analytic(x_interval, v0, x0)
    height = [y[0] for y in ys]
    vel = [y[1] for y in ys]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xs, height)
    ax[0].plot(x_interval, y_ref, "--")

    rel_error = (height - analytic(np.array(xs), v0, x0)) / analytic(np.array(xs), v0, x0)
    ax[1].plot(xs, rel_error)
    ax[1].axhline(0, c="k", ls="--")

    plt.show()


if __name__ == "__main__":
    main()
