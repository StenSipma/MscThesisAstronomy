from collections.abc import Callable

import constants as cst
import num_utils
import numba
import numpy as np
from models import nfw_potential_gradient_scalar
from numpy.typing import NDArray

import debug

# Type definitions for the function to be integrated
Dargs = None # tuple[NDArray, NDArray, float, float]
Dfunc = None # Callable[[float, NDArray, Dargs], NDArray]

# Arguments needed for integration
Iargs = None #tuple[float, float, float, float, float, Dfunc, Dargs, float, float]

# List of np arrays storing the k values
Klist = None #tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
RcontList = None #tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]

DEBUG = True
from numba import optional

@numba.jit("float64[:](float64, float64[:], float64[:], float64[:], float64, float64)", nopython=True, cache=True)
def he_derivs_inner(r: float, y: NDArray, M_old, sig_old, r_s, rho_s):
    """
    Derivatives of the equation to be integrated:
        y[0] P and in the return dP/dr
        y[1] M and in the return dM/dr

    We need to pass dargs as an explicit argument because closures do not work with numba
    """
    P, M = y
#    M_old, sig_old, r_s, rho_s = dargs

    # Linear interpolation
    # print("DERIVS:", P, M)
    # print(M_old)
    # print(sig_old)
    sig = num_utils.linear_spline(M, x_dat=M_old, y_dat=sig_old)

    g = -nfw_potential_gradient_scalar(r, r_s=r_s, rho_s=rho_s)
    rho = (cst.C2 * P / sig) ** (1 / cst.gamma)
    return np.array([g * rho * cst.C4, 4.0 * np.pi * r**2 * rho * cst.C5])

@numba.jit(nopython=True, cache=True)
def he_derivs(r: float, y: NDArray, dargs: Dargs) -> NDArray:
    """
    Derivatives of the equation to be integrated:
        y[0] P and in the return dP/dr
        y[1] M and in the return dM/dr

    We need to pass dargs as an explicit argument because closures do not work with numba
    """
#    P, M = y
    M_old, sig_old, r_s, rho_s = dargs
    return he_derivs_inner(r, y, M_old, sig_old, r_s, rho_s) 

@numba.jit(nopython=True, cache=True)
def he_to_solve(P0: float, i_args: Iargs):
    x0, h0, M0, atol, rtol, derivs, dargs, M_tot, P_inf = i_args

    _, y = stepper_dopr_853_hydrostat(P0, i_args)
    # Take last iteration (-1) and then P component (0)
    P_found = y[-1][0]
    return P_found - P_inf

def print_types(**kwargs):
    for name, arg in kwargs.items():
        print(f"{name}: {type(arg)}")
    print("-------------")

def integrate_he(P0_initial: float, M, sig, r_s: float, rho_s: float, M_tot: float, P_inf: float, h0: float = 1.0, r0: float=0.0, M0: float=0.0):
    """
    Just a wrapper around stepper_dopr_853_hydrostat which is a bit more convenient to use than
    the stepper_dopr interface.
    """
    # Tolerances for the stepping algorithm
    atol = P0_initial * 1e-12
    rtol = 0.0 #np.float64(1e-8)

    dargs = (M, sig, np.float64(r_s), np.float64(rho_s))
    x0 = np.float64(r0)  # UNIT: pc
    h0 = np.float64(h0)  # UNIT: pc
    i_args = (x0, h0, np.float64(M0), atol, rtol, he_derivs, dargs, np.float64(M_tot), np.float64(P_inf))

    # Actual integration
    x, y = stepper_dopr_853_hydrostat(P0_initial, i_args)

    x_ret = np.array(x)
    y_ret = np.vstack(y)

    return x_ret, y_ret

def integrate_hydrostatic_equilibrium(P0_initial: float, M: NDArray, sig: NDArray, r_s: float, rho_s: float, M_tot: float, P_inf: float, h0: float=1.0, r0: float=0.0, M0: float=0.0):
    """
    Integrate Hydrostatic Equilibrium together with Mass Shells using a shooting method:
    to satisfy P(R(M_tot)) = P_inf & M(0) = 0
    """
    # Tolerances for the stepping algorithm
    atol = P0_initial * 1e-12
    rtol = 0.0 #np.float64(1e-8)

    dargs = (M, sig, np.float64(r_s), np.float64(rho_s))
    x0 = np.float64(r0)
    h0 = np.float64(h0)  # UNIT: pc
    i_args = (x0, h0, np.float64(M0), atol, rtol, he_derivs, dargs, np.float64(M_tot), np.float64(P_inf))
    if debug.LOG_LEVEL <= debug.DEBUG:
        print_types(x0=x0, h0=h0, M0=M0, atol=atol, rtol=rtol, he_derivs=he_derivs, dargs=dargs, M_tot=np.float64(M_tot), P_inf=np.float64(P_inf))

    # Arguments for the bisect
    bound = 100.0
    a = np.float64(P0_initial) / bound
    b = np.float64(P0_initial) * bound
    x_tol = np.float64(1e-10)  # TODO: Verify what is needed!
#     y_tol = np.float64(P_inf) / 1e6
    y_tol = np.float64(P_inf) / 1e6

    debug.log_info("Starting integration")
    if debug.LOG_LEVEL <= debug.DEBUG:
        print_types(he_to_solve=he_to_solve, args=i_args, a=a, b=b, x_tol=x_tol, y_tol=y_tol)
    P0 = num_utils.bisect(he_to_solve, args=i_args, a=a, b=b, x_tol=x_tol, y_tol=y_tol)
    x, y = stepper_dopr_853_hydrostat(P0, i_args)

    x_ret = np.array(x)
    y_ret = np.vstack(y)

    return x_ret, y_ret


@numba.jit(nopython=True, cache=True)
def interpol_mass_difference(x_guess: float, args):
    """
    Helper function which gives the difference to M_tot using the interpolated values.
    """
    x, h, M_tot, rcont = args
    return evaluate_dense(x=x_guess, xold=x, h=h, rcont=rcont)[1] - M_tot

@numba.jit(nopython=True, cache=True)
def stepper_dopr_853_hydrostat(P0: float, i_args: Iargs) -> tuple[list[float], list[NDArray]]:
    """
    Integration of Hydrostatic Equilibrium and Mass shells. Results in:
        - P(r)   (= y[0])
        - M(r)   (= y[1])
    """
    # x0: float, h0: float, M0: float, atol: float, rtol: float, derivs: Dfunc, dargs: Dargs, M_tot: float
    x0, h0, M0, atol, rtol, derivs, dargs, M_tot, P_inf = i_args
    x = x0
    y = np.array([P0, M0])
    h = h0
    errold = 1e-4

    debug.log_debug("Initial conditions: x =", x, "y = ", y)
    dydx = derivs(x, y, dargs)
    ys = [y]
    xs = [x]

    debug.log_debug("Starting the integration!")
    while True:
        debug.log_debugv("STEP:", x, h, y)
        xnew, yout, dydxnew, hnext, errold, ks = step(x=x, y=y, dydx=dydx, h_try=h, derivs=derivs, dargs=dargs, atol=atol, rtol=rtol, errold=errold)

        # Stopping condition
        if yout[1] > M_tot:
            debug.log_debug("Stopping and finding root")
#             print("BEFORE: ", x, x+h)
            rcont = prepare_dense(h=h, x=x, y=y, yout=yout, dydx=dydx, dydxnew=dydxnew, ks=ks, derivs=derivs, dargs=dargs)
#             print("RC", rcont)

            # TODO: specify x_tol, y_tol
            x_root = num_utils.find_root_secant(interpol_mass_difference, a=x, b=x + h, args=(x, h, M_tot, rcont), x_tol=1e-10, y_tol=M_tot / 1e6)
            y_ins = evaluate_dense(x=x_root, xold=x, h=h, rcont=rcont)
            #y_ins[1] = M_tot # Set the mass to this value, since that is what we are searching for.
#             print("ROOT:", x, x_root, h, "gives:", y_ins)
            
            xs.append(x_root)
            ys.append(y_ins)
            break

        if yout[0] < (P_inf / 1e3):
            debug.log_warn("Pressure low, stopping early")
            # The Pressure is too low already, just stop.
            xs.append(xnew)
            ys.append(yout)
            break

        h = hnext
        # xold = x
        x = xnew
        y = yout
        dydx = dydxnew

        ys.append(y)
        xs.append(x)

    return xs, ys


EPS = 1e-16  # TODO: specify


@numba.jit(nopython=True, cache=True)
def step(x: float, y: NDArray, dydx: NDArray, h_try: float, derivs: Dfunc, dargs: Dargs, atol: float, rtol: float, errold: float) -> tuple[float, NDArray, NDArray, float, float, Klist]:
    reject = False  # Should be False (since the previous step succeeded)

    h = h_try
    hnext = 0.0  # Some initial value ?
    while True:
        yout, yerr, yerr2, ks = dy(x=x, y=y, dydx=dydx, h=h, derivs=derivs, dargs=dargs)
        err = error(h=h, atol=atol, rtol=rtol, yerr=yerr, yerr2=yerr2, y=y, yout=yout)
        success, h, hnext, errold, reject = controller_success(err, errold, h, reject)
        debug.log_debugvv("INDv. STEP:", success, h, hnext, errold, reject, x,  y)
        if success:
            break
        if np.isnan(h): 
            raise ValueError("stepsize is NAN, exiting...")
        if np.abs(h) <= np.abs(x) * EPS:
            raise ValueError("stepsize underflow in integration")
    dydxnew = derivs(x + h, yout, dargs)
    # Prepare dense if needed ?
    # Maybe store the new h:  hdid = h ?

    # Return the 'new': x, xold, y, dydx

    return x + h, yout, dydxnew, hnext, errold, ks


@numba.jit(nopython=True, cache=True)
def controller_success(err, errold, h, reject) -> tuple[bool, float, float, float, bool]:
    # Maybe specify these yourself ?
    beta = 0.0
    alpha = 1.0 / 8.0 - beta * 0.2
    safe = 0.9
    minscale = 0.333
    maxscale = 6.0

    scale = 0.0  # Initialize
    if err <= 1.0:
        if err == 0.0:
            scale = maxscale
        else:
            scale = safe * pow(err, -alpha) * pow(errold, beta)
            if scale < minscale:
                scale = minscale
            elif scale > maxscale:
                scale = maxscale
        if reject:
            hnext = h * min(scale, 1.0)
        else:
            hnext = h * scale
        errold = max(err, 1e-4)  # ???????????
        reject = False
        return True, h, hnext, errold, reject

    scale = max(safe * pow(err, -alpha), minscale)
    h *= scale
    reject = True

    # Pass 0.0 as hnext to satisfy the type system
    return False, h, 0.0, errold, reject


@numba.jit(nopython=True, cache=True)
def error(h: float, atol: float, rtol: float, yerr: NDArray, yerr2: NDArray, y: NDArray, yout: NDArray):
    sk = atol + rtol * np.maximum(np.abs(y), np.abs(yout))
    err2 = np.sum((yerr2 / sk) ** 2)
    err = np.sum((yerr / sk) ** 2)

    denom = err + 0.01 * err2
    if denom <= 0.0:
        denom = 1.0

    n = yerr.size
    return np.abs(h) * err * np.sqrt(1.0 / (n * denom))


@numba.jit(nopython=True, cache=True)
def dy(x: float, y: NDArray, dydx: NDArray, h: float, dargs: Dargs, derivs: Dfunc) -> tuple[NDArray, NDArray, NDArray, Klist]:
    k2 = derivs(x + c2 * h, y + h * a21 * dydx, dargs)
    k3 = derivs(x + c3 * h, y + h * (a31 * dydx + a32 * k2), dargs)
    k4 = derivs(x + c4 * h, y + h * (a41 * dydx + a43 * k3), dargs)
    k5 = derivs(x + c5 * h, y + h * (a51 * dydx + a53 * k3 + a54 * k4), dargs)
    k6 = derivs(x + c6 * h, y + h * (a61 * dydx + a64 * k4 + a65 * k5), dargs)
    k7 = derivs(x + c7 * h, y + h * (a71 * dydx + a74 * k4 + a75 * k5 + a76 * k6), dargs)
    k8 = derivs(x + c8 * h, y + h * (a81 * dydx + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7), dargs)
    k9 = derivs(x + c9 * h, y + h * (a91 * dydx + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8), dargs)
    k10 = derivs(x + c10 * h, y + h * (a101 * dydx + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7 + a108 * k8 + a109 * k9), dargs)
    k2 = derivs(x + c11 * h, y + h * (a111 * dydx + a114 * k4 + a115 * k5 + a116 * k6 + a117 * k7 + a118 * k8 + a119 * k9 + a1110 * k10), dargs)

    xph = x + h
    k3 = derivs(xph, y + h * (a121 * dydx + a124 * k4 + a125 * k5 + a126 * k6 + a127 * k7 + a128 * k8 + a129 * k9 + a1210 * k10 + a1211 * k2), dargs)
    k4 = b1 * dydx + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k2 + b12 * k3
    yout = y + h * k4

    # Two error estimators.
    yerr = k4 - bhh1 * dydx - bhh2 * k9 - bhh3 * k3
    yerr2 = er1 * dydx + er6 * k6 + er7 * k7 + er8 * k8 + er9 * k9 + er10 * k10 + er11 * k2 + er12 * k3
    ks = (k2, k3, k4, k5, k6, k7, k8, k9, k10)
    return yout, yerr, yerr2, ks


@numba.jit(nopython=True, cache=True)
def prepare_dense(h: float, x: float, y: NDArray, yout: NDArray, dydx: NDArray, dydxnew: NDArray, ks: Klist, derivs: Dfunc, dargs: Dargs) -> RcontList:
    k2, k3, k4, k5, k6, k7, k8, k9, k10 = ks
    # Prepare stuff
    rcont1 = y
    ydiff = yout - y
    rcont2 = ydiff
    bspl = h * dydx - ydiff
    rcont3 = bspl
    rcont4 = ydiff - h * dydxnew - bspl
    rcont5 = d41 * dydx + d46 * k6 + d47 * k7 + d48 * k8 + d49 * k9 + d410 * k10 + d411 * k2 + d412 * k3
    rcont6 = d51 * dydx + d56 * k6 + d57 * k7 + d58 * k8 + d59 * k9 + d510 * k10 + d511 * k2 + d512 * k3
    rcont7 = d61 * dydx + d66 * k6 + d67 * k7 + d68 * k8 + d69 * k9 + d610 * k10 + d611 * k2 + d612 * k3
    rcont8 = d71 * dydx + d76 * k6 + d77 * k7 + d78 * k8 + d79 * k9 + d710 * k10 + d711 * k2 + d712 * k3

    k10 = derivs(x + c14 * h, y + h * (a141 * dydx + a147 * k7 + a148 * k8 + a149 * k9 + a1410 * k10 + a1411 * k2 + a1412 * k3 + a1413 * dydxnew), dargs)
    k2 = derivs(x + c15 * h, y + h * (a151 * dydx + a156 * k6 + a157 * k7 + a158 * k8 + a1511 * k2 + a1512 * k3 + a1513 * dydxnew + a1514 * k10), dargs)
    k3 = derivs(x + c16 * h, y + h * (a161 * dydx + a166 * k6 + a167 * k7 + a168 * k8 + a169 * k9 + a1613 * dydxnew + a1614 * k10 + a1615 * k2), dargs)

    rcont5 = h * (rcont5 + d413 * dydxnew + d414 * k10 + d415 * k2 + d416 * k3)
    rcont6 = h * (rcont6 + d513 * dydxnew + d514 * k10 + d515 * k2 + d516 * k3)
    rcont7 = h * (rcont7 + d613 * dydxnew + d614 * k10 + d615 * k2 + d616 * k3)
    rcont8 = h * (rcont8 + d713 * dydxnew + d714 * k10 + d715 * k2 + d716 * k3)

    return rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8


@numba.jit(nopython=True, cache=True)
def evaluate_dense(x: float, xold: float, h: float, rcont: RcontList) -> NDArray:
    """
    Evaluate a the cubic polynomial at x (xold <= x <= xold + h) for y[i]
    """
    rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8 = rcont

    s = (x - xold) / h
    s1 = 1.0 - s
    return rcont1 + s * (rcont2 + s1 * (rcont3 + s * (rcont4 + s1 * (rcont5 + s * (rcont6 + s1 * (rcont7 + s * rcont8))))))


######################## CONSTANTS for integration ##############################
c2 = 0.526001519587677318785587544488e-01
c3 = 0.789002279381515978178381316732e-01
c4 = 0.118350341907227396726757197510e00
c5 = 0.281649658092772603273242802490e00
c6 = 0.333333333333333333333333333333e00
c7 = 0.25e00
c8 = 0.307692307692307692307692307692e00
c9 = 0.651282051282051282051282051282e00
c10 = 0.6e00
c11 = 0.857142857142857142857142857142e00
c14 = 0.1e00
c15 = 0.2e00
c16 = 0.777777777777777777777777777778e00
b1 = 5.42937341165687622380535766363e-2
b6 = 4.45031289275240888144113950566e0
b7 = 1.89151789931450038304281599044e0
b8 = -5.8012039600105847814672114227e0
b9 = 3.1116436695781989440891606237e-1
b10 = -1.52160949662516078556178806805e-1
b11 = 2.01365400804030348374776537501e-1
b12 = 4.47106157277725905176885569043e-2
bhh1 = 0.244094488188976377952755905512e00
bhh2 = 0.733846688281611857341361741547e00
bhh3 = 0.220588235294117647058823529412e-01
er1 = 0.1312004499419488073250102996e-01
er6 = -0.1225156446376204440720569753e01
er7 = -0.4957589496572501915214079952e00
er8 = 0.1664377182454986536961530415e01
er9 = -0.3503288487499736816886487290e00
er10 = 0.3341791187130174790297318841e00
er11 = 0.8192320648511571246570742613e-01
er12 = -0.2235530786388629525884427845e-01
a21 = 5.26001519587677318785587544488e-2
a31 = 1.97250569845378994544595329183e-2
a32 = 5.91751709536136983633785987549e-2
a41 = 2.95875854768068491816892993775e-2
a43 = 8.87627564304205475450678981324e-2
a51 = 2.41365134159266685502369798665e-1
a53 = -8.84549479328286085344864962717e-1
a54 = 9.24834003261792003115737966543e-1
a61 = 3.7037037037037037037037037037e-2
a64 = 1.70828608729473871279604482173e-1
a65 = 1.25467687566822425016691814123e-1
a71 = 3.7109375e-2
a74 = 1.70252211019544039314978060272e-1
a75 = 6.02165389804559606850219397283e-2
a76 = -1.7578125e-2
a81 = 3.70920001185047927108779319836e-2
a84 = 1.70383925712239993810214054705e-1
a85 = 1.07262030446373284651809199168e-1
a86 = -1.53194377486244017527936158236e-2
a87 = 8.27378916381402288758473766002e-3
a91 = 6.24110958716075717114429577812e-1
a94 = -3.36089262944694129406857109825e0
a95 = -8.68219346841726006818189891453e-1
a96 = 2.75920996994467083049415600797e1
a97 = 2.01540675504778934086186788979e1
a98 = -4.34898841810699588477366255144e1
a101 = 4.77662536438264365890433908527e-1
a104 = -2.48811461997166764192642586468e0
a105 = -5.90290826836842996371446475743e-1
a106 = 2.12300514481811942347288949897e1
a107 = 1.52792336328824235832596922938e1
a108 = -3.32882109689848629194453265587e1
a109 = -2.03312017085086261358222928593e-2
a111 = -9.3714243008598732571704021658e-1
a114 = 5.18637242884406370830023853209e0
a115 = 1.09143734899672957818500254654e0
a116 = -8.14978701074692612513997267357e0
a117 = -1.85200656599969598641566180701e1
a118 = 2.27394870993505042818970056734e1
a119 = 2.49360555267965238987089396762e0
a1110 = -3.0467644718982195003823669022e0
a121 = 2.27331014751653820792359768449e0
a124 = -1.05344954667372501984066689879e1
a125 = -2.00087205822486249909675718444e0
a126 = -1.79589318631187989172765950534e1
a127 = 2.79488845294199600508499808837e1
a128 = -2.85899827713502369474065508674e0
a129 = -8.87285693353062954433549289258e0
a1210 = 1.23605671757943030647266201528e1
a1211 = 6.43392746015763530355970484046e-1
a141 = 5.61675022830479523392909219681e-2
a147 = 2.53500210216624811088794765333e-1
a148 = -2.46239037470802489917441475441e-1
a149 = -1.24191423263816360469010140626e-1
a1410 = 1.5329179827876569731206322685e-1
a1411 = 8.20105229563468988491666602057e-3
a1412 = 7.56789766054569976138603589584e-3
a1413 = -8.298e-3
a151 = 3.18346481635021405060768473261e-2
a156 = 2.83009096723667755288322961402e-2
a157 = 5.35419883074385676223797384372e-2
a158 = -5.49237485713909884646569340306e-2
a1511 = -1.08347328697249322858509316994e-4
a1512 = 3.82571090835658412954920192323e-4
a1513 = -3.40465008687404560802977114492e-4
a1514 = 1.41312443674632500278074618366e-1
a161 = -4.28896301583791923408573538692e-1
a166 = -4.69762141536116384314449447206e0
a167 = 7.68342119606259904184240953878e0
a168 = 4.06898981839711007970213554331e0
a169 = 3.56727187455281109270669543021e-1
a1613 = -1.39902416515901462129418009734e-3
a1614 = 2.9475147891527723389556272149e0
a1615 = -9.15095847217987001081870187138e0
d41 = -0.84289382761090128651353491142e01
d46 = 0.56671495351937776962531783590e00
d47 = -0.30689499459498916912797304727e01
d48 = 0.23846676565120698287728149680e01
d49 = 0.21170345824450282767155149946e01
d410 = -0.87139158377797299206789907490e00
d411 = 0.22404374302607882758541771650e01
d412 = 0.63157877876946881815570249290e00
d413 = -0.88990336451333310820698117400e-01
d414 = 0.18148505520854727256656404962e02
d415 = -0.91946323924783554000451984436e01
d416 = -0.44360363875948939664310572000e01
d51 = 0.10427508642579134603413151009e02
d56 = 0.24228349177525818288430175319e03
d57 = 0.16520045171727028198505394887e03
d58 = -0.37454675472269020279518312152e03
d59 = -0.22113666853125306036270938578e02
d510 = 0.77334326684722638389603898808e01
d511 = -0.30674084731089398182061213626e02
d512 = -0.93321305264302278729567221706e01
d513 = 0.15697238121770843886131091075e02
d514 = -0.31139403219565177677282850411e02
d515 = -0.93529243588444783865713862664e01
d516 = 0.35816841486394083752465898540e02
d61 = 0.19985053242002433820987653617e02
d66 = -0.38703730874935176555105901742e03
d67 = -0.18917813819516756882830838328e03
d68 = 0.52780815920542364900561016686e03
d69 = -0.11573902539959630126141871134e02
d610 = 0.68812326946963000169666922661e01
d611 = -0.10006050966910838403183860980e01
d612 = 0.77771377980534432092869265740e00
d613 = -0.27782057523535084065932004339e01
d614 = -0.60196695231264120758267380846e02
d615 = 0.84320405506677161018159903784e02
d616 = 0.11992291136182789328035130030e02
d71 = -0.25693933462703749003312586129e02
d76 = -0.15418974869023643374053993627e03
d77 = -0.23152937917604549567536039109e03
d78 = 0.35763911791061412378285349910e03
d79 = 0.93405324183624310003907691704e02
d710 = -0.37458323136451633156875139351e02
d711 = 0.10409964950896230045147246184e03
d712 = 0.29840293426660503123344363579e02
d713 = -0.43533456590011143754432175058e02
d714 = 0.96324553959188282948394950600e02
d715 = -0.39177261675615439165231486172e02
d716 = -0.14972683625798562581422125276e03
