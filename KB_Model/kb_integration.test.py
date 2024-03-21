import constants as cst
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as aconsts
from astropy import units as u
from kb_integration import integrate_hydrostatic_equilibrium


def kb_initial_conditions():
    P_inf_ref = 2.6e-11  # erg / cm3
    P_0_ref = 8.8e-10  # erg / cm3

    M_tot = 5.8e12 * cst.solmass  # g
    R_out = 230e3 * cst.pc_to_cm  # cm

    epsilon = 1.48
    s_0 = 17e31  # cm4 / g(2/3) / s2
    s_max = 13.5 * s_0

    A = M_tot / (s_max - s_0) ** epsilon
    #     A = M_tot / (s_max - s_0)**epsilon

    Omega_L0 = 0.7
    Omega_m0 = 0.3
    z = 0.0538

    dM = 1e8 * cst.solmass

    # Make sure to properly sample the low mass (< dM) range
    M_low = 10 ** np.arange(-3, np.log10(dM / cst.solmass), 0.2) * cst.solmass
    M_high = np.arange(dM, M_tot + 1, dM)

    M = np.hstack([0, M_low, M_high])
    sig = (M / A) ** (1 / epsilon) + s_0

    H0 = 65 * u.km / u.s / u.Mpc  # Value from KB

    H = H0 * np.sqrt(Omega_m0 * (1 + z) ** 3 + Omega_L0)

    # Use values from KB instead
    delta_c = 8.5e4
    r_s = 79.3e3 * cst.pc_to_cm

    rho_crit = (3 * H**2 / (8 * np.pi * aconsts.G)).cgs.value

    rho_s = rho_crit * delta_c / 4  # Does not use concentration parameter

    # Convert back to right units:
    M_tot /= 1e8 * cst.solmass
    R_out /= cst.pc_to_cm
    r_s /= cst.pc_to_cm
    rho_s /= cst.proton_mass
    P_inf_ref /= 1e-13
    P_0_ref /= 1e-13
    M /= 1e8 * cst.solmass
    sig /= 1e30
    return (M_tot, R_out, r_s, rho_s, P_inf_ref, P_0_ref, M, sig)


def run_save():
    M_tot, R_out, r_s, rho_s, P_inf_ref, P_0_ref, M, sig = kb_initial_conditions()
    results = integrate_hydrostatic_equilibrium(P_0_ref, M=M, sig=sig, r_s=r_s, rho_s=rho_s, M_tot=M_tot, P_inf=P_inf_ref)
    r, y = results
    print(r.size)
    print(y.size)
    print(r.max())

    to_save = np.vstack([r, y.T[0], y.T[1]])
    np.savetxt("./data/kb-test.dat", to_save.T)


def plot():
    r, P, M = np.loadtxt("./data/kb-test.dat").T

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(r, P)
    ax[0].set_ylabel("P [$10^{-13}$ erg / cm3]")
    ax[0].set_xlabel("r [pc]")

    ax[1].plot(r, M)
    ax[1].set_ylabel(r"M [$10^8 M_\odot$]")
    ax[1].set_xlabel("r [pc]")

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    plt.show()


def main():
    # run_save()
    plot()


if __name__ == "__main__":
    main()
