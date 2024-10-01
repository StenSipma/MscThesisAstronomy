#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
from astropy import units as u
from astropy import constants as aconsts
from scipy.interpolate import interp1d


# ## Util

# In[2]:


# We need to add the path to the myutils.py file, in order to import it
from pathlib import Path
import sys
module_location = Path('/net/virgo01/data/users/sipma/MscThesis/MscThesisAstronomy/KB_Model')
sys.path.append(str(module_location))


# In[3]:


# Custom imports
from dataclasses import dataclass
from misc import Parameters, compute_quantities
from models import beta_model, nfw_potential_gradient, nfw_potential_gradient_scalar
import constants as cst

from caching import Cache
# Init
cache = Cache(Path.home() / 'dataserver' / 'MscThesis' / 'data' / 'cache')


# In[4]:


def save_name(fname, ext="pdf", path=None):
    if path is None:
        path = Path("./figures/milkyway")
    return path / f"{fname}.{ext}"

def savefig(fig, name, ext="png", dpi=300, *args, **kwargs):
    sname = save_name(name, ext=ext)
    fig.savefig(sname, dpi=dpi)


# ### Cooling Functions

# In[5]:


from cooling_func import CoolingFunc
cooling_sd = CoolingFunc(data_location="/net/virgo01/data/users/sipma/data/SutherlandCoolingFunctions")


# FeH -> log10(0.2)
def cooling_f_sd(T, n_e, FeH=-0.6989700043360187):
    T_in = np.log10(T * cst.TEMP_UNIT)  # Convert to log & set to correct unit
    L = cooling_sd(FeH, T_in)[0]
    return 10**L


# In[6]:


from MscThesisAstronomy.Cloudy.custom_read import CloudyCoolingFunc
cooling_cloudy = CloudyCoolingFunc(data_location=Path("MscThesisAstronomy/Cloudy/FinalTable.txt"), interp_method='pchip')

## Maps the metallicity to the n_H = a x rho relation (the number here is a)
## input is metallicity in relative solar metallicity
DENSITY_NH_MAP = {
    0.05 : 1.321933959813,
    0.10 : 1.327257504256,
    0.20 : 1.346317884938,
    0.30 : 1.367010767026,
    0.40 : 1.377914209247,
}


def cooling_f_cloudy(T, n_H, Z=0.2):
    """
    Returns cooling - heating for the cloudy cooling func
    
    T: not in log space
    n_H: not in log space
    Z: nog in log space, relative to solar metallicity (= Z*Z_sol)
    """
    T_in = T * cst.TEMP_UNIT  # Convert to correct unit
    
    # Convert to correct shape (if not in correct shape)
    n_H_in = np.broadcast_to(np.array(n_H), T.shape)
    
    # Convert to correct shape
    Z_in = np.broadcast_to(np.array(Z), T.shape)
    
    heat = cooling_cloudy.heating(n_H_in, Z_in, T_in)
    cool = cooling_cloudy.cooling(n_H_in, Z_in, T_in)
    return cool - heat, heat, cool


# ## Initial Conditions

# ### Milky Way

# In[7]:


from scipy.integrate import cumulative_trapezoid, trapezoid


def mw_initial_conditions(R_ref=None, R_out=None, beta_param=None, T_ref=None, electron=True):
    ## For the NFW profile
    M_vir = 1.3e12 * cst.solmass
    r_vir = 282e3 * cst.pc_to_cm
    c_vir = 10

    r_s = r_vir / c_vir

    rho_s = M_vir / (16 * np.pi * r_s**3 * (np.log(1 + c_vir) - c_vir /
                                            (1 + c_vir)))
    NFW_param_mw = (r_s, rho_s)

    if T_ref is None:
        T_ref = 10**6.3  # Kelvin

    if R_out is None:
        R_out = r_vir

    if R_ref is None:
        R_out0 = R_out
    else:
        R_out0 = R_ref

    if beta_param:
        n_0, r_c, beta = beta_param
    else:
        n_0 = 2.6e-3  # cm-3
        r_c = 3.0  # kpc
        beta = 0.5

    dR = 1 * cst.pc_to_cm  # pc

    ## Density profile
    # Make spacing better
    #     M_low = 10**np.arange(-3, np.log10(dM / solmass), .2) * solmass
    #     r = np.linspace(0, R_out, 10000)
    r1 = 10**np.arange(-4, np.log10(dR / cst.pc_to_cm), .2) * cst.pc_to_cm
    r2 = np.arange(dR, R_out0 + 1, dR)
    r = np.append(r1, r2)

    n_e = beta_model(r / cst.pc_to_cm / 1e3, n0=n_0, rc=r_c, beta=beta)

    #     T_inf = 10**6.3 # Kelvin

    n_frac = 11 / 6  # Conversion of n = frac * n_e
    mu = 0.62
    if electron:
        n = n_frac * n_e
    else:
        n = n_e
    rho = mu * cst.proton_mass * n

    # Calculate based on a constant T profile and given density
    P_ref = n[-1] * cst.k_B * T_ref

    # Find optimal P_0
    #     g = np.array([-nfw_potential_gradient_scalar(i, r_s=r_s, rho_s=rho_s) for i in r])
    g = -nfw_potential_gradient(r, r_s=r_s, rho_s=rho_s)
    rhs_P = g * rho

    P_guess = cumulative_trapezoid(rhs_P, r, initial=0)
    P_0 = P_ref - P_guess[-1]
    P = P_guess + P_0

    P_0_ref = P_0

    ### Now we have P_0, so we can integrate further to get P_inf at an actual outer radius
    ### we also integrate Mass now

    r1 = 10**np.arange(-4, np.log10(dR / cst.pc_to_cm), .2) * cst.pc_to_cm
    r2 = np.arange(dR, R_out + 1, dR)
    r = np.append(r1, r2)

    n_e = beta_model(r / cst.pc_to_cm / 1e3, n0=n_0, rc=r_c, beta=beta)
    if electron:
        n = n_frac * n_e
    else:
        n = n_e
    rho = mu * cst.proton_mass * n

    # Integrate rho (TODO: integration can also be analytic?)
    rhs_M = 4 * np.pi * r**2 * rho
    M = cumulative_trapezoid(rhs_M, r, initial=0)
    M_tot = M[-1]

    # Integrate P (with known P_0)
    #     g = np.array([-nfw_potential_gradient_scalar(i, r_s=r_s, rho_s=rho_s) for i in r])
    g = -nfw_potential_gradient(r, r_s=r_s, rho_s=rho_s)
    rhs_P = g * rho
    P = cumulative_trapezoid(rhs_P, r, initial=0) + P_0_ref

    P_inf_ref = P[-1]
    sig = P / rho**cst.gamma

    # Convert back to right units:
    M_tot /= cst.MASS_UNIT
    R_out /= cst.LEN_UNIT
    r_s /= cst.LEN_UNIT
    rho_s /= cst.DENS_UNIT
    P_inf_ref /= cst.PRES_UNIT
    P_0_ref /= cst.PRES_UNIT
    M /= cst.MASS_UNIT
    sig /= cst.SIG_UNIT

    P /= cst.PRES_UNIT
    n_e /= cst.NDENS_UNIT
    r /= cst.LEN_UNIT

    return P, M, n_e, r, Parameters(M_tot=M_tot,
                                    R_out=R_out,
                                    r_s=r_s,
                                    rho_s=rho_s,
                                    P_inf=P_inf_ref,
                                    P_0=P_0_ref,
                                    M=M,
                                    sig=sig,
                                    r=r)


# In[8]:


# Initial conditions:

def ic_mb2016():
    rc = 2.12 # kpc
    beta = 0.49
    C = 0.0135
    n0 = C / (rc ** (3*beta)) # cm-3

    Z = 0.3 # Zsol
    return (n0, rc, beta), Z

def ic_mb2016_rescaled():
    scale = 0.3 / 0.09 # To get  M_cgm / (Mb - Mdisk) ~ 0.3

    rc = 2.12 # kpc
    beta = 0.49
    C = scale*0.0135

    n0 = C / (rc ** (3*beta)) # cm-3
    
    Z = 0.1 # Zsol
    return (n0, rc, beta), Z


# ## Integration

# In[9]:


from kb_integration import integrate_hydrostatic_equilibrium


# ## Convection

# In[10]:


import numba


@numba.jit(nopython=True)
def equalize_entropy(sig, T, M):
    sig = sig.copy()
    restart = True
    Mshell = M[1:] - M[:-1]
    Mshell = np.append([0], Mshell)
    while restart:  # Keep looping until everything is flat.
        restart = False
        s_prev = sig[0]
        for i, s in enumerate(sig[1:], 1):
            if s_prev > s:
                # Instabiel, zei de kat
                start = i - 1

                # Sum variables
                sTM = 0
                TM = 0
                s_prev_inner = s_prev + 1
                for j, s_inner in enumerate(sig[start:], start):
                    #                     if s_inner > s_prev:
                    if s_inner > s_prev_inner:
                        end = j
                        break
                    sTM += np.log(s_inner) * T[j] * Mshell[j]
                    TM += T[j] * Mshell[j]

                    s_prev_inner = s_inner
                else:
                    end = len(sig)
                smean = sTM / TM  # = ln (sig_mean), so exponentiate later

                sig[start:end] = np.exp(smean)

                # Restart the loop
                restart = True
                break

            s_prev = s
    return sig


def apply_convection(r, P, M, sig, param, **kwargs):
    rho, n_e, T = compute_quantities(r, P, M, sig, mu=0.62)

    sig_new = equalize_entropy(sig, T, M)
    if any(sig_new[:-1] > sig_new[1:]):
        print("WARN: Negative gradient after convection calc")

    sig_func = interp1d(M, sig_new, kind="linear", fill_value="extrapolate")
    r, y = integrate_hydrostatic_equilibrium(
        P[0],
        M=M,
        sig=sig_new,
        r_s=param.r_s,
        rho_s=param.rho_s,
        M_tot=param.M_tot,
        P_inf=param.P_inf,
        **kwargs,
    )
    P, M = y.T

    # Updates sigma + updated masses
    sig_new = sig_func(M)
    if any(sig_new[:-1] > sig_new[1:]):
        print("WARN: Negative gradient after H.E. integration")

    return r, P, M, sig_new


# ## Evolution

# ### Updating Entropy

# In[11]:


def update_entropy_sd(r, P, M, sig, temperature_limit, dt_lower_factor, dt, Z):
    rho, n_e, T = compute_quantities(r, P, M, sig)

    # Apply a limit to the cooling
#     lamb = np.zeros_like(T)

    idx_good = T > temperature_limit
    lamb = cooling_f_sd(T[idx_good], None, FeH=np.log10(Z))

    # Calculate the cooling time for cells above the temperature limit.
    # Cells above will be set to inf (because cooling = 0)
    t_cool = np.full_like(sig, np.inf)
    t_cool[idx_good] = (3 / 2) * 2.2 / n_e[idx_good] * T[idx_good] / lamb * cst.C3
    min_t_cool = t_cool.min()

    if min_t_cool / dt_lower_factor <= dt:
        dt = min_t_cool / dt_lower_factor  # Can refine this guess ?

    sig_new = sig*(1 - dt / t_cool)

    return sig_new, dt, t_cool


# In[12]:


def update_entropy_cloudy(r, P, M, sig, temperature_limit, dt_lower_factor, dt, Z, onlycool=False):
    rho, n_e, T = compute_quantities(r, P, M, sig, mu=0.62)

    # Apply a limit to the cooling
    a = DENSITY_NH_MAP[Z]  # From relation between n_H and rho for different Z
    n_H = rho / a
    lamb, heat, cool, = cooling_f_cloudy(T, n_H, Z=Z)
    if onlycool:
        lamb = cool
    
    mu = 0.62 # Use same value as used in compute_quantities
    n = rho / mu # TODO: check if this is correct (since we use rho differently before)
    
    # Calculate cooling time (used for updating and checking lower limit)
    # Can be inf if the cooling is below the temperature limit, so if
    # used in the update, set those components to 0 again.
    t_cool = (3 / 2) * n * T / lamb * cst.C3
#     min_t_cool = t_cool[t_cool > 0].min()
    min_t_cool = np.abs(t_cool).min()

    if min_t_cool / dt_lower_factor <= dt:
        dt = min_t_cool / dt_lower_factor  # Can refine this guess ?

    sig = sig*(1 - dt / t_cool)
    return sig, dt, t_cool


# In[13]:


def gaussian_3d(r, sigma, norm=1):
    return norm * (2 * np.pi * sigma**2)**(-3 / 2) * np.exp(-r**2 / (2 * sigma**2))


def sech(x):
    return 1 / np.cosh(x)


def sech2_3d(r, sigma, norm=1):
    return 3 * norm / (np.pi**3 * sigma**3) * sech(r / sigma)**2


def fermi_bubble_heat(r, scale=3e3, E_tot=1e39, kernel_func=gaussian_3d):
    # Scale: units of pc
    # E_tot: units of erg / s
    # r    : units of pc
    kernel = kernel_func(r, sigma=scale, norm=1)  # unit in 1 / pc3
    kernel /= (cst.pc_to_cm)**3  # Convert 1/pc3 to 1/cm3
    return E_tot * kernel  # Unit erg / s / cm3


def update_entropy_bubble(r,
                          P,
                          M,
                          sig,
                          temperature_limit,
                          dt_lower_factor,
                          dt,
                          Z,
                          fb_scale=3e3,
                          fb_heat=1e39,
                          kernel_func=gaussian_3d):
    rho, n_e, T = compute_quantities(r, P, M, sig, mu=0.62)

    # Apply a limit to the cooling
    a = DENSITY_NH_MAP[Z]  # From relation between n_H and rho for different Z
    n_H = rho / a
    lamb, _, _, = cooling_f_cloudy(T, n_H, Z=Z)

    # Also add fermi bubble heating
    fb_heat = fermi_bubble_heat(r, scale=fb_scale, E_tot=fb_heat, kernel_func=kernel_func)
    lamb -= fb_heat

    mu = 0.62  # Use same value as used in compute_quantities
    n = rho / mu  # TODO: check if this is correct (since we use rho differently before)

    # Calculate cooling time (used for updating and checking lower limit)
    # Can be inf if the cooling is below the temperature limit, so if
    # used in the update, set those components to 0 again.
    t_cool = (3 / 2) * n * T / lamb * cst.C3
    #     min_t_cool = t_cool[t_cool > 0].min()
    min_t_cool = np.abs(t_cool).min()

    if min_t_cool / dt_lower_factor <= dt:
        dt = min_t_cool / dt_lower_factor  # Can refine this guess ?

    sig = sig * (1 - dt / t_cool)
    return sig, dt, t_cool


# In[14]:


def update_entropy_stopcool(r, P, M, sig, temperature_limit, dt_lower_factor, dt, Z):
    rho, n_e, T = compute_quantities(r, P, M, sig)

    R_LIMIT = 1e3
    # Apply a limit to the cooling
    idx_good = r > R_LIMIT
    a = DENSITY_NH_MAP[Z]  # From relation between n_H and rho for different Z
    n_H = rho / a
    lamb, _, _, = cooling_f_cloudy(T[idx_good], n_H[idx_good], Z=Z)

    mu = 0.62 # Use same value as used in compute_quantities
    n = rho / mu # TODO: check if this is correct (since we use rho differently before)
    
    # Calculate the cooling time for cells above the temperature limit.
    # Cells above will be set to inf (because cooling = 0)
    t_cool = np.full_like(sig, np.inf)
    t_cool[idx_good] = (3 / 2) * n[idx_good] * T[idx_good] / lamb * cst.C3
    min_t_cool = t_cool.min()

    if min_t_cool / dt_lower_factor <= dt:
        dt = min_t_cool / dt_lower_factor  # Can refine this guess ?

    sig_new = sig*(1 - dt / t_cool)

    return sig_new, dt, t_cool


# In[15]:


import traceback
def print_types(**kwargs):
    for name, arg in kwargs.items():
        print(f"{name}: {type(arg)}")
    print("-------------")


# ### Adaptive Evolution

# In[16]:


def evolve_adaptive(
    update_entropy,
    dt_start,
    t_max,
    metallicity,
    param: Parameters = None,
    max_niter=40,
    temperature_limit=None,
    dt_lower_factor=2,
    dt_limit=0.01, # TODO: see if this is a good limit?
    convection=False,
    r_inner=0.0,
    entropy_kwargs=None,
):
    entropy_kwargs = {} if entropy_kwargs is None else entropy_kwargs
    
    # TODO: add docstring
    M0 = param.M
    sig0 = param.sig
    r_s, rho_s = param.r_s, param.rho_s
    P_inf_ref = param.P_inf
    P_0_guess = param.P_0
    R_out = param.R_out
    M_tot = param.M_tot
    
    ## Boundary conditions at r > 0, will also give a nonzero mass
    M_inner = 0.0
    if r_inner > 0.0:
        M_func = interp1d(param.r, param.M, kind="linear")
        M_inner = float(M_func(r_inner))
        print(f"Integrating from nonzero r: {r_inner}, found mass: {M_inner} 10^8 Mo")

    sig_func = interp1d(M0, sig0, kind="linear", fill_value="extrapolate")
    r, y = integrate_hydrostatic_equilibrium(
        P_0_guess,
        M=M0,
        sig=sig0,
        r_s=r_s,
        rho_s=rho_s,
        M_tot=M_tot,
        P_inf=P_inf_ref,
        r0=r_inner,
        M0=M_inner,
    )
    P, M = y.T

    sig = sig_func(M)

    if convection:
        print("Applying convection")
        r, P, M, sig = apply_convection(r, P, M, sig, param, r0=r_inner, M0=M_inner)

    # Store initial values
    radii = [r]
    pressures = [P]
    masses = [M]
    sigmas = [sig]
    # Empty first, will be added at the start of the first iteration.
    t_cools = []

    dt = dt_start
    t = 0.
    time_steps = [t]

    niter = 0

    try:
        while niter < max_niter and t < t_max:
            print(f"Iteration {niter:>3d}, at {t:g} Myr")

            sig, dt, t_cool = update_entropy(r, P, M, sig, temperature_limit,
                                             dt_lower_factor, dt, metallicity, **entropy_kwargs)
            t_cools.append(t_cool)

            if dt < dt_limit:
                print(f"Lower limit for timestep reached: {dt} Myr < {dt_limit} Myr. Stopping evolution.")
                break

            if np.any(sig < 0):
                print("Negative entropy found stopping evolution")
                raise ValueError()

            ## To resynchronize later
            sig_func = interp1d(M,
                                sig,
                                kind="linear",
                                fill_value="extrapolate")
            # Updated sigma, old masses
            P_0_guess = P[0]  #* 1e1
            r, y = integrate_hydrostatic_equilibrium(
                P_0_guess,
                M=M,
                sig=sig,
                r_s=r_s,
                rho_s=rho_s,
                M_tot=M_tot,
                P_inf=P_inf_ref,
                r0=r_inner,
                M0=M_inner,
            )
            P, M = y.T

            # Updated sigma + updated masses
            sig = sig_func(M)

            if convection:
                print("Applying convection")
                r, P, M, sig = apply_convection(r, P, M, sig, param, r0=r_inner, M0=M_inner)

            # Update time
            t += dt
            dt = dt_start  # reset
            niter += 1

            sigmas.append(sig)
            radii.append(r)
            masses.append(M)
            pressures.append(P)
            time_steps.append(t)
        else:
            if not niter < max_niter:
                print(f"\nReached max number of iterations: {max_niter}")
            elif not t < t_max:
                print(f"\nReached max time: {t_max} Myr")
            else:
                # Should not happen...
                print("\nLoop exited for unknown reason")
    except Exception as e:
        print(
            f"Evolution failed at time {t:.2f} Myr with exception {e}                  "
        )
        traceback.print_exc()

    print("\nDone")

    #     return {'time': np.array(time_steps), 'radius': radii, 'pressure': pressures, 'mass': masses, 'sig':sigmas, 't_cool': t_cools}
    return np.array(time_steps), radii, pressures, masses, sigmas, t_cools


# ## Execution

# ### Test
# Also partly to let the Numba routines compile once.

# In[17]:


from time import process_time

TEMPERATURE_LIM = 10**(4.2) / cst.TEMP_UNIT


# In[18]:


from multiprocessing import Pool


def evolve_test(cache,
                     name,
                     entropy_func,
                     *args,
                     dt=1., # Myr
                     t_max=1000., # Myr
                     max_niter=200,
                     dt_lower_factor=10.,
                     convection=True,
                     entropy_kwargs=None,
                     r_inner=0.,
                     **kwargs):

#     @cache.named(name)
    def func():
        print("Running", name)
        start = process_time()

        beta_param, Z = ic_mb2016()
        *_, param = mw_initial_conditions(beta_param=beta_param)


        # Limit in temperature unit (1 / 1e6)
        TEMPERATURE_LIM = 10**(4.2) / cst.TEMP_UNIT

        print(f"Running with {dt = }")
        evolve_result = evolve_adaptive(
            param=param,
            update_entropy=entropy_func,
            dt_start=dt,
            t_max=t_max,
            max_niter=max_niter,
            metallicity=Z,
            temperature_limit=TEMPERATURE_LIM,
            dt_lower_factor=dt_lower_factor,
            convection=convection,
            entropy_kwargs=entropy_kwargs,
            r_inner=r_inner,
            #                 r_inner=1.0e3,
            #                 entropy_kwargs={
            #                     'fb_scale': 3e3,
            #                     'fb_heat': 1e40,
            #                     'kernel_func': sech2_3d
            #                 },
        )

        duration = process_time() - start
        print("-------------------------------------------")
        print(f"Process finished in {duration:.1f} seconds. ")
        print(f": {duration / 60:.1f} minutes")
        print(f": {duration / 3600:.1f} hours")

        return param, evolve_result

    return func(*args, **kwargs)


# In[19]:


def mw_adaptive_test(*args, **kwargs):
    entropy_funcs = [ update_entropy_sd]
    names = [f"MW-TEST-1"]
    dts = [0.1]

    evolutions = {}

    for name, entropy_func, dt in zip(names, entropy_funcs, dts):
        print(f"Submitting {name}")

        func = evolve_test

        all_args = (cache, name, entropy_func, *args)
        all_kwargs = {'dt': dt, 't_max': 10, 'max_niter': 3, **kwargs}

        evolutions[name] = func(*all_args, **all_kwargs)

#         print(f"All done, collecting results...")
#         for name, fut in evolutions_fut.items():
#             evolutions[name] = fut.get() # Get the result from the Future
#             if evolutions[name]['stderr'] != "":
#                 print(f"Error for {name}")
#                 print(evoltions[name]['stderr'])
    return evolutions



# In[20]:


from multiprocessing import Pool


def evolve_ic_mb2016(cache,
                     name,
                     entropy_func,
                     *args,
                     dt=1., # Myr
                     t_max=1000., # Myr
                     max_niter=200,
                     dt_lower_factor=10.,
                     convection=True,
                     entropy_kwargs=None,
                     r_inner=0.,
                     **kwargs):

    @cache.named(name)
    def func(*args, **kwargs):
        print("Running", name)
        start = process_time()

        beta_param, Z = ic_mb2016()
        *_, param = mw_initial_conditions(beta_param=beta_param)


        # Limit in temperature unit (1 / 1e6)
        TEMPERATURE_LIM = 10**(4.2) / cst.TEMP_UNIT

        print(f"Running with {dt = }")
        evolve_result = evolve_adaptive(
            *args,
            param=param,
            update_entropy=entropy_func,
            dt_start=dt,
            t_max=t_max,
            max_niter=max_niter,
            metallicity=Z,
            temperature_limit=TEMPERATURE_LIM,
            dt_lower_factor=dt_lower_factor,
            convection=convection,
            entropy_kwargs=entropy_kwargs,
            r_inner=r_inner,
            **kwargs,
            #                 r_inner=1.0e3,
            #                 entropy_kwargs={
            #                     'fb_scale': 3e3,
            #                     'fb_heat': 1e40,
            #                     'kernel_func': sech2_3d
            #                 },
        )

        duration = process_time() - start
        print("-------------------------------------------")
        print(f"Process finished in {duration:.1f} seconds. ")
        print(f": {duration / 60:.1f} minutes")
        print(f": {duration / 3600:.1f} hours")

        return param, evolve_result

    return func(*args, **kwargs)


# In[21]:


def mw_adaptive_test_parallel(*args, **kwargs):
    entropy_funcs = [
        update_entropy_sd, update_entropy_sd, update_entropy_sd,
        update_entropy_sd
    ]
    names = [f"MW-TEST-1", "MW-TEST-2", "MW-TEST-3", "MW-TEST-4"]
    dts = [0.1, 0.5, 1, 5]

    evolutions = {}
    evolutions_fut = {}

    with Pool(processes=4) as pool:
        for name, entropy_func, dt in zip(names, entropy_funcs, dts):
            print(f"Submitting {name}")
            
            func = evolve_ic_mb2016

            all_args = (cache, name, entropy_func, *args)
            all_kwargs = {'dt': dt, 't_max': 10, 'max_niter': 3, **kwargs}

            evolutions_fut[name] = pool.apply_async(
                func, args=all_args, kwds=all_kwargs)  # Start the processes and get the Future

        print(f"Closing and waiting...")
        pool.close()  # Avoid more processes from entering the pool
        pool.join()  # Wait until all are done

        print(f"All done, collecting results...")
        for name, fut in evolutions_fut.items():
            evolutions[name] = fut.get() # Get the result from the Future
            if evolutions[name]['stderr'] != "":
                print(f"Error for {name}")
                print(evoltions[name]['stderr'])
    return evolutions


# res_mw_test


# ### MB2016 (Cooling + Convection)

# In[22]:


def mw_adaptive_mb2016_cooling(pool, *args, force_run=False, **kwargs):
    profile_name = "mb2016"
    entropy_funcs = [ update_entropy_sd, update_entropy_cloudy, update_entropy_sd, update_entropy_cloudy, update_entropy_cloudy ]
    convection = [False, False, True, True, True]
    names = [ f"MW-SD-{profile_name}", f"MW-CLOUDY-{profile_name}", f"MW-SD-{profile_name}-conv", f"MW-CLOUDY-{profile_name}-conv", f"MW-CLOUDY-{profile_name}-cool-conv" ]
    
    entropy_kwargs = [ None, None, None, None, {'onlycool': True}]

    dts = [0.1, 0.1, 0.1, 0.1, 0.1]

    if isinstance(force_run, bool):
        force_run = [force_run] * len(entropy_funcs)
    elif isinstance(force_run, list):
        assert len(force_run) == len(entropy_funcs)
    else:
        print(f"force_run has unknown type {type(force_run)}")

    evolutions_fut = {}

    for name, entropy_func, f_run, conv, dt, e_kw in zip(names, entropy_funcs, force_run, convection, dts, entropy_kwargs):
        print(f"Submitting {name}")

        func = evolve_ic_mb2016

        all_args = (cache, name, entropy_func, *args)
        all_kwargs = {'dt': dt, 'convection':conv, 'max_niter':4000, 'entropy_kwargs':e_kw, 'force_run': f_run, **kwargs}

        evolutions_fut[name] = pool.apply_async(
            func, args=all_args, kwds=all_kwargs)  # Start the processes and get the Future
    return evolutions_fut


# ### MB2016-reschaled (Cooling + Convection)

# In[25]:


def evolve_ic_mb2016_rescaled(cache,
                     name,
                     entropy_func,
                     *args,
                     dt=1., # Myr
                     t_max=1000., # Myr
                     max_niter=200,
                     dt_lower_factor=10.,
                     convection=True,
                     entropy_kwargs=None,
                     r_inner=0.,
                     **kwargs):

    @cache.named(name)
    def func(*args, **kwargs):
        print("Running", name)
        start = process_time()

        beta_param, Z = ic_mb2016_rescaled()
        *_, param = mw_initial_conditions(beta_param=beta_param)


        # Limit in temperature unit (1 / 1e6)
        TEMPERATURE_LIM = 10**(4.2) / cst.TEMP_UNIT

        print(f"Running with {dt = }")
        evolve_result = evolve_adaptive(
            *args,
            param=param,
            update_entropy=entropy_func,
            dt_start=dt,
            t_max=t_max,
            max_niter=max_niter,
            metallicity=Z,
            temperature_limit=TEMPERATURE_LIM,
            dt_lower_factor=dt_lower_factor,
            convection=convection,
            entropy_kwargs=entropy_kwargs,
            r_inner=r_inner,
            **kwargs,
        )

        duration = process_time() - start
        print("-------------------------------------------")
        print(f"Process finished in {duration:.1f} seconds. ")
        print(f": {duration / 60:.1f} minutes")
        print(f": {duration / 3600:.1f} hours")

        return param, evolve_result

    return func(*args, **kwargs)


# In[26]:


def mw_adaptive_mb2016_rescaled_cooling(pool, *args, force_run=False, **kwargs):
    profile_name = "mb2016rescaled"
    entropy_funcs = [ update_entropy_sd, update_entropy_cloudy, update_entropy_sd, update_entropy_cloudy, update_entropy_cloudy ]
    convection = [False, False, True, True, True]
    names = [ f"MW-SD-{profile_name}", f"MW-CLOUDY-{profile_name}", f"MW-SD-{profile_name}-conv", f"MW-CLOUDY-{profile_name}-conv", f"MW-CLOUDY-{profile_name}-cool-conv" ]
    
    entropy_kwargs = [ None, None, None, None, {'onlycool': True}]

    dts = [0.1, 0.1, 0.1, 0.1, 0.1]

    if isinstance(force_run, bool):
        force_run = [force_run] * len(entropy_funcs)
    elif isinstance(force_run, list):
        assert len(force_run) == len(entropy_funcs)
    else:
        print(f"force_run has unknown type {type(force_run)}")

    evolutions_fut = {}

    for name, entropy_func, f_run, conv, dt, e_kw in zip(names, entropy_funcs, force_run, convection, dts, entropy_kwargs):
        print(f"Submitting {name}")

        func = evolve_ic_mb2016_rescaled

        all_args = (cache, name, entropy_func, *args)
        all_kwargs = {'dt': dt, 'convection':conv, 'max_niter':4000, 'entropy_kwargs':e_kw, 'force_run': f_run, **kwargs}

        evolutions_fut[name] = pool.apply_async(
            func, args=all_args, kwds=all_kwargs)  # Start the processes and get the Future
    return evolutions_fut

def mw_adaptive_nonzero_bound(pool, *args, force_run=False, **kwargs):
    r0s = ["0.1", "1", "10"] # In kpc
    profile_name_fmt = "MW-SD-mb2016{rescaled}-radius{radius}e3"
    entropy_func = update_entropy_sd
    conv = True
    dt = 0.1

    evolutions_fut = {}

    if isinstance(force_run, bool):
        force_run = [force_run] * len(r0s) * 2
    elif isinstance(force_run, list):
        assert len(force_run) == len(r0s) * 2
    else:
        print(f"force_run has unknown type {type(force_run)}")

    for rescaled in ["", "rescaled"]:
        if rescaled == "":
            func = evolve_ic_mb2016
        elif rescaled == "rescaled":
            func = evolve_ic_mb2016_rescaled
        else:
            print(f"Invalid {rescaled}, skipping")
            continue

        for f_run, r0 in zip(force_run, r0s):

            name = profile_name_fmt.format(rescaled=rescaled, radius=r0)
            print(f"Submitting {name}")

            r_inner = float(r0)*1e3 # Convert to number (in pc)

            all_args = (cache, name, entropy_func, *args)
            all_kwargs = {'r_inner':r_inner, 'dt': dt, 'convection': conv, 'max_niter':4000, 'force_run': f_run, **kwargs}

            evolutions_fut[name] = pool.apply_async(
                func, args=all_args, kwds=all_kwargs)  # Start the processes and get the Future
    return evolutions_fut

def mw_adaptive_dt_lower(pool, *args, force_run=False, **kwargs):
    profile_name_fmt = "MW-{method}-mb2016{rescaled}-dtlim{dt_limit}"
    conv = True
    dt = 0.1
    dtlims = [1e-2, 1e-3, 1e-4]

    evolutions_fut = {}

    if isinstance(force_run, bool):
        force_run = [force_run] * len(dtlims) * 4
    elif isinstance(force_run, list):
        assert len(force_run) == len(dtlims) * 4
    else:
        print(f"force_run has unknown type {type(force_run)}")

    for rescaled in ["", "rescaled"]:
        if rescaled == "":
            func = evolve_ic_mb2016
        elif rescaled == "rescaled":
            func = evolve_ic_mb2016_rescaled
        else:
            print(f"Invalid {rescaled}, skipping")
            continue

        for method in ["SD", "CLOUDY"]:
            if method == "SD":
                entropy_func = update_entropy_sd
            elif method == "CLOUDY":
                entropy_func = update_entropy_cloudy
            else:
                print(f"Invalid {method}, skipping")
                continue


            for f_run, dt_limit in zip(force_run, dtlims):

                name = profile_name_fmt.format(method=method, rescaled=rescaled, dt_limit=dt_limit)
                print(f"Submitting {name}")


                all_args = (cache, name, entropy_func, *args)
                all_kwargs = {'dt_limit': dt_limit, 'dt': dt, 'convection': conv, 'max_niter':4000, 'force_run': f_run, **kwargs}

                evolutions_fut[name] = pool.apply_async(
                    func, args=all_args, kwds=all_kwargs)  # Start the processes and get the Future
    return evolutions_fut



# ### Run for specific Energy

# In[27]:


def create_name(rescaled: bool, fb_heat: str, fb_scale: str, prefix="MW-", suffix=""):
    if rescaled:
        profile_name = "mb2016rescaled"
    else:
        profile_name = "mb2016"

    name = f"{prefix}{profile_name}-E{fb_heat}-S{fb_scale}{suffix}"
    return name


# In[28]:


def mw_adaptive_energy(
    pool,
    cache,
    rescaled: bool,
    fb_heat: str,
    fb_scale: str,
    *args,
    convection=True,
    dt=0.1,
    max_niter=4000,
    **kwargs,
):
    name = create_name(rescaled=rescaled, fb_heat=fb_heat, fb_scale=fb_scale)
    
    entropy_func = update_entropy_bubble

    if rescaled:
        func = evolve_ic_mb2016_rescaled
    else:
        func = evolve_ic_mb2016

    e_kw = {
        'fb_heat': float(fb_heat),
        'fb_scale': float(fb_scale),
    }
    
    all_args = (cache, name, entropy_func, *args)
    all_kwargs = {
        'dt': dt,
        'convection': convection,
        'max_niter': 4000,
        'entropy_kwargs': e_kw,
        **kwargs
    }
    
    print(f"Submitting {name}")

    # Start the processes and return the Future
    return name, pool.apply_async(func, args=all_args, kwds=all_kwargs)


# In[29]:


def mw_adaptive_mb2016_heating(pool, cache, *args, energies=None, fb_scale="8e3", **kwargs):
    if energies is None:
        energies = [
            "5e43",
            "1e43",
            "5e42",
            "1e42",
            "5e41",
            "1e41",
            "5e40",
            "1e40",
            "5e39",
            "1e39",
        ]

    evolutions_fut = {}
    for resc in [False, True]:  # Rescaled profile or not
        for e in energies:
            name, fut = mw_adaptive_energy(pool,
                                           cache,
                                           *args,
                                           rescaled=resc,
                                           fb_heat=e,
                                           fb_scale=fb_scale,
                                           **kwargs)
            evolutions_fut[name] = fut

    return evolutions_fut


# ## Run all

# In[ ]:


def main():
    ################# Run testing first, and then make sure #############
    print("--------- TESTING FIRST... ------------")

    res_mw_test = mw_adaptive_test()

    print("--------- TESTING Parallel ------------")

    res_mw_test = mw_adaptive_test_parallel(force_run=True)
    for k, v in res_mw_test.items():
        print(f"{k} evolved for times: {v['result'][1][0]}")
        if v['stderr'] != "":
            print(f"ERROR: {k} has an issue, force stopping")
            return

    response = input("Continue? [Y/n] ")
    if response != "" and response != "y":
        print("Stopping")
        return

    print("--------- Running for real ------------")
    evolutions = {}

    with Pool(processes=10) as pool:
        evolutions_fut = {}
        
        ### Cooling case
        #fut_mb2016 = mw_adaptive_mb2016_cooling(pool)
        #evolutions_fut.update(fut_mb2016)
        
        #fut_mb2016rescaled = mw_adaptive_mb2016_rescaled_cooling(pool)
        #evolutions_fut.update(fut_mb2016rescaled)

        ### Outer bound radius
        #fut_nonzero_bound = mw_adaptive_nonzero_bound(pool)
        #evolutions_fut.update(fut_nonzero_bound)

        ### Lower dt limits
        fut_lower_dt_limit = mw_adaptive_dt_lower(pool)
        evolutions_fut.update(fut_lower_dt_limit)
        
        ### Heating case (for multiple energies 1e39 - 1e43)
        #for scale in ["8e3", "5e3"]:
        #    fut_mb2016_heat = mw_adaptive_mb2016_heating(pool, cache, fb_scale=scale)
        #    evolutions_fut.update(fut_mb2016_heat)

        #### tmp:  FORCE RUNNING!
        # name, fut = mw_adaptive_energy(pool, cache, rescaled=True, fb_heat="5e41", fb_scale="5e3", dt=0.05)
        # evolutions_fut[name] = fut

        ### For scale = 8e3, rerun for more fine energies to find cooling / heating balance.
        ### Values for energies obtained by earlier runs
        
        # mb2016
        # for e in ['2e41', '3e41', '4e41']:
        #     name, fut = mw_adaptive_energy(pool, cache, rescaled=False, fb_heat=e, fb_scale="8e3")
        #     evolutions_fut[name] = fut
        # mb2016rescaled
        # for e in ['2e42', '3e42', '4e42']:
        #     name, fut = mw_adaptive_energy(pool, cache, rescaled=True, fb_heat=e, fb_scale="8e3")
        #     evolutions_fut[name] = fut

        print("Closing and waiting...")
        pool.close()
        pool.join()

        print(f"All done, collecting results...")
        for name, fut in evolutions_fut.items():
            evolutions[name] = fut.get()  # Get the result from the Future
            if evolutions[name]['stderr'] != "":
                print(f"Error for {name}")
                print(evolutions[name]['stderr'])


if __name__ == '__main__':
    response = input("Is anaconda3/2024.02 loaded? [Y/n] ")
    if response != "" and response != "y":
        print("Not loaded, exiting")
        exit(1)

    main()
