import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colormaps

import numpy as np
from misc import Parameters, compute_quantities

import constants as cst

def index_closest(x, test_points):
    """
    Assuming x is always increasing, create a mask which selects the elements in
    x which are closest to the test points.
    """
    # Maybe there is a better way to do this :')
    test = np.array(test_points)
    diff = np.abs(x[:, np.newaxis] - test[np.newaxis, :])
    indices = diff.argmin(axis=0)
    mask = np.isin(x, x[indices])
    return mask



def full_plot_all(result, plot_mask=None, plot_times=None, time_cmap='rainbow', **kwargs):
    param, _ = result

    if plot_mask is not None:
        selected = [ [qi for qi, m in zip(quantity, plot_mask) if m] for quantity in result[1] ]
        selected[0] = t[plot_mask]
    elif plot_times is not None:
        t = result[1][0]
        plot_mask = index_closest(t, plot_times)
        selected = [
            [qi for qi, m in zip(quantity, plot_mask) if m] for quantity in result[1]
        ]
        selected[0] = t[plot_mask]
    else:
        selected = result[1]

    ts, rs, ps, ms, ss, tcs = selected

    figax = None
    
    cmap = colormaps[time_cmap].resampled(64)
    norm = Normalize(vmin=ts.min(), vmax=ts.max())
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    
    figax = plt.subplots(3, 2, figsize=(2*3.5, 3*3.5), sharex=True)
    for t, r, p, m, s, tcs in zip(*selected):
        color = mapper.to_rgba(t)
        figax = full_plot(r, p, m, s, param=param, figax=figax, options={'label': f"{t:.0f} Myr", 'color': color})
        fig, ax = figax
    
    # Add the colourbar
    
    # left, bottom, width, height
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
    fig.colorbar(mapper, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel(r't [Myr]')
    
    for a in ax.flat[:-2]:
        a.tick_params(axis="x",direction="in", pad=-15)
    
    fig.subplots_adjust(bottom=0.18, wspace=0.32, hspace=0)
    return figax

def recreate_plot_all(result, plot_mask=None, plot_times=None, time_cmap='rainbow', **kwargs):
    param, _ = result
    cmap = colormaps[time_cmap].resampled(8)

    if plot_mask is not None:
        selected = [ [qi for qi, m in zip(quantity, plot_mask) if m] for quantity in result[1] ]
        selected[0] = t[plot_mask]
    elif plot_times is not None:
        t = result[1][0]
        plot_mask = index_closest(t, plot_times)
        selected = [
            [qi for qi, m in zip(quantity, plot_mask) if m] for quantity in result[1]
        ]
        selected[0] = t[plot_mask]
    else:
        selected = result[1]

    ts, rs, ps, ms, ss, tcs = selected

    cmap = colormaps[time_cmap].resampled(64)
    norm = Normalize(vmin=ts.min(), vmax=ts.max())
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    
    figax = plt.subplots(2, 2, figsize=(2*4.5, 2*3.5))
    
    for t, r, p, m, s, tc in zip(*selected):
        color = mapper.to_rgba(t)
        figax = recreate_plot(r, p, m, s, param=param, figax=figax, temp_unit='K', options={'label': f"{t:.0f} Myr", 'color': color}, **kwargs)
        fig, ax = figax
        
    sig0 = [s[0] for s in ss]
    ax[1, 1].plot(ts, sig0, c='k')
    ax[1, 1].set_ylabel(r'$\sigma_0$')
    ax[1, 1].set_xlabel(r't $[Myr]$')
    # left, bottom, width, height
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
    fig.colorbar(mapper, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel(r't [Myr]')
    fig.subplots_adjust(bottom=0.20, wspace=0.3, hspace=0.31)
    
    return figax


def full_plot(r, P, M, sig, figax=None, param:Parameters = None, options=None, log_x=True, r_min=None, r_max=None):
    P_0_ref = param.P_0
    P_inf_ref = param.P_inf
    M_tot = param.M_tot
    R_out = param.R_out
    
    if figax is None:
        figax = plt.subplots(3, 2, tight_layout=True, sharex=True, figsize=(2*3.5, 3*3.5))
    
#     options_def = {'c': 'tab:blue', 'ls':'--'}
    options_def = dict()
    if options is not None:
        for k, v in options.items():
            options_def[k] = v
    
    fig, ax = figax
    
    rho, n_e, T = compute_quantities(r, P, M, sig)

    r_plot = r / 1e3

    ax[0, 0].plot(r_plot, P, **options_def)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel(r'Pressure, $P\,\,\,[10^{-13} {\rm erg\,\, cm}^{-3}]$')

    ax[1, 0].plot(r_plot, M, **options_def)
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_ylabel(r'Cumulative Mass, $M$  [$10^8\,\,M_\odot$]')

    ax[0, 1].plot(r_plot, rho, **options_def)
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_ylabel(r'Gas density, $\rho \,\,[m_p \,\,{\rm cm}^{-3}]$')

    ax[1, 1].plot(r_plot, n_e, **options_def)
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_ylabel(r'$e^-$ number density, $n_e \,\,[{\rm cm}^{-3}]$')

    ax[2, 0].plot(r_plot, sig, **options_def)
    ax[2, 0].set_yscale('log')
    ax[2, 0].set_ylabel(r'Entropy index, $\sigma\,\,\,[{\rm cm}^4\,\,{\rm g}^{-2/3}\,\,{\rm s}^{-2}]$')

    ax[2, 1].plot(r_plot, T, **options_def)
    ax[2, 1].set_yscale('log')
    ax[2, 1].set_ylabel(r'Temperature, $[10^6 {\rm K}]$')

    ax[2, 0].set_xlabel('r [kpc]')
    ax[2, 1].set_xlabel('r [kpc]')
    
    if r_min is not None:
        ax[0,0].set_xlim(r_min, ax[0,0].get_xlim()[1])

    if r_max is not None:
        ax[0,0].set_xlim(ax[0,0].get_xlim()[0], r_max)
        
    if log_x:
        ax[0, 0].set_xscale('log') # All axes are connected

    #ax[0, 0].axhline(P_0_ref, ls=':', c='k')
    ax[0, 0].axhline(P_inf_ref, ls=':', c='k', alpha=0.5)
    ax[1, 0].axhline(M_tot, ls=':', c='k', alpha=0.5)
    #ax[1, 0].axvline(R_out / 1e3, ls='--', c='k')
    return fig, ax


def recreate_plot(r, P, M, sig, temp_unit="kev", kb_ylims=True, figax=None, options=None, param: Parameters = None):
    # Unpack parameters
    P_0_ref = param.P_0
    P_inf_ref = param.P_inf
    M_tot = param.M_tot
    R_out = param.R_out

    if figax is None:
        figax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 6))

    options_def = dict()
    if options is not None:
        for k, v in options.items():
            options_def[k] = v

    fig, ax = figax

    rho, n_e, T = compute_quantities(r, P, M, sig)

    r_plot = r /  1e3

    # Density:
    ax[0, 0].plot(r_plot, n_e, **options_def)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylabel(r'$e^{-}$ density [${\rm cm}^{-3}$]')
    ax[0, 0].set_xlabel('r [kpc]')
    #ax[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    #ax[0, 0].yaxis.set_major_formatter(FormatStrFormatter("%g"))
    
    electron_volt = 1.602177e-12 # erg

    # Temperature
    ylims = np.array([0, 4])
    if temp_unit == "kev":
        T_plot = cst.k_B * T*cst.TEMP_UNIT / electron_volt / 1e3
        ylabel = 'Temperature [keV]'
    else: # Assuming Kelvin
        T_plot = T*cst.TEMP_UNIT
        ylims = ylims / cst.k_B * electron_volt * 1e3
        ylabel = 'Temperature [K]'

    ax[0, 1].plot(r_plot, T_plot, **options_def)
    ax[0, 1].set_ylabel(ylabel)
    ax[0, 1].set_xlabel('r [kpc]')

    s_0 = param.sig[0]
#     ax[1, 0].plot((sig - s_0) / s_0, M / M_tot, **options_def)
    sig_plot = (sig - sig[0]) / s_0
    M_plot = M / M_tot
    ax[1, 0].plot(sig_plot[1:], M_plot[1:], **options_def)

    if kb_ylims:
        ax[0, 0].set_ylim(0.002, 0.4)
        ax[0, 1].set_ylim(*ylims)
        ax[0, 1].set_xlim(0, param.R_out / 1e3)
        ax[0, 0].set_xlim(2, param.R_out / 1e3)

    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xscale('log')
#     ax[1, 0].set_ylim(0.0001, 1)
#     ax[1, 0].set_xlim(0.1, 10.0)
    ax[1, 0].set_ylabel(r"$M(\sigma)\, /\, M_{tot}$")
    ax[1, 0].set_xlabel(r"$(\sigma - \sigma_0(t))\, /\, \sigma_0(0)$")
    #ax[1, 0].yaxis.set_major_formatter(FormatStrFormatter("%g"))
    #ax[1, 0].yaxis.set_major_formatter(FormatStrFormatter("%g"))

    return fig, ax
