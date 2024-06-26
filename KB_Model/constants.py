import astropy.constants as aconsts
from astropy import units as u
import math

# Provide all constants in CGS (without astropy units)
proton_mass = aconsts.m_p.cgs.value
k_B = aconsts.k_B.cgs.value
G = aconsts.G.cgs.value

electron_volt = u.eV.to('erg')
pc_to_cm = u.pc.to('cm')
solmass = u.solMass.to('g')
yr_to_s = u.yr.to('s')  # s

gamma = 5 / 3


## Units for quantities:
MASS_UNIT = 1e8 * solmass # g
TEMP_UNIT = 1e6           # K
PRES_UNIT = 1e-13         # erg / cm3
SIG_UNIT = 1e30           # ...
LEN_UNIT = pc_to_cm       # cm
DENS_UNIT = proton_mass   # g / cm3
NDENS_UNIT = 1            # 1 / cm3
TIME_UNIT = u.Myr.to("s") # s
VEL_UNIT = u.km.to("cm") / u.s.to("s") # s

# Ideal gas law
C1 = k_B * (NDENS_UNIT * TEMP_UNIT / PRES_UNIT)

# entropy calculation
C2 = PRES_UNIT / (SIG_UNIT * DENS_UNIT**gamma)

# cooling time
C3 = k_B * TEMP_UNIT / (TIME_UNIT * NDENS_UNIT)

# hydrostatic equilibrium
C4 = DENS_UNIT**2 * LEN_UNIT**2 / PRES_UNIT

# mass shells
C5 = DENS_UNIT * LEN_UNIT ** 3 / MASS_UNIT

# inflow velocity
C6 = MASS_UNIT / (VEL_UNIT * TIME_UNIT * LEN_UNIT **2 * DENS_UNIT)

# sound speed
C7 = math.sqrt( k_B * TEMP_UNIT / proton_mass ) /  VEL_UNIT

if __name__ == '__main__':
    print("Constants for calculations:")
    print(f"| {C1 = :.8e}")
    print(f"| {C2 = :.8e}")
    print(f"| {C3 = :.8e}")
    print(f"| {C4 = :.8e}")
    print(f"| {C5 = :.8e}")
    print(f"| {C6 = :.8e}")
    print(f"| {C7 = :.8e}")
