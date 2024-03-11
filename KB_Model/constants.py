import astropy.constants as aconsts
from astropy import units as u

# Provide all constants in CGS (without astropy units)
proton_mass = aconsts.m_p.cgs.value
k_B = aconsts.k_B.cgs.value
G = aconsts.G.cgs.value
electron_volt = 1.602177e-12  # erg

pc_to_cm = (1 * u.pc).cgs.value
solmass = (1 * u.solMass).cgs.value

yr_to_s = 365.25 * 24 * 60 * 60  # s

gamma = 5 / 3
