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


C1 = k_B * 1e-7  # Ideal gas law
C2 = 1e-43 * proton_mass ** (-gamma)  # entropy calculation
C3 = 1e13 / u.Myr.to("s")  # cooling time
C4 = proton_mass * pc_to_cm**2 * 1e13 * (1e43) ** (-1 / gamma)  # hydrostatic equilibrium
C5 = proton_mass * pc_to_cm**3 / (1e8 * solmass)  # mass shells
# print(C1, C2, C3, C4, C5)
