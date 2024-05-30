import constants as cst
import numpy as np

def compute_quantities(r, P, M, sig, mu=0.62):
    """
    Assumes: 
        r: [pc]
        P: [1e-13 erg/cm3]
        M: [1e8 Mo]
        sig: [1e30 cm4 / g(2/3) / s2 ]
	
	For Kaiser-Binney, the default is 0.62 (fully ionized)
    """
    rho = (cst.C2 * P / sig)**(1/cst.gamma)
    n = rho / mu
    n_e = n / (1 + 1/1.2)
    
    T = P / (n*cst.C1)
    return rho, n_e, T

from dataclasses import dataclass

@dataclass
class Parameters:
    # Profile
    M_tot: float
    R_out: float
    r_s: float
    rho_s: float
    # Pressure
    P_inf: float
    P_0: float
    # Entropy profile
    M: np.ndarray
    sig: np.ndarray
        
    def nfw_param(self):
        return (self.r_s, self.rho_s)
    
    def __str__(self):
        return f"""Parameters:
  M_tot = {self.M_tot:.2e} 1e8 Mo
  R_out = {self.R_out / 1e3:.2f} kpc
  r_s   = {self.r_s / 1e3 :.2f} kpc
  rho_s = {self.rho_s :.2e} m_p / cm3 
  P_inf = {self.P_inf :.2e} 1e-13 erg / cm3 
  P_0   = {self.P_0 :.2e} 1e-13 erg / cm3 
        """
