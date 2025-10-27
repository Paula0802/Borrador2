# dispersion.py - modelos vectorizados
import numpy as np
from .constants import hbar_eVs

def omega_from_wl_nm(wl_nm):
    # approximate photon energy in eV from wavelength nm: E(eV) = 1240 / lambda(nm)
    return 1240.0 / (np.array(wl_nm, dtype=float))

def cauchy(wl_nm, A, B, C):
    # Cauchy commonly with lambda in micrometers. Here accept wl_nm -> convert um
    lam_um = np.array(wl_nm, dtype=float) * 1e-3
    n = A + B/(lam_um**2 + 1e-30) + C/(lam_um**4 + 1e-30)
    return n + 1j * np.zeros_like(n)

def drude_epsilon(omega_eV, eps_inf, omega_p_eV, gamma_eV):
    # eps(omega) = eps_inf - omega_p^2 / (omega^2 + i gamma omega)
    omega = np.array(omega_eV, dtype=complex)
    num = omega_p_eV**2
    den = (omega**2 + 1j * gamma_eV * omega)
    eps = eps_inf - num/den
    return eps

def epsilon_to_nk(eps):
    # vector eps -> nk complex
    eps = np.array(eps, dtype=complex)
    mag = np.sqrt(eps.real**2 + eps.imag**2)
    n = np.sqrt((mag + eps.real)/2.0)
    k = np.sqrt((mag - eps.real)/2.0)
    return n + 1j*k
