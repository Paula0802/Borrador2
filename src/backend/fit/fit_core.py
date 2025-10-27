# fit_core.py - simple fit for single-thickness film example
import numpy as np
from scipy.optimize import least_squares
from ..tmm.tmm import tmm_spectrum
from ..io.file_utils import read_nk_file

def param_builder_single_thickness(p, geometry, wl):
    # geometry: dict with layers and materials; p[0] is thickness nm for layer index 1 (first film)
    layers = geometry["layers"]
    thicknesses = geometry.get("thicknesses", [])
    # replace thickness[0] with p[0]
    if len(thicknesses) == 0:
        thicknesses = [p[0]]
    else:
        thicknesses[0] = p[0]
    # build n_stack fast: assume materials files for MVP
    from ..tmm.dispersion import cauchy
    n_stack = []
    for mat in layers:
        # expect file path stored externally, but for MVP we treat any "material" as constant
        # Here just use air(1.0) for outer and a sample film with cauchy parameters if name starts with 'film'
        if mat.lower() == "air":
            n_stack.append(1.0+0j)
        elif mat.lower().startswith("film"):
            # sample cauchy
            n_stack.append(cauchy(wl, 1.8, 0.001, 0.0))
        else:
            # substrate
            n_stack.append(1.52+0j)
    return n_stack, thicknesses

def fit_reflectance_simple_thickness(geometry, wl_exp, R_exp, initial_guess):
    # wl_exp: array
    wl = wl_exp
    def resid(p):
        n_stack, dlist = param_builder_single_thickness(p, geometry, wl)
        R_model, _, _ = tmm_spectrum(wl, n_stack, dlist, theta_inc_deg=geometry.get("angle_deg",0.0), pol=geometry.get("polarization","s"))
        return (R_model - R_exp)
    p0 = initial_guess if len(initial_guess)>0 else [100.0]
    res = least_squares(resid, p0, method='lm')
    # pack results
    return {"x": wl.tolist(), "R_model": resid(res.x).tolist(), "params": res.x.tolist(), "cost": float(res.cost)}
