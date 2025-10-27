# endpoints.py - rutas principales
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import uuid, os, json
from ..tmm.tmm import tmm_spectrum
from ..io.file_utils import read_nk_file, read_reflectance_file
from ..tmm.dispersion import cauchy, drude_epsilon, epsilon_to_nk
from ..fit.fit_core import fit_reflectance_simple_thickness as fit_reflectance


router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# simple in-memory "materials" registry (for MVP)
MATERIALS = {}

@router.post("/materials/upload")
async def upload_material(file: UploadFile = File(...), material_name: str = Form(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".txt", ".xlsx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    token = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{material_name}_{token}{ext}"
    contents = await file.read()
    path.write_bytes(contents)
    MATERIALS[material_name] = {"type":"file", "path": str(path)}
    return {"material": material_name, "path": str(path), "status":"uploaded"}

@router.post("/materials/define")
def define_material(model_def: dict):
    # model_def: name, model, parameters
    name = model_def.get("name")
    if name is None:
        raise HTTPException(status_code=400, detail="Name required")
    MATERIALS[name] = {"type":"model", "model": model_def.get("model"), "params": model_def.get("parameters")}
    return {"status":"ok", "name": name}

@router.post("/geometry/calc")
def geometry_calc(geom: dict):
    """
    Receive geometry JSON -> compute theoretical R,T,A on requested wavelength grid.
    Minimal: expects materials already registered (via upload or define)
    """
    # parse
    layers = geom["layers"]
    thicknesses = geom.get("thicknesses", [])
    angle = geom.get("angle_deg", 0.0)
    wl_cfg = geom.get("wavelength", {"start_nm":400,"end_nm":800,"points":401})
    pol = geom.get("polarization","s")
    wl = list(range(wl_cfg["start_nm"], wl_cfg["end_nm"]+1, max(1,int((wl_cfg["end_nm"]-wl_cfg["start_nm"])//(wl_cfg["points"]-1) or 1))))
    # build n_stack: for each layer get n(λ)+ik(λ)
    n_stack = []
    from ..tmm.dispersion import epsilon_to_nk
    for mat in layers:
        if mat not in MATERIALS:
            raise HTTPException(status_code=400, detail=f"Material {mat} not defined")
        info = MATERIALS[mat]
        if info["type"]=="file":
            wls, ns, ks = read_nk_file(info["path"])
            # interpolate to wl
            import numpy as np
            n_interp = np.interp(wl, wls, ns)
            k_interp = np.interp(wl, wls, ks)
            n_stack.append(n_interp + 1j*k_interp)
        else:
            model = info["model"]
            params = info["params"]
            import numpy as np
            if model=="cauchy":
                arr = cauchy(wl, params.get("A",1.5), params.get("B",0.0), params.get("C",0.0))
                n_stack.append(arr)
            elif model=="drude":
                # params: eps_inf, omega_p_eV, gamma_eV ; compute epsilon(omega) from omega array in eV
                from ..tmm.dispersion import drude_epsilon, omega_from_wl_nm
                omega_eV = omega_from_wl_nm(np.array(wl))
                eps = drude_epsilon(omega_eV, params.get("eps_inf",1.0), params.get("omega_p",1.0), params.get("gamma",0.1))
                nk = epsilon_to_nk(eps)
                n_stack.append(nk)
            else:
                raise HTTPException(status_code=400, detail=f"Model {model} not supported yet")
    # call TMM
    R,T,A = tmm_spectrum(wl, n_stack, thicknesses, theta_inc_deg=angle, pol=pol)
    return {"wl": wl, "R": R.tolist(), "T": T.tolist(), "A": A.tolist()}

@router.post("/fit/run")
def fit_run(req: dict):
    """
    Simplified fit runner for MVP.
    Expects:
      - geometry (same as in /geometry/calc)
      - exp_path (path to reflectance CSV with two columns x,y)
      - fit_targets, initial_guess, bounds optional
    """
    geometry = req.get("geometry")
    exp_path = req.get("exp_path")
    initial_guess = req.get("initial_guess", [])
    bounds = req.get("bounds", None)
    if geometry is None or exp_path is None:
        raise HTTPException(status_code=400, detail="geometry and exp_path required")
    # read experimental
    wl_exp, R_exp = read_reflectance_file(exp_path)
    # wrapper: our fit_core expects param_builder known; for MVP fit a single thickness
    # assume fit_targets = ["thickness[0]"]
    from ..fit.fit_core import fit_reflectance_simple_thickness
    res = fit_reflectance_simple_thickness(geometry, wl_exp, R_exp, initial_guess)
    return {"status":"done", "result": res}
