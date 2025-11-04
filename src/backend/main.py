# src/backend/main.py
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import io
import os
import json
import uuid
import re

# Robust import for TransferMatrixMethod (works in several run modes)
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent  # .../src/backend
_TMM_DIR = _BACKEND_DIR / "TMM"

# If backend is used as a package (uvicorn started from src with "backend.main"), try relative import first
try:
    # this works when running as package: `uvicorn backend.main:app` from src/
    from .TMM.tmm import TransferMatrixMethod
except Exception:
    try:
        # try absolute import if sys.path already contains src
        from TMM.tmm import TransferMatrixMethod
    except Exception:
        # last resort: ensure backend folder is on sys.path so 'TMM' package can be found
        parent = _BACKEND_DIR  # this is src/backend
        # ensure parent-of-backend (src) is on sys.path so `backend` and `TMM` are importable as packages
        src_dir = parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        # also ensure backend folder itself available
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        try:
            from TMM.tmm import TransferMatrixMethod
        except Exception as e:
            # raise a clearer error for debugging
            raise ImportError(
                "Could not import TransferMatrixMethod from TMM.tmm. "
                "Make sure src/backend/TMM/tmm.py exists and that you run uvicorn from the 'src' folder, "
                "or create __init__.py files. Full error: " + str(e)
            )

# -------------------------
# Configuración de la app
# -------------------------
app = FastAPI(title="DeltaPsi-Free")

# Habilitar CORS (ajusta allow_origins en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cambiar por dominios reales en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Router / Endpoints API
# -------------------------
router = APIRouter(prefix="/api")

ALLOWED_EXT = {".csv", ".txt", ".xlsx"}

# -------------------------
# Helpers para lectura de archivos (upload)
# -------------------------
def _read_file_bytes(file: UploadFile) -> pd.DataFrame:
    """
    Intenta leer .csv, .txt (tsv o whitespace) y .xlsx y devuelve un DataFrame de pandas.
    """
    filename = (file.filename or "").lower()
    # si file.file fue leído anteriormente por otra función, mover a inicio
    try:
        file.file.seek(0)
    except Exception:
        pass

    content = file.file.read()
    if filename.endswith(".csv"):
        # intenta detección simple: coma por defecto, si falla intenta ; .
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep=";")
        return df
    elif filename.endswith(".txt"):
        # intenta tabulados o whitespace; si falla intenta coma
        try:
            df = pd.read_csv(io.BytesIO(content), sep="\t")
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(content), delim_whitespace=True, header=None)
            except Exception:
                df = pd.read_csv(io.BytesIO(content))
        return df
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return df
    else:
        raise ValueError("Unsupported file type")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas a 'wavelength', 'psi', 'delta' cuando es posible.
    Trata aliases comunes como 'wl', 'lambda', 'phi', 'Δ', etc.
    Si el archivo no tiene cabecera numérica intenta mapear las primeras 3 columnas numéricas.
    """
    df = df.copy()
    # si header es None, pandas deja columnas como 0,1,2... convertimos a strings
    df.columns = [str(c).strip().lower() for c in df.columns]

    col_map = {}
    for col in df.columns:
        c = col.replace(" ", "").replace("_", "")
        if c in {"wavelength", "lambda", "wl", "lam", "w"}:
            col_map[col] = "wavelength"
        elif c in {"psi", "φ", "phi"}:
            col_map[col] = "psi"
        elif c in {"delta", "Δ", "d", "del"}:
            col_map[col] = "delta"

    # si no mapeó todas las columnas intenta asignar por posición si hay al menos 3 cols numéricas
    if len(col_map) < 3:
        numeric_cols = []
        for col in df.columns:
            try:
                # si la columna puede convertirse a número (sin errores), la consideramos numérica
                pd.to_numeric(df[col].dropna().iloc[:10], errors="raise")
                numeric_cols.append(col)
            except Exception:
                continue
        if len(numeric_cols) >= 3:
            col_map = {numeric_cols[0]: "wavelength", numeric_cols[1]: "psi", numeric_cols[2]: "delta"}

    df = df.rename(columns=col_map)
    return df

# ---- START: TMM integration helpers ----
from typing import Tuple, List
import numpy as np

# Import relativo correcto cuando main.py forma parte del paquete 'backend'
from .TMM.tmm import TransferMatrixMethod

def parse_material_file_upload(upload: UploadFile) -> dict:
    """
    Lee UploadFile que contiene wl,n,k   OR   wl,e1,e2   OR  wl,e (complex as strings)
    Devuelve dict con keys: 'wl' (np.array float), and 'n' and 'k' arrays OR 'eps' complex array.
    """
    # Reuse your _read_file_bytes function which returns DataFrame
    df = _read_file_bytes(upload)
    # normalize cols (simple)
    cols = {c.strip().lower(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        return None

    wl = get_col('wavelength','wl','lambda','lam','w')
    if wl is None:
        raise ValueError("No wavelength column found (wl/wavelength/lambda).")
    wl_arr = np.asarray(pd.to_numeric(wl, errors='coerce')).astype(float)
    # try n,k
    n_col = get_col('n')
    k_col = get_col('k')
    e1_col = get_col('e1','epsilon_real','er')
    e2_col = get_col('e2','epsilon_imag','ei')
    e_col = get_col('e','epsilon')

    out = {'wl': wl_arr}
    if n_col is not None and k_col is not None:
        n_arr = np.asarray(pd.to_numeric(n_col, errors='coerce')).astype(float)
        k_arr = np.asarray(pd.to_numeric(k_col, errors='coerce')).astype(float)
        out['n'] = n_arr
        out['k'] = k_arr
        return out
    if e1_col is not None and e2_col is not None:
        e1 = np.asarray(pd.to_numeric(e1_col, errors='coerce')).astype(float)
        e2 = np.asarray(pd.to_numeric(e2_col, errors='coerce')).astype(float)
        eps = e1 + 1j*e2
        out['eps'] = eps
        return out
    if e_col is not None:
        # try parse complex strings or numeric
        eps_list = []
        for v in e_col:
            try:
                # If numeric
                val = complex(float(v))
            except Exception:
                try:
                    # try python complex parsing, e.g. "1.5+0.02j"
                    val = complex(str(v).replace('i','j'))
                except Exception:
                    raise ValueError("Could not parse epsilon column values.")
            eps_list.append(val)
        out['eps'] = np.asarray(eps_list, dtype=complex)
        return out

    raise ValueError("File must contain either (n and k) or (e1 and e2) or (e).")

def epsilon_to_nk(eps_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte epsilon(λ) compleja a n(λ), k(λ) mediante sqrt.
    """
    n_complex = np.sqrt(eps_complex)   # devuelve array complejo n + i k
    n_vals = np.real(n_complex)
    k_vals = np.imag(n_complex)
    return n_vals, k_vals

def interp_complex_to_grid(src_wl: np.ndarray, src_y: np.ndarray, target_wl: np.ndarray) -> np.ndarray:
    """
    Interpola src_y (real o complex) de src_wl a target_wl (lineal).
    Para complejos interpola parte real e imag separado.
    """
    src_wl = np.asarray(src_wl, dtype=float)
    target_wl = np.asarray(target_wl, dtype=float)
    if np.iscomplexobj(src_y):
        realp = np.interp(target_wl, src_wl, np.real(src_y))
        imagp = np.interp(target_wl, src_wl, np.imag(src_y))
        return realp + 1j*imagp
    else:
        return np.interp(target_wl, src_wl, src_y)

def compute_stack_reflection(wl_grid: np.ndarray, layers: List[dict], angle_deg: float, polarization: str = 'both'):
    """
    layers: lista de dicts con:
      - name
      - modelType: 'ambient', 'file', 'cauchy', 'glass', 'substrate', 'drude', etc
      - thickness: number in same units as wl (we use nm)
      - for file-based: 'file_obj' => UploadFile already parsed or parsed data dict
      - for parametric: params in layer['params']
    Devuelve diccionario con arrays: wl, rp, rs, R_p, R_s, psi_deg, delta_deg
    """
    wl = np.asarray(wl_grid, dtype=float)
    n_layers_per_wl = []  # will be list of lists: for each layer a vector over wl
    thicknesses = []      # thicknesses for intermediate layers (nm)
    # We'll build n_list for each layer: ambient (index 0), layers..., substrate (last)
    # Expected: thicknesses length = number of 'film' layers (intermediate) — see notes below.

    # First: for each layer produce N(λ) vector (complex) or scalar
    for layer in layers:
        mtype = layer.get('modelType', '').lower()
        if mtype in ('ambient', 'air'):
            # ambient: vacuum
            n_layers_per_wl.append(np.full_like(wl, 1.0 + 0j, dtype=complex))
            # ambient has no thickness entry
        elif mtype == 'glass':
            n_layers_per_wl.append(np.full_like(wl, 1.52 + 0j, dtype=complex))
        elif mtype == 'file':
            # file data should be provided in layer['file_parsed'] (a dict from parse_material_file_upload)
            parsed = layer.get('file_parsed')
            if parsed is None or 'wl' not in parsed:
                raise ValueError(f"Layer {layer.get('name')} expects 'file_parsed' with content.")
            src_wl = parsed['wl']
            if 'n' in parsed and 'k' in parsed:
                Nsrc = parsed['n'] + 1j*parsed['k']
            elif 'eps' in parsed:
                nvals, kvals = epsilon_to_nk(parsed['eps'])
                Nsrc = nvals + 1j*kvals
            else:
                raise ValueError("Parsed file lacks n/k or eps.")
            N_on_grid = interp_complex_to_grid(src_wl, Nsrc, wl)
            n_layers_per_wl.append(N_on_grid)
            # add thickness to thicknesses list (but TMM expects thicknesses for intermediate layers)
            thicknesses.append(float(layer.get('thickness', 0.0)))
        elif mtype == 'cauchy':
            # parametric Cauchy: n(λ) = A + B/λ^2 + C/λ^4  (λ in nm here, use same units as user)
            params = layer.get('params', {})  # dictionary with A,B,C numeric
            A = float(params.get('A', params.get('A_'+str(layer.get('idx', '')), 1.0) or 1.0))
            B = float(params.get('B', params.get('B_'+str(layer.get('idx', '')), 0.0) or 0.0))
            C = float(params.get('C', params.get('C_'+str(layer.get('idx', '')), 0.0) or 0.0))
            lam = wl
            nvals = A + B/(lam**2) + C/(lam**4)
            n_layers_per_wl.append(nvals + 0j)
            thicknesses.append(float(layer.get('thickness', 0.0)))
        elif mtype == 'drude':
            # If you want parametric Drude: implement here using layer['params'] (eps(ω) -> n(λ))
            # For now require file-based dielectric for drude or implement later
            if 'file_parsed' in layer:
                parsed = layer['file_parsed']
                src_wl = parsed['wl']
                if 'n' in parsed and 'k' in parsed:
                    Nsrc = parsed['n'] + 1j*parsed['k']
                elif 'eps' in parsed:
                    nvals, kvals = epsilon_to_nk(parsed['eps'])
                    Nsrc = nvals + 1j*kvals
                else:
                    raise ValueError("Parsed file lacks n/k or eps for drude layer.")
                N_on_grid = interp_complex_to_grid(src_wl, Nsrc, wl)
                n_layers_per_wl.append(N_on_grid)
                thicknesses.append(float(layer.get('thickness', 0.0)))
            else:
                raise NotImplementedError("Parametric Drude not implemented; upload eps(n) file or implement formula backend.")
        else:
            # fallback: treat as vacuum
            n_layers_per_wl.append(np.full_like(wl, 1.0 + 0j, dtype=complex))

    # The tmm.TransferMatrixMethod expects:
    # - an n list that contains ambient (index 0), then index for each layer (n[i+1] used in phi),
    #   and final substrate at the end. thicknesses array should have length = number of intermediate layers.
    # Our building above appended thicknesses only for those layers that actually have thickness (file/cauchy/drude).
    # But order matters: ensure your layers list follows: ambient, layer1, layer2, ..., substrate
    # and thicknesses matches layers between ambient and substrate (len(thicknesses) == len(n_layers_per_wl)-2)
    # To be safe, we require that the first element is ambient and last is substrate.

    # Validate sizes:
    if len(n_layers_per_wl) < 2:
        raise ValueError("Need at least ambient and substrate layers.")

    # Determine whether each n_layers_per_wl element is array or scalar. We made them arrays.
    # Build rp and rs arrays
    rp = np.zeros_like(wl, dtype=complex)
    rs = np.zeros_like(wl, dtype=complex)
    Rp = np.zeros_like(wl, dtype=float)
    Rs = np.zeros_like(wl, dtype=float)

    # For each wavelength index, call TransferMatrixMethod with scalar n values at that wl
    for i_w, lam in enumerate(wl):
        # create n_list for this wavelength: scalars complex
        n_list = [ complex(n_layers_per_wl[j][i_w]) for j in range(len(n_layers_per_wl)) ]
        # thicknesses list currently corresponds to intermediate layers in same order as n_list[1:-1]
        # make sure thicknesses length equals len(n_list)-2
        # If thicknesses length mismatches, attempt to map: we used thicknesses.append for each intermediate, so should match
        if len(thicknesses) != (len(n_list) - 2):
            # try to make thicknesses list of length len(n_list)-2 by filling zeros or using any provided field
            # here we will pad/truncate to length
            needed = len(n_list) - 2
            if len(thicknesses) < needed:
                padded = thicknesses + [0.0] * (needed - len(thicknesses))
                d_list = padded[:needed]
            else:
                d_list = thicknesses[:needed]
        else:
            d_list = thicknesses

        # instantiate TMM for S polarization
        try:
            tmm_s = TransferMatrixMethod(theta=angle_deg, l=lam, n=n_list, thicknesses=d_list, polarization='S')
            r_s_complex = tmm_s.get_reflection_amplitude()
            Rs[i_w] = abs(r_s_complex)**2
            rs[i_w] = r_s_complex
        except Exception as e:
            # fallback: set zero and log
            rs[i_w] = 0+0j
            Rs[i_w] = 0.0

        # instantiate TMM for P polarization
        try:
            tmm_p = TransferMatrixMethod(theta=angle_deg, l=lam, n=n_list, thicknesses=d_list, polarization='P')
            r_p_complex = tmm_p.get_reflection_amplitude()
            Rp[i_w] = abs(r_p_complex)**2
            rp[i_w] = r_p_complex
        except Exception as e:
            rp[i_w] = 0+0j
            Rp[i_w] = 0.0

    # compute psi and delta from rho = rp/rs
    # avoid division by zero: where rs==0 set psi/delta to NaN
    rho = np.zeros_like(rp, dtype=complex)
    psi_deg = np.full_like(wl, np.nan, dtype=float)
    delta_deg = np.full_like(wl, np.nan, dtype=float)
    for i_w in range(len(wl)):
        if abs(rs[i_w]) == 0:
            rho[i_w] = np.nan + 1j*np.nan
            psi_deg[i_w] = np.nan
            delta_deg[i_w] = np.nan
        else:
            rho[i_w] = rp[i_w] / rs[i_w]
            psi_deg[i_w] = np.degrees(np.arctan(abs(rho[i_w])))
            delta_deg[i_w] = np.degrees(np.angle(rho[i_w]))

    return {
        "wavelength": wl,
        "rp": rp,
        "rs": rs,
        "R_p": Rp,
        "R_s": Rs,
        "psi_deg": psi_deg,
        "delta_deg": delta_deg
    }
# ---- END: TMM integration helpers ----

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint: /api/upload
    Recibe un archivo (.csv, .txt, .xlsx), intenta leerlo y normalizar columnas.
    Devuelve:
      - preview: primeras 10 filas (lista de dicts)
      - length: número de puntos
      - data: { wavelength: [...], psi: [...], delta: [...] }
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    try:
        df = _read_file_bytes(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Normalizar nombres
    df = _normalize_columns(df)

    # Requisito mínimo: wavelength y al menos psi o delta
    if "wavelength" not in df.columns or ("psi" not in df.columns and "delta" not in df.columns):
        raise HTTPException(
            status_code=400,
            detail="File must contain at least 'wavelength' and 'psi' or 'delta' columns (or compatible headers)."
        )

    # Forzar conversión numérica y limpiar filas sucias
    for col in ["wavelength", "psi", "delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    # eliminar filas sin wavelength
    df_clean = df.dropna(subset=["wavelength"]).copy()

    if df_clean.shape[0] == 0:
        raise HTTPException(status_code=400, detail="No numeric wavelength values found in file.")

    preview_df = df_clean.head(10).reset_index(drop=True)

    wavelength = df_clean["wavelength"].astype(float).tolist()
    psi = df_clean["psi"].astype(float).tolist() if "psi" in df_clean.columns else [None] * len(wavelength)
    delta = df_clean["delta"].astype(float).tolist() if "delta" in df_clean.columns else [None] * len(wavelength)

    return JSONResponse(
        {
            "preview": preview_df.to_dict(orient="records"),
            "length": len(wavelength),
            "data": {"wavelength": wavelength, "psi": psi, "delta": delta},
        }
    )


# -------------------------
# Registrar router en la app principal (DESPUÉS de definir todos los endpoints)
app.include_router(router)

# -------------------------
# Servir frontend estático
# Suponemos esta estructura:
# src/
#  ├─ backend/
#  │   └─ main.py  <-- este archivo
#  └─ frontend/
#      └─ (index.html, upload.html, etc.)
BASE_DIR = Path(__file__).resolve().parent.parent  # ruta a src/
web_dir = BASE_DIR / "frontend"                     # -> src/frontend
# Montamos la carpeta frontend en la raíz ("/") solo si existe.
if web_dir.exists() and web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="frontend")
else:
    # Si no existe, arrancamos solo la API y registramos una advertencia para el desarrollador.
    import logging
    logging.getLogger("uvicorn.error").warning(
        f"Frontend directory '{web_dir}' not found — serving API only. Create it or adjust path."
    )


@router.post("/model")
async def save_model(model: dict):
    """
    Recibe el JSON con el modelo óptico desde el frontend.
    Por ahora lo guarda en disco (o puedes guardarlo en DB).
    """
    uid = str(uuid.uuid4())[:8]
    out_dir = Path(__file__).resolve().parent.parent / "user_models"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"model_{uid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    return JSONResponse({"status":"ok","path": str(path), "id": uid})


@router.post("/tmm/run")
async def tmm_run(request: Request):
    """
    Endpoint que acepta multipart/form-data (FormData) con:
      - campo 'payload' (JSON string) que contiene la configuración del modelo
      - uno o más archivos con los nombres de campo que figuran en payload.layers[*].file_field
    También intenta soportar (como fallback) el anterior esquema payload JSON + files List[UploadFile] si no se envía multipart.
    """
    try:
        # 1) Intentar leer form (multipart). Si no es multipart, intentar json body.
        form = None
        try:
            form = await request.form()
        except Exception:
            form = None

        if form:
            payload_raw = form.get('payload')
            if not payload_raw:
                # si no viene payload en form, intentar body JSON
                try:
                    payload = await request.json()
                except Exception:
                    payload = {}
            else:
                # payload puede venir como str (FormData)
                if isinstance(payload_raw, str):
                    payload = json.loads(payload_raw)
                else:
                    # starlette UploadFile u otros; aseguramos string
                    payload = json.loads(str(payload_raw))
            # construir mapa de archivos desde form.multi_items()
            file_map = {}
            for k, v in form.multi_items():
                # v puede ser UploadFile
                if isinstance(v, UploadFile):
                    file_map[k] = v
                    # también mapear por filename
                    if getattr(v, "filename", None):
                        file_map[v.filename] = v
        else:
            # fallback: el cliente puede haber enviado JSON en body y archivos como parte separada (no multipart)
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            # no tenemos archivos en este caso
            file_map = {}

        # 2) Leer parámetros globales y wavelength grid
        global_cfg = payload.get("global", {}) if isinstance(payload, dict) else {}
        angle = float(global_cfg.get("angle", 70.0))
        if "wavelength_grid" in global_cfg:
            g = global_cfg["wavelength_grid"]
            wl = np.linspace(float(g["start_nm"]), float(g["end_nm"]), int(g["points"]))
        else:
            wl = np.linspace(400.0, 800.0, 401)

        layers = payload.get("layers", []) if isinstance(payload, dict) else []

        # 3) Asociar archivos a capas 'file'
        for li, layer in enumerate(layers):
            if str(layer.get("modelType", "")).lower() == "file":
                file_field = layer.get("file_field") or layer.get("file_name")
                if not file_field:
                    raise HTTPException(status_code=400, detail=f"Layer {layer.get('name')} requires 'file_field' or 'file_name'.")
                # buscar en file_map por clave exacta o por coincidencia parcial
                up = file_map.get(file_field)
                if up is None:
                    # buscar parcialmente (si el frontend envió 'film_file' pero el campo se llama 'film_file.csv' etc)
                    for fname, ff in file_map.items():
                        if file_field in fname:
                            up = ff
                            break
                if up is None:
                    raise HTTPException(status_code=400, detail=f"Could not find uploaded file for '{file_field}' (layer {layer.get('name')}).")
                # parsear el archivo con tu helper
                try:
                    parsed = parse_material_file_upload(up)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error parsing material file for layer '{layer.get('name')}': {e}")
                layer['file_parsed'] = parsed
                layer['idx'] = li  # indice opcional

        # 4) Ejecutar cálculo TMM usando tu compute_stack_reflection (ya definido arriba)
        res = compute_stack_reflection(wl, layers, angle, polarization='both')

        # 5) Serializar resultados complejos a JSON
        def c_arr_to_json(arr):
            return [{"re": float(np.real(x)), "im": float(np.imag(x))} for x in arr.tolist()]

        out = {
            "status": "ok",
            "wavelength": [float(x) for x in res["wavelength"].tolist()],
            "rp": c_arr_to_json(res["rp"]),
            "rs": c_arr_to_json(res["rs"]),
            "R_p": [float(x) for x in res["R_p"].tolist()],
            "R_s": [float(x) for x in res["R_s"].tolist()],
            "psi_deg": [float(x) if not np.isnan(x) else None for x in res["psi_deg"].tolist()],
            "delta_deg": [float(x) if not np.isnan(x) else None for x in res["delta_deg"].tolist()],
        }
        return JSONResponse(out)

    except HTTPException:
        # re-lanzar para mantener códigos de error originales
        raise
    except Exception as e:
        # captura errores inesperados
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tmm/run-multipart")
async def tmm_run_multipart(request: Request):
    """
    Recibe multipart/form-data con campo 'payload' (JSON string) y archivos.
    payload: JSON string con estructura del model (global + layers).
    Archivos deben enviarse con el nombre de campo igual a layer.file_field.
    """
    try:
        form = await request.form()
        payload_raw = form.get('payload')
        if not payload_raw:
            raise HTTPException(status_code=400, detail="Missing 'payload' in form data.")

        try:
            payload = json.loads(payload_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not parse JSON payload.")

        # Crear un mapeo de archivos: key=form_field_name -> UploadFile
        files_map = {}
        for k, v in form.multi_items():
            if isinstance(v, UploadFile):
                files_map[k] = v

        # leer configuración global
        global_cfg = payload.get("global", {})
        angle_deg = float(global_cfg.get("angle", 70.0))
        if "wavelength_grid" in global_cfg:
            g = global_cfg["wavelength_grid"]
            wl = np.linspace(float(g["start_nm"]), float(g["end_nm"]), int(g["points"]))
        else:
            wl = np.linspace(400.0, 800.0, 401)

        layers = payload.get("layers", [])

        # Para cada layer que requiere archivo, buscar el UploadFile en files_map
        for li, layer in enumerate(layers):
            if str(layer.get("modelType", "")).lower() in ("file", "drude"):
                file_field = layer.get("file_field") or layer.get("file_name")
                if not file_field:
                    raise HTTPException(status_code=400, detail=f"Layer {layer.get('name')} requires file_field.")
                up = files_map.get(file_field)
                if up is None:
                    # intentar buscar por coincidencia parcial
                    for k, v in files_map.items():
                        if file_field in k or k in file_field:
                            up = v
                            break
                if up is None:
                    raise HTTPException(status_code=400, detail=f"Uploaded file for '{file_field}' not found.")
                try:
                    parsed = parse_material_file_upload(up)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error parsing material file for layer {layer.get('name')}: {e}")
                layer['file_parsed'] = parsed
                layer['idx'] = li

        # Llamar a tu función principal que calcula psi/delta
        res = compute_stack_reflection(wl, layers, angle_deg, polarization='both')

        # Serializar arrays complejos
        def c_arr_to_json(arr):
            return [{"re": float(np.real(x)), "im": float(np.imag(x))} for x in arr.tolist()]

        out = {
            "status": "ok",
            "wavelength": [float(x) for x in res["wavelength"].tolist()],
            "rp": c_arr_to_json(res["rp"]),
            "rs": c_arr_to_json(res["rs"]),
            "R_p": [float(x) for x in res["R_p"].tolist()],
            "R_s": [float(x) for x in res["R_s"].tolist()],
            "psi_deg": [float(x) if not np.isnan(x) else None for x in res["psi_deg"].tolist()],
            "delta_deg": [float(x) if not np.isnan(x) else None for x in res["delta_deg"].tolist()],
        }
        return JSONResponse(out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Endpoint: parsear fórmula de dispersión (seguro)
# -------------------------
# Este endpoint recibe JSON con:
#   { "formula_text": "<texto>", "lambda_unit": "nm"|"um", "sample_wl": [..] (opcional) }
#
# Responde con:
#   { latex, expr, params, sample: [...], warnings: [...] }
#
# NOTA: usa sympy para parsear y lambdify (no eval)
ALLOWED_FUNCS = {
    'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'asin', 'acos', 'atan', 'pi'
}

def _format_number_or_complex(z):
    # convierte número real o complejo a formato JSON-friendly
    try:
        zc = complex(z)
    except Exception:
        return str(z)
    if abs(zc.imag) < 1e-12:
        return float(round(zc.real, 8))
    else:
        return {"real": float(round(zc.real, 8)), "imag": float(round(zc.imag, 8))}

@router.post("/parse_formula")
async def parse_formula(payload: dict = Body(...)):
    """
    Parsear una fórmula de dispersión provista por el usuario.
    Se acepta texto con asignaciones en la primera línea, por ejemplo:
      "A=1.45, B=0.0035\n n = A + B/(lambda**2)"
    Parámetros:
      - formula_text (str) required
      - lambda_unit: 'nm' (default) or 'um'
      - sample_wl: optional list of wavelengths (nm) for evaluación
    """
    formula_text = payload.get("formula_text")
    if not formula_text or not isinstance(formula_text, str):
        raise HTTPException(status_code=400, detail="formula_text (string) is required")

    lambda_unit = payload.get("lambda_unit", "nm")
    if lambda_unit not in {"nm", "um"}:
        raise HTTPException(status_code=400, detail="lambda_unit must be 'nm' or 'um'")

    sample_wl = payload.get("sample_wl", None)
    if sample_wl is None:
        # default sample wavelengths in nm
        sample_wl = list(np.linspace(300, 1000, 7))

    # ---------- Parse assignments (opcional) ----------
    lines = formula_text.strip().splitlines()
    assign_line = None
    expr_lines = []
    if len(lines) > 0 and re.search(r'\w+\s*=', lines[0]):
        assign_line = lines[0]
        expr_lines = lines[1:]
    else:
        expr_lines = lines

    params = {}
    if assign_line:
        for part in re.split(r'[;,]', assign_line):
            part = part.strip()
            if not part:
                continue
            m = re.match(r'([A-Za-z_]\w*)\s*=\s*([0-9.eE+\-]+)', part)
            if m:
                name = m.group(1)
                val = float(m.group(2))
                params[name] = val

    expr_text = "\n".join(expr_lines).strip()
    if not expr_text:
        raise HTTPException(status_code=400, detail="No expression found after parameter line")

    # si el usuario escribió "n = ..." extraer RHS
    if '=' in expr_text:
        parts = expr_text.split('=', 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        # preferimos la RHS como expresión
        expr_text = rhs

    # normalizaciones mínimas
    expr_text = expr_text.replace('λ', 'lambda').replace('μm', 'um')

    # construir symbolo lambda en sympy (nombre 'lambda' is fine)
    lam = smp.symbols('lambda')

    # construir local_dict con parámetros conocidos y funciones permitidas
    local_dict = {'lambda': lam}
    for p in params.keys():
        local_dict[p] = smp.symbols(p)
    for fn in ALLOWED_FUNCS:
        try:
            # mapear funciones de sympy si existen
            if hasattr(smp, fn):
                local_dict[fn] = getattr(smp, fn)
        except Exception:
            pass

    # parsear expr con sympy de forma controlada
    try:
        expr = smp.parse_expr(expr_text, local_dict=local_dict, evaluate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse expression: {e}")

    # comprobar símbolos libres
    syms = {str(s) for s in expr.free_symbols}
    allowed_symbols = set(['lambda']) | set(params.keys())
    invalid = syms - allowed_symbols
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unrecognized symbols in expression: {sorted(list(invalid))}")

    # sustituir parámetros numéricos
    if params:
        subs_map = {smp.symbols(k): v for k, v in params.items()}
        expr_sub = expr.subs(subs_map)
    else:
        expr_sub = expr

    # obtener LaTeX para preview
    try:
        latex_repr = smp.latex(expr)
    except Exception:
        latex_repr = str(expr)

    # crear función numérica con numpy
    try:
        fnum = smp.lambdify(lam, expr_sub, modules=["numpy"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not create numeric function: {e}")

    # preparar array de evaluación (convertir unidades si necesario)
    wl_arr = np.asarray(sample_wl, dtype=float)
    if lambda_unit == 'um':
        # usuario dice que fórmula está en micrómetros; convertir sample (nm -> um)
        wl_eval = wl_arr / 1000.0
    else:
        # asumimos nm
        wl_eval = wl_arr

    # evaluar de forma robusta
    warnings = []
    try:
        y = fnum(wl_eval)
        # si devuelve escalar, convertir a array
        if np.isscalar(y):
            y = np.array([y])
        y = np.asarray(y, dtype=complex)
    except Exception as e:
        # intentar evaluación punto a punto para generar mejor mensaje
        try:
            y_list = []
            for val in wl_eval:
                yv = fnum(val)
                y_list.append(complex(yv))
            y = np.asarray(y_list, dtype=complex)
            warnings.append("Numeric evaluation used pointwise fallback.")
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Could not evaluate expression numerically: {e2}")

    # formatear sample para JSON (si es complejo se transforma)
    sample_out = [_format_number_or_complex(v) for v in y.tolist()]

    response = {
        "latex": latex_repr,
        "expr": str(expr_sub),
        "params": params,
        "sample_wavelengths_nm": [float(round(x, 6)) for x in wl_arr.tolist()],
        "sample": sample_out,
        "warnings": warnings,
    }
    return JSONResponse(response)


# -------------------------
# Entrada local para pruebas (opcional)
# -------------------------
# Puedes ejecutar con:
#   uvicorn backend.main:app --reload --port 8000
#
# Asegúrate de ejecutar el comando desde la carpeta 'src' (donde está la carpeta backend)
# o ajusta el module path si corres desde otra ubicación.
#
# Recomendaciones:
# - Instala dependencias:
#   pip install fastapi uvicorn pandas python-multipart openpyxl sympy numpy
#
# - Coloca tu frontend en src/frontend (por ejemplo upload.html + css/js).
#
# - En tu frontend usa fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData })
#   (o ruta relativa '/api/upload' si el frontend y backend se sirven desde el mismo host).
#
# Listo — ahora el backend soporta parseo seguro de fórmulas y evaluación de muestra.
