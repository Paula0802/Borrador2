# src/backend/main.py
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import io
import os

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


def _read_file_bytes(file: UploadFile) -> pd.DataFrame:
    """
    Intenta leer .csv, .txt (tsv o whitespace) y .xlsx y devuelve un DataFrame de pandas.
    """
    filename = (file.filename or "").lower()
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


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint: /api/upload
    Recibe un archivo (.csv, .txt, .xlsx), intenta leerlo y normalizar columnas.
    Devuelve:
      - preview: primeras 5 filas (lista de dicts)
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


# Registrar router en la app principal
app.include_router(router)

# -------------------------
# Servir frontend estático
# -------------------------
# Suponemos esta estructura:
# src/
#  ├─ backend/
#  │   └─ main.py  <-- este archivo
#  └─ frontend/
#      └─ (index.html, upload.html, etc.)
BASE_DIR = Path(__file__).resolve().parent.parent  # ruta a src/
web_dir = BASE_DIR / "frontend"                     # -> src/frontend
if not web_dir.exists():
    # Si no existe la carpeta, intentamos una ruta alternativa (por si ejecutas desde otra ubicación)
    # pero preferimos lanzar error para que el desarrollador lo arregle.
    raise RuntimeError(f"Frontend directory '{web_dir}' does not exist. Create it or adjust path.")
# Montamos la carpeta frontend en la raíz ("/")
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="frontend")


@router.post("/model")
async def save_model(model: dict):
    """
    Recibe el JSON con el modelo óptico desde el frontend.
    Por ahora lo guarda en disco (o puedes guardarlo en DB).
    """
    import json, uuid, os
    uid = str(uuid.uuid4())[:8]
    out_dir = Path(__file__).resolve().parent.parent / "user_models"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"model_{uid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    return JSONResponse({"status":"ok","path": str(path), "id": uid})


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
#   pip install fastapi uvicorn pandas python-multipart openpyxl
#
# - Coloca tu frontend en src/frontend (por ejemplo upload.html + css/js).
#
# - En tu frontend usa fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData })
#   (o ruta relativa '/api/upload' si el frontend y backend se sirven desde el mismo host).
#
# Eso es todo — este archivo fusiona el comportamiento que solicitaste.
