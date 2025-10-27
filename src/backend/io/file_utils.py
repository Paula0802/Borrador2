# file_utils.py - leer formatos nk y reflectance
import pandas as pd
import numpy as np
from pathlib import Path

def read_nk_file(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path, sep=None, engine='python', comment='#')
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type")
    cols = [c.lower() for c in df.columns]
    if "wl" in cols and "n" in cols:
        wl = df.iloc[:, cols.index("wl")].values.astype(float)
        n = df.iloc[:, cols.index("n")].values.astype(float)
        if "k" in cols:
            k = df.iloc[:, cols.index("k")].values.astype(float)
        else:
            k = np.zeros_like(n)
        return wl, n, k
    else:
        # try plain two/three column detection
        arr = df.values
        if arr.shape[1] >= 2:
            wl = arr[:,0].astype(float)
            n = arr[:,1].astype(float)
            if arr.shape[1] >= 3:
                k = arr[:,2].astype(float)
            else:
                k = np.zeros_like(n)
            return wl, n, k
        else:
            raise ValueError("File format not recognized")

def read_reflectance_file(path):
    # expects two columns: x (wl or angle), R (0..1)
    path = Path(path)
    ext = path.suffix.lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path, sep=None, engine='python', comment='#', header=None)
        arr = df.values
        x = arr[:,0].astype(float)
        y = arr[:,1].astype(float)
        return x, y
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path, header=None)
        arr = df.values
        return arr[:,0].astype(float), arr[:,1].astype(float)
    else:
        raise ValueError("Unsupported")
