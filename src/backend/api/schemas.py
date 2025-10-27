# schemas.py - pydantic models
from pydantic import BaseModel
from typing import List, Optional

class ModelDefine(BaseModel):
    name: str
    model: str  # e.g., "cauchy", "drude", "lorentz", "sellmeier"
    parameters: dict

class GeometryDefine(BaseModel):
    geometry_name: str
    layers: List[str]        # list of material names, including superstrate and substrate
    thicknesses: List[float] # nm for middle layers only
    angle_deg: float
    wavelength: dict         # {"start_nm":400,"end_nm":800,"points":401}
    polarization: str = "s"

class FitRequest(BaseModel):
    geometry_name: str
    exp_file_path: str
    fit_targets: List[str]      # e.g., ["thickness[0]","film.model.B"]
    initial_guess: List[float]
    bounds: Optional[List[List[float]]] = None
    algorithm: Optional[str] = "least_squares"
