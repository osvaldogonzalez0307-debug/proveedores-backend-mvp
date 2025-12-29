from __future__ import annotations

import os
import re
from typing import Optional, List, Dict, Any
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Usa el geocodificado por OSM (como ya lo traes)
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "proveedores_mvp_geocoded_osm.xlsx")

# Puedes cambiarlo con variable de entorno si un día usas otro nombre:
# set DATA_FILE=C:\...\archivo.xlsx
DATA_FILE = os.environ.get("DATA_FILE", DEFAULT_DATA_FILE)

# Columnas esperadas (tolerante a variaciones)
COL_ESTADO = "estado"
COL_MUNICIPIO = "municipio"
COL_PROVEEDOR = "proveedor"
COL_SERVICIO = "servicio_std"  # GRUA/LLANTA/GASOLINA/CERRAJERIA
COL_LAT = "lat"
COL_LON = "lon"

# Teléfonos (pueden existir tel_1..tel_n)
TEL_PREFIX = "tel_"
COL_EMAIL = "email"
COL_NOTAS = "notas"  # si existe, se usa

# Limpieza proveedor: quitar prefijo "costos" (en cualquier combinación)
COSTOS_RE = re.compile(r"^\s*costos\s*", re.IGNORECASE)


# =========================
# UTILS
# =========================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2) + cos(radians(lat1)) * cos(radians(lat2)) * (sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def clean_provider_name(name: Any) -> str:
    s = "" if pd.isna(name) else str(name)
    s = COSTOS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_service(s: Any) -> str:
    if pd.isna(s):
        return ""
    t = str(s).strip().upper()
    # Normalizaciones por si vienen variantes
    if "GRU" in t:
        return "GRUA"
    if "LLAN" in t:
        return "LLANTA"
    if "GAS" in t:
        return "GASOLINA"
    if "CERR" in t:
        return "CERRAJERIA"
    return t


def pick_tel_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).lower().startswith(TEL_PREFIX)]
    # ordenar tel_1, tel_2...
    def key(c: str):
        m = re.search(r"(\d+)$", c)
        return int(m.group(1)) if m else 999
    return sorted(cols, key=key)


def read_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No existe el archivo de datos: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE, dtype=str)  # todo como texto para no perder teléfonos
    # Normaliza nombres de columnas a minúsculas (por si vienen raras)
    df.columns = [str(c).strip() for c in df.columns]

    # Asegura columnas mínimas
    for col in [COL_ESTADO, COL_MUNICIPIO, COL_PROVEEDOR, COL_SERVICIO, COL_LAT, COL_LON]:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida '{col}' en {DATA_FILE}")

    # Limpieza y casting
    df[COL_PROVEEDOR] = df[COL_PROVEEDOR].apply(clean_provider_name)
    df[COL_SERVICIO] = df[COL_SERVICIO].apply(normalize_service)

    # Lat/Lon a float (tolerante)
    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")

    # Limpia estado/municipio
    df[COL_ESTADO] = df[COL_ESTADO].fillna("").astype(str).str.strip()
    df[COL_MUNICIPIO] = df[COL_MUNICIPIO].fillna("").astype(str).str.strip()

    return df


# Cache en memoria
DATA_DF: Optional[pd.DataFrame] = None
TEL_COLS: List[str] = []


def ensure_loaded():
    global DATA_DF, TEL_COLS
    if DATA_DF is None:
        DATA_DF = read_data()
        TEL_COLS = pick_tel_columns(DATA_DF)


# =========================
# API MODELS
# =========================
class Punto(BaseModel):
    lat: float
    lon: float


class RecomendarRequest(BaseModel):
    servicio: str = Field(..., description="GRUA, LLANTA, GASOLINA, CERRAJERIA")
    top_n: int = Field(10, ge=1, le=50)
    max_km: Optional[float] = Field(None, ge=0)
    origen: Punto
    destino: Optional[Punto] = None  # solo para grúa


class ProveedorOut(BaseModel):
    proveedor: str
    servicio: str
    dist_km: float
    estado: str
    municipio: str
    ubicacion: str
    lat: float
    lon: float
    telefonos: List[str]
    email: Optional[str] = None
    notas: str = ""


# =========================
# APP
# =========================
app = FastAPI(title="API Proveedores", version="1.0")

# CORS: para que funcione con Live Server/Netlify sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en producción luego se restringe a tu dominio
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "message": "API Proveedores activa"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
def stats():
    ensure_loaded()
    assert DATA_DF is not None
    by_service = DATA_DF[COL_SERVICIO].value_counts(dropna=False).to_dict()
    return {
        "rows": int(len(DATA_DF)),
        "by_service": by_service,
        "data_file": DATA_FILE,
        "tel_cols": TEL_COLS,
    }


@app.post("/recomendar", response_model=List[ProveedorOut])
def recomendar(req: RecomendarRequest):
    ensure_loaded()
    assert DATA_DF is not None

    servicio = normalize_service(req.servicio)
    if servicio not in {"GRUA", "LLANTA", "GASOLINA", "CERRAJERIA"}:
        raise HTTPException(status_code=422, detail=f"Servicio inválido: {req.servicio}")

    df = DATA_DF.copy()
    df = df[df[COL_SERVICIO] == servicio]

    # Filtra geocodificados
    df = df[df[COL_LAT].notna() & df[COL_LON].notna()]

    if df.empty:
        return []

    o_lat, o_lon = req.origen.lat, req.origen.lon

    # distancia: por ahora usamos origen (para grúa podrías mejorar con promedio origen/destino)
    df["_dist_km"] = df.apply(lambda r: haversine_km(o_lat, o_lon, float(r[COL_LAT]), float(r[COL_LON])), axis=1)

    if req.max_km is not None:
        df = df[df["_dist_km"] <= float(req.max_km)]

    df = df.sort_values("_dist_km", ascending=True).head(int(req.top_n))

    out: List[ProveedorOut] = []
    for _, r in df.iterrows():
        tels: List[str] = []
        for c in TEL_COLS:
            val = r.get(c, None)
            if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
                continue
            s = str(val).strip()
            if not s:
                continue
            # limpia separadores raros
            s = re.sub(r"\s+", "", s)
            # deja solo dígitos si trae basura
            s_digits = re.sub(r"[^\d]", "", s)
            if len(s_digits) >= 7:
                tels.append(s_digits)

        ubic = f"{r.get(COL_ESTADO,'').strip()} / {r.get(COL_MUNICIPIO,'').strip()}".strip(" /")

        notas = ""
        if COL_NOTAS in DATA_DF.columns:
            notas = "" if pd.isna(r.get(COL_NOTAS)) else str(r.get(COL_NOTAS)).strip()

        email = None
        if COL_EMAIL in DATA_DF.columns:
            ev = r.get(COL_EMAIL)
            if ev is not None and not pd.isna(ev):
                email = str(ev).strip() or None

        out.append(
            ProveedorOut(
                proveedor=str(r.get(COL_PROVEEDOR, "")).strip(),
                servicio=servicio,
                dist_km=float(r["_dist_km"]),
                estado=str(r.get(COL_ESTADO, "")).strip(),
                municipio=str(r.get(COL_MUNICIPIO, "")).strip(),
                ubicacion=ubic,
                lat=float(r[COL_LAT]),
                lon=float(r[COL_LON]),
                telefonos=tels,
                email=email,
                notas=notas,
            )
        )
    return out
