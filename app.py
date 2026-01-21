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
    if "CERR" in t or "APERTURA" in t or t == "APERTURA_AUTO":
        return "CERRAJERIA"
    if "PASO" in t and "CORRIENTE" in t:
        return "PASO_CORRIENTE"
    if t == "CORRIENTE" or t == "BATERIA" or t == "ARRANQUE":
        return "PASO_CORRIENTE"
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

    try:
        df = pd.read_excel(DATA_FILE, dtype=str)  # todo como texto para no perder teléfonos
    except Exception as e:
        raise ValueError(f"Error al leer el archivo Excel {DATA_FILE}: {str(e)}") from e

    # Normaliza nombres de columnas: quita espacios, convierte a minúsculas
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Verifica que el DataFrame no esté vacío
    if df.empty:
        raise ValueError(f"El archivo Excel {DATA_FILE} está vacío o no contiene datos")

    # Asegura columnas mínimas (municipio puede venir como "ciudad" en algunos Excel)
    if COL_MUNICIPIO not in df.columns:
        if "ciudad" in df.columns:
            df[COL_MUNICIPIO] = df["ciudad"]  # fallback: ciudad -> municipio
        else:
            raise ValueError(f"Falta columna requerida '{COL_MUNICIPIO}' (o 'ciudad') en {DATA_FILE}. Columnas disponibles: {list(df.columns)}")

    # Verifica otras columnas requeridas
    for col in [COL_ESTADO, COL_PROVEEDOR, COL_SERVICIO, COL_LAT, COL_LON]:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida '{col}' en {DATA_FILE}. Columnas disponibles: {list(df.columns)}")

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
        try:
            DATA_DF = read_data()
            TEL_COLS = pick_tel_columns(DATA_DF)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al cargar los datos: {str(e)}"
            ) from e


# =========================
# API MODELS
# =========================
class Punto(BaseModel):
    lat: float
    lon: float


class RecomendarRequest(BaseModel):
    servicio: str = Field(..., description="GRUA, LLANTA, GASOLINA, CERRAJERIA, PASO_CORRIENTE, APERTURA_AUTO")
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
    if DATA_DF is None:
        raise HTTPException(status_code=500, detail="Error: datos no cargados")
    by_service = DATA_DF[COL_SERVICIO].value_counts(dropna=False).to_dict()
    
    # Contar cuántos tienen coordenadas válidas
    with_coords = DATA_DF[(DATA_DF[COL_LAT].notna()) & (DATA_DF[COL_LON].notna())]
    by_service_with_coords = with_coords[COL_SERVICIO].value_counts(dropna=False).to_dict()
    
    return {
        "rows": int(len(DATA_DF)),
        "rows_with_coords": int(len(with_coords)),
        "by_service": by_service,
        "by_service_with_coords": by_service_with_coords,
        "data_file": DATA_FILE,
        "tel_cols": TEL_COLS,
        "unique_services": sorted(DATA_DF[COL_SERVICIO].dropna().unique().tolist()),
    }


@app.post("/recomendar", response_model=List[ProveedorOut])
def recomendar(req: RecomendarRequest):
    ensure_loaded()
    if DATA_DF is None:
        raise HTTPException(status_code=500, detail="Error: datos no cargados")

    servicio = normalize_service(req.servicio)
    servicios_validos = {"GRUA", "LLANTA", "GASOLINA", "CERRAJERIA", "PASO_CORRIENTE"}
    if servicio not in servicios_validos:
        raise HTTPException(status_code=422, detail=f"Servicio inválido: {req.servicio}. Servicios válidos: {', '.join(sorted(servicios_validos))}")

    df = DATA_DF.copy()
    
    # Debug: ver qué servicios hay en el Excel
    servicios_disponibles = df[COL_SERVICIO].unique().tolist()
    
    df = df[df[COL_SERVICIO] == servicio]
    
    # Debug: contar antes de filtrar geocodificados
    count_antes_geo = len(df)

    # Filtra geocodificados
    df = df[df[COL_LAT].notna() & df[COL_LON].notna()]
    
    # Debug: contar después de filtrar geocodificados
    count_despues_geo = len(df)

    if df.empty:
        # Log útil para debug (en producción podrías usar logging)
        print(f"[DEBUG] Servicio '{servicio}' (normalizado de '{req.servicio}'): "
              f"0 resultados. Servicios disponibles en Excel: {servicios_disponibles[:10]}. "
              f"Antes de filtrar geo: {count_antes_geo}, después: {count_despues_geo}")
        return []

    o_lat, o_lon = req.origen.lat, req.origen.lon

    # Validación de coordenadas de origen
    if not (-90 <= o_lat <= 90) or not (-180 <= o_lon <= 180):
        raise HTTPException(status_code=422, detail="Coordenadas de origen inválidas")

    # distancia: por ahora usamos origen (para grúa podrías mejorar con promedio origen/destino)
    try:
        df["_dist_km"] = df.apply(lambda r: haversine_km(o_lat, o_lon, float(r[COL_LAT]), float(r[COL_LON])), axis=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al calcular distancias: {str(e)}") from e

    if req.max_km is not None:
        df = df[df["_dist_km"] <= float(req.max_km)]

    df = df.sort_values("_dist_km", ascending=True).head(int(req.top_n))

    out: List[ProveedorOut] = []
    for _, r in df.iterrows():
        try:
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

            # Validación de coordenadas antes de agregar
            lat_val = float(r[COL_LAT])
            lon_val = float(r[COL_LON])
            if not (-90 <= lat_val <= 90) or not (-180 <= lon_val <= 180):
                continue  # Salta filas con coordenadas inválidas

            out.append(
                ProveedorOut(
                    proveedor=str(r.get(COL_PROVEEDOR, "")).strip(),
                    servicio=servicio,
                    dist_km=float(r["_dist_km"]),
                    estado=str(r.get(COL_ESTADO, "")).strip(),
                    municipio=str(r.get(COL_MUNICIPIO, "")).strip(),
                    ubicacion=ubic,
                    lat=lat_val,
                    lon=lon_val,
                    telefonos=tels,
                    email=email,
                    notas=notas,
                )
            )
        except Exception as e:
            # Log del error pero continúa con las demás filas
            # En producción podrías usar logging aquí
            continue
    return out
