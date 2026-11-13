"""Caricamento configurazione e percorsi del progetto."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / os.environ.get("METEOITA_DATA", "data")   # demo: METEOITA_DATA=demo_data
FC_DIR = DATA / "forecasts"          # previsioni orarie archiviate (parquet mensili)
OBS_DIR = DATA / "observations"      # osservazioni orarie (parquet mensili)
CSV_DIR = DATA / "csv"               # export leggibili (CSV / Excel)
BUILD = ROOT / "build"               # file generati per il sito (non versionati)
SITE_DATA = BUILD / "site" / "data"

with open(ROOT / "config.yaml", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

LEADS = CFG["lead_days"]
AGGS = CFG["aggregations"]


def providers(kind=None, only_active=True):
    """Lista dei provider dal config. kind = 'openmeteo' | 'snapshot' | None."""
    out = []
    for p in CFG["providers"]:
        if only_active and not p.get("enabled", True):
            continue
        if kind and p["type"] != kind:
            continue
        if only_active and p.get("api_key_env") and not os.environ.get(p["api_key_env"]):
            continue                      # chiave mancante -> provider saltato
        out.append(p)
    return out


def gridded_ids():
    """Provider disponibili su qualsiasi punto del loro dominio (mappe, GPR, interrogazioni)."""
    return [p["id"] for p in providers("openmeteo")]


def provider_names():
    return {p["id"]: p["name"] for p in CFG["providers"]}


def target_vars(agg):
    """Variabili verificate per ciascuna aggregazione."""
    v = ["t2m", "prcp"]
    if agg == "24h":
        v = ["t2m", "tmax", "tmin", "prcp"]
    return v


for _d in (FC_DIR, OBS_DIR, CSV_DIR, SITE_DATA):
    _d.mkdir(parents=True, exist_ok=True)
