"""Regioni (Italia, Mondo, ...): poligoni, stazioni e provider di ciascuna."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import box, shape
from shapely.prepared import prep

from .config import CFG, ROOT, providers

REGIONS = [r for r in CFG["regions"] if r.get("enabled", True)]
RID = [r["id"] for r in REGIONS]


def is_points(rid):
    return region(rid).get("type") == "points"


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def sites(rid):
    return pd.DataFrame(region(rid)["sites"])


def verified(rid):
    """La regione viene verificata con osservazioni? (default si')"""
    return region(rid).get("verify", True)


def archive_points():
    """Siti delle regioni non verificate: le previsioni vi vengono solo archiviate.
    Formato come stations.csv (station = id del sito)."""
    rows = []
    for r in REGIONS:
        if r.get("type") == "points" and not r.get("verify", True):
            for s in r["sites"]:
                rows.append({"station": s["id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"],
                             "elev": s.get("elev", 0), "region": r["id"]})
    df = pd.DataFrame(rows, columns=["station", "name", "lat", "lon", "elev", "region"])
    df["tz"] = station_tz(df) if len(df) else []
    return df


def region(rid):
    return next(r for r in CFG["regions"] if r["id"] == rid)


@lru_cache(None)
def polygon(rid, buffer=None):
    """Poligono della regione; di default allargato un po' per includere
    stazioni costiere e piccole isole."""
    with open(ROOT / region(rid)["polygon"]) as f:
        gj = json.load(f)
    g = shape(gj["features"][0]["geometry"])
    if rid == "mondo":                            # esclusa l'Antartide
        g = g.intersection(box(-180, -60, 180, 84))
    b = (0.25 if rid != "mondo" else 0.5) if buffer is None else buffer
    return g.buffer(b) if b else g


def inside(rid, lat, lon):
    """Maschera booleana: punti dentro la regione (per le regioni a punti:
    entro radius_km da almeno un sito)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if is_points(rid):
        st = sites(rid)
        d = haversine_km(lat[:, None], lon[:, None], st["lat"].values[None, :], st["lon"].values[None, :])
        return (d <= region(rid).get("radius_km", 150)).any(axis=1)
    return contains_xy(polygon(rid), lon, lat)


def covers(p, lat, lon):
    """Il dominio del provider contiene i punti?"""
    c = p.get("coverage", "global")
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if c == "global":
        return np.ones(lat.shape, bool)
    x0, y0, x1, y1 = c
    return (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)


def region_providers(rid, kind="openmeteo"):
    """Provider il cui dominio copre tutta la regione."""
    if is_points(rid):
        st = sites(rid)
        b = (st["lon"].min(), st["lat"].min(), st["lon"].max(), st["lat"].max())
    else:
        b = polygon(rid).bounds                  # lon_min, lat_min, lon_max, lat_max
    out = []
    for p in providers(kind):
        c = p.get("coverage", "global")
        if c == "global" or (c[0] <= b[0] and c[1] <= b[1] and c[2] >= b[2] and c[3] >= b[3]):
            out.append(p["id"])
    return out


def region_stations(rid, stations):
    return stations[inside(rid, stations["lat"].values, stations["lon"].values)]


def tz_from_lon(lon):
    """Fuso approssimato dalla longitudine (usato per i punti griglia senza fuso)."""
    off = int(np.round(lon / 15.0))
    return "UTC" if off == 0 else f"Etc/GMT{'-' if off > 0 else '+'}{abs(off)}"


def station_tz(stations):
    tz = stations["tz"] if "tz" in stations else pd.Series([None] * len(stations), index=stations.index)
    return [t if isinstance(t, str) and t else tz_from_lon(lo) for t, lo in zip(tz, stations["lon"])]
