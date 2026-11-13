"""Osservazioni orarie da stazioni Meteostat (SYNOP/METAR di tutto il mondo).

Meteostat redistribuisce dati di servizi pubblici (Aeronautica Militare via
NOAA ISD / DWD). Le serie sono in UTC; 'prcp' e' la pioggia dell'ora che
termina all'orario indicato, come per le previsioni.
"""
import gzip
import io
import json

import numpy as np
import pandas as pd

from .config import CFG
from .providers import http_get

OBS = CFG["observations"]
LEGACY_COLS = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", "wdir",
               "wspd", "wpgt", "pres", "tsun", "coco"]


def meteostat_stations(min_end_year):
    """Elenco mondiale delle stazioni Meteostat con dati orari recenti."""
    raw = http_get(OBS["url_stations"]).content
    js = json.loads(gzip.decompress(raw))
    rows = []
    for s in js:
        inv = (s.get("inventory") or {}).get("hourly") or {}
        end = inv.get("end") or ""
        if not end or int(end[:4]) < min_end_year:
            continue
        loc = s.get("location") or {}
        ident = s.get("identifiers") or {}
        rows.append({"station": str(s["id"]), "name": (s.get("name") or {}).get("en", s["id"]),
                     "country": s.get("country"), "lat": loc.get("latitude"), "lon": loc.get("longitude"),
                     "elev": loc.get("elevation"), "tz": s.get("timezone"), "wmo": ident.get("wmo"),
                     "icao": ident.get("icao"), "hourly_start": inv.get("start"), "hourly_end": end})
    return pd.DataFrame(rows)


def spread_selection(df, n, start=(41.9, 12.5)):
    """Sceglie n stazioni ben distribuite sulla sfera (farthest point sampling)."""
    la, lo = np.radians(df["lat"].values), np.radians(df["lon"].values)
    xyz = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    s0 = np.radians(start)
    p0 = np.array([np.cos(s0[0]) * np.cos(s0[1]), np.cos(s0[0]) * np.sin(s0[1]), np.sin(s0[0])])
    chosen = [int(np.argmin(np.linalg.norm(xyz - p0, axis=1)))]
    d = np.linalg.norm(xyz - xyz[chosen[0]], axis=1)
    while len(chosen) < min(n, len(df)):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(xyz - xyz[i], axis=1))
    return df.iloc[sorted(chosen)].reset_index(drop=True)


def _parse_new(content):
    df = pd.read_csv(io.BytesIO(gzip.decompress(content)))
    cols = set(df.columns)
    if {"year", "month", "day", "hour"} <= cols:
        t = pd.to_datetime(df[["year", "month", "day", "hour"]])
    elif {"date", "hour"} <= cols:
        t = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    elif "time" in cols:
        t = pd.to_datetime(df["time"])
    else:
        raise ValueError(f"formato Meteostat sconosciuto: {list(df.columns)[:8]}")
    out = pd.DataFrame({"valid_time": t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC"),
                        "t2m": df.get("temp"), "prcp": df.get("prcp")})
    if OBS.get("drop_model_sources", True):
        for v, src in (("t2m", "temp_source"), ("prcp", "prcp_source")):
            if src in df:
                is_model = df[src].astype(str).str.contains("model", case=False, na=False)
                out.loc[is_model.values, v] = np.nan
    return out


def _parse_legacy(content):
    df = pd.read_csv(io.BytesIO(gzip.decompress(content)), header=None, names=LEGACY_COLS)
    t = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    return pd.DataFrame({"valid_time": t.dt.tz_localize("UTC"), "t2m": df["temp"], "prcp": df["prcp"]})


def station_hourly(station, start, end):
    """Osservazioni orarie di una stazione tra start ed end (UTC)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for year in range(start.year, end.year + 1):
        try:
            r = http_get(OBS["url_hourly"].format(year=year, station=station), tries=2)
            parts.append(_parse_new(r.content))
        except Exception as e:                       # noqa: BLE001
            print(f"   - {station} {year}: dump nuovo non disponibile ({e.__class__.__name__})")
    if not parts:
        try:
            r = http_get(OBS["url_hourly_legacy"].format(station=station), tries=2)
            parts.append(_parse_legacy(r.content))
        except Exception as e:                       # noqa: BLE001
            print(f"   - {station}: nessun dato ({e.__class__.__name__})")
            return pd.DataFrame(columns=["station", "valid_time", "t2m", "prcp"])
    df = pd.concat(parts, ignore_index=True)
    s0 = start.tz_localize("UTC") if start.tz is None else start
    s1 = end.tz_localize("UTC") if end.tz is None else end
    df = df[(df["valid_time"] >= s0) & (df["valid_time"] <= s1)].copy()
    df["station"] = str(station)
    # controlli di plausibilita'
    df.loc[(df["t2m"] < -40) | (df["t2m"] > 50), "t2m"] = np.nan
    df.loc[(df["prcp"] < 0) | (df["prcp"] > 200), "prcp"] = np.nan
    return df[["station", "valid_time", "t2m", "prcp"]].dropna(subset=["t2m", "prcp"], how="all")
