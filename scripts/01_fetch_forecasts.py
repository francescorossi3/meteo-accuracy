# -*- coding: utf-8 -*-
"""
01 - Download delle previsioni sulle stazioni.

* Modelli Open-Meteo: Previous Runs API -> per ogni ora valida si ottiene
  la previsione emessa 0, 1, ..., 7 giorni prima. Funziona anche per il
  passato (dati dal 2024): con --start/--end si ricostruisce lo storico.
* Provider "snapshot": si salva la previsione di oggi (non e' possibile
  recuperarla dopo), con lead = giorni di calendario (0 oggi, 1 domani...).

Uso:
    python scripts/01_fetch_forecasts.py                      # ultimi 3 giorni + snapshot
    python scripts/01_fetch_forecasts.py --start 2025-01-01 --end 2025-06-30 --mode openmeteo
In Spyder: modificare i valori di default qui sotto ed eseguire.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from meteoita.config import DATA, LEADS, providers
from meteoita.providers import (SNAPSHOT_APIS, SNAPSHOT_LIMITS, calendar_lead,
                                openmeteo_previous_runs)
from meteoita.regions import archive_points, covers, station_tz
from meteoita.storage import write_forecasts

today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
ap = argparse.ArgumentParser()
ap.add_argument("--start", default=str((today - pd.Timedelta(days=3)).date()))
ap.add_argument("--end", default=str((today - pd.Timedelta(days=1)).date()))
ap.add_argument("--mode", default="all", choices=["all", "openmeteo", "snapshot"])
args, _ = ap.parse_known_args()

stations = pd.read_csv(DATA / "stations.csv", dtype={"station": str})
stations["tz"] = station_tz(stations)
# siti senza verifica (RadCalNet): si scaricano e archiviano le previsioni dei modelli
sites = archive_points()
if len(sites):
    print(f"+ {len(sites)} siti archiviati: {', '.join(sites['station'])}")
print(f"{len(stations)} stazioni, periodo {args.start} -> {args.end}, modo {args.mode}")

# %% ---------------------------------------------------- modelli Open-Meteo
if args.mode in ("all", "openmeteo"):
    # blocchi di 31 giorni per non fare richieste troppo grandi
    edges = list(pd.date_range(args.start, args.end, freq="31D")) + [pd.Timestamp(args.end) + pd.Timedelta(days=1)]
    allpts = pd.concat([stations, sites[stations.columns.intersection(sites.columns)]], ignore_index=True)
    for p in providers("openmeteo"):
        sub = allpts[covers(p, allpts["lat"], allpts["lon"])].reset_index(drop=True)          # solo nel dominio
        tot = 0
        for a, b in zip(edges[:-1], edges[1:]):
            s, e = a.date(), (b - pd.Timedelta(days=1)).date()
            try:
                df = openmeteo_previous_runs(p["model"], sub["lat"].tolist(), sub["lon"].tolist(), s, e, LEADS)
            except Exception as ex:                     # noqa: BLE001
                print(f"  ! {p['id']} {s}..{e}: {ex}")
                continue
            if df.empty:
                continue
            df["station"] = sub["station"].values[df["idx"].values]
            df["provider"] = p["id"]
            tot += write_forecasts(df.drop(columns="idx"))
        print(f"  {p['id']:<12} {len(sub):>4} stazioni {tot:>9} righe")

# %% ---------------------------------------------------- provider snapshot
if args.mode in ("all", "snapshot"):
    issued = pd.Timestamp.now(tz="UTC")
    for p in providers("snapshot"):
        fn = SNAPSHOT_APIS[p["api"]]
        max_st, pause = SNAPSHOT_LIMITS.get(p["api"], (None, 1.0))
        key = None
        if p.get("api_key_env"):
            import os
            key = os.environ[p["api_key_env"]]
        cov = stations[covers(p, stations["lat"], stations["lon"])].reset_index(drop=True)
        sub = cov if max_st is None or len(cov) <= max_st else cov.iloc[np.linspace(0, len(cov) - 1, max_st).astype(int)]
        parts = []
        for _, st in sub.iterrows():
            try:
                d = fn(st["lat"], st["lon"], st["elev"], key)
            except Exception as ex:                     # noqa: BLE001
                print(f"  ! {p['id']} {st['station']}: {ex}")
                continue
            d["station"], d["tz"] = st["station"], st["tz"]
            parts.append(d)
            time.sleep(pause)
        if not parts:
            print(f"  {p['id']:<12} nessun dato")
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df[df["valid_time"] > issued]
        df["lead_day"] = calendar_lead(df["valid_time"], issued, df["tz"].values)
        df = df[df["lead_day"].isin(LEADS)].drop(columns="tz")
        df["provider"] = p["id"]
        print(f"  {p['id']:<12} {write_forecasts(df):>9} righe (snapshot)")
