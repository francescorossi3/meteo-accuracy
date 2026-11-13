# -*- coding: utf-8 -*-
"""
02 - Download delle osservazioni orarie (Meteostat) sulle stazioni.

Di default riscarica gli ultimi 10 giorni (i dati arrivano con qualche
giorno di ritardo e vengono completati nel tempo).
    python scripts/02_fetch_observations.py --start 2025-01-01 --end 2025-12-31
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from meteoita.config import DATA
from meteoita.observations import station_hourly
from meteoita.storage import write_observations

today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
ap = argparse.ArgumentParser()
ap.add_argument("--start", default=str((today - pd.Timedelta(days=10)).date()))
ap.add_argument("--end", default=str((today + pd.Timedelta(days=1)).date()))
args, _ = ap.parse_known_args()

stations = pd.read_csv(DATA / "stations.csv", dtype={"station": str})
parts = []
for _, st in stations.iterrows():
    d = station_hourly(st["station"], args.start, args.end)
    print(f"  {st['station']:<6} {st['name'][:25]:<25} {len(d):>6} ore")
    parts.append(d)

obs = pd.concat(parts, ignore_index=True)
print(f"Osservazioni scritte: {write_observations(obs)}")
