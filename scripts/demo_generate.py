# -*- coding: utf-8 -*-
"""
DEMO - genera dati SINTETICI (stazioni, osservazioni, previsioni) per provare
l'intera catena senza connessione e vedere il sito in locale:

    python scripts/demo_generate.py
    python scripts/03_verify.py
    python scripts/04_train_gpr.py
    python scripts/05_build_maps.py --demo
    python scripts/06_build_site.py
    python -m http.server -d build/site 8000     ->  http://localhost:8000

Scrive in una cartella separata (demo_data/) per non mescolare i dati veri:
impostare la variabile d'ambiente METEOITA_DATA=demo_data prima di lanciare
gli script (lo fa run_demo.sh).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from meteoita import demo
from meteoita.config import DATA, LEADS, providers
from meteoita.providers import calendar_lead
from meteoita.regions import covers
from meteoita.storage import write_forecasts, write_observations

DAYS = 75
rng = np.random.default_rng(0)
it = pd.DataFrame(demo.SEED_STATIONS, columns=["station", "name", "lat", "lon", "elev"])
it["tz"] = "Europe/Rome"
st = pd.concat([it, pd.DataFrame(demo.WORLD_SEED, columns=["station", "name", "lat", "lon", "elev", "tz"])],
               ignore_index=True)
from meteoita.regions import RID, archive_points, inside, verified
st = st[np.any([inside(r, st["lat"], st["lon"]) for r in RID if verified(r)], axis=0)].reset_index(drop=True)
st.to_csv(DATA / "stations.csv", index=False)
print("stazioni:", len(st))

end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
times = pd.date_range(end - pd.Timedelta(days=DAYS), end, freq="h")
lat, lon, el = st["lat"].values, st["lon"].values, st["elev"].values

# osservazioni = verita' + errore di misura + microclima di stazione + buchi
t, p = demo.truth(lat, lon, el, times)
micro = rng.normal(0, 0.8, len(st))[:, None]
t_obs = t + micro + rng.normal(0, 0.3, t.shape)
p_obs = np.clip(p * rng.lognormal(0, 0.3, p.shape), 0, None)
p_obs[p_obs < 0.1] = 0.0
obs = pd.DataFrame({"station": np.repeat(st["station"].values, len(times)),
                    "valid_time": np.tile(times, len(st)), "t2m": t_obs.ravel(), "prcp": p_obs.ravel()})
obs = obs.sample(frac=0.97, random_state=1)                    # 3% di dati mancanti
print("osservazioni:", write_observations(obs))

# siti senza verifica (RadCalNet): solo previsioni archiviate, nessuna osservazione
sites = archive_points()
lat = np.r_[lat, sites["lat"].values]; lon = np.r_[lon, sites["lon"].values]; el = np.r_[el, sites["elev"].values]
st = pd.concat([st, sites[["station", "name", "lat", "lon", "elev", "tz"]]], ignore_index=True)

# previsioni dei modelli "su griglia" (lead fisso come la Previous Runs API)
for prov in providers("openmeteo"):
    if prov["id"] not in demo.MODELS:
        continue
    cv = covers(prov, lat, lon)                                    # solo stazioni nel dominio
    parts = []
    for L in LEADS:
        if prov["id"] == "icon2i" and L > 2:                    # ICON-2I arriva a 72 h
            continue
        tf, pf = demo.forecast(prov["id"], L, lat[cv], lon[cv], el[cv], times)
        parts.append(pd.DataFrame({"station": np.repeat(st["station"].values[cv], len(times)),
                                   "valid_time": np.tile(times, int(cv.sum())), "lead_day": L,
                                   "t2m": tf.ravel(), "prcp": np.clip(pf.ravel(), 0, None)}))
    df = pd.concat(parts)
    df["provider"] = prov["id"]
    print(prov["id"], write_forecasts(df))

# un provider "snapshot" (metno): lead in giorni di calendario, emissione alle 06 UTC
days = pd.date_range(times[0].normalize(), end.normalize(), freq="D")
parts = []
for d in days:
    issued = d + pd.Timedelta(hours=6)
    vt = pd.date_range(issued + pd.Timedelta(hours=1), issued + pd.Timedelta(days=8), freq="h")
    vt = vt[vt <= end]
    if len(vt) == 0:
        continue
    lead = calendar_lead(pd.Series(vt), issued, "UTC")
    for L in np.unique(lead):
        s = vt[lead == L]
        n = len(st) - len(sites)                                 # solo stazioni (niente snapshot sui siti)
        tf, pf = demo.forecast("metno", L, lat[:n], lon[:n], el[:n], s)
        parts.append(pd.DataFrame({"station": np.repeat(st["station"].values[:n], len(s)),
                                   "valid_time": np.tile(s, n), "lead_day": L,
                                   "t2m": tf.ravel(), "prcp": np.clip(pf.ravel(), 0, None)}))
df = pd.concat(parts)
df = df[df["lead_day"].isin(LEADS)]
df["provider"] = "metno"
print("metno", write_forecasts(df))
