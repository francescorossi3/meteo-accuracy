# -*- coding: utf-8 -*-
"""
00 - Scelta delle stazioni di verifica.

Per ogni regione di config.yaml sceglie n_stations stazioni Meteostat con dati
orari recenti, ben distribuite sul territorio (preferendo le sinottiche
ufficiali con codice WMO). Le stazioni italiane valgono anche per il Mondo.
Risultato: data/stations.csv (apribile in Excel, modificabile a mano).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from meteoita.config import DATA
from meteoita.observations import meteostat_stations, spread_selection
from meteoita.regions import REGIONS, haversine_km, inside, is_points, sites, station_tz, verified

try:
    allst = meteostat_stations(min_end_year=pd.Timestamp.now().year - 1).dropna(subset=["lat", "lon"])
    print(f"Stazioni Meteostat con dati orari recenti: {len(allst)}")
except Exception as ex:                              # noqa: BLE001
    # elenco non raggiungibile: si usano le stazioni sinottiche di riserva
    from meteoita.demo import SEED_STATIONS, WORLD_SEED
    print(f"! elenco stazioni non disponibile ({ex}): uso l'elenco di riserva")
    it = pd.DataFrame(SEED_STATIONS, columns=["station", "name", "lat", "lon", "elev"])
    it["tz"], it["country"] = "Europe/Rome", "IT"
    allst = pd.concat([it, pd.DataFrame(WORLD_SEED, columns=["station", "name", "lat", "lon", "elev", "tz"])])
    allst["wmo"] = allst["station"]

parts = []
for r in REGIONS:
    if not verified(r["id"]):                    # es. RadCalNet: solo archivio delle previsioni
        continue
    if is_points(r["id"]):
        # regione a punti: le stazioni piu' vicine a ciascun sito (entro radius_km)
        sel = []
        for _, s in sites(r["id"]).iterrows():
            d = haversine_km(allst["lat"].values, allst["lon"].values, s["lat"], s["lon"])
            near = allst.assign(dist_km=d).sort_values("dist_km")
            near = near[near["dist_km"] <= r.get("radius_km", 150)].head(r.get("stations_per_site", 2))
            print(f"  {s['name']:<28} " + ", ".join(f"{n} ({k:.0f} km)" for n, k in zip(near["name"], near["dist_km"])))
            sel.append(near)
        parts.append(pd.concat(sel).drop_duplicates("station").drop(columns="dist_km"))
        continue
    pool = allst[inside(r["id"], allst["lat"].values, allst["lon"].values)]
    if r.get("country") and "country" in pool:
        pool = pool[pool["country"] == r["country"]]
    taken = set().union(*[set(p["station"]) for p in parts]) if parts else set()
    pool = pool[~pool["station"].isin(taken)]
    already = sum(inside(r["id"], p["lat"].values, p["lon"].values).sum() for p in parts)
    n = max(0, r["n_stations"] - int(already))
    if n == 0 or pool.empty:
        continue
    if pool["wmo"].notna().sum() >= n:
        pool = pool[pool["wmo"].notna()]
    sel = spread_selection(pool.reset_index(drop=True), n)
    print(f"  {r['name']:<10} {len(sel)} stazioni")
    parts.append(sel)

st = pd.concat(parts, ignore_index=True).drop_duplicates("station")
st["elev"] = st["elev"].fillna(0)
st["tz"] = station_tz(st)
st.to_csv(DATA / "stations.csv", index=False)
print(f"Salvate {len(st)} stazioni in {DATA / 'stations.csv'}")
