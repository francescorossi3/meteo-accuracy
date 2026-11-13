# -*- coding: utf-8 -*-
"""
05 - Mappe di previsione per ogni regione (Italia, Mondo, ...).

Scarica la previsione corrente dei modelli della regione su una griglia
regolare (config: regions[].grid_res) e calcola, per ogni periodo dei
prossimi giorni e per ogni aggregazione:
    best      -> provider migliore per quel caso
    weighted  -> media pesata (pesi 1/MSE)
    gpr       -> stima GPR
    gpr_unc   -> incertezza GPR (semi-ampiezza intervallo 68%)
    gpr_p     -> (solo pioggia) probabilita' di superare la 1a soglia
Uscita: build/site/data/maps/<regione>/<var>_<agg>.json
Fusi: con map_timezone "local" i giorni sono quelli locali di ogni punto e
le mappe orarie e a 6 ore sono in UTC (stessa ora per tutto il mondo).
    python scripts/05_build_maps.py          (dati reali)
    python scripts/05_build_maps.py --demo   (mondo sintetico, senza internet)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from meteoita import gp
from meteoita.config import AGGS, CFG, DATA, LEADS, SITE_DATA, providers, target_vars
from meteoita.grid import region_grid
from meteoita.providers import openmeteo_forecast
from meteoita.regions import REGIONS, region_providers, tz_from_lon, verified
from meteoita.verify import aggregate

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
args, _ = ap.parse_known_args()

DAYS = CFG["grid"]["forecast_days"]
now = pd.Timestamp.now(tz="UTC").floor("h")
weights_all = pd.DataFrame(json.load(open(SITE_DATA / "weights.json")))
rnd = lambda a: [None if not np.isfinite(v) else round(float(v), 2) for v in a]     # noqa: E731
if args.demo:
    from meteoita import demo

for R in REGIONS:
    rid = R["id"]
    grid = region_grid(rid)
    lats, lons = grid["lat"].values, grid["lon"].values
    provs = [p for p in providers("openmeteo") if p["id"] in region_providers(rid)]
    mtz = R.get("map_timezone", "UTC")
    tz_sub = "UTC" if mtz == "local" else mtz                     # fuso di mappe orarie e 6 h
    today = pd.Timestamp(now.tz_convert(tz_sub).strftime("%Y-%m-%d"))
    print(f"\n{R['name']}: {len(grid)} punti, {len(provs)} modelli, emissione {now:%Y-%m-%d %H:%M} UTC")

    # %% previsioni sulla griglia
    parts = []
    if args.demo:
        grid["elev"] = demo.elevation(lats, lons)
        times = pd.date_range(now + pd.Timedelta(hours=1), periods=24 * DAYS, freq="h")
        lead = np.clip(((times - now) / pd.Timedelta(hours=24)).astype(int), 0, max(LEADS))
        for p in provs:
            T = np.full((len(grid), len(times)), np.nan)
            P = T.copy()
            for L in np.unique(lead):
                sel = lead == L
                T[:, sel], P[:, sel] = demo.forecast(p["id"], L, lats, lons, grid["elev"].values, times[sel])
            parts.append(pd.DataFrame({"idx": np.repeat(np.arange(len(grid)), len(times)),
                                       "valid_time": np.tile(times, len(grid)), "t2m": T.ravel(),
                                       "prcp": np.clip(P.ravel(), 0, None), "provider": p["id"]}))
    else:
        for p in provs:
            try:
                df, elev = openmeteo_forecast(p["model"], lats, lons, DAYS)
            except Exception as ex:                                  # noqa: BLE001
                print(f"  ! {p['id']}: {ex}")
                continue
            df["provider"] = p["id"]
            parts.append(df)
            if grid["elev"].isna().any():
                grid["elev"] = grid["elev"].fillna(pd.Series(elev))
            print(f"  {p['id']:<12} ok")
        grid.to_csv(DATA / f"grid_{rid}.csv", index=False)
    if not parts:
        continue
    fcg = pd.concat(parts, ignore_index=True)
    fcg["valid_time"] = pd.to_datetime(fcg["valid_time"], utc=True).dt.as_unit("ns")
    fcg = fcg[fcg["valid_time"] > now]
    point_tz = np.array([tz_from_lon(x) for x in lons], dtype=object) if mtz == "local" else None
    fcg["tz"] = point_tz[fcg["idx"].values] if mtz == "local" else mtz

    weights = weights_all[weights_all["region"] == rid]
    gpdir = SITE_DATA / "gp" / rid
    OUT = SITE_DATA / "maps" / rid
    OUT.mkdir(parents=True, exist_ok=True)

    # %% calcolo per aggregazione e variabile
    for agg in AGGS:
        a = aggregate(fcg, agg, ["idx", "provider"], t_cols=("t2m",), p_cols=("prcp",),
                      tz="tz" if agg == "24h" else tz_sub)
        a["lead_day"] = (a["period"].dt.normalize() - today).dt.days
        a = a[a["lead_day"].isin(LEADS)]
        for var in target_vars(agg):
            W = a.pivot_table(index=["period", "idx"], columns="provider", values=var, aggfunc="first")
            # periodi coperti da meno di meta' dei punti (es. "oggi" gia' iniziato in parte del mondo) -> esclusi
            npts = W.notna().any(axis=1).groupby(level=0).sum()
            periods = sorted(npts[npts >= 0.5 * len(grid)].index)
            ver = verified(rid)
            layers = ({k: [] for k in ("best", "weighted", "gpr", "gpr_unc", "gpr_lo95", "gpr_hi95")}
                      if ver else {"mean": [], "spread": []})            # non verificata: solo i modelli
            if var == "prcp" and ver:
                layers["gpr_p"] = []
            prov_layers = {p: [] for p in W.columns} if (agg != "1h" or not ver) else {}
            best_names, leads_out, model_cache = [], [], {}
            for per in periods:
                lead = int((per.normalize() - today).days)
                Wp = W.xs(per, level=0).reindex(range(len(grid)))
                if not ver:                                   # media e dispersione dei modelli
                    layers["mean"].append(rnd(Wp.mean(axis=1).values))
                    layers["spread"].append(rnd(Wp.std(axis=1).values))
                    for p in prov_layers:
                        prov_layers[p].append(rnd(Wp[p].values))
                    best_names.append(None); leads_out.append(lead)
                    continue
                sel = weights[(weights["agg"] == agg) & (weights["var"] == var) & (weights["lead_day"] == lead)]
                # migliore e media pesata
                if not sel.empty:
                    best = sel.loc[sel["is_best"], "provider"].iloc[0]
                    pv = [p for p in sel["provider"] if p in Wp]
                    w = sel.set_index("provider").loc[pv, "weight"].values
                    F = Wp[pv].values
                    M = ~np.isnan(F)
                    ws = M.astype(float) @ w
                    wavg = np.where(ws > 0.5, (np.where(M, F, 0) @ w) / np.maximum(ws, 1e-9), np.nan)
                    bvals = Wp[best].values if best in Wp else np.full(len(grid), np.nan)
                else:
                    best, bvals = None, np.full(len(grid), np.nan)
                    wavg = Wp.mean(axis=1).values
                # GPR
                mfile = gpdir / f"{var}_{agg}_d{lead}.json"
                if mfile.exists():
                    if mfile not in model_cache:
                        m = json.load(open(mfile))
                        model_cache[mfile] = (m, gp._unpack(m))
                    m, cache = model_cache[mfile]
                    df = Wp.copy()
                    df["lat"], df["lon"], df["elev"] = lats, lons, grid["elev"].values
                    X, ok = gp.design(df, m["providers"], m["transform"])
                    mu, sd = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
                    if ok.any():
                        mu[ok], sd[ok] = gp.predict_export(m, X[ok], cache)
                    o = gp.to_original(mu, sd, m["transform"])
                    layers["gpr"].append(rnd(o["est"])); layers["gpr_unc"].append(rnd(o["unc"]))
                    layers["gpr_lo95"].append(rnd(o["lo95"])); layers["gpr_hi95"].append(rnd(o["hi95"]))
                    if var == "prcp":
                        thr = CFG["verification"]["prcp_thresholds"][agg][0]
                        layers["gpr_p"].append(rnd(100 * gp.prob_exceed(mu, sd, thr, m["transform"])))
                else:
                    for k in layers:
                        if k.startswith("gpr"):
                            layers[k].append([None] * len(grid))
                layers["best"].append(rnd(bvals)); layers["weighted"].append(rnd(wavg))
                for p in prov_layers:
                    prov_layers[p].append(rnd(Wp[p].values))
                best_names.append(best); leads_out.append(lead)
            out = {"region": rid, "verified": ver, "var": var, "agg": agg, "issued": now.isoformat(),
                   "timezone": "locale" if (agg == "24h" and mtz == "local") else tz_sub,
                   "periods": [p.strftime("%Y-%m-%dT%H:%M") for p in periods], "lead_day": leads_out,
                   "best_provider": best_names, "lat": lats.tolist(), "lon": lons.tolist(),
                   "res": float(R.get("grid_res", 0)), "layers": layers, "providers": prov_layers,
                   "names": grid["name"].tolist() if "name" in grid else None}
            with open(OUT / f"{var}_{agg}.json", "w") as f:
                json.dump(out, f, separators=(",", ":"))
            print(f"  mappa {var:<5} {agg:<4} {len(periods):>3} periodi")
