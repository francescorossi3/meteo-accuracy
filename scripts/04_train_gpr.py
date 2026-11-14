# -*- coding: utf-8 -*-
"""
04 - Modelli GPR (uno per variabile x aggregazione x lead day).

Per ciascun caso:
  a) addestra sui giorni PRIMA dell'holdout e predice l'holdout
     -> errori e calibrazione (copertura degli intervalli 68% e 95%):
        e' il controllo che dice se stime e incertezze GPR sono affidabili
  b) riaddestra sugli ultimi train_days (holdout incluso) ed esporta il
     modello in JSON per il sito (mappe + interrogazione di un punto)
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from meteoita import gp
from meteoita.config import AGGS, BUILD, CFG, CSV_DIR, DATA, LEADS, SITE_DATA, gridded_ids, provider_names, target_vars
from meteoita.regions import REGIONS, region_providers
from meteoita.verify import by_region, daily_stats, merge_stats, pairs_for, rankings, score_table, station_table

warnings.filterwarnings("ignore")                       # avvisi di convergenza sklearn
os.environ["PYTHONWARNINGS"] = "ignore"                  # anche nei processi paralleli
G = CFG["gpr"]

split = pd.read_parquet(BUILD / "split.parquet").iloc[0]
hold_start, last_day, end = split["hold_start"], split["last_day"], split["end"]
st_all = station_table()
stations = st_all.set_index("station")
GRID = gridded_ids()
T0 = end - pd.Timedelta(days=G["train_days"] + CFG["verification"]["holdout_days"] + 1)
MAX_KEYS = 8000          # (stazione, periodo) campionati per aggregazione: il GPR usa <= n_train_max punti


def sample_pairs(agg, seed):
    """Coppie di un'aggregazione, campionate per (stazione, periodo) per contenere la memoria:
    stessi campioni per tutti i provider e anticipi."""
    p = pairs_for(T0, end, providers=GRID, aggs=[agg])
    keys = p[["station", "period", "date"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    keep = []
    for part in (keys[keys["date"] < hold_start], keys[keys["date"] >= hold_start]):
        keep.append(part.iloc[rng.choice(len(part), min(len(part), MAX_KEYS), replace=False)] if len(part) else part)
    keep = pd.concat(keep)
    return by_region(p.merge(keep[["station", "period"]], on=["station", "period"]), st_all)
OUT = SITE_DATA / "gp"
OUT.mkdir(parents=True, exist_ok=True)


def wide_table(sub):
    w = sub.pivot_table(index=["station", "period", "date"], columns="provider", values="fc", aggfunc="first")
    w["obs"] = sub.groupby(["station", "period", "date"])["obs"].first()
    w = w.reset_index()
    w = w.join(stations[["lat", "lon", "elev"]], on="station")
    return w.dropna(subset=["lat", "lon", "obs"])


def run_case(rid, agg, var, lead, sub, seed):
    """Validazione su holdout + modello finale per un caso (regione, var, agg, lead)."""
    rng = np.random.default_rng(seed)
    take = lambda idx, n: idx if len(idx) <= n else rng.choice(idx, n, replace=False)   # noqa: E731
    res = {"valid": None, "hold": None, "st_err": None, "index": None, "log": ""}
    tr = gp.transform_of(var)
    w = wide_table(sub)
    cov = {p: w[p].notna().mean() for p in region_providers(rid) if p in w}
    provs = [p for p, c in cov.items() if c >= G["min_model_coverage"]]
    if not provs:
        return res
    X, ok = gp.design(w, provs, tr)
    w, X = w[ok].reset_index(drop=True), X[ok]
    y = gp.fwd(w["obs"].values, tr)

    # ---- a) validazione su holdout (training sui giorni precedenti)
    itr = np.where((w["date"] < hold_start) & (w["date"] >= hold_start - pd.Timedelta(days=G["train_days"])))[0]
    iho = np.where(w["date"] >= hold_start)[0]
    if len(itr) >= 30 and len(iho) >= 10:
        itr = take(itr, G["n_train_max"])
        model, mu, sc = gp.fit(X[itr], y[itr], G["restarts"], seed, tr)
        pm, ps = gp.predict_export(gp.export(model, mu, sc, {}), X[iho])
        v = gp.validation(w["obs"].values[iho], pm, ps, tr)
        v.update(region=rid, agg=agg, var=var, lead_day=lead, n_train=len(itr))
        res["valid"] = v
        hp = w.iloc[iho][["station", "period", "obs"]].copy()
        hp["fc"] = gp.to_original(pm, ps, tr)["est"]
        hp["provider"], hp["region"], hp["agg"], hp["var"], hp["lead_day"] = "GPR", rid, agg, var, lead
        res["hold"] = hp
        se = hp.assign(ae=np.abs(hp["fc"] - hp["obs"])).groupby("station")["ae"].mean()
        res["st_err"] = pd.DataFrame({"station": se.index, "mae": se.values, "region": rid, "agg": agg,
                                        "var": var, "lead_day": lead})

    # ---- b) modello finale per il sito (ultimi train_days, holdout incluso)
    ifin = np.where(w["date"] > last_day - pd.Timedelta(days=G["train_days"]))[0]
    if len(ifin) < 30:
        return res
    ifin = take(ifin, G["n_train_max"])
    model, mu, sc = gp.fit(X[ifin], y[ifin], G["restarts"], seed, tr)
    meta = {"region": rid, "var": var, "agg": agg, "lead_day": int(lead), "providers": provs,
            "features": gp.GEO + provs, "transform": tr, "trained_until": str(last_day.date()),
            "n_train": int(len(ifin))}
    m = gp.export(model, mu, sc, meta)
    name = f"{rid}/{var}_{agg}_d{lead}.json"
    (OUT / rid).mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(m, f, separators=(",", ":"))
    res["index"] = {k: meta[k] for k in ("region", "var", "agg", "lead_day", "providers", "n_train")} | {"file": name}
    res["log"] = f"  {rid:<10}{var:<5}{agg:>4} d{lead}  n={len(ifin):<4} provider={len(provs):<2} rumore={m['kernel']['noise']:.3f}"
    return res


# un job per caso, in parallelo su tutti i core
t0 = time.time()
results = []
for k, agg in enumerate(AGGS):                    # un'aggregazione alla volta (memoria)
    pairs = sample_pairs(agg, G["random_seed"] + k)
    print(f"{agg}: {len(pairs):,} coppie campionate")
    groups = {key: g for key, g in pairs.groupby(["region", "agg", "var", "lead_day"])}
    del pairs
    cases = [(r["id"], agg, var, lead) for r in REGIONS if r.get("verify", True) for var in target_vars(agg) for lead in LEADS
             if (r["id"], agg, var, lead) in groups]
    results += Parallel(n_jobs=-1, verbose=0)(
        delayed(run_case)(*c, groups[c], G["random_seed"] + 100 * k + i) for i, c in enumerate(cases))
    del groups
for r in results:
    if r["log"]:
        print(r["log"])
valid_rows = [r["valid"] for r in results if r["valid"] is not None]
hold_pairs = [r["hold"] for r in results if r["hold"] is not None]
station_err = [r["st_err"] for r in results if r["st_err"] is not None]
index = [r["index"] for r in results if r["index"] is not None]

with open(OUT / "index.json", "w") as f:
    json.dump(index, f)

# %% export validazione
val = pd.DataFrame(valid_rows)
if not val.empty:
    val = val[["region", "var", "agg", "lead_day", "n_train", "n", "bias", "mae", "rmse", "r2", "crps", "cov68", "cov95", "mean_unc"]]
    val.round(4).to_csv(CSV_DIR / "gpr_validazione.csv", index=False)
    print("\nCalibrazione GPR (ideale: cov68 ~ 0.68, cov95 ~ 0.95):")
    print(val.groupby(["region", "var", "agg"])[["mae", "r2", "cov68", "cov95"]].mean().round(3).to_string())

# classifica holdout completa: provider + BEST + WEIGHTED + GPR
hold = pd.read_parquet(BUILD / "holdout_pairs.parquet")          # tutti i provider (anche snapshot)
combo = pd.read_parquet(BUILD / "holdout_combo.parquet")
allh = pd.concat([hold, combo] + hold_pairs, ignore_index=True)
hs = score_table(allh).round(4)
hs.to_csv(CSV_DIR / "classifica_holdout.csv", index=False)

# GPR fuori campione nello storico -> entra nelle classifiche normali (30/90/365 giorni, tutto)
STATS = DATA / "stats" / "daily_stats.parquet"
if hold_pairs:
    gpr_pairs = pd.concat(hold_pairs, ignore_index=True)
    stats = merge_stats(pd.read_parquet(STATS), daily_stats(gpr_pairs))
    stats.to_parquet(STATS, index=False, compression="zstd")
    scores, trend = rankings(stats, provider_names())
    scores.round(4).to_csv(CSV_DIR / "classifica.csv", index=False)
    trend.round(4).to_csv(CSV_DIR / "trend_mensile.csv", index=False)
    for name, df in (("scores.json", scores), ("trend.json", trend)):
        with open(SITE_DATA / name, "w") as f:
            json.dump(json.loads(df.round(4).to_json(orient="records")), f, separators=(",", ":"))
    print("Classifiche aggiornate con il GPR")
with open(SITE_DATA / "scores_holdout.json", "w") as f:
    json.dump(json.loads(hs.to_json(orient="records")), f, separators=(",", ":"))
with open(SITE_DATA / "gpr_validation.json", "w") as f:
    json.dump(json.loads(val.round(4).to_json(orient="records")) if not val.empty else [], f)
se = pd.concat(station_err, ignore_index=True) if station_err else pd.DataFrame()
with open(SITE_DATA / "gpr_station_errors.json", "w") as f:
    json.dump(json.loads(se.round(3).to_json(orient="records")) if not se.empty else [], f)
print(f"\nModelli GPR esportati: {len(index)} in {time.time() - t0:.0f}s")
