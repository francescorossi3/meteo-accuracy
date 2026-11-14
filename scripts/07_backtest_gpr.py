# -*- coding: utf-8 -*-
"""
07 - Backtest storico del GPR (validazione su tutto lo storico).

Per ogni "piega" (fold) di test di 14 giorni, andando indietro nel tempo:
  * training sui 45 giorni PRECEDENTI (nessun dato del futuro)
  * il "migliore" e i pesi della media pesata sono scelti anch'essi solo
    sul periodo di training (ex-ante, come succede davvero ogni giorno)
  * si confrontano sulle STESSE righe di test:
        GPR, migliore ex-ante, media pesata, ogni singolo modello,
        e il migliore ex-post (il modello che a posteriori e' andato meglio:
        un riferimento "imbattibile" che nessun metodo puo' conoscere prima)
Risultati: data/csv/gpr_backtest.csv (ogni fold) e gpr_backtest_sintesi.csv,
           build/site/data/gpr_backtest.json per il sito.

E' pesante: di default gira al massimo una volta a settimana (o con --force).
    python scripts/07_backtest_gpr.py --force --folds 6
"""
import argparse
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
from meteoita.config import AGGS, CFG, CSV_DIR, DATA, LEADS, SITE_DATA, gridded_ids, target_vars
from meteoita.regions import REGIONS, region_providers
from meteoita.verify import best_and_weights, by_region, daily_stats, pairs_for, score_table, station_table

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
G = CFG["gpr"]
TEST_DAYS = CFG["verification"]["holdout_days"]

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=12, help="numero massimo di fold (14 giorni l'uno)")
ap.add_argument("--force", action="store_true", help="esegui anche se l'ultimo backtest ha meno di 7 giorni")
args, _ = ap.parse_known_args()

OUT_CSV = CSV_DIR / "gpr_backtest.csv"
if OUT_CSV.exists() and not args.force:
    last = pd.read_csv(OUT_CSV, usecols=["run_date"])["run_date"].max()
    if (pd.Timestamp.now() - pd.Timestamp(last)).days < 7:
        print(f"Backtest recente ({last}): salto (usa --force per rifarlo)")
        # copia comunque il JSON per il sito
        if (CSV_DIR / "gpr_backtest_sintesi.csv").exists():
            s = pd.read_csv(CSV_DIR / "gpr_backtest_sintesi.csv")
            with open(SITE_DATA / "gpr_backtest.json", "w") as f:
                json.dump(json.loads(s.to_json(orient="records")), f)
        sys.exit(0)

stats = pd.read_parquet(DATA / "stats" / "daily_stats.parquet")
first_day, last_day = stats["date"].min(), stats["date"].max()
st_all = station_table()
stations = st_all.set_index("station")
GRID = gridded_ids()
print(f"Storico disponibile: {first_day:%Y-%m-%d} -> {last_day:%Y-%m-%d}")

# fold: blocchi di test di 14 giorni, il piu' recente finisce prima dell'holdout corrente
folds = []
t_end = last_day - pd.Timedelta(days=TEST_DAYS)
while len(folds) < args.folds:
    t0 = t_end - pd.Timedelta(days=TEST_DAYS)
    if t0 - pd.Timedelta(days=G["train_days"]) < first_day:
        break
    folds.append((t0, t_end))
    t_end = t0
print(f"Fold disponibili: {len(folds)}")
if not folds:
    print("Storico troppo corto per il backtest (servono almeno ~2 mesi): lancia la ricostruzione dello storico.")
    sys.exit(0)


def wide(sub):
    w = sub.pivot_table(index=["station", "period", "date"], columns="provider", values="fc", aggfunc="first")
    w["obs"] = sub.groupby(["station", "period", "date"])["obs"].first()
    w = w.reset_index().join(stations[["lat", "lon", "elev"]], on="station")
    return w.dropna(subset=["lat", "lon", "obs"])


def run_fold_case(k, rid, agg, var, lead, sub, t0, seed):
    """Un fold per un caso: GPR vs migliore/pesata ex-ante vs modelli, sulle stesse righe."""
    rng = np.random.default_rng(seed)
    tr = gp.transform_of(var)
    w = wide(sub)
    train = w[w["date"] < t0]
    test = w[w["date"] >= t0]
    provs = [p for p in region_providers(rid) if p in w and train[p].notna().mean() >= G["min_model_coverage"]]
    if not provs or len(train) < 30 or len(test) < 10:
        return None
    # scelte ex-ante dal periodo di training
    tr_long = sub[sub["date"] < t0].copy()
    tr_long["region"] = rid
    sel = best_and_weights(score_table(tr_long))
    if sel.empty:
        return None
    best = sel.loc[sel["is_best"], "provider"].iloc[0]
    wts = sel.set_index("provider")["weight"]
    # GPR
    Xtr, oktr = gp.design(train, provs, tr)
    Xte, okte = gp.design(test, provs, tr)
    itr = np.where(oktr)[0]
    itr = itr if len(itr) <= G["n_train_max"] else rng.choice(itr, G["n_train_max"], replace=False)
    model, mu, sc = gp.fit(Xtr[itr], gp.fwd(train["obs"].values[itr], tr), G["restarts"], seed, tr)
    test = test[okte].copy()
    pm, ps = gp.predict_export(gp.export(model, mu, sc, {}), Xte[okte])
    o = gp.to_original(pm, ps, tr)
    test["GPR"], test["_mu"], test["_sd"] = o["est"], pm, ps
    # media pesata (pesi rinormalizzati sui modelli presenti)
    pw = [p for p in wts.index if p in test]
    F = test[pw].values
    M = ~np.isnan(F)
    wv = wts.loc[pw].values
    ws = M.astype(float) @ wv
    test["WEIGHTED"] = np.where(ws > 0.5, (np.where(M, F, 0) @ wv) / np.maximum(ws, 1e-9), np.nan)
    test["BEST"] = test[best] if best in test else np.nan
    # stesse righe per tutti: dove GPR, BEST e WEIGHTED esistono
    t = test.dropna(subset=["GPR", "BEST", "WEIGHTED"])
    if len(t) < 10:
        return None
    rmse = lambda c: float(np.sqrt(np.nanmean((t[c] - t["obs"]) ** 2)))          # noqa: E731
    sst = float(np.sum((t["obs"] - t["obs"].mean()) ** 2))
    r2 = lambda c: float(1 - np.nansum((t[c] - t["obs"]) ** 2) / sst) if sst > 0 else np.nan   # noqa: E731
    per_model = {p: rmse(p) for p in provs if t[p].notna().mean() > 0.9}
    v = gp.validation(t["obs"].values, t["_mu"].values, t["_sd"].values, tr)
    # coppie fuori campione (per lo storico delle classifiche)
    oos = pd.concat([pd.DataFrame({"station": t["station"], "period": t["period"], "obs": t["obs"], "fc": t[c],
                                   "provider": c}) for c in ("GPR", "BEST", "WEIGHTED")], ignore_index=True)
    oos["region"], oos["agg"], oos["var"], oos["lead_day"] = rid, agg, var, lead
    return oos, {"fold": k, "test_start": str(t0.date()), "region": rid, "var": var, "agg": agg, "lead_day": lead,
            "n": len(t), "rmse_gpr": rmse("GPR"), "rmse_best_exante": rmse("BEST"), "best_exante": best,
            "rmse_weighted": rmse("WEIGHTED"),
            "rmse_best_expost": min(per_model.values()) if per_model else np.nan,
            "best_expost": min(per_model, key=per_model.get) if per_model else None,
            "mae_gpr": float(np.mean(np.abs(t["GPR"] - t["obs"]))),
            "r2_gpr": r2("GPR"), "r2_best_exante": r2("BEST"), "r2_weighted": r2("WEIGHTED"),
            "sse_gpr": float(np.sum((t["GPR"] - t["obs"]) ** 2)), "sse_best": float(np.sum((t["BEST"] - t["obs"]) ** 2)),
            "sse_weighted": float(np.sum((t["WEIGHTED"] - t["obs"]) ** 2)), "sst": sst,
            "cov68": v["cov68"], "cov95": v["cov95"]}


rows, oos_all = [], []
t_start = time.time()
for agg in AGGS:
    for k, (t0, t1) in enumerate(folds):
        p = pairs_for(t0 - pd.Timedelta(days=G["train_days"]), t1, providers=GRID, aggs=[agg])
        if p.empty:
            continue
        p = p[p["date"] < t1]
        # campionamento per (stazione, periodo) come nel GPR giornaliero (memoria)
        keys = p[["station", "period"]].drop_duplicates()
        if len(keys) > 12000:
            keys = keys.sample(12000, random_state=k)
            p = p.merge(keys, on=["station", "period"])
        p = by_region(p, st_all)
        groups = {key: g for key, g in p.groupby(["region", "var", "lead_day"])}
        cases = [(r["id"], var, lead) for r in REGIONS if r.get("verify", True)
                 for var in target_vars(agg) for lead in LEADS if (r["id"], var, lead) in groups]
        res = Parallel(n_jobs=-1)(delayed(run_fold_case)(k, rid, agg, var, lead, groups[(rid, var, lead)], t0,
                                                         G["random_seed"] + 1000 * k + i)
                                  for i, (rid, var, lead) in enumerate(cases))
        rows += [r[1] for r in res if r]
        oos_all += [r[0] for r in res if r]
        print(f"  {agg:>4} fold {k + 1}/{len(folds)} ({t0:%Y-%m-%d}): {sum(r is not None for r in res)} casi "
              f"({time.time() - t_start:.0f}s)")

bt = pd.DataFrame(rows)
if bt.empty:
    print("Nessun risultato")
    sys.exit(0)
bt["run_date"] = str(pd.Timestamp.now().date())

# le previsioni fuori campione dei blocchi riempiono lo storico di GPR/BEST/WEIGHTED nelle classifiche
# (solo nei giorni che non le hanno gia' dalla validazione giornaliera)
if oos_all:
    new = daily_stats(pd.concat(oos_all, ignore_index=True))
    have = set(zip(stats["date"], stats["provider"]))
    new = new[[k not in have for k in zip(new["date"], new["provider"])]]
    stats = pd.concat([stats, new], ignore_index=True).sort_values(["date", "region", "provider", "agg", "var", "lead_day"])
    stats.to_parquet(DATA / "stats" / "daily_stats.parquet", index=False, compression="zstd")
    print(f"Storico classifiche: aggiunte {len(new)} righe di GPR/BEST/WEIGHTED fuori campione")
bt["gpr_beats_best"] = bt["rmse_gpr"] < bt["rmse_best_exante"]
bt["gpr_beats_weighted"] = bt["rmse_gpr"] < bt["rmse_weighted"]
bt.round(4).to_csv(OUT_CSV, index=False)

# sintesi su tutti i fold (RMSE medio quadratico pesato sul numero di righe)
def agg_rmse(g, c):
    return float(np.sqrt(np.average(g[c] ** 2, weights=g["n"])))


summ = []
for (rid, var, agg, lead), g in bt.groupby(["region", "var", "agg", "lead_day"]):
    summ.append({"region": rid, "var": var, "agg": agg, "lead_day": lead, "folds": len(g), "n": int(g["n"].sum()),
                 "rmse_gpr": agg_rmse(g, "rmse_gpr"), "rmse_best_exante": agg_rmse(g, "rmse_best_exante"),
                 "rmse_weighted": agg_rmse(g, "rmse_weighted"), "rmse_best_expost": agg_rmse(g, "rmse_best_expost"),
                 # R² complessivo: 1 - somma SSE / somma SST su tutti i blocchi
                 "r2_gpr": float(1 - g["sse_gpr"].sum() / g["sst"].sum()) if g["sst"].sum() > 0 else np.nan,
                 "r2_best_exante": float(1 - g["sse_best"].sum() / g["sst"].sum()) if g["sst"].sum() > 0 else np.nan,
                 "r2_weighted": float(1 - g["sse_weighted"].sum() / g["sst"].sum()) if g["sst"].sum() > 0 else np.nan,
                 "share_gpr_beats_best": float(g["gpr_beats_best"].mean()),
                 "share_gpr_beats_weighted": float(g["gpr_beats_weighted"].mean()),
                 "cov68": float(np.average(g["cov68"], weights=g["n"])),
                 "cov95": float(np.average(g["cov95"], weights=g["n"])),
                 "first_test": g["test_start"].min(), "last_test": g["test_start"].max()})
summ = pd.DataFrame(summ)
summ["skill_vs_best_pct"] = 100 * (1 - summ["rmse_gpr"] / summ["rmse_best_exante"])
summ.round(4).to_csv(CSV_DIR / "gpr_backtest_sintesi.csv", index=False)
with open(SITE_DATA / "gpr_backtest.json", "w") as f:
    json.dump(json.loads(summ.round(4).to_json(orient="records")), f)
print("\nGPR vs migliore ex-ante (RMSE, media sui fold):")
print(summ.groupby(["region", "var", "agg"])[["rmse_gpr", "rmse_best_exante", "rmse_weighted", "r2_gpr",
                                                "r2_best_exante", "share_gpr_beats_best", "cov95"]].mean().round(3).to_string())
print(f"Backtest completato in {time.time() - t_start:.0f}s")
