# -*- coding: utf-8 -*-
"""
03 - Verifica: coppie previsione/osservazione, classifiche, migliore e pesi.

Passi:
 1. coppie orarie -> aggregate a 1h / 6h / 24h (ora locale) per ogni lead day
 2. statistiche giornaliere salvate in data/stats/daily_stats.parquet (storico completo)
 3. classifiche su finestre mobili (30 / 90 / 365 giorni + tutto lo storico)
 4. scelta del provider MIGLIORE e dei PESI (1/MSE) sul periodo di training,
    valutazione di BEST e WEIGHTED sugli ultimi giorni NON usati (holdout)
 5. migliore e pesi "operativi" (ultima finestra) -> sito e mappe
 6. export CSV / Excel / JSON
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from meteoita.config import (AGGS, BUILD, CFG, CSV_DIR, DATA, SITE_DATA, provider_names)
from meteoita.storage import read_observations
from meteoita.regions import REGIONS, archive_points, region_providers
from meteoita.storage import read_forecasts
from meteoita.verify import (COMBOS, best_and_weights, by_region, combo_forecasts, daily_stats, merge_stats,
                             pairs_for, rankings, score_table, station_table, stats_to_scores)

V = CFG["verification"]
WIN, HOLD = V["primary_window_days"], V["holdout_days"]
ap = argparse.ArgumentParser()
ap.add_argument("--recompute-days", type=int, default=max(15, HOLD + 1),
                help="giorni (dall'ultima osservazione) di cui ricalcolare le statistiche")
ap.add_argument("--from", dest="from_date", default=None,
                help="ricalcola le statistiche da questa data (dopo un backfill)")
ap.add_argument("--to", dest="to_date", default=None,
                help="ricalcola solo fino a questa data (backfill a blocchi)")
args, _ = ap.parse_known_args()

stations = station_table()
names = provider_names()

# %% 1) coppie dei giorni recenti (o dal --from in poi, un mese alla volta)
now = pd.Timestamp.now(tz="UTC")
end = read_observations(now - pd.Timedelta(days=40), now)["valid_time"].max()
start = pd.Timestamp(args.from_date, tz="UTC") if args.from_date else end - pd.Timedelta(days=args.recompute_days)
rec_end = min(end, pd.Timestamp(args.to_date, tz="UTC") + pd.Timedelta(hours=23)) if args.to_date else end
print(f"Ricalcolo coppie: {start:%Y-%m-%d} -> {rec_end:%Y-%m-%d %H:%M} UTC")

STATS = DATA / "stats" / "daily_stats.parquet"
STATS.parent.mkdir(exist_ok=True)
stats = pd.read_parquet(STATS) if STATS.exists() else pd.DataFrame()
chunks = [t for t in pd.date_range(start, rec_end, freq="30D")] + [rec_end]
for a, b in zip(chunks[:-1], chunks[1:]):
    pairs = pairs_for(a, b)
    if pairs.empty:
        continue
    new = daily_stats(by_region(pairs, stations))
    if not stats.empty:                          # le combinazioni (BEST, WEIGHTED, GPR) restano
        stats = stats[~(stats["date"].isin(new["date"].unique()) & ~stats["provider"].isin(COMBOS))]
    stats = pd.concat([stats, new], ignore_index=True)
    print(f"  {a:%Y-%m-%d}..{b:%Y-%m-%d}: {len(pairs):,} coppie")
stats = stats.sort_values(["date", "region", "provider", "agg", "var", "lead_day"]).reset_index(drop=True)
stats.to_parquet(STATS, index=False, compression="zstd")
# per holdout e combinazioni servono le coppie degli ultimi HOLD giorni
pairs = by_region(pairs_for(end - pd.Timedelta(days=HOLD + 1), end), stations)
last_day = stats["date"].max()

# %% 3) training / holdout: BEST e WEIGHTED scelti PRIMA dell'holdout e valutati sull'holdout
hold_start = last_day - pd.Timedelta(days=HOLD - 1)
tr_stats = stats[(stats["date"] < hold_start) & (stats["date"] >= hold_start - pd.Timedelta(days=WIN))]
sel_train = best_and_weights(stats_to_scores(tr_stats)) if not tr_stats.empty else pd.DataFrame()
hold = pairs[pairs["date"] >= hold_start]
hold_combo = combo_forecasts(hold, sel_train) if not sel_train.empty else pd.DataFrame()
hold_scores = score_table(pd.concat([hold, hold_combo], ignore_index=True))
# le combinazioni fuori campione entrano nello storico -> compaiono nelle classifiche normali
if not hold_combo.empty:
    stats = merge_stats(stats, daily_stats(hold_combo))
    stats.to_parquet(STATS, index=False, compression="zstd")

# %% 4) classifiche su finestre mobili (provider + BEST, WEIGHTED, GPR) e trend mensile
scores, trend = rankings(stats, names)

# %% 5) selezione operativa (ultima finestra, holdout incluso)
sel = best_and_weights(stats_to_scores(stats[stats["date"] > last_day - pd.Timedelta(days=WIN)]))
for r in REGIONS:
    b = sel[sel["is_best"] & (sel["region"] == r["id"])]
    if not b.empty:
        print(f"\nMigliore per caso - {r['name']} (finestra operativa):")
        print(b.pivot_table(index=["var", "agg"], columns="lead_day", values="provider", aggfunc="first").to_string())

BUILD.mkdir(exist_ok=True)
hold_combo.to_parquet(BUILD / "holdout_combo.parquet", index=False)
hold.to_parquet(BUILD / "holdout_pairs.parquet", index=False)
pd.DataFrame({"hold_start": [hold_start], "last_day": [last_day], "end": [end]}).to_parquet(BUILD / "split.parquet")

# %% 6) export
r = lambda df: df.round(4)                                           # noqa: E731
r(scores).to_csv(CSV_DIR / "classifica.csv", index=False)
r(hold_scores).to_csv(CSV_DIR / "classifica_holdout.csv", index=False)
r(sel).to_csv(CSV_DIR / "migliore_e_pesi.csv", index=False)
r(trend).to_csv(CSV_DIR / "trend_mensile.csv", index=False)
stations.to_csv(CSV_DIR / "stazioni.csv", index=False)

# coppie giornaliere in formato largo (una colonna per provider) -> CSV mensili per Excel
daily = pairs_for(start, rec_end, aggs=["24h"])
wide = daily.pivot_table(index=["date", "station", "var", "lead_day"], columns="provider",
                         values="fc", aggfunc="first")
wide.insert(0, "osservato", daily.groupby(["date", "station", "var", "lead_day"])["obs"].first())
wide = wide.reset_index()
(CSV_DIR / "coppie_giornaliere").mkdir(exist_ok=True)
for ym, g in wide.groupby(wide["date"].dt.strftime("%Y-%m")):
    g = g.copy()
    g["date"] = g["date"].dt.strftime("%Y-%m-%d")
    path = CSV_DIR / "coppie_giornaliere" / f"{ym}.csv"
    if path.exists():                                # si aggiornano solo i giorni ricalcolati
        old = pd.read_csv(path, dtype={"station": str})
        g = pd.concat([old[~old["date"].isin(g["date"].unique())], g], ignore_index=True)
    g = g.sort_values(["date", "station", "var", "lead_day"])
    g.round(2).to_csv(path, index=False)

# %% 6b) archivio delle previsioni dei modelli sui siti non verificati (RadCalNet): CSV mensili
sites = archive_points()
for rid, sg in (sites.groupby("region") if len(sites) else []):
    out_dir = CSV_DIR / rid
    out_dir.mkdir(exist_ok=True)
    for m0 in pd.period_range(start.tz_localize(None).to_period("M"), rec_end.tz_localize(None).to_period("M"), freq="M"):
        a = max(pd.Timestamp(m0.start_time, tz="UTC"), start - pd.Timedelta(days=1))
        b = min(pd.Timestamp(m0.end_time, tz="UTC"), rec_end + pd.Timedelta(days=8))
        f = read_forecasts(a, b, region_providers(rid))
        f = f[f["station"].isin(sg["station"])]
        if f.empty:
            continue
        w = f.pivot_table(index=["valid_time", "station", "lead_day"], columns="provider",
                          values=["t2m", "prcp"], aggfunc="first")
        w.columns = [f"{v}_{p}" for v, p in w.columns]
        w = w.reset_index().rename(columns={"station": "sito"})
        w.insert(0, "valid_time_utc", w.pop("valid_time").dt.strftime("%Y-%m-%d %H:%M"))
        w.insert(2, "nome", w["sito"].map(dict(zip(sg["station"], sg["name"]))))
        path = out_dir / f"{m0}.csv"
        if path.exists():                       # si aggiornano solo le ore ricalcolate
            old = pd.read_csv(path)
            key = set(zip(w["valid_time_utc"], w["sito"], w["lead_day"]))
            old = old[[k not in key for k in zip(old["valid_time_utc"], old["sito"], old["lead_day"])]]
            w = pd.concat([old, w], ignore_index=True)
        w.sort_values(["valid_time_utc", "sito", "lead_day"]).round(2).to_csv(path, index=False)
    print(f"Archivio {rid}: {out_dir}")

# %% 7) verifica per eventi significativi, su tutto lo storico delle coppie giornaliere
from meteoita.events import event_tables
files = sorted((CSV_DIR / "coppie_giornaliere").glob("*.csv"))
if files:
    allday = pd.concat([pd.read_csv(f, dtype={"station": str}) for f in files], ignore_index=True)
    prov_cols = [c for c in allday.columns if c not in ("date", "station", "var", "lead_day", "osservato")]
    dl = allday.melt(id_vars=["date", "station", "var", "lead_day", "osservato"], value_vars=prov_cols,
                     var_name="provider", value_name="fc").dropna(subset=["fc", "osservato"])
    dl = by_region(dl.rename(columns={"osservato": "obs"}), stations)
    ev_scores, ev_list = event_tables(dl)
else:
    ev_scores, ev_list = pd.DataFrame(), pd.DataFrame()
ev_scores.round(4).to_csv(CSV_DIR / "eventi_punteggi.csv", index=False)
if not ev_list.empty:
    ev_list = ev_list.merge(stations[["station", "name"]], on="station", how="left")
    ev_list.round(2).to_csv(CSV_DIR / "eventi_elenco.csv", index=False)
    print("Eventi osservati:", ev_list.drop_duplicates(["region", "event", "date", "station"])
          .groupby(["region", "event"]).size().to_dict())

# file per il sito
def dump(name, obj):
    with open(SITE_DATA / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False, default=str)

clean = lambda df: json.loads(r(df).to_json(orient="records", date_format="iso"))   # noqa: E731
dump("scores.json", clean(scores))
dump("scores_holdout.json", clean(hold_scores))
dump("trend.json", clean(trend))
dump("weights.json", clean(sel))
dump("stations.json", clean(stations))
dump("events_scores.json", clean(ev_scores) if not ev_scores.empty else [])
dump("events_list.json", clean(ev_list.sort_values("date", ascending=False).head(3000)) if not ev_list.empty else [])
dump("events_def.json", CFG.get("events", []))
dump("meta_verify.json", {"last_day": str(last_day.date()), "hold_start": str(hold_start.date()),
                          "window_days": WIN, "holdout_days": HOLD,
                          "providers": [{"id": p["id"], "name": p["name"], "type": p["type"]}
                                        for p in CFG["providers"] if p.get("enabled", True)],
                          "prcp_thresholds": V["prcp_thresholds"],
                          "regions": [{"id": r["id"], "name": r["name"], "center": r.get("center"),
                                       "type": r.get("type", "area"), "sites": r.get("sites", []),
                                       "verify": r.get("verify", True),
                                       "radius_km": r.get("radius_km"),
                                       "providers": region_providers(r["id"]),
                                       "n_stations": int(len(stations[stations["station"].isin(
                                           pairs.loc[pairs["region"] == r["id"], "station"])]))}
                                      for r in REGIONS]})
print("\nExport completato:", CSV_DIR)
