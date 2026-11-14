# -*- coding: utf-8 -*-
"""
06 - Assembla il sito statico in build/site (pubblicato su GitHub Pages).
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from meteoita.config import AGGS, BUILD, CFG, CSV_DIR, DATA, LEADS, SITE_DATA, gridded_ids

SITE = BUILD / "site"
shutil.copytree(ROOT / "web", SITE, dirs_exist_ok=True)

repo = os.environ.get("GITHUB_REPOSITORY", "")            # es. "utente/meteo-accuracy"
branch = os.environ.get("GITHUB_REF_NAME", "main")
meta = {
    "title": CFG["site"]["title"],
    "built": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
    "aggs": AGGS, "leads": LEADS,
    "openmeteo_url": CFG["site"]["openmeteo_url"],
    "providers": [{"id": p["id"], "name": p["name"], "type": p["type"], "model": p.get("model"),
                   "coverage": p.get("coverage", "global"), "gridded": p["id"] in gridded_ids()}
                  for p in CFG["providers"] if p.get("enabled", True)],
    "prcp_thresholds": CFG["verification"]["prcp_thresholds"],
    "data_url": f"https://github.com/{repo}/tree/{branch}/data/csv" if repo else "../data/csv",
    "raw_url": f"https://raw.githubusercontent.com/{repo}/{branch}/data/csv" if repo else "",
}
mv = SITE_DATA / "meta_verify.json"
if mv.exists():
    meta.update({k: v for k, v in json.load(open(mv)).items() if k != "providers"})
with open(SITE_DATA / "meta.json", "w") as f:
    json.dump(meta, f)
# osservazioni recenti delle stazioni (ultimi 35 giorni) per i grafici "passato" del sito
from meteoita.storage import read_observations
import numpy as np
end = pd.Timestamp.now(tz="UTC").floor("h")
obs = read_observations(end - pd.Timedelta(days=35), end)
st_csv = DATA / "stations.csv"
if len(obs) and st_csv.exists():
    st = pd.read_csv(st_csv, dtype={"station": str})
    t0 = obs["valid_time"].min()
    idx = pd.date_range(t0, end, freq="h")
    out = {"start_ms": int(t0.value // 1_000_000), "stations": [], "t2m": {}, "prcp": {}}
    for sid, g in obs.groupby("station"):
        row = st[st["station"] == sid]
        if row.empty:
            continue
        g = g.set_index("valid_time").reindex(idx)
        out["stations"].append({"id": sid, "name": row["name"].iloc[0], "lat": float(row["lat"].iloc[0]),
                                "lon": float(row["lon"].iloc[0])})
        for v in ("t2m", "prcp"):
            out[v][sid] = [None if not np.isfinite(x) else round(float(x), 1) for x in g[v].values]
    with open(SITE_DATA / "obs_recent.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Osservazioni recenti: {len(out['stations'])} stazioni")

# riepilogo Excel con le tabelle principali (un foglio per tabella)
sheets = {"classifica_90g": "classifica.csv", "holdout_e_combinazioni": "classifica_holdout.csv",
          "migliore_e_pesi": "migliore_e_pesi.csv", "gpr_validazione": "gpr_validazione.csv",
          "trend_mensile": "trend_mensile.csv", "eventi_punteggi": "eventi_punteggi.csv",
          "eventi_elenco": "eventi_elenco.csv", "stazioni": "stazioni.csv"}
with pd.ExcelWriter(CSV_DIR / "riepilogo.xlsx", engine="openpyxl") as xw:
    for sh, fn in sheets.items():
        if (CSV_DIR / fn).exists():
            df = pd.read_csv(CSV_DIR / fn)
            if sh == "classifica_90g" and "window" in df:
                df = df[df["window"] == f"{CFG['verification']['primary_window_days']}d"]
            df.to_excel(xw, sheet_name=sh, index=False)
            ws = xw.sheets[sh]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(9, min(28, len(str(col[0].value)) + 3))
print("Sito pronto in", SITE)
