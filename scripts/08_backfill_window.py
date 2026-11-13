# -*- coding: utf-8 -*-
"""
08 - Sceglie il prossimo blocco di storico da ricostruire (backfill automatico).

Stato in data/backfill_state.json: {"earliest": "AAAA-MM-GG"} = primo giorno
gia' ricostruito. Ogni esecuzione propone i days_per_run giorni precedenti,
fino a backfill.start. Con --done aggiorna lo stato dopo un backfill riuscito.
Scrive start/end/todo in $GITHUB_OUTPUT (se esiste) e a schermo.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from meteoita.config import CFG, DATA

B = CFG.get("backfill", {"start": "2025-06-01", "days_per_run": 20})
STATE = DATA / "backfill_state.json"
ap = argparse.ArgumentParser()
ap.add_argument("--done", default=None, help="data iniziale appena ricostruita (aggiorna lo stato)")
args, _ = ap.parse_known_args()

if args.done:
    STATE.write_text(json.dumps({"earliest": args.done}))
    print("Stato aggiornato:", args.done)
    sys.exit(0)

if STATE.exists():
    earliest = pd.Timestamp(json.loads(STATE.read_text())["earliest"])
else:
    st = DATA / "stats" / "daily_stats.parquet"
    earliest = pd.read_parquet(st, columns=["date"])["date"].min() if st.exists() \
        else pd.Timestamp.now().normalize() - pd.Timedelta(days=3)
stop = pd.Timestamp(B["start"])
end = earliest - pd.Timedelta(days=1)
start = max(stop, end - pd.Timedelta(days=int(B["days_per_run"]) - 1))
todo = end >= stop
out = {"start": str(start.date()), "end": str(end.date()), "todo": str(todo).lower()}
print(out if todo else f"Storico completo fino a {stop.date()}: niente da fare")
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for k, v in out.items():
            f.write(f"{k}={v}\n")
