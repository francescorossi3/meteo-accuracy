# -*- coding: utf-8 -*-
"""
09 - Riusa le mappe gia' pubblicate (nessuna chiamata alle API meteo).

Serve al workflow "Ricostruisci sito (senza API)": classifiche, GPR e pagine
vengono rigenerati dai dati del repository, mentre le mappe di previsione
(che richiederebbero di riscaricare i modelli) si copiano dal sito online.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

from meteoita.config import AGGS, SITE_DATA, target_vars
from meteoita.regions import REGIONS

repo = os.environ.get("GITHUB_REPOSITORY", "francescorossi3/meteo-accuracy")
owner, name = repo.split("/")
base = os.environ.get("SITE_URL", f"https://{owner}.github.io/{name}") + "/data/maps"
n = 0
for r in REGIONS:
    (SITE_DATA / "maps" / r["id"]).mkdir(parents=True, exist_ok=True)
    for agg in AGGS:
        for var in target_vars(agg):
            url = f"{base}/{r['id']}/{var}_{agg}.json"
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"  - {url}: {resp.status_code}")
                continue
            json.loads(resp.text)                    # controllo che sia JSON valido
            (SITE_DATA / "maps" / r["id"] / f"{var}_{agg}.json").write_text(resp.text)
            n += 1
print(f"Mappe riusate dal sito pubblicato: {n}")
