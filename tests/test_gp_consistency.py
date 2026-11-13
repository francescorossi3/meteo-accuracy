# -*- coding: utf-8 -*-
"""Verifica che la predizione GPR sia identica in sklearn, in meteoita.gp
(modello esportato in JSON) e nel browser (web/gp.js, eseguito con node)."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from meteoita import gp

rng = np.random.default_rng(1)
n, provs = 200, ["a", "b", "c"]
df = pd.DataFrame({"lat": rng.uniform(37, 47, n), "lon": rng.uniform(7, 18, n), "elev": rng.uniform(0, 1500, n)})
truth = 15 - 0.6 * (df["lat"] - 37) - 0.006 * df["elev"] + rng.normal(0, 3, n)
for i, p in enumerate(provs):
    df[p] = truth + rng.normal(0.3 * i, 1 + 0.3 * i, n)
df.loc[rng.choice(n, 20, replace=False), "c"] = np.nan          # valori mancanti -> imputazione
obs = truth + rng.normal(0, 0.5, n)

X, ok = gp.design(df, provs, "none")
model, mu, sc = gp.fit(X[:150], obs.values[:150], restarts=0)
m = gp.export(model, mu, sc, {"providers": provs, "transform": "none"}, sig=10)

m_sk, s_sk = model.predict((X[150:] - mu) / sc, return_std=True)
m_py, s_py = gp.predict_export(m, X[150:])
print("sklearn vs export:  max|dmean| = %.2e  max|dstd| = %.2e" % (np.abs(m_sk - m_py).max(), np.abs(s_sk - s_py).max()))
assert np.allclose(m_sk, m_py, atol=1e-5) and np.allclose(s_sk, s_py, atol=1e-5)

node = shutil.which("node")
if node:
    with tempfile.TemporaryDirectory() as d:
        json.dump(m, open(f"{d}/m.json", "w"))
        feats = df.iloc[150:][["lat", "lon", "elev"] + provs].to_dict("records")
        json.dump([{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in feats], open(f"{d}/f.json", "w"))
        js = f"""const GP=require('{ROOT}/web/gp.js');const fs=require('fs');
          const m=GP.unpack(JSON.parse(fs.readFileSync('{d}/m.json')));
          const f=JSON.parse(fs.readFileSync('{d}/f.json'));
          console.log(JSON.stringify(f.map(x=>{{const r=GP.predictRaw(m,GP.design(m,x));return [r.mu,r.sd]}})));"""
        out = np.array(json.loads(subprocess.check_output([node, "-e", js])))
    print("export vs gp.js:    max|dmean| = %.2e  max|dstd| = %.2e" % (np.abs(out[:, 0] - m_py).max(), np.abs(out[:, 1] - s_py).max()))
    assert np.allclose(out[:, 0], m_py, atol=1e-6) and np.allclose(out[:, 1], s_py, atol=1e-6)
print("OK: le tre implementazioni coincidono")
