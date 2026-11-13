"""Archivio su disco: parquet mensili compatti (valori x10 in Int16).

Formato in memoria ("long"):
    previsioni  -> station, provider, valid_time (UTC), lead_day, t2m, prcp
    osservazioni-> station, valid_time (UTC), t2m, prcp

Su disco le previsioni sono "larghe" (una colonna per lead day) e scalate
x10 su interi a 16 bit: ~10 volte piu' piccole di un CSV e quindi adatte a
stare dentro un repository git per anni.
"""
import numpy as np
import pandas as pd

from .config import FC_DIR, OBS_DIR

VARS = ["t2m", "prcp"]
KEY_FC = ["station", "provider", "valid_time", "lead_day"]


def utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _to_i16(s):
    return pd.array(np.round(s.astype(float) * 10), dtype="Int16")


def _from_i16(s):
    return s.astype("float32") / 10.0


def _month_files(base, start, end, pattern):
    months = pd.period_range(utc(start).tz_localize(None).to_period("M"),
                             utc(end).tz_localize(None).to_period("M"), freq="M")
    return [base / pattern.format(ym=str(m)) for m in months]


# ------------------------------------------------------------------ previsioni
def write_forecasts(df):
    """Aggiunge/aggiorna previsioni (formato long). Le righe duplicate vengono sovrascritte."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.as_unit("ns")
    df["ym"] = df["valid_time"].dt.strftime("%Y-%m")
    n = 0
    for (prov, ym), part in df.groupby(["provider", "ym"]):
        wide = part.pivot_table(index=["station", "valid_time"], columns="lead_day",
                                values=VARS, aggfunc="last")
        wide.columns = [f"{v}_d{int(l)}" for v, l in wide.columns]
        wide = wide.reset_index()
        path = FC_DIR / prov / f"{ym}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_parquet(path)
            for c in old.columns:
                if c not in ("station", "valid_time"):
                    old[c] = _from_i16(old[c])
            old["valid_time"] = pd.to_datetime(old["valid_time"], utc=True).dt.as_unit("ns")
            wide = (pd.concat([old.set_index(["station", "valid_time"]),
                               wide.set_index(["station", "valid_time"])])
                    .groupby(level=[0, 1]).last().reset_index())
        cols = sorted([c for c in wide.columns if c not in ("station", "valid_time")],
                      key=lambda c: (c.split("_d")[0], int(c.split("_d")[1])))
        wide = wide[["station", "valid_time"] + cols].sort_values(["station", "valid_time"])
        for c in cols:
            wide[c] = _to_i16(wide[c])
        wide["station"] = wide["station"].astype(str)
        wide.to_parquet(path, index=False, compression="zstd")
        n += len(part)
    return n


def read_forecasts(start, end, providers=None):
    """Legge le previsioni con valid_time in [start, end] (UTC) in formato long."""
    start, end = utc(start), utc(end)
    provs = providers or [p.name for p in FC_DIR.iterdir() if p.is_dir()]
    out = []
    for prov in provs:
        for path in _month_files(FC_DIR / prov, start, end, "{ym}.parquet"):
            if not path.exists():
                continue
            w = pd.read_parquet(path)
            w["valid_time"] = pd.to_datetime(w["valid_time"], utc=True).dt.as_unit("ns")
            w = w[(w["valid_time"] >= start) & (w["valid_time"] <= end)]
            if w.empty:
                continue
            long = w.melt(id_vars=["station", "valid_time"], var_name="col", value_name="val")
            long["var"] = long["col"].str.split("_d").str[0]
            long["lead_day"] = long["col"].str.split("_d").str[1].astype(int)
            long["val"] = _from_i16(long["val"])
            long = long.pivot_table(index=["station", "valid_time", "lead_day"], columns="var",
                                    values="val", aggfunc="first").reset_index()
            long["provider"] = prov
            out.append(long)
    if not out:
        return pd.DataFrame(columns=KEY_FC + VARS)
    df = pd.concat(out, ignore_index=True)
    for v in VARS:
        if v not in df:
            df[v] = np.nan
    df.columns.name = None
    return df[KEY_FC + VARS]


# --------------------------------------------------------------- osservazioni
def write_observations(df):
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True).dt.as_unit("ns")
    df["ym"] = df["valid_time"].dt.strftime("%Y-%m")
    for ym, part in df.groupby("ym"):
        part = part[["station", "valid_time"] + VARS]
        path = OBS_DIR / f"{ym}.parquet"
        if path.exists():
            old = read_observations_file(path)
            part = pd.concat([old, part]).drop_duplicates(["station", "valid_time"], keep="last")
        part = part.sort_values(["station", "valid_time"]).copy()
        for v in VARS:
            part[v] = _to_i16(part[v])
        part["station"] = part["station"].astype(str)
        part.to_parquet(path, index=False, compression="zstd")
    return len(df)


def read_observations_file(path):
    o = pd.read_parquet(path)
    o["valid_time"] = pd.to_datetime(o["valid_time"], utc=True).dt.as_unit("ns")
    for v in VARS:
        o[v] = _from_i16(o[v])
    return o


def read_observations(start, end):
    start, end = utc(start), utc(end)
    out = [read_observations_file(p) for p in _month_files(OBS_DIR, start, end, "{ym}.parquet") if p.exists()]
    if not out:
        return pd.DataFrame(columns=["station", "valid_time"] + VARS)
    o = pd.concat(out, ignore_index=True)
    return o[(o["valid_time"] >= start) & (o["valid_time"] <= end)]
