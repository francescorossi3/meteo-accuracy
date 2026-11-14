"""Accoppiamento previsioni-osservazioni, aggregazioni, metriche e combinazioni."""
import numpy as np
import pandas as pd

from .config import AGGS, CFG, DATA, target_vars
from .providers import SNAPSHOT_HOURLY_MAX_LEAD

VCFG = CFG["verification"]
BLOCK_H = {"1h": 1, "6h": 6, "24h": 24}
MIN_T = {"1h": 1, "6h": 5, "24h": 20}      # ore minime per medie / max / min di temperatura
MIN_P = {"1h": 1, "6h": 6, "24h": 23}      # ore minime per somme di pioggia (23: giorno con cambio ora)


def _local(valid_time, tz):
    """Ora locale naive di (valid_time - 1 h). tz: stringa unica o sequenza per riga."""
    v = pd.Series(pd.to_datetime(valid_time, utc=True)).reset_index(drop=True) - pd.Timedelta(hours=1)
    if isinstance(tz, str):
        return v.dt.tz_convert(tz).dt.tz_localize(None)
    tz = pd.Series(np.asarray(tz))
    out = pd.Series(pd.NaT, index=v.index, dtype="datetime64[ns]")
    for z in tz.unique():
        m = (tz == z).values
        out[m] = v[m].dt.tz_convert(z).dt.tz_localize(None).values
    return out


def period_of(valid_time, agg, tz="UTC"):
    """Etichetta del periodo in ora locale (inizio periodo, naive).
    valid_time e' fine ora (UTC) -> si sottrae 1 h prima di assegnare il blocco."""
    loc = _local(valid_time, tz)
    if agg == "1h":
        return loc
    if agg == "6h":
        return loc.dt.floor("6h")
    return loc.dt.floor("D")


def aggregate(df, agg, keys, t_cols=(), p_cols=(), tz="UTC"):
    """Aggrega valori orari per blocchi locali.
    tz: fuso unico (stringa) oppure nome di una colonna di df con il fuso di ogni riga.
    t_cols: colonne di temperatura -> media (+ max/min per 24h, prefisso tmax/tmin)
    p_cols: colonne di pioggia -> somma. Blocchi incompleti -> NaN."""
    d = df.reset_index(drop=True).copy()
    d["period"] = period_of(d["valid_time"], agg, d[tz].values if tz in d.columns else tz).values
    g = d.groupby(list(keys) + ["period"], sort=False)
    parts = {}
    for c in t_cols:
        n = g[c].count()
        ok = n >= MIN_T[agg]
        parts[c] = g[c].mean().where(ok)
        if agg == "24h":
            parts[c.replace("t2m", "tmax")] = g[c].max().where(ok)
            parts[c.replace("t2m", "tmin")] = g[c].min().where(ok)
    for c in p_cols:
        n = g[c].count()
        parts[c] = g[c].sum(min_count=1).where(n >= MIN_P[agg])
    return pd.DataFrame(parts).reset_index()


# ---------------------------------------------------------------- coppie
def station_table():
    from .regions import station_tz
    st = pd.read_csv(DATA / "stations.csv", dtype={"station": str})
    st["tz"] = station_tz(st)
    return st


def hourly_pairs(fc, obs, stations=None):
    st = station_table() if stations is None else stations
    p = fc.merge(obs, on=["station", "valid_time"], suffixes=("_fc", "_obs"), how="inner")
    p = p.merge(st[["station", "tz"]], on="station", how="inner")
    for v in ("t2m", "prcp"):                       # stesso set di ore per prev. e oss.
        bad = p[f"{v}_fc"].isna() | p[f"{v}_obs"].isna()
        p.loc[bad, [f"{v}_fc", f"{v}_obs"]] = np.nan
    return p


def aggregate_pairs(hp, agg):
    """-> coppie long: station, provider, lead_day, period, agg, var, fc, obs"""
    hp = hp
    if agg == "1h":                                 # niente 1h per provider a passo 3-6 h
        lim = hp["provider"].map(SNAPSHOT_HOURLY_MAX_LEAD).fillna(99)
        hp = hp[hp["lead_day"] <= lim]
    a = aggregate(hp, agg, ["station", "provider", "lead_day"],
                  t_cols=("t2m_fc", "t2m_obs"), p_cols=("prcp_fc", "prcp_obs"), tz="tz")
    out = []
    for v in target_vars(agg):
        x = a[["station", "provider", "lead_day", "period", f"{v}_fc", f"{v}_obs"]].dropna()
        x.columns = ["station", "provider", "lead_day", "period", "fc", "obs"]
        x["var"] = v
        out.append(x)
    res = pd.concat(out, ignore_index=True)
    res["agg"] = agg
    return res


def pairs_for(start, end, providers=None, aggs=AGGS):
    """Coppie aggregate per [start, end] (UTC), elaborate un mese alla volta
    per contenere la memoria."""
    from .storage import read_forecasts, read_observations, utc
    start, end = utc(start), utc(end)
    edges = list(pd.date_range(start.normalize(), end, freq="MS", tz="UTC"))
    edges = sorted(set([start] + [e for e in edges if e > start] + [end + pd.Timedelta(seconds=1)]))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        a2, b2 = a - pd.Timedelta(days=1), min(b + pd.Timedelta(days=1), end)   # margine: blocchi a cavallo
        fc = read_forecasts(a2, b2, providers)
        if fc.empty:
            continue
        obs = read_observations(a2, b2)
        hp = hourly_pairs(fc, obs)
        out += [aggregate_pairs(hp, g) for g in aggs]
    if not out:
        return pd.DataFrame(columns=["station", "provider", "lead_day", "period", "fc", "obs", "var", "agg", "date"])
    res = pd.concat(out, ignore_index=True)
    res = res.drop_duplicates(["station", "provider", "lead_day", "period", "agg", "var"], keep="last")
    lo = (start - pd.Timedelta(hours=14)).tz_localize(None)          # fusi da UTC-12 a UTC+14
    res = res[res["period"] >= lo.floor("D")]
    res["date"] = res["period"].dt.normalize()
    return res


# ---------------------------------------------------------------- metriche
def r2_score(obs, fc):
    """R² = 1 - SSE/SST: quota della varianza osservata spiegata dalla previsione
    (1 = perfetta, 0 = come prevedere sempre la media osservata, < 0 = peggio)."""
    obs, fc = np.asarray(obs, float), np.asarray(fc, float)
    sst = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((fc - obs) ** 2) / sst) if sst > 0 else np.nan


def _cat_scores(fc, obs, thr):
    h = np.sum((fc >= thr) & (obs >= thr)); f = np.sum((fc >= thr) & (obs < thr))
    m = np.sum((fc < thr) & (obs >= thr)); c = np.sum((fc < thr) & (obs < thr))
    n = h + f + m + c
    hr = (h + f) * (h + m) / n if n else np.nan
    div = lambda a, b: a / b if b else np.nan       # noqa: E731
    return {"pod": div(h, h + m), "far": div(f, h + f), "csi": div(h, h + f + m),
            "ets": div(h - hr, h + f + m - hr), "fbias": div(h + f, h + m)}


def score_table(pairs, by=("region", "provider", "agg", "var", "lead_day")):
    rows = []
    for key, g in pairs.groupby(list(by), sort=True):
        if len(g) < VCFG["min_pairs"]:
            continue
        e = g["fc"].values - g["obs"].values
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update(n=len(g), bias=e.mean(), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                 corr=np.corrcoef(g["fc"], g["obs"])[0, 1] if g["obs"].std() > 0 and g["fc"].std() > 0 else np.nan,
                 r2=r2_score(g["obs"].values, g["fc"].values),
                 obs_mean=g["obs"].mean())
        if r["var"] == "prcp":
            for i, thr in enumerate(VCFG["prcp_thresholds"][r["agg"]]):
                for k, val in _cat_scores(g["fc"].values, g["obs"].values, thr).items():
                    r[f"{k}_{i + 1}"] = val
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- combinazioni
def best_and_weights(scores):
    """Per ogni (regione, agg, var, lead) -> provider migliore e pesi 1/MSE.
    Entrano solo i modelli che coprono tutta la regione, cosi' le
    combinazioni sono calcolabili ovunque (mappe e click sul sito)."""
    from .regions import region_providers
    rows = []
    for (rid, agg, var, lead), g in scores.groupby(["region", "agg", "var", "lead_day"]):
        g = g[g["provider"].isin(region_providers(rid))]
        metric = VCFG["primary_score"].get(var, "rmse")
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        best = g.loc[g[metric].idxmin(), "provider"]
        w = 1.0 / (g["rmse"] ** 2)
        w = w / w.sum()
        for prov, wi, sc in zip(g["provider"], w, g[metric]):
            rows.append({"region": rid, "agg": agg, "var": var, "lead_day": int(lead), "provider": prov,
                         "weight": float(wi), "score": float(sc), "metric": metric,
                         "is_best": prov == best})
    return pd.DataFrame(rows)


def combo_forecasts(pairs, sel):
    """Crea le serie BEST e WEIGHTED dalle coppie dei singoli provider."""
    keys = ["region", "station", "lead_day", "period", "agg", "var"]
    wide = pairs.pivot_table(index=keys, columns="provider", values="fc", aggfunc="first")
    obs = pairs.groupby(keys)["obs"].first()
    out = []
    for (rid, agg, var, lead), s in sel.groupby(["region", "agg", "var", "lead_day"]):
        try:
            sub = wide.xs((rid, lead, agg, var), level=("region", "lead_day", "agg", "var"), drop_level=False)
        except KeyError:
            continue
        provs = [p for p in s["provider"] if p in sub.columns]
        if not provs:
            continue
        best = s.loc[s["is_best"], "provider"].iloc[0]
        w = s.set_index("provider").loc[provs, "weight"].values
        F = sub[provs].values
        M = ~np.isnan(F)
        wf = np.where(M, F, 0) @ w
        ws = M.astype(float) @ w
        weighted = np.where(ws > 0.5, wf / np.maximum(ws, 1e-9), np.nan)  # serve >= meta' del peso
        b = sub[best].values if best in sub.columns else np.full(len(sub), np.nan)
        for name, vals in (("BEST", b), ("WEIGHTED", weighted)):
            d = sub.index.to_frame(index=False)
            d["provider"], d["fc"] = name, vals
            d["obs"] = obs.loc[sub.index].values
            out.append(d.dropna(subset=["fc", "obs"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- statistiche giornaliere
# Per scalare a anni di dati non si tengono tutte le coppie: per ogni giorno si
# salvano solo le "statistiche sufficienti" (somme) da cui si ricavano bias,
# MAE, RMSE, correlazione e tabelle di contingenza su qualsiasi finestra.
def daily_stats(pairs):
    d = pairs.copy()
    d["date"] = d["period"].dt.normalize()
    d["e"] = d["fc"] - d["obs"]
    d["ae"], d["e2"] = d["e"].abs(), d["e"] ** 2
    d["f2"], d["o2"], d["fo"] = d["fc"] ** 2, d["obs"] ** 2, d["fc"] * d["obs"]
    keys = ["region", "provider", "agg", "var", "lead_day", "date"]
    s = d.groupby(keys).agg(n=("e", "size"), se=("e", "sum"), sae=("ae", "sum"), se2=("e2", "sum"),
                            sf=("fc", "sum"), so=("obs", "sum"), sf2=("f2", "sum"), so2=("o2", "sum"),
                            sfo=("fo", "sum"))
    p = d[d["var"] == "prcp"]
    if not p.empty:
        cats = []
        for agg, g in p.groupby("agg"):
            g = g.copy()
            for i, thr in enumerate(VCFG["prcp_thresholds"][agg]):
                fy, oy = g["fc"] >= thr, g["obs"] >= thr
                g[f"h_{i + 1}"], g[f"f_{i + 1}"] = (fy & oy), (fy & ~oy)
                g[f"m_{i + 1}"], g[f"c_{i + 1}"] = (~fy & oy), (~fy & ~oy)
            cats.append(g.groupby(keys)[[c for c in g.columns if c[:2] in ("h_", "f_", "m_", "c_")]].sum())
        s = s.join(pd.concat(cats))
    return s.reset_index()


def stats_to_scores(stats, by=("region", "provider", "agg", "var", "lead_day")):
    num = [c for c in stats.columns if c not in ("region", "provider", "agg", "var", "lead_day", "date") + tuple(by)]
    s = stats.groupby(list(by))[num].sum(min_count=1).reset_index()
    s = s[s["n"] >= VCFG["min_pairs"]].copy()
    n = s["n"]
    s["bias"], s["mae"], s["rmse"] = s["se"] / n, s["sae"] / n, np.sqrt(s["se2"] / n)
    cov = n * s["sfo"] - s["sf"] * s["so"]
    den = np.sqrt((n * s["sf2"] - s["sf"] ** 2).clip(lower=0) * (n * s["so2"] - s["so"] ** 2).clip(lower=0))
    s["corr"] = np.where(den > 0, cov / den.replace(0, np.nan), np.nan)
    sst = (s["so2"] - s["so"] ** 2 / n).clip(lower=0)               # somma dei quadrati totale
    s["r2"] = np.where(sst > 0, 1 - s["se2"] / sst.replace(0, np.nan), np.nan)
    s["obs_mean"] = s["so"] / n
    for i in (1, 2, 3):
        if f"h_{i}" not in s:
            continue
        h, f, m, c = (s[f"{k}_{i}"] for k in "hfmc")
        tot = h + f + m + c
        hr = (h + f) * (h + m) / tot.replace(0, np.nan)
        s[f"pod_{i}"] = h / (h + m).replace(0, np.nan)
        s[f"far_{i}"] = f / (h + f).replace(0, np.nan)
        s[f"csi_{i}"] = h / (h + f + m).replace(0, np.nan)
        s[f"ets_{i}"] = (h - hr) / (h + f + m - hr).replace(0, np.nan)
        s[f"fbias_{i}"] = (h + f) / (h + m).replace(0, np.nan)
    keep = list(by) + ["n", "bias", "mae", "rmse", "corr", "r2", "obs_mean"] + \
        [c for c in s.columns if c.split("_")[0] in ("pod", "far", "csi", "ets", "fbias")]
    return s[keep]


def by_region(pairs, stations):
    """Duplica le coppie per ogni regione che contiene la stazione (colonna 'region')."""
    from .regions import RID, region_stations, verified
    out = []
    for rid in [r for r in RID if verified(r)]:
        ids = set(region_stations(rid, stations)["station"])
        p = pairs[pairs["station"].isin(ids)].copy()
        p["region"] = rid
        out.append(p)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- classifiche (stats -> tabelle)
COMBOS = ("BEST", "WEIGHTED", "GPR")


def merge_stats(stats, new):
    """Sostituisce in stats le righe (data, provider) presenti in new e aggiunge le nuove."""
    if new is None or new.empty:
        return stats
    if stats is None or stats.empty:
        return new.copy()
    key_new = set(zip(new["date"], new["provider"]))
    keep = [k not in key_new for k in zip(stats["date"], stats["provider"])]
    out = pd.concat([stats[keep], new], ignore_index=True)
    return out.sort_values(["date", "region", "provider", "agg", "var", "lead_day"]).reset_index(drop=True)


def rankings(stats, provider_names=None):
    """Classifiche su finestre mobili + trend mensile, provider e combinazioni (BEST, WEIGHTED, GPR)."""
    last_day = stats["date"].max()
    boards = []
    for w in VCFG["score_windows_days"] + ["all"]:
        s = stats if w == "all" else stats[stats["date"] > last_day - pd.Timedelta(days=w)]
        sc = stats_to_scores(s)
        sc.insert(0, "window", f"{w}d" if w != "all" else "all")
        boards.append(sc)
    scores = pd.concat(boards, ignore_index=True)
    if provider_names:
        scores["provider_name"] = scores["provider"].map(provider_names).fillna(scores["provider"])
    st_m = stats.copy()
    st_m["month"] = st_m["date"].dt.to_period("M").astype(str)
    trend = stats_to_scores(st_m, by=("region", "provider", "agg", "var", "lead_day", "month"))
    return scores, trend
