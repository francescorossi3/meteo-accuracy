"""Verifica per eventi significativi (pioggia forte, caldo, gelo, ...).

Partendo dalle coppie giornaliere (osservato + previsione di ogni provider):
  * evento osservato  = l'osservazione supera la soglia
  * evento previsto   = la previsione supera la soglia
Per provider x tipo di evento x anticipo:
  POD  (eventi colti / eventi osservati)
  FAR  (falsi allarmi / eventi previsti)
  CSI  (colti / (colti + mancati + falsi allarmi))
  errore medio e MAE sui soli giorni con evento osservato
"""
import numpy as np
import pandas as pd

from .config import CFG

EVENTS = CFG.get("events", [])


def _hit(x, op, thr):
    return (x >= thr) if op == ">=" else (x <= thr)


def event_tables(daily):
    """daily: coppie long con colonne region, date, station, var, lead_day, provider, fc, obs."""
    scores, lists = [], []
    for rid, dr in daily.groupby("region"):
        sc, ls = _event_tables(dr)
        sc.insert(0, "region", rid)
        if not ls.empty:
            ls.insert(0, "region", rid)
        scores.append(sc); lists.append(ls)
    return (pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(),
            pd.concat(lists, ignore_index=True) if lists else pd.DataFrame())


def _event_tables(daily):
    scores, lists = [], []
    for ev in EVENTS:
        d = daily[daily["var"] == ev["var"]].copy()
        if d.empty:
            continue
        d["o"] = _hit(d["obs"], ev["op"], ev["thr"])
        d["f"] = _hit(d["fc"], ev["op"], ev["thr"])
        for (prov, lead), g in d.groupby(["provider", "lead_day"]):
            h = int((g["o"] & g["f"]).sum()); m = int((g["o"] & ~g["f"]).sum()); fa = int((~g["o"] & g["f"]).sum())
            e = (g["fc"] - g["obs"])[g["o"]]
            scores.append({"event": ev["id"], "event_name": ev["name"], "provider": prov, "lead_day": int(lead),
                           "n_days": len(g), "observed": h + m, "hits": h, "misses": m, "false_alarms": fa,
                           "pod": h / (h + m) if h + m else np.nan,
                           "far": fa / (h + fa) if h + fa else np.nan,
                           "csi": h / (h + m + fa) if h + m + fa else np.nan,
                           "bias_on_event": e.mean() if len(e) else np.nan,
                           "mae_on_event": e.abs().mean() if len(e) else np.nan})
        # elenco eventi osservati con le previsioni di ogni provider (formato largo)
        obs_ev = d[d["o"]]
        if not obs_ev.empty:
            w = obs_ev.pivot_table(index=["date", "station", "lead_day"], columns="provider",
                                   values="fc", aggfunc="first")
            w.insert(0, "osservato", obs_ev.groupby(["date", "station", "lead_day"])["obs"].first())
            w = w.reset_index()
            w.insert(0, "event", ev["id"])
            lists.append(w)
    sc = pd.DataFrame(scores)
    ls = pd.concat(lists, ignore_index=True) if lists else pd.DataFrame()
    return sc, ls
