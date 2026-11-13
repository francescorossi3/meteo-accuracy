"""Download delle previsioni.

Convenzioni comuni a tutti i provider (formato restituito):
    valid_time : UTC, ora di fine del periodo
    t2m        : temperatura a 2 m istantanea a valid_time [°C]
    prcp       : precipitazione nell'ora che termina a valid_time [mm]

Per aggiungere un nuovo provider "snapshot" basta scrivere una funzione
    fetch_<api>(lat, lon, elev, key) -> DataFrame(valid_time, t2m, prcp)
e registrarla nel dizionario SNAPSHOT_APIS in fondo al file.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from .config import CFG

UA = os.environ.get("METEO_USER_AGENT", "meteo-accuracy/1.0 (github.com)")
OM_KEY = os.environ.get("OPENMETEO_API_KEY", "")


def om_url(url):
    """Con chiave a pagamento si usano gli endpoint 'customer-' di Open-Meteo."""
    return url.replace("https://", "https://customer-", 1) if OM_KEY else url


def om_params(params):
    return dict(params, apikey=OM_KEY) if OM_KEY else params
SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA

OM_VARS = {"t2m": "temperature_2m", "prcp": "precipitation"}


def http_get(url, params=None, headers=None, tries=4, timeout=60):
    """GET con retry ed attesa sui limiti di frequenza (HTTP 429)."""
    for k in range(tries):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            print(f"   ! rete: {e}; ritento")
            time.sleep(5 * (k + 1))
            continue
        if r.status_code == 429:
            print("   ! troppe richieste (429): attendo 65 s")
            time.sleep(65)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (k + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Richiesta fallita dopo {tries} tentativi: {url}")


# =====================================================================
#  OPEN-METEO
# =====================================================================
def _om_locations(js):
    return js if isinstance(js, list) else [js]


def _om_key(hourly, base, model):
    """Le chiavi possono avere il suffisso _<model> se si chiedono piu' modelli."""
    for k in (f"{base}_{model}", base):
        if k in hourly:
            return k
    return None


def openmeteo_previous_runs(model, lats, lons, start_date, end_date, leads, chunk=50):
    """Previsioni a lead fisso (Previous Runs API) per piu' punti.

    previous_dayN = valore previsto N*24 h prima dell'ora valida
    (N=0 -> run piu' recente). Ritorna long df: idx, valid_time, lead_day, t2m, prcp
    """
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    names = []
    for v in OM_VARS.values():
        for n in leads:
            names.append(v if n == 0 else f"{v}_previous_day{n}")
    out = []
    for i0 in range(0, len(lats), chunk):
        la, lo = lats[i0:i0 + chunk], lons[i0:i0 + chunk]
        params = {"latitude": ",".join(f"{x:.4f}" for x in la),
                  "longitude": ",".join(f"{x:.4f}" for x in lo),
                  "hourly": ",".join(names), "models": model,
                  "start_date": str(start_date), "end_date": str(end_date),
                  "timezone": "GMT"}
        js = http_get(om_url(url), om_params(params)).json()
        for j, loc in enumerate(_om_locations(js)):
            h = loc.get("hourly", {})
            t = pd.to_datetime(h.get("time", []), utc=True)
            for n in leads:
                row = {"idx": i0 + j, "valid_time": t, "lead_day": n}
                for short, v in OM_VARS.items():
                    base = v if n == 0 else f"{v}_previous_day{n}"
                    k = _om_key(h, base, model)
                    row[short] = np.array(h[k], dtype=float) if k else np.full(len(t), np.nan)
                out.append(pd.DataFrame(row))
        time.sleep(1)
    if not out:
        return pd.DataFrame(columns=["idx", "valid_time", "lead_day", "t2m", "prcp"])
    df = pd.concat(out, ignore_index=True)
    return df.dropna(subset=["t2m", "prcp"], how="all")


def openmeteo_forecast(model, lats, lons, days=8, chunk=100):
    """Previsione corrente (Forecast API) su molti punti -> (long df, elevazioni)."""
    url = CFG["site"]["openmeteo_url"]
    out, elev = [], {}
    for i0 in range(0, len(lats), chunk):
        la, lo = lats[i0:i0 + chunk], lons[i0:i0 + chunk]
        params = {"latitude": ",".join(f"{x:.4f}" for x in la),
                  "longitude": ",".join(f"{x:.4f}" for x in lo),
                  "hourly": ",".join(OM_VARS.values()), "models": model,
                  "forecast_days": days, "timezone": "GMT"}
        js = http_get(om_url(url), om_params(params)).json()
        for j, loc in enumerate(_om_locations(js)):
            h = loc.get("hourly", {})
            t = pd.to_datetime(h.get("time", []), utc=True)
            d = {"idx": i0 + j, "valid_time": t}
            for short, v in OM_VARS.items():
                k = _om_key(h, v, model)
                d[short] = np.array(h[k], dtype=float) if k else np.full(len(t), np.nan)
            out.append(pd.DataFrame(d))
            elev[i0 + j] = loc.get("elevation", np.nan)
        time.sleep(1)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return df, elev


# =====================================================================
#  SNAPSHOT PROVIDERS (servizi con sola previsione corrente)
# =====================================================================
def _assemble(temp_times, temps, prcp_ends, prcp_hours, prcp_vals):
    """Porta a passo orario: temperatura interpolata (max 6 h), pioggia di
    periodi lunghi (3 h, 6 h) ripartita uniformemente sulle ore del periodo."""
    tt = pd.Series(np.asarray(temps, float), index=pd.to_datetime(temp_times, utc=True)).sort_index()
    tt = tt[~tt.index.duplicated()]
    rows = {}
    for end, nh, val in zip(pd.to_datetime(prcp_ends, utc=True), prcp_hours, prcp_vals):
        if val is None or nh is None or nh <= 0 or (isinstance(val, float) and np.isnan(val)):
            continue
        for k in range(int(nh)):
            rows[end - pd.Timedelta(hours=k)] = float(val) / nh
    pp = pd.Series(rows, dtype=float).sort_index()
    if tt.empty and pp.empty:
        return pd.DataFrame(columns=["valid_time", "t2m", "prcp"])
    lo = min([x.min() for x in (tt.index, pp.index) if len(x)])
    hi = max([x.max() for x in (tt.index, pp.index) if len(x)])
    idx = pd.date_range(lo.floor("h"), hi.ceil("h"), freq="h")
    t2m = tt.reindex(tt.index.union(idx)).interpolate(method="time", limit=6, limit_area="inside").reindex(idx)
    return pd.DataFrame({"valid_time": idx, "t2m": t2m.values, "prcp": pp.reindex(idx).values})


def fetch_metno(lat, lon, elev, key=None):
    js = http_get("https://api.met.no/weatherapi/locationforecast/2.0/complete",
                  {"lat": round(lat, 4), "lon": round(lon, 4), "altitude": int(elev or 0)}).json()
    tt, tv, pe, ph, pv = [], [], [], [], []
    for s in js["properties"]["timeseries"]:
        t = pd.Timestamp(s["time"])
        d = s["data"]
        tt.append(t)
        tv.append(d["instant"]["details"].get("air_temperature", np.nan))
        for nh, k in ((1, "next_1_hours"), (6, "next_6_hours")):
            if k in d:
                pe.append(t + pd.Timedelta(hours=nh)); ph.append(nh)
                pv.append(d[k]["details"].get("precipitation_amount", np.nan))
                break
    return _assemble(tt, tv, pe, ph, pv)


def fetch_openweathermap(lat, lon, elev, key):
    js = http_get("https://api.openweathermap.org/data/2.5/forecast",
                  {"lat": lat, "lon": lon, "units": "metric", "appid": key}).json()
    tt, tv, pe, ph, pv = [], [], [], [], []
    for s in js["list"]:
        t = pd.Timestamp(s["dt"], unit="s", tz="UTC")
        tt.append(t); tv.append(s["main"]["temp"])
        amount = s.get("rain", {}).get("3h", 0.0) + s.get("snow", {}).get("3h", 0.0)
        pe.append(t); ph.append(3); pv.append(amount)          # "volume nelle ultime 3 ore"
    return _assemble(tt, tv, pe, ph, pv)


def fetch_weatherapi(lat, lon, elev, key):
    js = http_get("https://api.weatherapi.com/v1/forecast.json",
                  {"key": key, "q": f"{lat},{lon}", "days": 3, "aqi": "no", "alerts": "no"}).json()
    tt, tv, pe, ph, pv = [], [], [], [], []
    for day in js["forecast"]["forecastday"]:
        for h in day["hour"]:
            t = pd.Timestamp(h["time_epoch"], unit="s", tz="UTC")
            tt.append(t); tv.append(h["temp_c"])
            pe.append(t + pd.Timedelta(hours=1)); ph.append(1); pv.append(h.get("precip_mm", np.nan))
    return _assemble(tt, tv, pe, ph, pv)


def fetch_tomorrowio(lat, lon, elev, key):
    js = http_get("https://api.tomorrow.io/v4/weather/forecast",
                  {"location": f"{lat},{lon}", "timesteps": "1h", "units": "metric", "apikey": key}).json()
    tt, tv, pe, ph, pv = [], [], [], [], []
    for h in js["timelines"]["hourly"]:
        t = pd.Timestamp(h["time"])
        v = h["values"]
        tt.append(t); tv.append(v.get("temperature", np.nan))
        rate = sum(v.get(k, 0.0) or 0.0 for k in ("rainIntensity", "sleetIntensity",
                                                   "freezingRainIntensity", "snowIntensity"))
        pe.append(t + pd.Timedelta(hours=1)); ph.append(1); pv.append(rate)
    return _assemble(tt, tv, pe, ph, pv)


def fetch_visualcrossing(lat, lon, elev, key):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/next7days"
    js = http_get(url, {"unitGroup": "metric", "include": "hours", "key": key,
                        "contentType": "json", "elements": "datetimeEpoch,temp,precip"}).json()
    tt, tv, pe, ph, pv = [], [], [], [], []
    for day in js["days"]:
        for h in day.get("hours", []):
            t = pd.Timestamp(h["datetimeEpoch"], unit="s", tz="UTC")
            tt.append(t); tv.append(h.get("temp", np.nan))
            pe.append(t + pd.Timedelta(hours=1)); ph.append(1); pv.append(h.get("precip", np.nan))
    return _assemble(tt, tv, pe, ph, pv)


SNAPSHOT_APIS = {
    "metno": fetch_metno,
    "openweathermap": fetch_openweathermap,
    "weatherapi": fetch_weatherapi,
    "tomorrowio": fetch_tomorrowio,
    "visualcrossing": fetch_visualcrossing,
}

# limiti dei piani gratuiti: (stazioni max, pausa tra chiamate in s)
SNAPSHOT_LIMITS = {
    "metno": (None, 0.5),
    "openweathermap": (None, 1.2),
    "weatherapi": (None, 1.0),
    "tomorrowio": (20, 150),          # piano free: 25 chiamate/ora
    "visualcrossing": (None, 1.0),
}

# fino a quale lead il provider ha davvero passo orario (oltre: 3-6 h ripartite)
# -> l'aggregazione 1h non viene verificata oltre questo lead
SNAPSHOT_HOURLY_MAX_LEAD = {"metno": 2, "openweathermap": -1, "weatherapi": 7,
                            "tomorrowio": 7, "visualcrossing": 7}


def calendar_lead(valid_time, issued, tz):
    """Lead in giorni di calendario locali (fuso tz, unico o per riga): 0 = oggi, 1 = domani, ..."""
    v = pd.Series(pd.to_datetime(valid_time, utc=True)).reset_index(drop=True) - pd.Timedelta(hours=1)
    tz = pd.Series(np.broadcast_to(np.asarray(tz, dtype=object), (len(v),)))
    out = np.zeros(len(v), int)
    for z in tz.unique():
        m = (tz == z).values
        vday = pd.to_datetime(v[m].dt.tz_convert(z).dt.strftime("%Y-%m-%d"))
        iday = pd.Timestamp(pd.Timestamp(issued).tz_convert(z).strftime("%Y-%m-%d"))
        out[m] = (vday - iday).dt.days.values
    return out
