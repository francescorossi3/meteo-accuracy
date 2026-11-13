"""Mondo sintetico per provare l'intera pipeline senza internet.

NON sono dati reali: servono solo a verificare che codice, file e sito
funzionino (python scripts/demo_generate.py).
"""
import zlib

import numpy as np
import pandas as pd

SEED_STATIONS = [  # stazioni sinottiche (WMO) usate anche come elenco di riserva
    ("16080", "Milano Linate", 45.433, 9.283, 107), ("16066", "Milano Malpensa", 45.617, 8.733, 211),
    ("16059", "Torino Caselle", 45.217, 7.650, 287), ("16120", "Genova Sestri", 44.417, 8.850, 3),
    ("16140", "Bologna", 44.533, 11.300, 36), ("16105", "Venezia Tessera", 45.500, 12.333, 6),
    ("16045", "Udine Rivolto", 45.983, 13.050, 51), ("16020", "Bolzano", 46.467, 11.333, 241),
    ("16090", "Verona Villafranca", 45.383, 10.867, 68), ("16076", "Bergamo Orio", 45.667, 9.700, 237),
    ("16170", "Firenze Peretola", 43.800, 11.200, 38), ("16158", "Pisa", 43.683, 10.400, 2),
    ("16191", "Ancona Falconara", 43.617, 13.367, 10), ("16181", "Perugia", 43.083, 12.500, 205),
    ("16230", "Pescara", 42.433, 14.200, 10), ("16242", "Roma Fiumicino", 41.800, 12.233, 3),
    ("16239", "Roma Ciampino", 41.783, 12.583, 105), ("16289", "Napoli Capodichino", 40.850, 14.300, 72),
    ("16252", "Campobasso", 41.567, 14.650, 793), ("16261", "Foggia Amendola", 41.533, 15.717, 60),
    ("16270", "Bari Palese", 41.133, 16.783, 34), ("16320", "Brindisi", 40.650, 17.950, 10),
    ("16332", "Lecce Galatina", 40.233, 18.150, 48), ("16300", "Potenza", 40.633, 15.800, 843),
    ("16350", "Crotone", 39.000, 17.067, 158), ("16422", "Reggio Calabria", 38.067, 15.650, 21),
    ("16405", "Palermo Punta Raisi", 38.183, 13.100, 21), ("16429", "Trapani Birgi", 37.917, 12.500, 7),
    ("16460", "Catania", 37.467, 15.050, 17), ("16560", "Cagliari Elmas", 39.250, 9.050, 4),
    ("16522", "Alghero", 40.633, 8.283, 23), ("16531", "Olbia", 40.900, 9.517, 13),
]

WORLD_SEED = [  # stazioni sinottiche mondiali (WMO) di riserva: (id, nome, lat, lon, quota, fuso)
    ("03772", "London Heathrow", 51.48, -0.45, 25, "Europe/London"), ("07150", "Paris Le Bourget", 48.97, 2.45, 52, "Europe/Paris"),
    ("10384", "Berlin", 52.47, 13.40, 49, "Europe/Berlin"), ("08221", "Madrid Barajas", 40.47, -3.56, 609, "Europe/Madrid"),
    ("17060", "Istanbul", 40.98, 28.82, 37, "Europe/Istanbul"), ("27612", "Moskva", 55.83, 37.62, 156, "Europe/Moscow"),
    ("01384", "Oslo Gardermoen", 60.20, 11.08, 202, "Europe/Oslo"), ("62366", "Cairo", 30.13, 31.40, 74, "Africa/Cairo"),
    ("63740", "Nairobi", -1.32, 36.92, 1624, "Africa/Nairobi"), ("68816", "Cape Town", -33.97, 18.60, 46, "Africa/Johannesburg"),
    ("61641", "Dakar", 14.73, -17.50, 24, "Africa/Dakar"), ("41194", "Dubai", 25.25, 55.33, 10, "Asia/Dubai"),
    ("43003", "Mumbai", 19.12, 72.85, 14, "Asia/Kolkata"), ("42182", "New Delhi", 28.58, 77.20, 216, "Asia/Kolkata"),
    ("48698", "Singapore", 1.37, 103.98, 16, "Asia/Singapore"), ("54511", "Beijing", 39.93, 116.28, 55, "Asia/Shanghai"),
    ("47662", "Tokyo", 35.68, 139.77, 36, "Asia/Tokyo"), ("94767", "Sydney", -33.95, 151.17, 6, "Australia/Sydney"),
    ("94610", "Perth", -31.93, 115.97, 15, "Australia/Perth"), ("93119", "Auckland", -37.00, 174.80, 7, "Pacific/Auckland"),
    ("72503", "New York LaGuardia", 40.78, -73.88, 3, "America/New_York"), ("72295", "Los Angeles", 33.94, -118.39, 30, "America/Los_Angeles"),
    ("72530", "Chicago O'Hare", 41.98, -87.90, 205, "America/Chicago"), ("71624", "Toronto", 43.68, -79.63, 173, "America/Toronto"),
    ("76679", "Ciudad de Mexico", 19.43, -99.07, 2230, "America/Mexico_City"), ("83746", "Rio de Janeiro", -22.82, -43.25, 9, "America/Sao_Paulo"),
    ("87576", "Buenos Aires Ezeiza", -34.82, -58.53, 20, "America/Argentina/Buenos_Aires"), ("85574", "Santiago", -33.38, -70.78, 476, "America/Santiago"),
    ("84628", "Lima", -12.00, -77.12, 13, "America/Lima"), ("70273", "Anchorage", 61.17, -150.02, 40, "America/Anchorage"),
    # vicino ai siti RadCalNet
    ("72486", "Ely (NV)", 39.30, -114.85, 1909, "America/Los_Angeles"), ("72485", "Tonopah (NV)", 38.05, -117.09, 1653, "America/Los_Angeles"),
    ("68104", "Walvis Bay", -22.98, 14.65, 92, "Africa/Windhoek"), ("07650", "Marseille Marignane", 43.44, 5.22, 9, "Europe/Paris"),
    ("07645", "Nimes Courbessac", 43.86, 4.41, 59, "Europe/Paris"), ("53446", "Baotou", 40.67, 109.85, 1067, "Asia/Shanghai"),
]

MODELS = {  # bias, rumore base, quanto "liscia" la quota (modelli a bassa risoluzione)
    "icon2i": (0.1, 0.9, 0.1), "ecmwf_ifs": (0.2, 1.0, 0.5), "ecmwf_aifs": (-0.1, 1.05, 0.6),
    "icon_eu": (0.3, 1.1, 0.4), "icon_global": (0.4, 1.3, 0.7), "gfs": (-0.5, 1.4, 0.7),
    "arpege": (0.3, 1.2, 0.5), "ukmo": (0.0, 1.1, 0.5), "gem": (-0.6, 1.6, 0.7),
    "knmi_eu": (0.2, 1.15, 0.3), "metno": (0.1, 1.0, 0.2),
}


def elevation(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    alps = 1800 * np.exp(-((lat - 46.4) / 0.45) ** 2) * (lon < 13.8)
    apen = 900 * np.exp(-(((lon - 13.6) - 0.9 * (42.5 - lat)) / 0.55) ** 2) * (lat < 44.5) * (lat > 38.0)
    return np.clip(alps + apen, 0, None)


def _smooth(seed, t_h, lat, lon, k=8, tscale=36.0, sscale=2.0):
    r = np.random.default_rng(seed)
    out = 0
    for _ in range(k):
        ft, fa, fo = r.normal(0, 1 / tscale), r.normal(0, 1 / sscale), r.normal(0, 1 / sscale)
        out = out + np.sin(2 * np.pi * (ft * t_h + fa * lat + fo * lon) + r.uniform(0, 2 * np.pi))
    return out * np.sqrt(2 / k)


def _grid(lat, lon, elev, times):
    t_h = (pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns").asi8 // 3_600_000_000_000).astype(float)
    T, LA = np.meshgrid(t_h, np.asarray(lat, float), indexing="xy")
    _, LO = np.meshgrid(t_h, np.asarray(lon, float), indexing="xy")
    _, EL = np.meshgrid(t_h, np.asarray(elev, float), indexing="xy")
    hour_local = (T + LO / 15.0) % 24
    doy = (T / 24) % 365.25
    return T, LA, LO, EL, hour_local, doy


def truth(lat, lon, elev, times):
    """Matrici (punti x tempi) di temperatura e pioggia oraria 'vere'."""
    T, LA, LO, EL, hl, doy = _grid(lat, lon, elev, times)
    t2m = (16 - 0.75 * (np.abs(LA) - 38) - 6.5 * EL / 1000 + 8 * np.sign(LA) * np.sin(2 * np.pi * (doy - 110) / 365.25)
           + 5 * np.sin(2 * np.pi * (hl - 9) / 24) + 3 * _smooth(1, T, LA, LO))
    lat_r = _smooth(2, T, LA, LO, tscale=20, sscale=1.5) + 0.0004 * EL
    prcp = np.clip(lat_r - 1.1, 0, None) ** 1.5 * 3
    return t2m, prcp


def forecast(model, lead, lat, lon, elev, times):
    bias, sig, smooth = MODELS.get(model, (0, 1.2, 0.5))
    T, LA, LO, EL, hl, doy = _grid(lat, lon, elev, times)
    s = 1 + 0.35 * lead
    h = zlib.crc32(f"{model}{int(lead)}".encode()) % 10_000
    t_true, _ = truth(lat, lon, elev, times)
    t2m = (t_true + bias + smooth * 6.5 * EL / 1000 * 0.4
           + sig * s * 0.8 * _smooth(100 + h, T, LA, LO, tscale=30)
           + sig * 0.4 * np.random.default_rng(h).normal(size=T.shape))
    lat_r = (_smooth(2, T, LA, LO, tscale=20, sscale=1.5) + 0.0004 * EL * (1 - smooth)
             + 0.25 * sig * s * _smooth(500 + h, T, LA, LO, tscale=15, sscale=1.0))
    prcp = np.clip(lat_r - 1.1, 0, None) ** 1.5 * 3 * (1 + 0.1 * bias)
    return t2m, prcp
