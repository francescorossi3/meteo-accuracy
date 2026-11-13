"""Griglie regolari di punti su terra per le mappe di ciascuna regione."""
import numpy as np
import pandas as pd
from shapely import contains_xy

from .config import DATA
from .regions import is_points, polygon, region, sites


def region_grid(rid):
    """Legge (o crea) data/grid_<regione>.csv: lat, lon, elev (quota dal primo download)."""
    r = region(rid)
    if is_points(rid):                               # regione a punti: la "griglia" sono i siti
        path = DATA / f"grid_{rid}.csv"
        g = sites(rid).rename(columns={"id": "site"})
        if path.exists():
            old = pd.read_csv(path)
            if "elev" in old and len(old) == len(g):
                g["elev"] = old["elev"].values
        if "elev" not in g:
            g["elev"] = np.nan
        g["res"] = 0.0
        return g
    res = float(r["grid_res"])
    path = DATA / f"grid_{rid}.csv"
    if path.exists():
        g = pd.read_csv(path)
        if len(g) and "res" in g and abs(g["res"].iloc[0] - res) < 1e-9:
            return g
    poly = polygon(rid, buffer=0.05 * res)
    x0, y0, x1, y1 = poly.bounds
    lats = np.arange(np.floor(y0 / res) * res + res / 2, y1, res)
    lons = np.arange(np.floor(x0 / res) * res + res / 2, x1, res)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    m = contains_xy(poly, LO.ravel(), LA.ravel())
    g = pd.DataFrame({"lat": np.round(LA.ravel()[m], 3), "lon": np.round(LO.ravel()[m], 3)})
    g["elev"], g["res"] = np.nan, res
    g.to_csv(path, index=False)
    return g
