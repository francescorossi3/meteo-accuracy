"""Regressione a processo gaussiano (GPR) che combina i provider.

Feature   : posizione sulla sfera (x, y, z: vale ovunque, anche a cavallo
            dell'antimeridiano e ai poli), quota + previsioni dei provider
Target    : osservazione (pioggia in radice quadrata, per gestire l'asimmetria)
Kernel    : C * RBF(ARD) + DotProduct + White
            - RBF con una lunghezza per feature: impara correzioni spaziali
              (es. bias di quota/costa) e non-linearita' tra modelli
            - DotProduct: combinazione lineare dei provider (tipo media pesata)
            - White: errore non spiegato -> entra nell'incertezza

Il modello viene esportato in JSON (punti di training, alpha, Cholesky) cosi'
che il sito possa calcolare stima e incertezza in qualsiasi punto, nel
browser, con lo stesso codice di predict_export() qui sotto (gp.js).
"""
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

GEO = ["gx", "gy", "gz", "elev"]


def transform_of(var):
    return "sqrt" if var == "prcp" else "none"


def fwd(x, transform):
    x = np.asarray(x, float)
    return np.sqrt(np.clip(x, 0, None)) if transform == "sqrt" else x


def design(df, providers, transform):
    """Matrice delle feature. I provider mancanti in una riga vengono
    sostituiti dalla media dei provider presenti in quella riga."""
    F = np.column_stack([fwd(df[p].values, transform) if p in df else np.full(len(df), np.nan)
                         for p in providers]) if providers else np.zeros((len(df), 0))
    ok = ~np.all(np.isnan(F), axis=1) if providers else np.ones(len(df), bool)
    if providers:
        with np.errstate(all="ignore"):
            rm = np.nanmean(np.where(ok[:, None], F, 0.0), axis=1)
        F = np.where(np.isnan(F), rm[:, None], F)
    la, lo = np.radians(df["lat"].values.astype(float)), np.radians(df["lon"].values.astype(float))
    geo = [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la), df["elev"].values.astype(float)]
    X = np.column_stack(geo + [F])
    return X, ok


def fit(X, y, restarts=2, seed=42, transform="none"):
    mu, sc = X.mean(0), X.std(0)
    sc[sc < 1e-9] = 1.0
    Xs = (X - mu) / sc
    d = X.shape[1]
    # limiti scelti per evitare che l'RBF "memorizzi" il rumore (soprattutto con
    # la pioggia, piena di zeri): scale >= 0.3 deviazioni standard, rumore minimo
    noise_min = 5e-2 if transform == "sqrt" else 1e-4
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(d) * 2.0, (0.3, 1e3))
              + DotProduct(1.0, (1e-3, 1e2)) + WhiteKernel(0.1, (noise_min, 1e1)))
    gpr = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=restarts,
                                   random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # avvisi di convergenza ai limiti: attesi
        gpr.fit(Xs, y)
    return gpr, mu, sc


def export(gpr, mu, sc, meta, sig=6):
    """Dizionario JSON-serializzabile con tutto il necessario per predire."""
    p = gpr.kernel_.get_params()
    L = gpr.L_
    r = lambda a: [float(f"{v:.{sig}g}") for v in np.ravel(a)]          # noqa: E731
    return dict(meta,
                x_mean=r(mu), x_scale=r(sc),
                X=[r(row) for row in gpr.X_train_],
                alpha=r(gpr.alpha_),
                L=r(L[np.tril_indices(L.shape[0])]),          # triangolo inferiore per righe
                y_mean=float(np.ravel(gpr._y_train_mean)[0]), y_std=float(np.ravel(gpr._y_train_std)[0]),
                kernel=dict(c=float(p["k1__k1__k1__constant_value"]),
                            ls=r(np.atleast_1d(p["k1__k1__k2__length_scale"])),
                            sigma0=float(p["k1__k2__sigma_0"]),
                            noise=float(p["k2__noise_level"])))


def _unpack(m):
    n = len(m["alpha"])
    L = np.zeros((n, n))
    L[np.tril_indices(n)] = m["L"]
    return np.array(m["X"]), np.array(m["alpha"]), L


def predict_export(m, X, cache=None, chunk=2000):
    """Predizione da modello esportato (stesse formule di web/gp.js).
    Ritorna media e deviazione standard nello spazio trasformato."""
    cache = cache if cache is not None else _unpack(m)
    if len(X) > chunk:
        parts = [predict_export(m, X[i:i + chunk], cache, chunk) for i in range(0, len(X), chunk)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    Xt, alpha, L = cache
    k = m["kernel"]
    Xs = (np.asarray(X, float) - np.array(m["x_mean"])) / np.array(m["x_scale"])
    ls = np.array(k["ls"])
    D = ((Xs[:, None, :] - Xt[None, :, :]) / ls) ** 2
    Ks = k["c"] * np.exp(-0.5 * D.sum(-1)) + k["sigma0"] ** 2 + Xs @ Xt.T
    mean = Ks @ alpha
    V = solve_triangular(L, Ks.T, lower=True)
    kss = k["c"] + k["sigma0"] ** 2 + (Xs ** 2).sum(1) + k["noise"]
    var = np.clip(kss - (V ** 2).sum(0), 0, None)
    return mean * m["y_std"] + m["y_mean"], np.sqrt(var) * m["y_std"]


def to_original(mu, sd, transform):
    """Stima, intervalli 68%/95% e incertezza (semi-ampiezza 68%) in unita' originali."""
    q = lambda z: mu + z * sd                                            # noqa: E731
    if transform == "sqrt":
        b = lambda x: np.clip(x, 0, None) ** 2                           # noqa: E731
        est = b(mu)
        lo68, hi68, lo95, hi95 = b(q(-1)), b(q(1)), b(q(-1.96)), b(q(1.96))
    else:
        est, lo68, hi68, lo95, hi95 = mu, q(-1), q(1), q(-1.96), q(1.96)
    return dict(est=est, lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95, unc=(hi68 - lo68) / 2)


def prob_exceed(mu, sd, thr, transform):
    t = np.sqrt(thr) if transform == "sqrt" else thr
    return 1 - norm.cdf((t - mu) / np.maximum(sd, 1e-9))


def validation(obs, mu, sd, transform):
    """Verifica su dati non usati nel training: errori e calibrazione.
    Per la pioggia si tollera 0.1 mm (risoluzione dei pluviometri)."""
    o = to_original(mu, sd, transform)
    e = o["est"] - obs
    tol = 0.1 if transform == "sqrt" else 0.0
    inside = lambda lo, hi: np.mean((obs >= lo - tol) & (obs <= hi + tol))     # noqa: E731
    res = dict(n=len(obs), bias=e.mean(), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
               cov68=inside(o["lo68"], o["hi68"]), cov95=inside(o["lo95"], o["hi95"]),
               mean_unc=np.mean(o["unc"]))
    z = (fwd(obs, transform) - mu) / np.maximum(sd, 1e-9)              # CRPS gaussiano (spazio del modello)
    res["crps"] = np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))
    return res
