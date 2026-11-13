// Predizione GPR nel browser: stesse formule di meteoita/gp.py::predict_export
// k(x,x') = c*exp(-0.5*sum(((x-x')/ls)^2)) + sigma0^2 + x.x'   (+ noise sulla diagonale)
"use strict";

const GP = (() => {
  const cache = new Map();

  function unpack(m) {
    if (m._L) return m;
    const n = m.alpha.length;
    const L = new Float64Array(n * n);
    let k = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * n + j] = m.L[k++];
    m._L = L; m._n = n;
    return m;
  }

  async function load(url) {
    if (!cache.has(url)) {
      cache.set(url, fetch(url).then(r => (r.ok ? r.json() : null)).then(m => (m ? unpack(m) : null)).catch(() => null));
    }
    return cache.get(url);
  }

  const fwd = (x, tr) => (tr === "sqrt" ? Math.sqrt(Math.max(x, 0)) : x);

  // features: oggetto {lat, lon, elev, <provider>: valore}; provider mancanti -> media dei presenti
  function design(m, f) {
    const vals = m.providers.map(p => (f[p] == null || !isFinite(f[p]) ? NaN : fwd(f[p], m.transform)));
    const ok = vals.filter(v => !isNaN(v));
    if (!ok.length) return null;
    const mean = ok.reduce((a, b) => a + b, 0) / ok.length;
    const la = f.lat * Math.PI / 180, lo = f.lon * Math.PI / 180;
    return [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la), f.elev, ...vals.map(v => (isNaN(v) ? mean : v))];
  }

  function predictRaw(m, x) {
    const d = x.length, n = m._n, K = m.kernel;
    const xs = x.map((v, j) => (v - m.x_mean[j]) / m.x_scale[j]);
    const ks = new Float64Array(n);
    let mean = 0;
    for (let i = 0; i < n; i++) {
      const xi = m.X[i];
      let s = 0, dot = 0;
      for (let j = 0; j < d; j++) {
        const t = (xs[j] - xi[j]) / K.ls[j];
        s += t * t; dot += xs[j] * xi[j];
      }
      ks[i] = K.c * Math.exp(-0.5 * s) + K.sigma0 * K.sigma0 + dot;
      mean += ks[i] * m.alpha[i];
    }
    // v = L^-1 ks (sostituzione in avanti)
    const v = new Float64Array(n);
    let vv = 0;
    for (let i = 0; i < n; i++) {
      let s = ks[i];
      const row = i * n;
      for (let j = 0; j < i; j++) s -= m._L[row + j] * v[j];
      v[i] = s / m._L[row + i];
      vv += v[i] * v[i];
    }
    let xx = 0;
    for (let j = 0; j < d; j++) xx += xs[j] * xs[j];
    const kss = K.c + K.sigma0 * K.sigma0 + xx + K.noise;
    const sd = Math.sqrt(Math.max(kss - vv, 0)) * m.y_std;
    return { mu: mean * m.y_std + m.y_mean, sd };
  }

  // risultato in unita' originali
  function predict(m, feats) {
    const x = design(m, feats);
    if (!x) return null;
    const { mu, sd } = predictRaw(m, x);
    const q = z => mu + z * sd;
    const b = m.transform === "sqrt" ? (y => Math.max(y, 0) ** 2) : (y => y);
    const r = { mu, sd, est: b(mu), lo68: b(q(-1)), hi68: b(q(1)), lo95: b(q(-1.96)), hi95: b(q(1.96)) };
    r.unc = (r.hi68 - r.lo68) / 2;
    return r;
  }

  // P(valore > soglia)
  function probExceed(r, thr, tr) {
    const t = tr === "sqrt" ? Math.sqrt(thr) : thr;
    return 1 - normCdf((t - r.mu) / Math.max(r.sd, 1e-9));
  }

  function normCdf(z) { // Abramowitz-Stegun 7.1.26
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }

  return { load, predict, probExceed, predictRaw, design, unpack };
})();

if (typeof module !== "undefined") module.exports = GP;
