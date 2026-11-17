"""2^3 full-factorial Design of Experiments for magnetron sputter deposition rate.

Factors (coded -1/+1): A = DC power, B = Ar pressure, C = target-substrate distance.
Replicated factorial runs give pure error for t-tests; centre points test curvature.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import stats

FACTORS = {
    "power_W": (200.0, 400.0),
    "pressure_mTorr": (2.0, 6.0),
    "distance_mm": (80.0, 120.0),
}
TERMS = ["A", "B", "C", "AB", "AC", "BC", "ABC"]


def decode(coded):
    out = {}
    for (name, (lo, hi)), x in zip(FACTORS.items(), coded):
        out[name] = lo + (x + 1) * (hi - lo) / 2
    return out


def true_rate(power_W, pressure_mTorr, distance_mm):
    """Hidden 'tool' response, nm/min: rate ~ power, gas scattering loss, ~1/d^2."""
    scatter = np.exp(-0.035 * pressure_mTorr * distance_mm / 100.0)
    return 0.14 * power_W * scatter * (100.0 / distance_mm) ** 2


def design(replicates=2, centre=3, seed=7):
    """Randomised run sheet: list of coded tuples."""
    runs = [c for c in product([-1, 1], repeat=3)] * replicates + [(0, 0, 0)] * centre
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(runs))
    return [runs[i] for i in order]


def run(sheet, noise_sd=1.5, seed=11):
    rng = np.random.default_rng(seed)
    y = np.array([true_rate(**decode(c)) for c in sheet])
    return y + noise_sd * rng.standard_normal(y.size)


def model_matrix(X):
    A, B, C = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([A, B, C, A * B, A * C, B * C, A * B * C])


@dataclass
class DoeResult:
    effects: dict           # term -> effect (high minus low), nm/min
    t_values: dict
    significant: list
    t_crit: float
    curvature: float        # centre mean - factorial mean
    curvature_t: float
    r2: float
    best_coded: tuple
    best_rate: float


def analyse(sheet, y, alpha=0.05) -> DoeResult:
    X = np.array(sheet, float)
    fact = np.any(X != 0, axis=1)
    Xf, yf = X[fact], y[fact]
    M = model_matrix(Xf)
    n = len(yf)
    effects = {t: 2 * (M[:, i] @ yf) / n for i, t in enumerate(TERMS)}

    # pure error from replicates at each factorial corner (+ centre points)
    groups = {}
    for x, v in zip(map(tuple, X), y):
        groups.setdefault(x, []).append(v)
    ss = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups.values())
    dof = sum(len(g) - 1 for g in groups.values())
    s = np.sqrt(ss / dof)
    se_eff = 2 * s / np.sqrt(n)
    t_vals = {k: v / se_eff for k, v in effects.items()}
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    sig = [k for k in TERMS if abs(t_vals[k]) > t_crit]

    yc = y[~fact]
    curv = yc.mean() - yf.mean()
    curv_t = curv / (s * np.sqrt(1 / n + 1 / len(yc)))

    beta = np.linalg.lstsq(np.column_stack([np.ones(n), M]), yf, rcond=None)[0]
    pred = np.column_stack([np.ones(n), M]) @ beta
    r2 = 1 - ((yf - pred) ** 2).sum() / ((yf - yf.mean()) ** 2).sum()

    corners = list(product([-1, 1], repeat=3))
    means = {c: np.mean(groups[c]) for c in corners}
    best = max(means, key=means.get)
    return DoeResult(effects, t_vals, sig, t_crit, curv, curv_t, r2, best, means[best])
