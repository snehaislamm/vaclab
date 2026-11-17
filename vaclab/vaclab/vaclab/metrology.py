"""Thin-film metrology: spectral reflectometry fit and wafer uniformity map.

Single transparent film (Cauchy n = A + B/lambda^2) on Si at normal incidence.
Thickness, A and B are fitted to the measured reflectance spectrum.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

WL = np.linspace(400.0, 900.0, 251)  # nm


def n_si(wl):
    """Simple real-index dispersion for crystalline Si over 400-900 nm."""
    return 3.75 + 9.0e4 / wl**2


def reflectance(thk_nm, A, B, wl=WL):
    n0, n1, n2 = 1.0, A + B / wl**2, n_si(wl)
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    phase = np.exp(-2j * (2 * np.pi * n1 * thk_nm / wl))
    r = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
    return np.abs(r) ** 2


def measure(thk_nm, A=1.98, B=1.6e4, noise=0.002, seed=0, wl=WL):
    rng = np.random.default_rng(seed)
    return reflectance(thk_nm, A, B, wl) + noise * rng.standard_normal(wl.size)


@dataclass
class FitResult:
    thickness_nm: float
    A: float
    B: float
    n_633: float
    rms: float


def fit(R, wl=WL, thk_range=(20.0, 400.0), dispersion=None) -> FitResult:
    """Coarse thickness grid (avoids fringe-order local minima), then least squares.

    dispersion=(A, B) fixes the film index (recipe mode: index calibrated once on a
    reference site), leaving thickness as the only free parameter. Thickness and
    index are strongly correlated for films near 100 nm, so fixing the index
    tightens the thickness precision.
    """
    A0, B0 = dispersion or (1.98, 1.6e4)
    grid = np.arange(*thk_range, 1.0)
    errs = [np.sum((reflectance(t, A0, B0, wl) - R) ** 2) for t in grid]
    t0 = grid[int(np.argmin(errs))]
    if dispersion is not None:
        res = least_squares(lambda p: reflectance(p[0], A0, B0, wl) - R, x0=[t0])
        return FitResult(res.x[0], A0, B0, A0 + B0 / 633.0**2,
                         float(np.sqrt(np.mean(res.fun**2))))
    res = least_squares(lambda p: reflectance(p[0], p[1], p[2] * 1e4, wl) - R,
                        x0=[t0, 1.98, 1.6], bounds=([1, 1.4, 0], [1000, 2.6, 5]))
    t, A, b = res.x
    return FitResult(t, A, b * 1e4, A + b * 1e4 / 633.0**2,
                     float(np.sqrt(np.mean(res.fun**2))))


def wafer_points(radius_mm=75.0, n_rings=4, edge_exclusion_mm=5.0):
    """49-point polar map (1 + 6 + 12 + 12 + 18), like a standard metrology recipe."""
    r_max = radius_mm - edge_exclusion_mm
    pts = [(0.0, 0.0)]
    for ring, k in zip(range(1, n_rings + 1), [6, 12, 12, 18]):
        r = r_max * ring / n_rings
        for j in range(k):
            a = 2 * np.pi * j / k
            pts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(pts)


def thickness_profile(xy, t0=102.6, radial_drop=0.04, tilt=0.01, radius_mm=75.0):
    r = np.hypot(xy[:, 0], xy[:, 1]) / radius_mm
    return t0 * (1 - radial_drop * r**2 + tilt * xy[:, 0] / radius_mm)


@dataclass
class UniformityResult:
    mean_nm: float
    sigma_pct: float
    range_pct: float   # (max - min) / (2 * mean), the usual +/- % spec
    max_fit_err_nm: float


def uniformity_map(seed=3, **kw):
    xy = wafer_points()
    truth = thickness_profile(xy, **kw)
    # calibrate index at the centre site (averaged spectra), then thickness-only fits
    ref = np.mean([measure(truth[0], seed=1000 + k) for k in range(10)], axis=0)
    cal = fit(ref)
    disp = (cal.A, cal.B)
    fitted = np.array([fit(measure(t, seed=seed + i), dispersion=disp).thickness_nm
                       for i, t in enumerate(truth)])
    res = UniformityResult(
        fitted.mean(),
        100 * fitted.std(ddof=1) / fitted.mean(),
        100 * (fitted.max() - fitted.min()) / (2 * fitted.mean()),
        float(np.max(np.abs(fitted - truth))),
    )
    return xy, truth, fitted, res
