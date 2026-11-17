"""Leak testing: rate-of-rise (pressure leak-back) and helium spray localisation.

Rate of rise: isolate the chamber, log P(t). Wall desorption saturates;
a real leak keeps rising linearly. Fit
    P(t) = P0 + A (1 - exp(-t/tau)) + b t
and report Q_leak = V * b  [mbar L/s].

Helium spray: spray each flange in turn with a mass-spec leak detector on
the chamber; a leaking flange gives a first-order rise during the dwell.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit


def ror_model(t, p0, a, tau, b):
    return p0 + a * (1.0 - np.exp(-t / tau)) + b * t


def simulate_ror(volume_l=100.0, leak=2e-6, desorb=5e-6, tau=120.0, p0=3e-7,
                 duration_s=900.0, dt=5.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s + dt, dt)
    p = ror_model(t, p0, desorb, tau, leak / volume_l)
    return t, p * (1.0 + noise * rng.standard_normal(t.size))


@dataclass
class RorResult:
    leak_mbar_l_s: float
    naive_leak_mbar_l_s: float   # straight line through the whole log
    desorb_mbar: float
    tau_s: float
    spec: float
    passed: bool


def analyse_ror(t, p, volume_l=100.0, spec=5e-6) -> RorResult:
    b0 = (p[-1] - p[len(p) // 2]) / (t[-1] - t[len(t) // 2])
    guess = [p[0], max(p[-1] - p[0] - b0 * t[-1], 1e-9), 100.0, max(b0, 1e-12)]
    popt, _ = curve_fit(ror_model, t, p, p0=guess,
                        bounds=([0, 0, 1, 0], [np.inf, np.inf, 5e3, np.inf]),
                        x_scale=np.abs(guess), maxfev=20000)
    naive = np.polyfit(t, p, 1)[0] * volume_l
    leak = popt[3] * volume_l
    return RorResult(leak, naive, popt[1], popt[2], spec, leak <= spec)


# ---------------- helium spray ----------------
@dataclass
class SprayLog:
    t: np.ndarray
    signal: np.ndarray       # leak detector reading, mbar L/s He
    flanges: list[str]
    windows: list[tuple[float, float]]


def simulate_spray(n_flanges=12, leaking=None, dwell_s=20.0, gap_s=30.0,
                   tau_s=4.0, baseline=5e-11, seed=1) -> SprayLog:
    leaking = leaking or {"F07": 3e-9}
    rng = np.random.default_rng(seed)
    names = [f"F{i:02d}" for i in range(1, n_flanges + 1)]
    period = dwell_s + gap_s
    t = np.arange(0.0, n_flanges * period + gap_s, 0.5)
    sig = baseline * (1 + 0.15 * rng.standard_normal(t.size)) + 1e-12 * t / t[-1]
    windows = []
    for i, name in enumerate(names):
        t0 = gap_s + i * period
        t1 = t0 + dwell_s
        windows.append((t0, t1))
        q = leaking.get(name, 0.0)
        if q:
            on = (t >= t0) & (t < t1)
            off = t >= t1
            sig[on] += q * (1 - np.exp(-(t[on] - t0) / tau_s))
            peak = q * (1 - np.exp(-dwell_s / tau_s))
            sig[off] += peak * np.exp(-(t[off] - t1) / tau_s)
    return SprayLog(t, sig, names, windows)


def locate_leaks(log: SprayLog, reject=1e-9, tau_s=4.0, lag_s=6.0):
    """Return {flange: estimated leak} for flanges whose response exceeds `reject`.

    The response window is shifted by `lag_s` to allow for detector delay, and
    the peak is corrected for the incomplete first-order rise in one dwell.
    """
    base = np.median(log.signal)
    found = {}
    for name, (t0, t1) in zip(log.flanges, log.windows):
        w = (log.t >= t0) & (log.t < t1 + lag_s)
        peak = log.signal[w].max() - base
        dwell = t1 - t0
        est = peak / (1 - np.exp(-dwell / tau_s))
        if est >= reject:
            found[name] = est
    return found
