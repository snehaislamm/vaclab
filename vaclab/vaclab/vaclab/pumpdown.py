"""Chamber pump-down model and vacuum-level certification.

Stages
  1. Roughing: atmosphere -> crossover pressure, volume-limited exponential.
  2. High vacuum: turbo through a conductance; pressure is gas-load limited
     (V / S_eff is well under a second), so P(t) = Q_total(t) / S_eff.
Units: mbar, litres, seconds, cm^2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ATM = 1013.0  # mbar


@dataclass
class Chamber:
    volume_l: float = 100.0            # chamber volume
    wall_area_cm2: float = 1.5e4       # internal surface area
    rough_speed_lps: float = 10.0      # roughing pump speed
    crossover_mbar: float = 1e-2       # rough -> turbo handover
    turbo_speed_lps: float = 700.0     # turbo pumping speed (N2)
    conductance_lps: float = 1500.0    # gate valve + port conductance
    q1_mbar_l_s_cm2: float = 1e-8      # outgassing rate after 1 h (unbaked SS)
    baked: bool = False                # bakeout cuts outgassing ~100x
    leak_mbar_l_s: float = 0.0         # real (air) leak

    @property
    def s_eff(self) -> float:
        """Effective speed at the chamber: series combination of pump and conductance."""
        return 1.0 / (1.0 / self.turbo_speed_lps + 1.0 / self.conductance_lps)

    def outgassing(self, t_hv_s: np.ndarray) -> np.ndarray:
        """Unbaked metal outgassing falls ~1/t (t in hours since pump-down)."""
        q1 = self.q1_mbar_l_s_cm2 * (0.01 if self.baked else 1.0)
        t_h = np.maximum(np.asarray(t_hv_s, float), 60.0) / 3600.0
        return q1 * self.wall_area_cm2 / t_h


def roughing_time(ch: Chamber) -> float:
    """t = (V/S) ln(P0/P1)."""
    return ch.volume_l / ch.rough_speed_lps * np.log(ATM / ch.crossover_mbar)


def pressure_curve(ch: Chamber, t_end_s: float = 12 * 3600, n: int = 2000):
    """Return (t, P) over the full pump-down."""
    t_r = roughing_time(ch)
    t = np.linspace(0.0, t_end_s, n)
    p = np.empty_like(t)
    rough = t <= t_r
    p[rough] = ATM * np.exp(-ch.rough_speed_lps / ch.volume_l * t[rough])
    hv = ~rough
    q = ch.outgassing(t[hv]) + ch.leak_mbar_l_s
    p[hv] = np.minimum(q / ch.s_eff, ch.crossover_mbar)
    return t, p


def time_to_pressure(ch: Chamber, target_mbar: float, t_max_s: float = 72 * 3600) -> float:
    """Seconds until P <= target; inf if the gas load never allows it."""
    t, p = pressure_curve(ch, t_max_s, 20000)
    idx = np.flatnonzero(p <= target_mbar)
    return float(t[idx[0]]) if idx.size else float("inf")


@dataclass
class CertResult:
    base_mbar: float
    spec_mbar: float
    leak_share: float  # fraction of the gas load that is the leak
    passed: bool


def certify(ch: Chamber, spec_mbar: float = 5e-8, after_h: float = 12.0) -> CertResult:
    """Chamber vacuum-level certification: base pressure after `after_h` hours."""
    t = after_h * 3600
    q_out = float(ch.outgassing(np.array([t]))[0])
    base = (q_out + ch.leak_mbar_l_s) / ch.s_eff
    share = ch.leak_mbar_l_s / (q_out + ch.leak_mbar_l_s)
    return CertResult(base, spec_mbar, share, base <= spec_mbar)


def max_allowed_leak(ch: Chamber, spec_mbar: float = 5e-8, after_h: float = 12.0) -> float:
    """Leak budget: largest leak that still certifies."""
    q_out = float(ch.outgassing(np.array([after_h * 3600]))[0])
    return max(spec_mbar * ch.s_eff - q_out, 0.0)
