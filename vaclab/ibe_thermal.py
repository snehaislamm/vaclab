"""Wafer temperature under a broad ion beam: 1-D series thermal resistance.

Beam power flux q'' = J * V_beam. Heat leaves through the wafer, the
wafer-to-holder contact (the weak link), the copper holder and the cooling water.
Same composite-wall method as a multilayer heat-conduction problem.
"""
from __future__ import annotations
import numpy as np

CONTACT_H = {           # W/m^2K, wafer-to-holder
    "dry clamp (vacuum)": 30.0,
    "thermal paste / pad": 250.0,
    "He backside gas": 600.0,
}


def beam_flux(j_mA_cm2=0.8, v_beam=500.0):
    """Power flux in W/m^2 (mA/cm^2 * V = 10 W/m^2 per unit)."""
    return j_mA_cm2 * v_beam * 10.0


def wafer_temperature(q_flux, h_contact, t_water=20.0, si_mm=0.675, cu_mm=10.0,
                      h_water=5000.0, k_si=150.0, k_cu=390.0):
    R = (si_mm * 1e-3) / k_si + 1 / h_contact + (cu_mm * 1e-3) / k_cu + 1 / h_water
    return t_water + q_flux * R


def time_constant(h_contact, si_mm=0.675, rho=2330.0, cp=700.0):
    """Lumped wafer time constant, s (Biot << 1)."""
    return rho * cp * si_mm * 1e-3 / h_contact


def table(j_mA_cm2=0.8, v_beam=500.0, resist_limit_c=120.0):
    q = beam_flux(j_mA_cm2, v_beam)
    rows = []
    for name, h in CONTACT_H.items():
        T = wafer_temperature(q, h)
        rows.append((name, h, T, time_constant(h), T <= resist_limit_c))
    return q, rows


def max_current_density(h_contact, v_beam=500.0, limit_c=120.0):
    """Largest beam current (mA/cm^2) that keeps the wafer at or below limit."""
    q_max = (limit_c - 20.0) / (wafer_temperature(1.0, h_contact) - 20.0)
    return q_max / (v_beam * 10.0)
