import numpy as np
import pytest

from vaclab import pumpdown as pd, leakcheck as lc, doe, metrology as m, ibe_thermal as th


# ---------- pump-down / certification ----------
def test_effective_speed_is_series_combination():
    ch = pd.Chamber(turbo_speed_lps=700, conductance_lps=1500)
    assert ch.s_eff == pytest.approx(477.27, rel=1e-3)
    assert ch.s_eff < min(700, 1500)


def test_roughing_time_formula():
    ch = pd.Chamber(volume_l=100, rough_speed_lps=10, crossover_mbar=1e-2)
    assert pd.roughing_time(ch) == pytest.approx(10 * np.log(1013 / 1e-2))


def test_pressure_is_monotonic_after_roughing():
    t, p = pd.pressure_curve(pd.Chamber())
    assert np.all(np.diff(p) <= 1e-15)


def test_bakeout_lowers_base_pressure_100x():
    a = pd.certify(pd.Chamber()).base_mbar
    b = pd.certify(pd.Chamber(baked=True)).base_mbar
    assert a / b == pytest.approx(100, rel=1e-6)


def test_leak_at_budget_just_certifies():
    budget = pd.max_allowed_leak(pd.Chamber())
    assert pd.certify(pd.Chamber(leak_mbar_l_s=0.99 * budget)).passed
    assert not pd.certify(pd.Chamber(leak_mbar_l_s=1.01 * budget)).passed


def test_unreachable_pressure_returns_inf():
    assert pd.time_to_pressure(pd.Chamber(leak_mbar_l_s=1e-3), 1e-8) == float("inf")


# ---------- leak testing ----------
@pytest.mark.parametrize("seed", range(5))
def test_ror_recovers_leak_within_10pct(seed):
    t, p = lc.simulate_ror(leak=2e-6, seed=seed)
    r = lc.analyse_ror(t, p)
    assert r.leak_mbar_l_s == pytest.approx(2e-6, rel=0.10)


def test_ror_naive_line_overstates_leak():
    t, p = lc.simulate_ror()
    r = lc.analyse_ror(t, p)
    assert r.naive_leak_mbar_l_s > r.leak_mbar_l_s


def test_ror_fails_large_leak():
    t, p = lc.simulate_ror(leak=2e-5)
    assert not lc.analyse_ror(t, p, spec=5e-6).passed


def test_spray_finds_only_the_leaking_flange():
    found = lc.locate_leaks(lc.simulate_spray(leaking={"F07": 3e-9}))
    assert list(found) == ["F07"]
    assert found["F07"] == pytest.approx(3e-9, rel=0.1)


def test_spray_finds_two_leaks_and_ignores_subthreshold():
    found = lc.locate_leaks(lc.simulate_spray(leaking={"F02": 5e-9, "F11": 2e-9, "F05": 2e-10}))
    assert set(found) == {"F02", "F11"}


# ---------- DOE ----------
def test_design_size_and_balance():
    s = np.array(doe.design(replicates=2, centre=3))
    assert len(s) == 19
    assert np.all(s.sum(axis=0) == 0)


def test_model_matrix_columns_orthogonal():
    s = np.array([c for c in doe.design(replicates=1, centre=0)], float)
    M = doe.model_matrix(s)
    assert np.allclose(M.T @ M, 8 * np.eye(7))


def test_doe_finds_physics():
    s = doe.design(); y = doe.run(s); r = doe.analyse(s, y)
    assert r.effects["A"] > 0 and r.effects["C"] < 0 and r.effects["B"] < 0
    assert {"A", "C"} <= set(r.significant)
    assert r.best_coded == (1, -1, -1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_noise_free_effects_match_hand_calc():
    s = doe.design(replicates=1, centre=1)
    y = np.array([doe.true_rate(**doe.decode(c)) for c in s])
    r = doe.analyse(s + [(0, 0, 0)], np.append(y, y[[i for i, c in enumerate(s) if c == (0, 0, 0)]]))
    hi = np.mean([doe.true_rate(**doe.decode((1, b, c))) for b in (-1, 1) for c in (-1, 1)])
    lo = np.mean([doe.true_rate(**doe.decode((-1, b, c))) for b in (-1, 1) for c in (-1, 1)])
    assert r.effects["A"] == pytest.approx(hi - lo)


# ---------- metrology ----------
def test_reflectance_bare_si_limit():
    R = m.reflectance(0.0, 1.98, 1.6e4)
    n = m.n_si(m.WL)
    assert np.allclose(R, ((1 - n) / (1 + n)) ** 2)


@pytest.mark.parametrize("thk", [60.0, 102.6, 250.0])
def test_fit_recovers_thickness(thk):
    f = m.fit(m.measure(thk, seed=5))
    assert f.thickness_nm == pytest.approx(thk, abs=1.5)
    assert f.n_633 == pytest.approx(1.98 + 1.6e4 / 633**2, abs=0.03)


def test_wafer_map_has_49_points_inside_edge_exclusion():
    xy = m.wafer_points()
    assert len(xy) == 49
    assert np.hypot(*xy.T).max() <= 70.0 + 1e-9


def test_uniformity_numbers():
    _, truth, fitted, u = m.uniformity_map()
    assert u.max_fit_err_nm < 1.0
    assert 0.5 < u.range_pct < 5


# ---------- IBE thermal ----------
def test_beam_flux_units():
    assert th.beam_flux(1.0, 1000.0) == pytest.approx(1e4)  # 1 W/cm^2


def test_dry_clamp_overheats_resist_but_helium_does_not():
    _, rows = th.table()
    ok = {name: good for name, _, _, _, good in rows}
    assert not ok["dry clamp (vacuum)"] and ok["He backside gas"]


def test_max_current_density_hits_limit():
    h = th.CONTACT_H["He backside gas"]
    j = th.max_current_density(h)
    assert th.wafer_temperature(th.beam_flux(j), h) == pytest.approx(120.0)
