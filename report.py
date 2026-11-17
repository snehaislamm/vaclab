"""Run every module and write figures + a one-page roundtable summary.

    python report.py            -> figures/*.png, roundtable.md
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vaclab import pumpdown as pd, leakcheck as lc, doe, metrology as m, ibe_thermal as th

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def fig_pumpdown():
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    rows = []
    for leak, label in [(0, "no leak"), (1e-5, "1e-5 mbar L/s leak"), (3e-5, "3e-5 mbar L/s leak")]:
        ch = pd.Chamber(leak_mbar_l_s=leak)
        t, p = pd.pressure_curve(ch)
        ax.semilogy(t / 3600, p, label=label)
        rows.append((leak, pd.certify(ch)))
    ax.axhline(5e-8, color="k", ls="--", lw=0.8, label="cert spec 5e-8 mbar @ 12 h")
    ax.set(xlabel="time (h)", ylabel="pressure (mbar)", ylim=(1e-8, 2e3), title="Pump-down and certification")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "pumpdown.png"); plt.close(fig)
    return rows, pd.max_allowed_leak(pd.Chamber()), pd.Chamber().s_eff


def fig_ror():
    t, p = lc.simulate_ror()
    r = lc.analyse_ror(t, p)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(t, p * 1e6, ".", ms=2, label="gauge")
    pfit = lc.ror_model(t, p[0], r.desorb_mbar, r.tau_s, r.leak_mbar_l_s / 100)
    ax.plot(t, pfit * 1e6, label="desorption + leak fit")
    ax.set(xlabel="time since isolation (s)", ylabel="pressure (1e-6 mbar)", title="Rate-of-rise (leak-back) test")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "rate_of_rise.png"); plt.close(fig)
    return r


def fig_spray():
    log = lc.simulate_spray()
    found = lc.locate_leaks(log)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.semilogy(log.t, log.signal)
    for name, (t0, t1) in zip(log.flanges, log.windows):
        ax.axvspan(t0, t1, color="orange" if name in found else "grey", alpha=0.15)
        ax.text((t0 + t1) / 2, 2e-11, name, ha="center", fontsize=6)
    ax.axhline(1e-9, color="k", ls="--", lw=0.8, label="reject 1e-9 mbar L/s")
    ax.set(xlabel="time (s)", ylabel="He signal (mbar L/s)", title="Helium spray, 12 flanges")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "he_spray.png"); plt.close(fig)
    return found


def fig_doe():
    sheet = doe.design()
    y = doe.run(sheet)
    r = doe.analyse(sheet, y)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    order = sorted(doe.TERMS, key=lambda k: abs(r.t_values[k]))
    ax.barh(order, [abs(r.t_values[k]) for k in order],
            color=["C3" if k in r.significant else "C7" for k in order])
    ax.axvline(r.t_crit, color="k", ls="--", lw=0.8, label=f"t crit = {r.t_crit:.2f}")
    ax.set(xlabel="|t| of effect", title="Pareto of effects: sputter deposition rate")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "doe_pareto.png"); plt.close(fig)
    return sheet, r


def fig_metrology():
    R = m.measure(102.6)
    f = m.fit(R)
    xy, truth, fitted, u = m.uniformity_map()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
    a1.plot(m.WL, R, ".", ms=2, label="measured")
    a1.plot(m.WL, m.reflectance(f.thickness_nm, f.A, f.B), label=f"fit {f.thickness_nm:.1f} nm")
    a1.set(xlabel="wavelength (nm)", ylabel="R", title="SiNx on Si reflectance"); a1.legend(fontsize=7)
    sc = a2.scatter(xy[:, 0], xy[:, 1], c=fitted, cmap="viridis", s=40)
    a2.add_patch(plt.Circle((0, 0), 75, fill=False, color="k"))
    a2.set(aspect="equal", xlim=(-80, 80), ylim=(-80, 80), title="49-point thickness map (nm)")
    fig.colorbar(sc, ax=a2, shrink=0.8)
    fig.tight_layout(); fig.savefig(OUT / "metrology.png"); plt.close(fig)
    return f, u


def main():
    cert_rows, budget, seff = fig_pumpdown()
    ror = fig_ror()
    found = fig_spray()
    sheet, d = fig_doe()
    f, u = fig_metrology()
    q, trows = th.table()

    L = ["# Weekly roundtable: vaclab results", ""]
    L += ["## 1. Chamber certification",
          f"- S_eff = {seff:.0f} L/s; roughing to 1e-2 mbar in {pd.roughing_time(pd.Chamber()):.0f} s",
          f"- Leak budget at spec 5e-8 mbar after 12 h: {budget:.2e} mbar L/s"]
    for leak, c in cert_rows:
        L.append(f"- leak {leak:.0e}: base {c.base_mbar:.2e} mbar, leak share {100*c.leak_share:.0f}% -> {'PASS' if c.passed else 'FAIL'}")
    L += ["", "## 2. Leak testing",
          f"- Rate of rise: fitted leak {ror.leak_mbar_l_s:.2e} mbar L/s (true 2.0e-6); "
          f"straight-line method says {ror.naive_leak_mbar_l_s:.2e} ({100*(ror.naive_leak_mbar_l_s/2e-6-1):+.0f}% bias from desorption)",
          f"- He spray: flagged {', '.join(f'{k} ({v:.1e})' for k, v in found.items())}; true leak F07 3.0e-9"]
    L += ["", "## 3. DOE: sputter deposition rate (2^3, 2 reps + 3 centre = %d runs)" % len(sheet),
          "| term | effect (nm/min) | t |", "|---|---|---|"]
    for k in doe.TERMS:
        L.append(f"| {k} | {d.effects[k]:+.2f} | {d.t_values[k]:+.1f}{' *' if k in d.significant else ''} |")
    L += [f"- significant at 95%: {', '.join(d.significant)}; R^2 = {d.r2:.3f}",
          f"- curvature (centre - factorial) = {d.curvature:+.2f} nm/min, t = {d.curvature_t:+.1f} -> "
          f"{'add axial runs (CCD)' if abs(d.curvature_t) > d.t_crit else 'linear model adequate'}",
          f"- fastest corner {d.best_coded} -> {doe.decode(d.best_coded)}: {d.best_rate:.1f} nm/min"]
    L += ["", "## 4. Metrology",
          f"- Reflectometry fit: {f.thickness_nm:.1f} nm (true 102.6), n(633) = {f.n_633:.3f}, rms {f.rms:.4f}",
          f"- 49-pt map: mean {u.mean_nm:.1f} nm, 1-sigma {u.sigma_pct:.2f}%, range +/-{u.range_pct:.2f}%, worst fit error {u.max_fit_err_nm:.2f} nm"]
    L += ["", f"## 5. IBE wafer heating (0.8 mA/cm^2, 500 V -> {q/1e4:.2f} W/cm^2)",
          "| contact | h (W/m^2K) | T wafer (C) | tau (s) | resist OK (<=120 C) |", "|---|---|---|---|---|"]
    for name, h, T, tau, ok in trows:
        L.append(f"| {name} | {h:.0f} | {T:.0f} | {tau:.1f} | {'yes' if ok else 'NO'} |")
    (Path(__file__).parent / "roundtable.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
