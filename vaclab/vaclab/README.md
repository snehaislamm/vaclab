# vaclab

A Python toolkit for the work a thin-film lab does every week: certifying a vacuum chamber, finding leaks, running a deposition Design of Experiments, and measuring the films that come out.

All tool data is **simulated** from physical models (no hardware). The analysis code is written to run unchanged on real gauge logs, leak-detector exports and reflectometer spectra.

```
pip install -r requirements.txt
pytest -q            # 28 tests
python report.py     # figures/*.png + roundtable.md
```

## Modules

| module | what it does | headline result |
|---|---|---|
| `pumpdown.py` | Roughing + turbo pump-down of a 100 L chamber (series conductance, 1/t outgassing, bakeout). Certification and leak budget. | Spec 5e-8 mbar at 12 h allows at most **1.1e-5 mbar·L/s** of leak; a 3e-5 leak fails. |
| `leakcheck.py` | **Rate-of-rise (leak-back)** fit that separates wall desorption from a real leak; **helium spray** log analysis that locates the leaking flange. | Fit recovers a 2.0e-6 leak to within 1%; a straight-line fit overstates it by 16%. He spray finds F07 (3.0e-9) among 12 flanges. |
| `doe.py` | 2³ full factorial (DC power, Ar pressure, target distance), 2 replicates + 3 centre points, randomised run order, effects, t-tests, curvature check. | Distance and power dominate; significant power × distance interaction; centre points show curvature, so the next step is a CCD. |
| `metrology.py` | Spectral reflectometry fit (Cauchy SiNx on Si) and a 49-point wafer map with edge exclusion. | Calibrating the index once at the centre site cut worst-case thickness error from 1.73 nm to **0.24 nm**; ±2.3% non-uniformity. |
| `ibe_thermal.py` | Wafer temperature under a broad ion beam: 1-D composite-wall resistance (wafer, contact, holder, water). | At 0.4 W/cm², a dry clamp reaches ~154 °C (past a 120 °C resist limit); He backside gas holds ~28 °C. |

## Figures

`figures/pumpdown.png`, `rate_of_rise.png`, `he_spray.png`, `doe_pareto.png`, `metrology.png`

## Why

The film thickness 102.6 nm used here is the SiNx thickness I measured by ellipsometry on an MIS capacitor in my NE 343 cleanroom lab. This repo is a rehearsal for doing the same measurements, plus the vacuum work around them, on production tools.
