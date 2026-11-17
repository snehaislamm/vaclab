# Weekly roundtable: vaclab results

## 1. Chamber certification
- S_eff = 477 L/s; roughing to 1e-2 mbar in 115 s
- Leak budget at spec 5e-8 mbar after 12 h: 1.14e-05 mbar L/s
- leak 0e+00: base 2.62e-08 mbar, leak share 0% -> PASS
- leak 1e-05: base 4.71e-08 mbar, leak share 44% -> PASS
- leak 3e-05: base 8.90e-08 mbar, leak share 71% -> FAIL

## 2. Leak testing
- Rate of rise: fitted leak 1.99e-06 mbar L/s (true 2.0e-6); straight-line method says 2.33e-06 (+16% bias from desorption)
- He spray: flagged F07 (3.0e-09); true leak F07 3.0e-9

## 3. DOE: sputter deposition rate (2^3, 2 reps + 3 centre = 19 runs)
| term | effect (nm/min) | t |
|---|---|---|
| A | +28.02 | +51.6 * |
| B | -6.04 | -11.1 * |
| C | -35.65 | -65.7 * |
| AB | -2.08 | -3.8 * |
| AC | -11.38 | -21.0 * |
| BC | +1.26 | +2.3 * |
| ABC | +0.01 | +0.0 |
- significant at 95%: A, B, C, AB, AC, BC; R^2 = 0.999
- curvature (centre - factorial) = -5.91 nm/min, t = -8.7 -> add axial runs (CCD)
- fastest corner (1, -1, -1) -> {'power_W': 400.0, 'pressure_mTorr': 2.0, 'distance_mm': 80.0}: 84.3 nm/min

## 4. Metrology
- Reflectometry fit: 103.4 nm (true 102.6), n(633) = 2.005, rms 0.0020
- 49-pt map: mean 100.7 nm, 1-sigma 1.40%, range +/-2.29%, worst fit error 0.24 nm

## 5. IBE wafer heating (0.8 mA/cm^2, 500 V -> 0.40 W/cm^2)
| contact | h (W/m^2K) | T wafer (C) | tau (s) | resist OK (<=120 C) |
|---|---|---|---|---|
| dry clamp (vacuum) | 30 | 154 | 36.7 | NO |
| thermal paste / pad | 250 | 37 | 4.4 | yes |
| He backside gas | 600 | 28 | 1.8 | yes |
