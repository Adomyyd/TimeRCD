# Stable pair: real data only

700 paired series, 11 equally weighted groups; probability scores, percent. Best epochs 19 / 23.

| Model | Affiliation-F | F1_T | Standard-F1 | VUS-PR |
|---|---:|---:|---:|---:|
| P0 stable | 81.28 | 45.20 | 43.11 | 47.21 |
| P1 stable | 79.33 | 44.09 | 42.75 | 48.42 |
| P1 minus P0 | -1.95 | -1.12 | -0.36 | 1.22 |

Per-dataset P1 minus P0 (percentage points):

| Dataset | Affiliation-F | F1_T | Standard-F1 | VUS-PR |
|---|---:|---:|---:|---:|
| IOPS | 1.59 | -12.24 | -8.34 | -4.52 |
| MGAB | -3.06 | -2.75 | -2.85 | -0.77 |
| NAB | -12.44 | 1.60 | -0.21 | -0.16 |
| NEK | 2.57 | -1.15 | 2.26 | 2.97 |
| Power | -4.92 | -4.91 | -4.93 | -4.97 |
| SED | 2.86 | 8.52 | 8.63 | 17.32 |
| Stock | -2.44 | -1.70 | -2.14 | -0.07 |
| TODS | 1.76 | 4.74 | 5.13 | 3.18 |
| UCR | -0.83 | -1.50 | -0.39 | -0.35 |
| WSD | -4.98 | -1.64 | -0.26 | 1.71 |
| YAHOO | -1.54 | -1.24 | -0.81 | -0.95 |

Source runs:
- [p0_official_stable_20261117_130401_eval_real700_20261121](../../runs/p0_official_stable_20261117_130401_eval_real700_20261121/)
- [p1_official_stable_20261117_130401_eval_real700_20261121](../../runs/p1_official_stable_20261117_130401_eval_real700_20261121/)
