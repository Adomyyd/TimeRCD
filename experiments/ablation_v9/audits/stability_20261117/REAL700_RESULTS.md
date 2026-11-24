# Stable trio: real data only

700 paired series, 11 equally weighted groups; probability scores, percent. Best epochs 19 / 23 / 33.

| Model | Affiliation-F | F1_T | Standard-F1 | VUS-PR |
|---|---:|---:|---:|---:|
| P0 stable | 81.28 | 45.20 | 43.11 | 47.21 |
| P1 stable | 79.33 | 44.09 | 42.75 | 48.42 |
| P2 stable | 78.32 | 39.86 | 38.62 | 43.32 |
| P1 minus P0 | -1.95 | -1.12 | -0.36 | 1.22 |
| P2 minus P0 | -2.96 | -5.35 | -4.49 | -3.89 |

Per-dataset deltas against P0 (percentage points):

| Dataset | Aff-F P1 | F1_T P1 | Std-F1 P1 | VUS-PR P1 | Aff-F P2 | F1_T P2 | Std-F1 P2 | VUS-PR P2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IOPS | 1.59 | -12.24 | -8.34 | -4.52 | 1.74 | -9.63 | -2.21 | 1.26 |
| MGAB | -3.06 | -2.75 | -2.85 | -0.77 | -4.73 | -7.67 | -7.48 | -1.96 |
| NAB | -12.44 | 1.60 | -0.21 | -0.16 | 1.72 | 4.43 | 3.98 | 6.07 |
| NEK | 2.57 | -1.15 | 2.26 | 2.97 | 3.23 | -3.53 | -1.78 | 0.45 |
| Power | -4.92 | -4.91 | -4.93 | -4.97 | -2.69 | -5.06 | -5.34 | -3.31 |
| SED | 2.86 | 8.52 | 8.63 | 17.32 | -24.15 | -30.75 | -30.93 | -41.33 |
| Stock | -2.44 | -1.70 | -2.14 | -0.07 | -2.90 | -1.94 | -2.22 | -0.43 |
| TODS | 1.76 | 4.74 | 5.13 | 3.18 | -3.16 | 1.44 | 0.54 | -0.43 |
| UCR | -0.83 | -1.50 | -0.39 | -0.35 | -0.72 | -6.94 | -6.34 | -5.67 |
| WSD | -4.98 | -1.64 | -0.26 | 1.71 | 0.33 | 2.05 | 3.10 | 2.62 |
| YAHOO | -1.54 | -1.24 | -0.81 | -0.95 | -1.21 | -1.20 | -0.65 | -0.05 |

`P1` and `P2` columns are each minus P0 on the same series and metric.

Source runs:
- [p0_official_stable_20261117_130401_eval_real700_20261121](../../runs/p0_official_stable_20261117_130401_eval_real700_20261121/)
- [p1_official_stable_20261117_130401_eval_real700_20261121](../../runs/p1_official_stable_20261117_130401_eval_real700_20261121/)
- [p2_official_stable_20261121_2320_eval_real700_20261124](../../runs/p2_official_stable_20261121_2320_eval_real700_20261124/)

Valid counts (probability, series level): all four headline metrics are 700/700 for every group; AUC-ROC is 694/700; Affiliation-P is 700/700 (P0), 687/700 (P1) and 699/700 (P2).
