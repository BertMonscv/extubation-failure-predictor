| Comparison | Spearman rho (95% CI) | RBO | Top-5 Jaccard | Note |
|---|---|---|---|---|
| C1 = rho(S_M, S_M->E) | 0.90 (0.87, 0.91) | 0.72 | 0.67 | frozen model | covariate shift |
| C2 = rho(S_M->E, S_E) [PRIMARY] | 0.48 (0.32, 0.57) | 0.58 | 0.25 | independent eICU model | same eval |
| C2 (same-hp eICU) | 0.47 | 0.61 | 0.43 | sensitivity (expected overfit) |
| C2 (drop mv_duration) | 0.33 | 0.57 | 0.25 | sensitivity |
| R0 permutation (99th pct) | 0.44 | - | - | floor; empirical p=0.0059 |
| R1 frozen / within-MIMIC | 1.00 (0.99, 1.00) | - | - | ceiling for C1 |
| R2 independent / within (~160 ev) | 0.29 (0.04, 0.55) | - | - | ceiling for C2 |
