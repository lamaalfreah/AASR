**Router V2 — grouped development CV and freeze**

Selected `small_mlp`; calibrated LONG threshold `0.91000000000000003`. Final artifact is a three-member calibrated ensemble of this lightweight model family.

Merged existing Router TRAIN and DEV: 2,400 rows across 768 anchors. Binary labels: 550 SHORT, 1,499 LONG; 351 FAILURE rows excluded from fitting/calibration but retained for final-answer scoring. No new questions. No CORE EVAL or TEST reads.

Five outer anchor folds estimate performance. Each outer training portion has four grouped inner folds for threshold selection; each inner estimator uses three further grouped folds for unweighted sigmoid calibration of class-balanced base fits. No held-out anchor enters fitting, preprocessing, calibration, or threshold selection for its outer prediction.

The predeclared one-standard-error rule selects lower Long-call rates, then lower regressions, among accuracy-eligible thresholds/models. All candidates use a fixed configuration. The final threshold uses pooled outer OOF scores; its apparent accuracy is a selection score and is reported separately from the nested estimate.

| Model | Nested adaptive | Long route rate | Regressions | Routing accuracy | LONG precision | LONG recall |
|---|---|---|---|---|---|---|
| logistic_regression | 84.29% | 74.75% | 15 | 93.61% | 97.11% | 94.06% |
| hist_gradient_boosting | 86.83% | 75.83% | 5 | 98.83% | 99.80% | 98.60% |
| small_mlp | 86.83% | 75.71% | 4 | 99.02% | 99.93% | 98.73% |

| Calibration (binary OOF) | Raw Brier | Calibrated Brier | Calibrated log loss | ECE, 10 bins |
|---|---|---|---|---|
| logistic_regression | 0.0507 | 0.0425 | 0.1248 | 0.0080 |
| hist_gradient_boosting | 0.0064 | 0.0050 | 0.0191 | 0.0054 |
| small_mlp | 0.0037 | 0.0041 | 0.0177 | 0.0075 |

| Policy, all 2400 development rows | Correct | Accuracy |
|---|---|---|
| Always Short | 602 | 25.08% |
| Always Long | 1984 | 82.67% |
| Selected family, nested CV | 2084 | 86.83% |
| Answer oracle | 2104 | 87.67% |

Nested estimated Short/Long answer routes: 24.29% / 75.71%. Short inference must run for every question to supply features. Long is additional. Rescues: 1486; regressions: 4; unnecessary Long calls: 48. Binary routing metrics exclude FAILURE. FAILURE nested analysis: `{"n": 351, "long_calls": 336, "adaptive_answer_correct": 51, "always_short_correct": 9, "always_long_correct": 49, "oracle_correct": 55}`.

Final fixed-threshold pooled OOF selection score: 2073/2400 (86.38%), Long rate 75.08%. Do not present this as an independently evaluated final-model score. Final model fit uses merged development data only.

**Selected inference-safe features**

All eight named operation probabilities; confidence; entropy; top1-top2 margin; query validity; predicted operation; separate missing-slot group indicators and their count; entity ambiguity; question length in characters and words; constraint count. Missingness is retained and imputed within each fit. No gold task/register/complexity/anchor/outcome enters the feature dictionary. Missing-slot groups remain the frozen extractor diagnostics, not invented exact partial slots.

The full operation vector was observed in one CPU forward pass of unchanged Short on the existing questions. Operation, query, status, reason and confidence matched all frozen Short predictions. The original outcomes and feature values are retained. No Long generation or GPU calls were needed.

**Uncertainty and limits**

Model-selection accuracy floor: 85.71%. Eligible families: hist_gradient_boosting, small_mlp. Fold metrics and thresholds are in `summary.json`; complete threshold curves and calibration group audits are in `folds/`.

Nested scores estimate this training-and-threshold procedure; choosing a model family from three CV results still introduces selection uncertainty. Existing controlled wording families are shared across anchors, so this CV does not test unseen language styles. Calibration estimates the binary SHORT/LONG target among non-FAILURE cases; FAILURE is an out-of-target input, not an abstention class. No conclusion about production data sufficiency follows from these template-derived folds alone. No new data were added.

Training warning counts: `{"Could not find the number of physical cores for the following reason:": 1}`. Final ensemble warnings: `[]`. No repeated tuning was performed after inspecting results.

Frozen artifact: `router_v2.joblib`; preprocessing, three classifiers, calibrators and threshold are bundled. Versions and hashes are in `requirements.txt` and `freeze_manifest.json`. The selected ensemble is not integrated into the Geo Engine.

**Stopped after freezing Router V2. CORE EVAL results remain untouched.**
