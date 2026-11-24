# Evaluation protocol

Chronological incident-level split (70/15/15) when the opening timestamp is sufficiently complete. Otherwise stratified split fallback is marked in reports. Model selection uses validation PR-AUC. Threshold selection uses validation F1 grid; test results are computed once with that threshold.

Reported: PR-AUC, ROC-AUC, precision, recall, F1, balanced accuracy, Brier score and confusion matrix. The target currently maps SLA-met to positive class. Convert to breach target consistently if the business goal is breach prediction; never mislabel positive-class probabilities.

Leakage: exclusions are conservative but do not prove availability at incident intake. Audit actual source semantics and point-in-time availability before claiming early prediction.
