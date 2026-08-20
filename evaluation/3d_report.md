# 3D Nodule Classifier — Test Set Evaluation Report

Checkpoint: `training\checkpoints\model_3d_best.pth`
Test candidates: 9190 across 13 scans

## Metrics

- ROC-AUC: 0.7771
- PR-AUC: 0.0262
- Sensitivity (recall): 0.6667
- Specificity: 0.7820
- Precision: 0.0030
- F1: 0.0060
- False positives / scan: 153.9231
- Decision threshold: 0.5

## Confusion matrix

- TP: 6
- FP: 2001
- FN: 3
- TN: 7180

## Notes

- Localization metrics (matching predictions to annotations.csv's real nodule coordinates) are NOT computed here -- candidates.csv already encodes candidate-vs-real-nodule labels, so sensitivity above is the nodule-detection-rate signal; per-nodule spatial localization scoring is deferred, not fabricated.
- This is a research/portfolio prototype, not a clinical device. Metrics reflect LUNA16 subset0's held-out test split only.