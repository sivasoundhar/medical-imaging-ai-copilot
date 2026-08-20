# Model Card

Research/portfolio prototype. **Not a clinical device — not validated for
diagnostic use.** See `README.md` for full project scope and disclaimers.

## 2D chest X-ray classifier

**Intended use:** Research/portfolio demonstration of transfer-learning
binary classification (NORMAL vs. PNEUMONIA) on chest X-rays. Not
validated against any clinical standard; paired with the LLM Copilot
layer for plain-language explanation of the model's own findings, never
an independent diagnosis.

**Architecture:** ResNet50, ImageNet-pretrained, fine-tuned end-to-end
(no frozen backbone in the final run).

**Training data:** Kaggle "Chest X-Ray Images (Pneumonia)" dataset —
**not** the official train/test split, which was found to leak patients
across splits (see `PROGRESS_LOG.md` Day 3). Re-split by patient ID from
scratch: 4,133 train / 808 val / 915 test images, verified
patient-disjoint.

**Training run:** Colab T4 GPU, 10 configured epochs, best checkpoint by
validation loss at epoch 8 (val_loss climbed on epochs 9–10 — mild
overfitting, correctly not selected). Checkpoint:
`training/checkpoints/model_2d_best.pth`.

**Test-set metrics** (915-image held-out test set, evaluated once,
`evaluation/2d_metrics.json`):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9919 |
| PR-AUC | 0.9946 |
| Sensitivity (recall) | 0.9883 |
| Specificity | 0.9356 |
| Precision | 0.9782 |
| F1 | 0.9832 |
| Confusion matrix | TN=218, FP=15, FN=8, TP=674 |
| Test set size | 915 images |

**Known limitations (real, measured — not hypothetical):**
- These metrics reflect one specific, fairly visually-separable Kaggle
  dataset — not a claim of clinical-grade generalization. No external
  validation set has been used.
- No bias/subgroup evaluation was possible: this dataset has no
  age/sex/demographic metadata per image, only the NORMAL/PNEUMONIA
  folder label — documented as a stated limitation, not silently
  skipped (see `PROGRESS_LOG.md` Day 4).
- Any previously published number on this exact Kaggle dataset using
  its *official* train/test split is potentially patient-leaked and
  optimistic — these numbers use a proper patient-disjoint split
  instead, so they may read lower than other reported results on the
  same dataset. That's more trustworthy, not a regression.
- No cross-validation, no clinical or radiologist review of any kind.
- **Grad-CAM has surfaced real shortcut-learning behavior on individual
  cases:** on at least one user-submitted X-ray, the model's Grad-CAM
  attention included a hot region over the burned-in "R" laterality
  marker and adjacent shoulder soft tissue — outside both lung fields —
  alongside a second, anatomically plausible hot region over the
  mediastinum. Preprocessing (`src/preprocessing/preprocess_2d.py`)
  performs no lung-field cropping or marker masking before resize, and
  the network is fine-tuned end-to-end, so any pixel (marker, shoulder,
  border) can carry gradient. This is consistent with the dataset-level
  confound documented in the literature for this exact Kaggle set
  (NORMAL/PNEUMONIA images pulled from different source cohorts, making
  non-anatomical cues like marker placement or patient positioning a
  viable shortcut). Not confirmed as systematic across the full test set
  — this is a real, observed instance, reported honestly rather than
  generalized without more Grad-CAM sampling. A real fix (lung-field
  ROI cropping/masking before resize) would be training-pipeline scope,
  not yet done.

## 3D CT nodule classifier

**Intended use:** Research/portfolio demonstration of a 3D CNN
classifying LUNA16 candidate patches (small cropped regions of a chest CT
scan) as nodule vs. non-nodule. Meant to be paired with an LLM layer that
explains these *grounded* vision-model findings in plain language — the
LLM does not diagnose from images directly, and this model does not
diagnose disease; it screens individual candidate locations already
identified by LUNA16's candidate-generation process, not raw scans.

**Training data:** LUNA16 `subset0` only (89 of the full dataset's CT
series), `candidates.csv` for labels (severely imbalanced, ~466:1
non-nodule:nodule). Series-safe train/val/test split (one series never
appears in more than one split), seed 42, 70/15/15. See
`PROGRESS_LOG.md` Day 6/7 entries for full data-verification detail.

**Training run:** 10 configured epochs, `early_stopping_patience: 3`,
selection metric `validation_roc_auc`. Real run on local hardware (GTX
1650) stopped early at epoch 8 (best epoch was 5, no improvement for 3
epochs after). Checkpoint: `training/checkpoints/model_3d_best.pth`
(epoch 5 weights).

**Test-set metrics** (held-out test split, `evaluation/3d_metrics.json`,
decision threshold 0.5 — not tuned):

| Metric | Value |
|---|---|
| ROC-AUC | 0.7771 |
| PR-AUC | 0.0262 |
| Sensitivity (recall) | 0.6667 |
| Specificity | 0.7820 |
| Precision | 0.0030 |
| F1 | 0.0060 |
| False positives / scan | 153.9 |
| Test set size | 9,190 candidates across 13 scans (9 real nodules) |

**Note on accuracy (deliberately not used as a headline metric):**
Raw accuracy on this test split is **78.2%** (7,186/9,190 correct: TN
7180, FP 2001, FN 3, TP 6) — sounds reasonable in isolation, but it's a
misleading number here and is why it isn't reported above. With only 9
real nodules in 9,190 candidates, a model that predicted "non-nodule"
for every single input would *also* score ~78.2% accuracy (7,183/9,190)
— i.e. this model's accuracy is statistically indistinguishable from a
model that learned nothing. ROC-AUC, sensitivity, and precision are
reported instead specifically because they don't hide this the way
accuracy does.

**Known limitations (real, measured — not hypothetical):**
- **Precision is very low (0.3%) and false-positives/scan is very high
  (~154/scan).** With only 9 true-nodule candidates in the whole test
  split (test-set imbalance ~1020:1), the default 0.5 threshold produces
  far more false positives than true positives (2001 FP vs. 6 TP). ROC-AUC
  (0.777) looks more encouraging than precision because ROC-AUC is
  threshold-independent and less sensitive to this level of imbalance —
  PR-AUC (0.026) is the more honest signal here.
  Threshold tuning (raising the decision threshold well above 0.5), a
  pretrained 3D backbone swap, and training on more LUNA16 subsets +
  a false-positive-reduction cascade were all considered (see
  `PROGRESS_LOG.md` Day 12.5) and deliberately **not pursued** — this
  is out of scope for a portfolio/research prototype, not an oversight.
  0.5 was never a meaningfully chosen operating point; this model ships
  as-is with the limitation documented rather than tuned.
- Trained only on `subset0` (89 series) of LUNA16's 10 subsets — not the
  full dataset. Generalization to the other subsets/full dataset is
  unverified.
- Only 9 positive (real nodule) examples in the entire test split — any
  single metric here (especially precision/F1) can swing a lot from one
  or two predictions. Treat these numbers as a first real measurement,
  not a stable estimate.
- Localization (matching a predicted-positive candidate to a real nodule's
  spatial coordinates in `annotations.csv`) is not scored — this model
  operates on LUNA16's pre-generated candidate patches, not raw scan
  localization.
- No cross-validation, no external test set, no clinical or radiologist
  review of any kind.

## Medical Copilot / LLM explanation layer

**Intended use:** Explains the vision models' *own, already-produced*
structured findings in plain language — never analyzes a raw image
itself, never invents a finding the vision pipeline didn't produce
(enforced in code by `src/safety/groundedness.py`, not just prompted
for). Provider-agnostic (Groq / Ollama / Claude) via `src/llm/gateway.py`.

**Safety architecture:** every LLM response is Pydantic-validated, then
checked for groundedness (claimed findings must be a subset of the real
vision output) and output safety (no confirmed-diagnosis language, no
ungrounded treatment claims, no "replaces a clinician" claims) before it
can reach a user. A response failing either check is retried once, then
rejected outright — never silently passed through degraded.

**Real evaluation (`evaluation/copilot_eval.py`), 8/8 required cases
pass** — including a prompt-injection case where the mock LLM is
scripted to fully comply with the injection (invents a diagnosis, claims
certainty, disables the professional-review flag), and the safety layer
still catches and rejects it. See `PROGRESS_LOG.md` Day 10 for full
per-case detail.

**Real provider comparison (`evaluation/provider_comparison.py`),
measured, not fabricated:** Ollama (local `llama3.2`) measured directly
— mean latency ~12s, schema-validity ~75% one-shot (small local models
occasionally violate the strict JSON schema; caught and correctly
rejected by the pipeline, not a pipeline bug). Groq and Claude are
reported as **not run** in this environment — no API credentials
configured — rather than estimated or fabricated.

**Known limitation:** small local models (e.g. `llama3.2`) sometimes
need the full retry budget to produce a compliant response, and can
occasionally exhaust it — this is a real, observed reliability
characteristic of the specific LLM, not a flaw in the validation logic,
which is doing exactly its job by rejecting non-compliant output.

**This is a research/portfolio prototype, not a clinical device.**
