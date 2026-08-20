# Pneumonia — General Educational Reference

**Source ID:** kb-pneumonia-001 · **Version:** 1.0 · **Status:** approved

## What it is

Pneumonia is an infection that inflames the air sacs (alveoli) in one or both lungs,
which can fill with fluid or pus. Common causes include bacteria, viruses, and fungi.
On a chest X-ray, pneumonia often appears as an area of increased opacity ("whiteness")
where normal air-filled lung would otherwise appear darker.

## Why an imaging model might flag it

A 2D classification model trained on chest X-rays learns visual patterns statistically
associated with pneumonia in its training data (e.g., regional opacity, consolidation
patterns). A model probability score reflects how strongly the image matches those
learned patterns — it is a pattern-matching output, not a clinical diagnosis, and does
not account for a patient's symptoms, history, labs, or physical exam.

## Limitations to always state

- A single chest X-ray, and a single model score, cannot confirm or rule out pneumonia
  on its own.
- Other conditions (e.g., atelectasis, pulmonary edema, scarring) can produce visually
  similar opacities.
- Model performance on any individual image can differ from its aggregate reported
  metrics.

## Not for treatment guidance

This reference describes what the finding means and why a model might flag it. It does
**not** contain treatment, medication, or dosing guidance, and none should be inferred
from it.
