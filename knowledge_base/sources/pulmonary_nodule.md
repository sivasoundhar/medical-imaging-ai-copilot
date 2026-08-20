# Pulmonary Nodules — General Educational Reference

**Source ID:** kb-pulmonary-nodule-001 · **Version:** 1.0 · **Status:** approved

## What it is

A pulmonary nodule is a small, roughly round spot in the lung, typically seen on a CT
scan. Nodules are common and are usually benign (non-cancerous) — most are caused by
old infections, scarring, or other non-threatening processes. A minority of nodules
turn out to be early-stage lung cancer, which is why suspicious nodules are tracked over
time or investigated further.

## Why an imaging model might flag a candidate location

This project's 3D model classifies a single, pre-identified candidate location within a
CT scan as "nodule" or "non-nodule" — it does not scan a whole volume to find candidates
on its own (see the project's model card for detail on this scope). A "nodule"
classification means the model's learned pattern-matching considers that specific
location more consistent with a nodule than the training data's non-nodule examples. It
is not a size measurement, a malignancy assessment, or a diagnosis.

## Limitations to always state

- Classifying a candidate location as "nodule" says nothing about whether it is benign
  or malignant.
- Nodule follow-up (repeat imaging, further workup) is a clinical decision based on
  size, growth, patient history, and risk factors — none of which this model considers.
- A single classification score for a single location is not equivalent to a full
  radiological read of the scan.

## Not for treatment guidance

This reference describes what a nodule classification means and why a model might
produce one. It does **not** contain treatment, follow-up-interval, or biopsy guidance,
and none should be inferred from it.
