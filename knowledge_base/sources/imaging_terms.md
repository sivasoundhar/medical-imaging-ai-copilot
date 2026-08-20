# Reading AI Imaging Output — Terms and Limitations

**Source ID:** kb-imaging-terms-001 · **Version:** 1.0 · **Status:** approved

This explains terms used in this system's own output. It is a project-authored guide to
reading the tool, not a clinical reference.

## Model probability / confidence score

A number between 0 and 1 (or 0–100%) representing how strongly the model's learned
pattern-matching associates the input with a given label. It is **not** the probability
that the patient actually has the condition, and it is **not** clinical certainty. A
0.87 score means the model's output for that label was 0.87 on its internal scale — nothing
more.

## Grad-CAM / heatmap

A visualization of which pixels most influenced the model's prediction (its
"activation" or "focus"). It shows where the model looked, not proof that a disease is
present there. A model can focus on the correct region and still be wrong, or focus on
an unrelated region and happen to be right.

## "Requires professional review"

Every output from this system carries this flag, always set to true. It means: this is
an AI-generated research/portfolio output, not a diagnosis, and any real clinical
decision requires a qualified healthcare professional's independent review.

## What this system's outputs are not

- Not a diagnosis
- Not a substitute for a radiologist or physician
- Not validated for clinical use
- Not a treatment recommendation
