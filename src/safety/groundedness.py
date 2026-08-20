"""Groundedness validator (Day 10, PROJECT_SPEC.md Section 3 rule 5 /
Section 49 / "Groundedness test examples").

Checks the LLM's claimed `findings` are a subset of the real vision-
model findings it was given -- this is what stops the LLM from
"inventing pleural effusion" or "inventing pneumothorax" (the spec's own
groundedness test examples) when the real vision output only said
pneumonia. Also satisfies Day 10's separately-listed "unsupported-
finding detection" bullet -- that's the same failure mode from the
other side (a finding present in the LLM's output but absent from the
real vision findings), not a second mechanism.
"""
from src.schemas.llm import MedicalReport


class GroundednessError(ValueError):
    """Raised when the LLM's response references a finding not present
    in the real vision-model output it was given."""


def check_groundedness(report: MedicalReport, allowed_labels: set[str]) -> None:
    """Raises `GroundednessError` if any claimed finding in
    `report.findings` isn't grounded in `allowed_labels` (the real
    vision findings' labels).

    Deliberately a simple case-insensitive containment check, not fuzzy
    NLP/embedding similarity -- matches Section 49's "deterministic,
    KB-driven" design: the same input always produces the same
    pass/fail, and a human can audit exactly why something failed.
    `allowed_labels` empty (e.g. "no abnormality" findings) means ANY
    non-empty claimed finding is ungrounded.
    """
    allowed_lower = {label.lower() for label in allowed_labels if label}
    for finding in report.findings:
        finding_lower = finding.lower()
        if not any(label in finding_lower for label in allowed_lower):
            raise GroundednessError(
                f"LLM claimed finding {finding!r} is not grounded in the real vision "
                f"findings {sorted(allowed_labels)!r}"
            )
