"""Output safety validator (Day 10, PROJECT_SPEC.md Section 22 "LLM
checks" / Section 3's numbered safety rules): reject responses
containing an unsupported diagnosis framing, ungrounded treatment
claims, absolute-certainty language, or claims that AI replaces a
clinician.

Deliberately reject-only, not "rewrite" -- Section 22 says "reject or
rewrite"; rewriting would mean another LLM call attempting to fix
unsafe output, which itself isn't guaranteed safe. Rejecting (raise, let
the caller retry the whole generation) is simpler to reason about and
never silently launders bad output into something that looks clean.
"""
import re

from src.schemas.llm import MedicalReport

# Patterns are intentionally simple/literal (not full NLP) -- same
# reasoning as groundedness.py: deterministic and auditable over fuzzy.
_ABSOLUTE_CERTAINTY_PATTERNS = [
    r"\bdefinitely\b",
    r"\bconfirmed diagnosis\b",
    r"\b100%\s*(certain|sure)\b",
    r"\bwithout (a )?doubt\b",
    r"\bcertainly (has|is)\b",
    r"\b(patient|they) (has|have)\b",  # asserting the patient HAS a condition, not "the model flagged"
]
_REPLACES_CLINICIAN_PATTERNS = [
    r"\bno need (for|to see) a (doctor|physician|radiologist)\b",
    r"\breplaces? (a |the )?(doctor|physician|radiologist|clinician)\b",
    r"\byou do not need (professional|clinical) (review|care)\b",
]
_TREATMENT_PATTERNS = [
    r"\byou should (take|start|begin)\b",
    r"\bprescri(be|ption|bed)\b",
    r"\brecommend(ed)? (treatment|therapy|medication|antibiotics?)\b",
    r"\b\d+\s*mg\b",  # a dosage
]


def check_output_safety(report: MedicalReport, kb_grounded_treatment: bool = False) -> list[str]:
    """Returns a list of human-readable violation descriptions (empty if
    clean). Scans `summary` + `findings` + `limitations` text.

    `kb_grounded_treatment=True` means the caller has already confirmed
    any treatment language traces back to approved KB content (Section 3
    rule 8) -- otherwise ANY treatment-shaped language is flagged,
    matching "avoid treatment recommendations unless explicitly grounded
    in approved reference material."
    """
    violations: list[str] = []
    full_text = " ".join([report.summary, *report.findings, *report.limitations])
    lower = full_text.lower()

    for pattern in _ABSOLUTE_CERTAINTY_PATTERNS:
        if re.search(pattern, lower):
            violations.append(f"Absolute-certainty / confirmed-diagnosis language detected ({pattern!r}).")

    for pattern in _REPLACES_CLINICIAN_PATTERNS:
        if re.search(pattern, lower):
            violations.append(f"Claims AI replaces a clinician ({pattern!r}).")

    if not kb_grounded_treatment:
        for pattern in _TREATMENT_PATTERNS:
            if re.search(pattern, lower):
                violations.append(f"Ungrounded treatment/medication claim ({pattern!r}).")

    if not report.requires_professional_review:
        violations.append("requires_professional_review must be true (Section 3 rule 7).")

    return violations
