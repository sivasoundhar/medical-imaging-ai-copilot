"""CI wrapper for the Day 10 failure-case evaluation
(evaluation/copilot_eval.py) -- asserts every required case (supported
finding, multiple findings, low confidence, no abnormality, missing
information, conflicting information, malformed vision output, prompt
injection) behaves correctly. Mocked only (MockLLMProvider) -- no real
network calls.
"""
import pytest

from evaluation.copilot_eval import EVAL_CASES, run_all_cases

_REQUIRED_CASE_NAMES = {
    "supported_finding",
    "multiple_findings",
    "low_confidence",
    "no_abnormality",
    "missing_information",
    "conflicting_information",
    "malformed_vision_output",
    "prompt_injection",
}


def test_eval_set_covers_every_required_case():
    assert {case.name for case in EVAL_CASES} == _REQUIRED_CASE_NAMES


@pytest.mark.asyncio
async def test_all_copilot_eval_cases_pass():
    results = await run_all_cases()
    failures = [r for r in results if not r.passed]
    assert not failures, "\n".join(f"{r.name}: {r.explanation}" for r in failures)
