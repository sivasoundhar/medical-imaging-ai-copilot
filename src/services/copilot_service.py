"""Day 10 Medical Copilot orchestration (PROJECT_SPEC.md Section 49's
flow): builds a constrained prompt from structured vision findings +
approved KB context, calls the Day 9 LLM gateway, then runs the full
validation pipeline -- Pydantic schema -> groundedness -> safety --
before anything reaches a caller.

The LLM is never given the raw image (Day 10: "The LLM must never
analyze the raw medical image directly") -- only what's built in
`src/llm/prompts.py` from already-structured vision output.

Fails closed: any malformed/ungrounded/unsafe response is retried once
per provider, then -- if a fallback provider is configured -- retried
on that provider too, before finally raising `CopilotError`. This
module never returns a report it couldn't validate, degraded or
otherwise.
"""
import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.llm.base import LLMError, LLMProvider
from src.llm.knowledge_base import retrieve_kb_passages
from src.llm.prompts import build_qa_prompt, build_report_prompt
from src.safety.groundedness import GroundednessError, check_groundedness
from src.safety.input_guard import InputValidationError, validate_question
from src.safety.output_guard import check_output_safety
from src.schemas.imaging import Finding
from src.schemas.llm import MedicalReport
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Section 49 determinism: "use low temperature, preferably 0 or the
# lowest supported value." Ignored by ClaudeProvider (see its module
# docstring) -- current Claude models reject non-default temperature.
DETERMINISTIC_TEMPERATURE = 0.0

# Section 21: "Invalid -> retry / fallback." One retry per provider, not
# an unbounded loop -- a provider that can't produce a valid/grounded/
# safe response twice in a row should be abandoned in favor of the
# fallback provider (if any), not hammered further.
MAX_ATTEMPTS = 2


class CopilotError(RuntimeError):
    """Wraps any pipeline failure -- LLM call failure, malformed JSON,
    schema validation failure, groundedness violation, or safety
    violation -- into one type callers can catch. Never raised with a
    usable-but-unvalidated report attached; there isn't one."""


@dataclass
class CopilotResult:
    report: MedicalReport
    provider: str
    model: str
    grounded: bool
    kb_sources_used: list[str] = field(default_factory=list)


def _extract_json_object(raw_text: str) -> str:
    """LLMs asked for "only JSON" sometimes still wrap it in markdown
    fences or add a stray sentence despite instructions -- try a direct
    parse first, then fall back to extracting the first {...} block
    rather than failing outright on cosmetic wrapping."""
    stripped = raw_text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        return match.group(0)
    return stripped  # let the caller's json.loads raise a clear error


async def _generate_validated_report(
    providers: list[LLMProvider], messages: list[dict], allowed_labels: set[str]
) -> tuple[MedicalReport, LLMProvider]:
    """Tries each provider in order (primary first, then any configured
    fallback), up to MAX_ATTEMPTS per provider, before giving up.
    Returns the report together with whichever provider actually
    produced it -- the caller must not assume it was the first one."""
    last_error: Exception | None = None
    tried: list[str] = []
    for provider in providers:
        tried.append(provider.provider_name)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw_text = await provider.generate(messages, temperature=DETERMINISTIC_TEMPERATURE)
                report = MedicalReport.model_validate_json(_extract_json_object(raw_text))
                check_groundedness(report, allowed_labels)
                violations = check_output_safety(report)
                if violations:
                    # A plain ValueError, not GroundednessError -- this is a
                    # distinct failure mode (unsafe language, not an
                    # ungrounded finding) that happens to be caught by the
                    # same retry logic below.
                    raise ValueError(f"Output safety check failed: {violations}")
                return report, provider
            except (LLMError, ValidationError, json.JSONDecodeError, GroundednessError, ValueError) as exc:
                logger.warning(
                    "Copilot attempt %d/%d on provider %r failed: %s",
                    attempt, MAX_ATTEMPTS, provider.provider_name, exc,
                )
                last_error = exc
                continue
        logger.warning(
            "Provider %r exhausted %d attempts; trying next provider if any.",
            provider.provider_name, MAX_ATTEMPTS,
        )
    raise CopilotError(
        f"Failed to produce a valid, grounded, safe response after trying "
        f"provider(s) {tried} ({MAX_ATTEMPTS} attempts each): {last_error}"
    ) from last_error


def _validate_vision_findings(findings: list[Finding]) -> None:
    """Guards against malformed vision output reaching the LLM at all --
    cheap, deterministic, no LLM call needed. Day 10's evaluation set
    explicitly includes a "malformed vision output" case."""
    for f in findings:
        if not f.label or not f.label.strip():
            raise CopilotError("Malformed vision findings: a finding has an empty label.")
        if not (0.0 <= f.probability <= 1.0):
            raise CopilotError(
                f"Malformed vision findings: probability {f.probability} for "
                f"{f.label!r} is outside [0, 1]."
            )


async def generate_report(
    providers: LLMProvider | list[LLMProvider],
    findings: list[Finding],
    localization: str | None = None,
    modality: str = "xray",
) -> CopilotResult:
    """Preliminary AI report generation (Section 8, Feature 4). Accepts
    either one provider or an ordered [primary, fallback] list -- a bare
    provider is normalized to a one-item list so existing single-provider
    callers don't need to change."""
    providers = [providers] if isinstance(providers, LLMProvider) else providers
    _validate_vision_findings(findings)
    allowed_labels = {f.label for f in findings}
    kb_passages = retrieve_kb_passages(list(allowed_labels))
    messages = build_report_prompt(findings, localization, modality, kb_passages)

    report, used_provider = await _generate_validated_report(providers, messages, allowed_labels)
    return CopilotResult(
        report=report,
        provider=used_provider.provider_name,
        model=used_provider.model,
        grounded=True,
        kb_sources_used=[p.source_id for p in kb_passages],
    )


async def answer_question(
    providers: LLMProvider | list[LLMProvider],
    question: str,
    findings: list[Finding],
    localization: str | None = None,
    modality: str = "xray",
    conversation_history: list[dict] | None = None,
) -> CopilotResult:
    """Copilot Q&A about the current analysis (Section 9, Feature 5).
    Raises `InputValidationError` for empty/oversized questions before
    any LLM call. Accepts either one provider or an ordered
    [primary, fallback] list -- see `generate_report`."""
    providers = [providers] if isinstance(providers, LLMProvider) else providers
    question = validate_question(question)
    _validate_vision_findings(findings)
    allowed_labels = {f.label for f in findings}
    kb_passages = retrieve_kb_passages(list(allowed_labels))
    messages = build_qa_prompt(
        question, findings, localization, modality, kb_passages, conversation_history
    )

    report, used_provider = await _generate_validated_report(providers, messages, allowed_labels)
    return CopilotResult(
        report=report,
        provider=used_provider.provider_name,
        model=used_provider.model,
        grounded=True,
        kb_sources_used=[p.source_id for p in kb_passages],
    )


__all__ = [
    "CopilotError",
    "CopilotResult",
    "InputValidationError",
    "generate_report",
    "answer_question",
]
