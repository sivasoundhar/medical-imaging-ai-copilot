"""Input validation (Day 10, PROJECT_SPEC.md Section 22 "Input checks").
File type/size are already enforced at the imaging layer (Day 8's
`/api/v1/imaging/analyze`) -- this covers the Copilot's question text.

Prompt-injection detection here is a cheap heuristic used for
logging/flagging, NOT the actual defense -- flagging input text is
easy to evade by rewording. The real defense is on the output side:
`src/safety/groundedness.py` and `output_guard.py` validate what the LLM
actually returns regardless of what prompted it, which is what still
catches an injection attempt that slips past this check (see the
"prompt injection" case in `evaluation/copilot_eval.py`).
"""
import re

MAX_QUESTION_LENGTH = 2000

_INJECTION_MARKERS = [
    r"\bignore (all |the )?(previous|prior|above) instructions\b",
    r"\bdisregard (all |the )?(previous|prior|above)\b",
    r"\byou are now\b",
    r"\bsystem prompt\b",
    r"\bnew instructions?:\s",
    r"\bthis is not ai.generated\b",
]


class InputValidationError(ValueError):
    pass


def validate_question(question: str) -> str:
    """Raises `InputValidationError` for empty or oversized input.
    Returns the stripped question on success."""
    question = question.strip()
    if not question:
        raise InputValidationError("Question must not be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise InputValidationError(
            f"Question exceeds max length ({MAX_QUESTION_LENGTH} chars): {len(question)} chars."
        )
    return question


def looks_like_prompt_injection(question: str) -> bool:
    """Heuristic only -- see module docstring. Used for logging/
    flagging, never as the sole safety mechanism."""
    lower = question.lower()
    return any(re.search(pattern, lower) for pattern in _INJECTION_MARKERS)
