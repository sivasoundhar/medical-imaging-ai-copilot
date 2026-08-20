"""Constrained prompt templates for the Day 10 Medical Copilot
(PROJECT_SPEC.md Section 49 "Core rule" / Section 9 Feature 5).

The LLM never sees the raw image (Day 10 instruction: "The LLM must
never analyze the raw medical image directly"). It only ever receives:
1. structured vision findings, 2. localization/Grad-CAM metadata,
3. approved KB passages, 4. current study context, 5. explicitly
allowed conversation context -- built here, nowhere else.
"""
import json

from src.llm.knowledge_base import KBPassage
from src.schemas.imaging import Finding

SYSTEM_PROMPT = """You are a medical imaging AI copilot. You explain the output of a \
separate, already-run vision model -- you do NOT analyze images yourself and have not \
seen any image.

STRICT RULES (violating any of these makes your response unusable):
1. You may reference ONLY the findings listed in "vision_findings" below. Never invent, \
infer, guess, or add any finding, diagnosis, or abnormality not explicitly present there.
2. Never state a confirmed diagnosis. A model probability is not clinical certainty --
say "the model flagged" / "the model's probability was", never "the patient has".
3. Never recommend a specific treatment, medication, or dosage unless it is explicitly \
present in "knowledge_base_context" below. If no KB context is provided, do not discuss \
treatment at all.
4. Never claim you or this system replaces a doctor, radiologist, or the need for \
professional review.
5. Always set "requires_professional_review" to true.
6. Respond with ONLY a single JSON object -- no markdown code fences, no text before or \
after it -- matching exactly this schema:
{"summary": "<1-3 sentence plain-language summary>", "findings": [<copy each label from \
"vision_findings" EXACTLY as written there, word for word>], "limitations": ["<at least \
one limitation/uncertainty statement>"], "requires_professional_review": true}
7. CRITICAL: "findings" is NOT a place for sentences, descriptions, or explanations -- it \
is an exact copy of the label strings already in "vision_findings" above, nothing else. \
If vision_findings contains the label "PNEUMONIA", the findings list must contain exactly \
"PNEUMONIA" -- not a sentence describing pneumonia, not a quote from the reference \
material below, not a rephrasing. ALL explanation, description, and detail -- including \
anything drawn from "knowledge_base_context" -- belongs in "summary" only, never in \
"findings".
8. "localization" (if present below) is context ONLY, telling you where on the image the \
finding is -- it is never itself a finding. Do NOT add "localization" (in any casing or \
form) as an entry in "findings". "findings" contains ONLY entries copied verbatim from \
"vision_findings" labels, and nothing else -- not localization, not modality, not \
anything from "knowledge_base_context"."""


def _vision_context(
    findings: list[Finding], localization: str | None, modality: str
) -> dict:
    return {
        "modality": modality,
        "vision_findings": [
            {"label": f.label, "probability": f.probability} for f in findings
        ],
        "localization": localization,
    }


def _kb_context(kb_passages: list[KBPassage] | None) -> list[dict]:
    if not kb_passages:
        return []
    return [
        {"source_id": p.source_id, "title": p.title, "topic": p.topic, "content": p.content}
        for p in kb_passages
    ]


def build_report_prompt(
    findings: list[Finding],
    localization: str | None = None,
    modality: str = "xray",
    kb_passages: list[KBPassage] | None = None,
) -> list[dict]:
    """Preliminary AI report generation (Section 8, Feature 4)."""
    context = {
        **_vision_context(findings, localization, modality),
        "knowledge_base_context": _kb_context(kb_passages),
        "task": "Generate a short preliminary AI report explaining these vision-model findings.",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context)},
    ]


def build_qa_prompt(
    question: str,
    findings: list[Finding],
    localization: str | None = None,
    modality: str = "xray",
    kb_passages: list[KBPassage] | None = None,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """Copilot Q&A about the current analysis (Section 9, Feature 5).
    `conversation_history` is prior turns of this same conversation --
    Section 49's "explicitly allowed conversation context" -- appended
    as-is; callers are responsible for only passing turns from THIS
    study's conversation, never another patient's."""
    context = {
        **_vision_context(findings, localization, modality),
        "knowledge_base_context": _kb_context(kb_passages),
        "task": "Answer the user's question below using only the context provided.",
        "question": question,
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": json.dumps(context)})
    return messages
