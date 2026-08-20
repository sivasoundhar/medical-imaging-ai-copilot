"""Approved knowledge-base retrieval (Day 10, PROJECT_SPEC.md Section
50). Deliberately NOT a RAG/vector-search pipeline -- Section 23 is
explicit that RAG must not become this project's center ("Do not turn
this project into a generic PDF chatbot"), and Section 49 wants the
Copilot "deterministic and knowledge-base driven." Retrieval here is a
plain, auditable keyword match against each source's `topic` field in
`knowledge_base/metadata/sources.json` -- given the same finding label,
it always returns the same passage(s), nothing fuzzier.

The KB is optional per Section 50 -- if `knowledge_base/` doesn't exist
or a label matches nothing, this returns an empty list rather than
fabricating reference content.
"""
import json
from dataclasses import dataclass
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent.parent.parent / "knowledge_base"


@dataclass
class KBPassage:
    source_id: str
    title: str
    version: str
    topic: str
    content: str


def retrieve_kb_passages(labels: list[str], kb_root: Path = KB_ROOT) -> list[KBPassage]:
    """Returns approved KB passages whose `topic` matches any of
    `labels` (case-insensitive substring match, deterministic order --
    sorted by `source_id`). Skips any source not marked `"status":
    "approved"` in the metadata, and any entry whose content file is
    missing on disk."""
    metadata_path = kb_root / "metadata" / "sources.json"
    if not metadata_path.exists():
        return []
    sources = json.loads(metadata_path.read_text(encoding="utf-8"))

    labels_lower = {label.lower() for label in labels if label}
    if not labels_lower:
        return []

    matches: list[KBPassage] = []
    for entry in sorted(sources, key=lambda e: e["source_id"]):
        if entry.get("status") != "approved":
            continue
        topic_lower = entry["topic"].lower()
        if not any(label in topic_lower or topic_lower in label for label in labels_lower):
            continue
        content_path = kb_root / "sources" / entry["file"]
        if not content_path.exists():
            continue
        matches.append(
            KBPassage(
                source_id=entry["source_id"],
                title=entry["title"],
                version=entry["version"],
                topic=entry["topic"],
                content=content_path.read_text(encoding="utf-8"),
            )
        )
    return matches
