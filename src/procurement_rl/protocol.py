"""Strict parser for the small Search-R1 action language.

Only assistant-generated messages belong here. Tool observations must never be
concatenated into text passed to this parser or trusted as assistant actions.
"""

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Action:
    kind: str
    query: str = ""
    answer: str = ""
    evidence: Tuple[str, ...] = ()
    error: str = ""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_TAGS = re.compile(r"</?(search|answer|evidence|information)\b[^>]*>", re.I)


def parse_action(text: str) -> Action:
    """Parse one search or one final answer; reject ambiguous/mixed actions.

    Free-form reasoning before the action is allowed. Final syntax is
    ``<evidence>id1,id2</evidence><answer>offer_id</answer>``.
    """
    if not isinstance(text, str) or not text.strip():
        return Action("invalid", error="empty_message")
    tags = [match.group(0).lower() for match in _TAGS.finditer(text)]
    if tags == ["<search>", "</search>"]:
        match = re.search(r"<search>(.*?)</search>\s*\Z", text, re.S)
        if match is None or not match.group(1).strip():
            return Action("invalid", error="invalid_search")
        query = match.group(1).strip()
        if len(query) > 2000 or "<" in query or ">" in query:
            return Action("invalid", error="invalid_search")
        return Action("search", query=query)
    if tags != ["<evidence>", "</evidence>", "<answer>", "</answer>"]:
        return Action("invalid", error="ambiguous_or_missing_action")
    match = re.search(r"<evidence>(.*?)</evidence>\s*<answer>(.*?)</answer>\s*\Z", text, re.S)
    if match is None:
        return Action("invalid", error="invalid_final")
    evidence = tuple(part.strip() for part in match.group(1).split(","))
    answer = match.group(2).strip()
    if not _ID.fullmatch(answer):
        return Action("invalid", error="invalid_offer_id")
    if not evidence or any(not _ID.fullmatch(item) for item in evidence):
        return Action("invalid", error="invalid_evidence_ids")
    if len(set(evidence)) != len(evidence):
        return Action("invalid", error="duplicate_evidence")
    return Action("answer", answer=answer, evidence=evidence)
