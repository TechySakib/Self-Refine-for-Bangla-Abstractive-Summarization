"""Conservative, deterministic XL-Sum Bengali article-body preprocessing.

This module performs no summarization, paraphrasing, inference, or LLM calls.
It normalizes Unicode/whitespace, removes clearly trailing related-link blocks,
and flags ambiguous inline related-link blocks for explicit manual review.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


PREPROCESSING_VERSION = "5.1.1"
# A bare trailing label with no text after it is unambiguous. Any marker
# followed by headline text is sent to manual review, even when it appears
# near the end of the article.

NAVIGATION_MARKER_RE = re.compile(
    r"(?P<label>(?:বিবিসি\s+বাংলা(?:য়|য়)\s+)?"
    r"আর(?:ও|ো)\s+প(?:ড়|ড়)(?:তে\s+পারেন|ুন)\s*[:：]?)"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_article_text(text: Any) -> str:
    """Apply only safe Unicode and whitespace normalization."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\ufeff", "").replace("\u00a0", " ")
    # Zero-width space is an accidental separator in many copied web texts;
    # preserve ZWJ/ZWNJ because they can be meaningful in Indic scripts.
    value = value.replace("\u200b", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def find_navigation_markers(text: str) -> List[re.Match]:
    return list(NAVIGATION_MARKER_RE.finditer(text))


def _find_unique_after(text: str, needle: str, start: int) -> int:
    needle = str(needle or "").strip()
    if not needle:
        raise ValueError("resume_at must contain an exact, non-empty article-text phrase.")
    positions: List[int] = []
    cursor = start
    while True:
        pos = text.find(needle, cursor)
        if pos < 0:
            break
        positions.append(pos)
        cursor = pos + 1
    if len(positions) != 1:
        raise ValueError(
            f"resume_at must occur exactly once after the navigation marker; found {len(positions)} matches. "
            "Use a longer, exact phrase copied from the article continuation."
        )
    return positions[0]


def clean_article_text(
    raw_text: Any,
    overrides: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Clean an article body and return (clean_text, metadata, review_rows).

    `overrides` maps zero-based marker occurrence indexes to CSV-like rows:
      - decision=remove_inline_block, resume_at=<exact unique article phrase>
      - decision=keep, review_note=<reason>

    Inline blocks are never guessed or silently truncated.
    """
    raw = str(raw_text or "")
    normalized = normalize_article_text(raw)
    matches = find_navigation_markers(normalized)
    overrides = overrides or {}

    operations: List[Tuple[int, int, str, int]] = []
    review_rows: List[Dict[str, Any]] = []
    marker_records: List[Dict[str, Any]] = []
    manually_kept: List[int] = []

    for marker_index, match in enumerate(matches):
        tail = normalized[match.end():]
        tail_words = len(tail.split())
        is_last_marker = marker_index == len(matches) - 1
        is_bare_trailing_label = is_last_marker and not tail.strip()

        override = overrides.get(marker_index) or {}
        decision = str(override.get("decision", "") or "").strip().lower()
        marker_info = {
            "marker_index": marker_index,
            "marker_text": match.group("label"),
            "marker_start": match.start(),
            "tail_characters": len(tail),
            "tail_whitespace_words": tail_words,
        }

        if is_bare_trailing_label:
            operations.append((match.start(), len(normalized), "auto_trailing_label", marker_index))
            marker_info["action"] = "auto_removed_bare_trailing_label"
            marker_records.append(marker_info)
            continue

        if decision == "remove_trailing_block":
            if not is_last_marker:
                raise ValueError("remove_trailing_block is allowed only for the final navigation marker.")
            operations.append((match.start(), len(normalized), "manual_trailing", marker_index))
            marker_info["action"] = "manually_removed_trailing_navigation_block"
            marker_records.append(marker_info)
            continue

        if decision == "remove_inline_block":
            resume_at = str(override.get("resume_at", "") or "").strip()
            resume_start = _find_unique_after(normalized, resume_at, match.end())
            if resume_start <= match.start():
                raise ValueError("resume_at must point to article text after the navigation marker.")
            # Do not allow a manual range to consume another navigation marker.
            next_marker = matches[marker_index + 1] if marker_index + 1 < len(matches) else None
            if next_marker and resume_start > next_marker.start():
                raise ValueError(
                    "resume_at crosses another navigation marker. Review each marker separately."
                )
            operations.append((match.start(), resume_start, "manual_inline", marker_index))
            marker_info["action"] = "manually_removed_inline_navigation"
            marker_info["resume_at"] = resume_at
            marker_records.append(marker_info)
            continue

        if decision == "keep":
            review_note = str(override.get("review_note", "") or "").strip()
            if not review_note:
                raise ValueError(
                    "A keep decision requires a short review_note explaining why the text was retained."
                )
            manually_kept.append(marker_index)
            marker_info["action"] = "manually_reviewed_and_kept"
            marker_info["review_note"] = review_note
            marker_records.append(marker_info)
            continue

        marker_info["action"] = "manual_review_required"
        marker_records.append(marker_info)
        before = normalized[max(0, match.start() - 180):match.start()]
        after = normalized[match.end():match.end() + 420]
        review_rows.append({
            "marker_index": marker_index,
            "marker_text": match.group("label"),
            "context_before": before,
            "context_after": after,
            "decision": "",
            "resume_at": "",
            "review_note": "",
        })

    # Apply removals from right to left so original character offsets remain valid.
    cleaned = normalized
    for start, end, _kind, _marker_index in sorted(operations, key=lambda item: item[0], reverse=True):
        cleaned = cleaned[:start].rstrip() + (" " + cleaned[end:].lstrip() if cleaned[end:].strip() else "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    unresolved = bool(review_rows)
    actions = [record["action"] for record in marker_records]
    if not matches:
        status = "normalized_only" if normalized != raw else "unchanged"
    elif unresolved:
        status = "manual_review_required"
    elif "manually_reviewed_and_kept" in actions:
        status = "manually_reviewed_kept"
    elif any(action in actions for action in (
        "manually_removed_inline_navigation",
        "manually_removed_trailing_navigation_block",
    )):
        status = "manual_navigation_cleaned"
    else:
        status = "auto_bare_trailing_label_cleaned"

    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "status": status,
        "review_required": unresolved,
        "original_characters": len(raw),
        "normalized_characters": len(normalized),
        "cleaned_characters": len(cleaned),
        "characters_removed_after_normalization": max(0, len(normalized) - len(cleaned)),
        "normalization_changed_text": normalized != raw,
        "original_sha256": _sha256(raw),
        "normalized_sha256": _sha256(normalized),
        "cleaned_sha256": _sha256(cleaned),
        "navigation_marker_count": len(matches),
        "navigation_markers": marker_records,
        "manual_review_reason": (
            "Related-reading marker followed by text detected; the boundary is not guessed. "
            "Use remove_inline_block with an exact resume_at phrase, remove_trailing_block "
            "only when the marker and all following text are related links, or keep after review."
            if unresolved else None
        ),
        "method": "deterministic_unicode_whitespace_and_related-link_cleanup_only",
    }
    return cleaned, metadata, review_rows


def apply_overrides_for_row(
    raw_text: Any,
    override_rows: Iterable[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    mapped: Dict[int, Dict[str, Any]] = {}
    for row in override_rows:
        try:
            index = int(row.get("marker_index", 0))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid marker_index in cleaning override: {row!r}")
        if index in mapped:
            raise ValueError(f"Duplicate cleaning override for marker_index={index}.")
        mapped[index] = row
    return clean_article_text(raw_text, mapped)
