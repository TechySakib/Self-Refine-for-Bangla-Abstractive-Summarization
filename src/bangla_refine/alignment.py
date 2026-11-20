import re
import unicodedata

BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Deliberately small stopword list: this is only a conservative screening aid.
STOPWORDS = {
    "এবং", "কিন্তু", "তবে", "যে", "যা", "এই", "ওই", "তার", "তাঁর",
    "তাদের", "তিনি", "তারা", "একটি", "এক", "জন্য", "থেকে", "সঙ্গে",
    "করে", "করেন", "হয়", "হয়", "ছিল", "হবে", "হয়েছে", "হয়েছে",
    "আর", "না", "এর", "এতে", "দিকে", "বলে", "যখন", "তখন",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "was", "were", "be", "has", "have", "had", "as", "by",
}


def _normalize(text):
    return unicodedata.normalize("NFC", str(text or "")).translate(BENGALI_DIGIT_MAP).lower()


def _numbers(text):
    return set(re.findall(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)", _normalize(text)))


def _content_tokens(text):
    normalized = _normalize(text)
    tokens = re.findall(r"[a-z]+|[\u0980-\u09FF]+|\d+", normalized)
    return {
        token for token in tokens
        if len(token) > 1 and token not in STOPWORDS
    }


def audit_reference_alignment(source_article, reference_summary):
    """Flag potential source/reference mismatches; never auto-exclude an example.

    Missing numeric/date tokens are a strong review cue. Lexical overlap is only
    a weak secondary cue because abstractive summaries can paraphrase heavily.
    """
    source = str(source_article or "")
    reference = str(reference_summary or "")
    source_numbers = _numbers(source)
    reference_numbers = _numbers(reference)
    missing_numbers = sorted(reference_numbers - source_numbers)

    ref_tokens = _content_tokens(reference)
    source_tokens = _content_tokens(source)
    overlap_tokens = sorted(ref_tokens & source_tokens)
    coverage = (
        len(overlap_tokens) / len(ref_tokens)
        if ref_tokens else None
    )

    flags = []
    if missing_numbers:
        flags.append("reference_contains_numeric_or_date_token_absent_from_source")
    if ref_tokens and len(ref_tokens) >= 5 and coverage is not None and coverage < 0.15:
        flags.append("very_low_lexical_overlap_review")

    return {
        "status": "review_required" if flags else "no_obvious_mismatch_detected",
        "review_required": bool(flags),
        "flags": flags,
        "reference_numeric_tokens": sorted(reference_numbers),
        "source_numeric_tokens": sorted(source_numbers),
        "reference_numeric_tokens_missing_from_source": missing_numbers,
        "reference_unique_content_tokens": len(ref_tokens),
        "shared_content_tokens": len(overlap_tokens),
        "lexical_content_token_coverage": round(coverage, 4) if coverage is not None else None,
        "interpretation": (
            "Advisory flag only. Do not automatically remove or rewrite the reference; "
            "manually inspect the original dataset record and source."
        ),
    }
