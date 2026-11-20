import json
import re
import unicodedata


# Corpus-level calibration from the 1,012-example XL-Sum Bengali validation split.
# These are flexible reference-distribution observations, not per-article quotas.
SUMMARY_LENGTH_GUIDANCE = """
XL-Sum Bengali validation split-এর পর্যবেক্ষিত reference-summary পরিসংখ্যান:
- নমুনা: ১,০১২টি
- whitespace-separated word median: ২১
- ৭৫তম percentile: ২৬
- ৯০তম percentile: ৩১
- observed range: ৭–৪৩ শব্দ
- sentence median: ১; p75/p90: ২টি sentence

এগুলো corpus-level calibration; সব article-এর জন্য একই word count বা hard cap নয়।
Article-এর কেন্দ্রীয় সংবাদ, অপরিহার্য তথ্যের সংখ্যা এবং সম্পর্ক অনুযায়ী দৈর্ঘ্য বেছে নিন।
ব্যবহারিক band (heuristic, measured conditional distribution নয়):
- একটি কেন্দ্রীয় ঘটনা/তথ্য: সাধারণত ৭–২১ শব্দ, ১টি sentence।
- ২–৩টি ঘনিষ্ঠভাবে সম্পর্কিত তথ্য: সাধারণত ২২–২৬ শব্দ, ১–২টি sentence।
- একাধিক স্বতন্ত্র গুরুত্বপূর্ণ তথ্য: সাধারণত ২৭–৩১ শব্দ, প্রয়োজনে ২টি sentence।
- ৩২–৪৩ শব্দ কেবল তখনই, যখন একাধিক অপরিহার্য তথ্য বাদ না দিয়ে সংক্ষেপ করা যায় না।

সংক্ষিপ্ততার নিয়ম:
1. প্রথমে article-এর প্রধান সংবাদ/মূল ঘটনা শনাক্ত করুন; article-এর প্রতিটি উপ-বিষয় তালিকাভুক্ত করবেন না।
2. প্রধান ঘটনা বোঝাতে প্রয়োজনীয় supporting fact ছাড়া জীবনী, দীর্ঘ পটভূমি বা গৌণ তথ্য যোগ করবেন না।
3. একটি sentence-এ অর্থ অক্ষুণ্ণ থাকলে অকারণে দুই বা তার বেশি sentence লিখবেন না।
4. ৩১ শব্দের বেশি হলে প্রতিটি clause অপরিহার্য কি না নিজে যাচাই করুন; অপ্রয়োজনীয় clause বাদ দিন।
5. ৩১ শব্দ অতিক্রম করা নিষিদ্ধ নয়; তবে অতিরিক্ত দৈর্ঘ্যের স্পষ্ট তথ্যগত কারণ থাকতে হবে।
6. শুধু target পূরণ করতে শব্দ যোগ করবেন না এবং সংক্ষিপ্ত করতে গিয়ে মূল তথ্য বিকৃত করবেন না।
"""

QUALIFIER_PRESERVATION_GUIDANCE = """
সংখ্যা ও qualifier সংরক্ষণের বাধ্যতামূলক পরীক্ষা:
- কমপক্ষে, অন্তত, প্রায়/প্রায়, আনুমানিক, সর্বোচ্চ, অনধিক, সর্বনিম্ন, মাত্র,
  এর বেশি/এর কম, প্রায় ... মতো, পর্যন্ত—এসব qualifier থাকলে সংশ্লিষ্ট সংখ্যা/পরিমাণের
  সঙ্গে তাদের অর্থও সংরক্ষণ করুন।
- at least, at most, approximately, about, around, up to, more than, less than,
  over, under, nearly, almost—একইভাবে সংরক্ষণ করুন।
- 'কমপক্ষে ৩০০০' কে শুধু '৩০০০' করবেন না; 'প্রায় ৫০০' কে নির্ভুল '৫০০' হিসেবে লিখবেন না।
- তারিখ, সময়, পরিমাণ, পরিসর, শতাংশ, তুলনা, negation এবং attribution বদলাবেন না।
- 'বিক্রি বেড়েছে' থেকে '২০১৪ সাল থেকেই বিক্রি বাড়তে শুরু করেছে'—এমন নতুন temporal
  relationship তৈরি করবেন না, যদি source তা স্পষ্টভাবে না বলে।
- source-এ 'নেতা' বলা হলে প্রমাণ ছাড়া 'প্রধান নেতা' লিখবেন না।
- কোনো সংখ্যা বা qualifier summary-তে না রাখলে সেটি বাদ দেওয়া কেন্দ্রীয় তথ্যের অর্থ বদলায় কি না যাচাই করুন।
"""

BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

QUALIFIER_FAMILIES = {
    "lower_bound": ["কমপক্ষে", "অন্তত", "at least", "no fewer than"],
    "upper_bound": ["সর্বোচ্চ", "অনধিক", "at most", "up to"],
    "approximate": ["প্রায়", "প্রায়", "আনুমানিক", "মতো", "প্রায়ই", "about", "around", "approximately", "nearly", "almost"],
    "greater_than": ["এর বেশি", "এরও বেশি", "বেশি", "অধিক", "more than", "over", "greater than"],
    "less_than": ["এর কম", "কম", "less than", "under", "fewer than"],
}


def summary_length_profile(text):
    text = str(text or "").strip()
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[।!?])\s+|(?<=[.!?])\s+|\n+", text)
        if part.strip()
    ]
    return {
        "characters": len(text),
        "whitespace_words": len(text.split()),
        "sentences": len(sentences),
    }


def length_review_diagnostics(text):
    profile = summary_length_profile(text)
    count = profile["whitespace_words"]
    return {
        **profile,
        "p90_review_threshold_words": 31,
        "observed_validation_max_words": 43,
        "p90_review_required": count > 31,
        "above_observed_validation_max": count > 43,
        "interpretation": (
            "Review each clause for necessity; not an automatic error or hard cap."
            if count > 31 else "Within the observed p90 word-count threshold."
        ),
    }


def make_length_review_issue(text):
    """Create a deterministic, non-factual review cue when Python counts >31 words.

    The p90 threshold prompts a clause-level review; it is not a hard maximum.
    """
    profile = summary_length_profile(text)
    if profile["whitespace_words"] <= 31:
        return None
    return {
        "issue_category": "conciseness / deterministic length review",
        "summary_span": str(text or "").strip(),
        "problem": (
            f"Python measured {profile['whitespace_words']} whitespace-separated words, "
            "above the validation p90 review threshold of 31. Review each clause for "
            "necessity; this is not an automatic error or hard cap."
        ),
        "proposed_correction": (
            "Remove only redundant or secondary clauses if they can be removed without "
            "losing the central event, essential facts, qualifiers, dates, attribution, "
            "or temporal relationships. If every clause is necessary, preserve the summary."
        ),
        "source_sentence_ids": [],
        "source_quote": "",
        "suggested_action": "review_and_compress_if_redundant",
        "origin": "deterministic_length_review",
        "measured_word_count": profile["whitespace_words"],
        "review_threshold_words": 31,
    }


def _normalize_digits(text):
    return unicodedata.normalize("NFC", str(text or "")).translate(BENGALI_DIGIT_MAP)


def split_source_sentences(text):
    raw_parts = re.split(r"(?<=[।!?])\s+|(?<=[.!?])\s+|\n+", text or "")
    sentences = [p.strip() for p in raw_parts if p.strip()]
    return {f"S{i:03d}": sentence for i, sentence in enumerate(sentences, start=1)}


def numbered_source(text):
    sentence_map = split_source_sentences(text)
    numbered = "\n".join(f"[{sid}] {sentence}" for sid, sentence in sentence_map.items())
    return numbered, sentence_map


def _qualifier_number_mentions(text):
    """Find explicit digit-based number + qualifier pairs for a conservative audit."""
    original = unicodedata.normalize("NFC", str(text or ""))
    normalized = _normalize_digits(original)
    number = r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    found = []
    phrases = []
    for family, values in QUALIFIER_FAMILIES.items():
        for phrase in values:
            phrases.append((family, phrase))

    # Qualifier before number: কমপক্ষে ৩০০০, at least 3,000.
    for family, phrase in phrases:
        pattern = re.compile(
            rf"(?P<qualifier>{re.escape(phrase)})\s*[:：-]?\s*{number}",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(normalized):
            start, end = match.span()
            found.append({
                "family": family,
                "qualifier": match.group("qualifier"),
                "number": match.group("number").replace(",", ""),
                "span": original[start:end],
                "start": start,
                "end": end,
            })

    # Qualifier after number: ৩০০০-এর বেশি, 500 or more, 500-এর মতো.
    suffixes = []
    for family, values in QUALIFIER_FAMILIES.items():
        for phrase in values:
            if phrase in {"কমপক্ষে", "অন্তত", "সর্বোচ্চ", "অনধিক", "প্রায়", "প্রায়", "আনুমানিক", "at least", "at most", "approximately", "about", "around", "nearly", "almost", "up to", "no fewer than"}:
                continue
            suffixes.append((family, phrase))
    for family, phrase in suffixes:
        pattern = re.compile(
            rf"{number}\s*(?:[-–—]?\s*(?:টি|জন|বার|ভাগ|শতাংশ|টাকা|ডলার))?\s*(?P<qualifier>{re.escape(phrase)})",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(normalized):
            start, end = match.span()
            found.append({
                "family": family,
                "qualifier": match.group("qualifier"),
                "number": match.group("number").replace(",", ""),
                "span": original[start:end],
                "start": start,
                "end": end,
            })

    # Deduplicate overlapping regex matches.
    unique = {}
    for item in found:
        key = (item["family"], item["number"], item["start"], item["end"])
        unique[key] = item
    return sorted(unique.values(), key=lambda x: (x["start"], x["end"]))


def find_qualifier_preservation_flags(source, summary):
    """Flag a source qualifier when the same digit-based number is in the summary
    but its qualifier family is absent nearby. This is a conservative heuristic,
    not a complete Bangla numerical-semantic parser.
    """
    source_mentions = _qualifier_number_mentions(source)
    summary_normalized = _normalize_digits(summary)
    summary_mentions = _qualifier_number_mentions(summary)
    flags = []

    for mention in source_mentions:
        number = mention["number"]
        number_pattern = re.compile(rf"(?<!\d){re.escape(number)}(?!\d)")
        matches = list(number_pattern.finditer(summary_normalized))
        if not matches:
            continue
        preserved = any(
            any(
                candidate["family"] == mention["family"]
                and candidate["number"] == number
                and abs(candidate["start"] - number_match.start()) <= 48
                for candidate in summary_mentions
            )
            for number_match in matches
        )
        if preserved:
            continue
        match = matches[0]
        context_start = max(0, match.start() - 24)
        context_end = min(len(summary_normalized), match.end() + 24)
        flags.append({
            "flag_type": "possible_qualifier_loss",
            "qualifier_family": mention["family"],
            "source_qualifier": mention["qualifier"],
            "number": number,
            "source_quote": mention["span"],
            "summary_span": summary_normalized[context_start:context_end].strip(),
            "message": (
                f"Source qualifies number {number} as '{mention['qualifier']}', "
                "but the same number appears in the summary without a nearby equivalent qualifier."
            ),
        })
    return flags


def find_sentence_ids_for_quote(sentence_map, quote):
    quote_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(quote or ""))).strip()
    matches = []
    for sid, sentence in sentence_map.items():
        sentence_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", sentence)).strip()
        if quote_norm and quote_norm in sentence_norm:
            matches.append(sid)
    if matches:
        return matches
    # A quote may span adjacent source sentences.
    ordered = list(sentence_map.items())
    for i in range(len(ordered)):
        for j in range(i + 1, min(i + 4, len(ordered)) + 1):
            span = " ".join(x[1] for x in ordered[i:j])
            span_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", span)).strip()
            if quote_norm and quote_norm in span_norm:
                return [x[0] for x in ordered[i:j]]
    return []


def make_qualifier_issues(flags, sentence_map):
    issues = []
    for flag in flags:
        quote = flag.get("source_quote", "")
        issues.append({
            "issue_category": "numerical qualifier omission",
            "summary_span": flag.get("summary_span", ""),
            "problem": flag.get("message", "Possible numerical qualifier loss."),
            "proposed_correction": (
                f"Preserve the source qualifier '{flag.get('source_qualifier')}' "
                f"with the number {flag.get('number')}."
            ),
            "source_sentence_ids": find_sentence_ids_for_quote(sentence_map, quote),
            "source_quote": quote,
            "suggested_action": "correct",
            "origin": "deterministic_qualifier_audit",
        })
    return issues


def format_qualifier_flags(flags):
    if not flags:
        return "কোনো deterministic numerical-qualifier flag পাওয়া যায়নি।"
    return json.dumps(flags, ensure_ascii=False, indent=2)


def article_prompt_text(row):
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    return f"শিরোনাম: {title}\n\nমূল সংবাদ:\n{text}"


def initial_summary_prompt(article):
    return f"""আপনি একজন পেশাদার বাংলা সংবাদ-সারসংক্ষেপ লেখক।
নিচের সংবাদটির একটি সংক্ষিপ্ত, তথ্যনিষ্ঠ, স্বাভাবিক বাংলা abstractive summary লিখুন।

কাজের পদ্ধতি:
1. প্রথমে article-এর প্রধান সংবাদ/মূল ঘটনা শনাক্ত করুন।
2. প্রধান ঘটনা বোঝাতে অপরিহার্য তথ্য এবং প্রয়োজন হলে ১টি supporting fact বেছে নিন।
3. article-এর সব উপ-বিষয়, দীর্ঘ জীবনী বা গৌণ পটভূমি এক summary-তে ঢোকাবেন না।
4. একটি sentence-এ কেন্দ্রীয় তথ্য বোঝানো গেলে অকারণে দ্বিতীয় sentence যোগ করবেন না।

নিয়ম:
- শুধু মূল সংবাদে থাকা তথ্য ব্যবহার করুন; বাইরের জ্ঞান যোগ করবেন না।
- ব্যক্তি, প্রতিষ্ঠান, স্থান, তারিখ, সংখ্যা, qualifier, negation এবং attribution ঠিক রাখুন।
- কোনো temporal, causal বা comparative relationship নিজে থেকে তৈরি করবেন না।
- 'নেতা'কে 'প্রধান নেতা' বা 'বিক্রি বেড়েছে'কে 'ওই বছর থেকেই বিক্রি বাড়তে শুরু করেছে' করবেন না, যদি source তা না বলে।
- স্বাভাবিক, প্রমিত ও সাবলীল বাংলা ব্যবহার করুন।
- শুধু summary text দিন; heading, label, ব্যাখ্যা বা মূল্যায়ন নয়।

{SUMMARY_LENGTH_GUIDANCE}
{QUALIFIER_PRESERVATION_GUIDANCE}

{article}

বাংলা সারসংক্ষেপ:
"""


def conventional_feedback_prompt(article, summary, qualifier_flags=None):
    flags = qualifier_flags or []
    length_info = length_review_diagnostics(summary)
    return f"""আপনি নিজের তৈরি বাংলা সংবাদ-সারসংক্ষেপের কঠোর কিন্তু ন্যায্য reviewer।
মূল সংবাদ ও summary তুলনা করে কেবল বাস্তব, সংশোধনযোগ্য সমস্যা শনাক্ত করুন।

অবশ্যই পরীক্ষা করুন:
- factual consistency এবং unsupported claim
- গুরুত্বপূর্ণ তথ্য বাদ পড়েছে কি না
- ব্যক্তি/প্রতিষ্ঠান/স্থান/তারিখ/সংখ্যা/শতাংশের ভুল
- কমপক্ষে/প্রায়/সর্বোচ্চ/এর বেশি/এর কমের মতো qualifier হারিয়েছে কি না
- তারিখ, সময়, causal/temporal relation, negation ও attribution বদলেছে কি না
- বাংলা ব্যাকরণ, স্বাভাবিকতা, redundancy ও অপ্রয়োজনীয় পটভূমি
- summary অতিরিক্ত দীর্ঘ বা অযথা সংক্ষিপ্ত কি না

{SUMMARY_LENGTH_GUIDANCE}
{QUALIFIER_PRESERVATION_GUIDANCE}

এই summary-এর length diagnostics:
{json.dumps(length_info, ensure_ascii=False, indent=2)}

Rule-based qualifier audit flags (এগুলো source-এর digit-based heuristic; যাচাই করে ব্যবহার করুন):
{format_qualifier_flags(flags)}

শুধু বাস্তবসম্মত issue লিখুন। কোনো flag ভুল হলে তা issue হিসেবে গ্রহণ করবেন না।
যদি flag সঠিক হয়, সংশ্লিষ্ট qualifier অবশ্যই preserve করার পরামর্শ দিন।
NO_ACTIONABLE_ISSUES কেবল তখনই লিখবেন, যখন কোনো factual, coverage, qualifier, language বা conciseness issue নেই।
অন্যথায় numbered actionable bullet দিন।

মূল সংবাদ:
{article}

বর্তমান summary:
{summary}

Feedback:
"""


def conventional_refine_prompt(article, summary, feedback, qualifier_flags=None):
    flags = qualifier_flags or []
    return f"""আপনি বাংলা সংবাদ-সারসংক্ষেপ সম্পাদনা করছেন।
মূল সংবাদ এবং feedback ব্যবহার করে বর্তমান summary-টি targetedভাবে উন্নত করুন।

নিয়ম:
1. source-এ সমর্থিত নয় এমন তথ্য যোগ করবেন না।
2. feedback-এর যথার্থ সমস্যাগুলোই সংশোধন করুন; ভুল flag অনুসরণ করবেন না।
3. নাম, সংখ্যা, qualifier, তারিখ, স্থান, negation, attribution ও সম্পর্ক সংরক্ষণ করুন।
4. temporal বা causal সম্পর্ক source-এ স্পষ্ট না থাকলে তৈরি করবেন না।
5. সঠিক তথ্য অকারণে rewrite করবেন না।
6. প্রধান সংবাদকে অগ্রাধিকার দিন; গৌণ জীবনী/পটভূমি বাদ দিন।
7. শুধু revised summary-এর মূল লেখা দিন; heading বা prefix নয়।
8. article-এর জটিলতা অনুযায়ী adaptive length বজায় রাখুন; শুধু ছোট করার জন্য জরুরি তথ্য বাদ দেবেন না।

{SUMMARY_LENGTH_GUIDANCE}
{QUALIFIER_PRESERVATION_GUIDANCE}

Rule-based qualifier audit flags:
{format_qualifier_flags(flags)}

মূল সংবাদ:
{article}

বর্তমান summary:
{summary}

Feedback:
{feedback}

Revised Bangla summary:
"""


def evidence_feedback_prompt(article, summary, sentence_map, qualifier_flags=None):
    numbered, _ = numbered_source(article)
    flags = qualifier_flags or []
    schema = {
        "overall_assessment": "short Bangla assessment",
        "length_assessment": "appropriate | overlong | too_short",
        "length_reason": "short Bangla reason; if over 31 words, assess each clause",
        "actionable_issues": [{
            "issue_category": "factual inconsistency | unsupported claim | information omission | entity/numerical error | numerical qualifier omission | temporal relation error | Bangla fluency | redundancy",
            "summary_span": "exact problematic span from the summary",
            "problem": "why it is a problem",
            "proposed_correction": "specific correction, or empty if not appropriate",
            "source_sentence_ids": ["S001"],
            "source_quote": "exact contiguous quotation from cited sentence(s); empty only for style-only issues",
            "suggested_action": "correct | add | remove | rewrite | preserve",
        }],
    }
    return f"""আপনি বাংলা সংবাদ-সারসংক্ষেপের একজন কঠোর, নির্ভুল এবং সতর্ক reviewer।
মূল সংবাদ ও summary তুলনা করে JSON feedback তৈরি করুন।

পরীক্ষার মাত্রা:
1. factual consistency ও unsupported claim
2. central information coverage
3. entity, numerical value ও qualifier preservation
4. temporal/causal/comparative relationship
5. negation এবং attribution
6. বাংলা fluency, redundancy ও conciseness
7. summary length-এর যৌক্তিকতা

{SUMMARY_LENGTH_GUIDANCE}
{QUALIFIER_PRESERVATION_GUIDANCE}

এই summary-এর length diagnostics:
{json.dumps(length_review_diagnostics(summary), ensure_ascii=False, indent=2)}

Rule-based qualifier audit flags:
{format_qualifier_flags(flags)}
প্রতিটি flag source-এর সাথে মিলিয়ে যাচাই করুন। সঠিক হলে actionable issue হিসেবে নিন; ভুল হলে বাদ দিন।

Evidence rules:
- factual, omission, entity, numerical, qualifier বা temporal issue-এর জন্য source_sentence_ids এবং হুবহু source_quote দিন।
- একাধিক sentence জুড়ে quote হলে সব সংশ্লিষ্ট sentence ID ক্রমানুসারে দিন।
- quote paraphrase করবেন না; quote exact না হলে বানিয়ে দেবেন না।
- style-only issue হলে source_sentence_ids=[] এবং source_quote="" দিন।
- issue-এর proposed_correction source থেকে সরাসরি সমর্থিত হতে হবে।
- quote থাকা মানেই claim সমর্থিত—এমন ধরে নেবেন না।
- নিশ্চিত না হলে issue তুলবেন না বা uncertainty problem-এ স্পষ্ট করুন।
- actionable issue না থাকলে actionable_issues=[] দিন।
- JSON ছাড়া অন্য কোনো লেখা দেবেন না।

JSON কাঠামো:
{json.dumps(schema, ensure_ascii=False, indent=2)}

মূল সংবাদ (sentence IDs সহ):
{numbered}

বর্তমান summary:
{summary}

JSON feedback:
"""


def semantic_support_prompt(article, summary, sentence_map, factual_issues):
    numbered, _ = numbered_source(article)
    items = []
    for index, issue in factual_issues:
        items.append({
            "issue_index": index,
            "issue_category": issue.get("issue_category", ""),
            "summary_span": issue.get("summary_span", ""),
            "problem": issue.get("problem", ""),
            "proposed_correction": issue.get("proposed_correction", ""),
            "source_sentence_ids": issue.get("source_sentence_ids", []),
            "source_quote": issue.get("source_quote", ""),
            "suggested_action": issue.get("suggested_action", ""),
        })
    schema = {
        "verdicts": [{
            "issue_index": 0,
            "original_claim_status": "supported | contradicted | not_stated | not_fully_supported | uncertain",
            "correction_status": "supported | contradicted | insufficient | uncertain",
            "semantic_equivalence": "equivalent | materially_different | not_applicable | uncertain",
            "decision": "approve_correction | preserve_original | reject_issue | uncertain",
            "reason": "brief Bangla explanation based only on the source",
        }]
    }
    return f"""আপনি একটি second-pass source-support এবং semantic-equivalence reviewer।
আপনার কাজ শুধু quote আছে কি না দেখা নয়। আপনাকে একই সঙ্গে যাচাই করতে হবে:
(A) বর্তমান summary-র original claim source দ্বারা সমর্থিত কি না,
(B) proposed correction source দ্বারা সমর্থিত কি না,
(C) original wording ও proposed correction অর্থগতভাবে একই কি না।

কঠোর নিয়ম:
1. শুধু supplied source ব্যবহার করুন; বাইরের জ্ঞান ব্যবহার করবেন না।
2. original summary span যদি source দ্বারা সমর্থিত হয় এবং proposed correction কেবল সমার্থক/শৈলীগত পুনর্লিখন হয়, semantic_equivalence='equivalent' এবং decision='preserve_original' দিন।
3. original claim ইতিমধ্যে সঠিক হলে শুধু অন্যভাবে বলা যায় বলে সেটি সংশোধন করবেন না।
4. original claim source-এর সঙ্গে সাংঘর্ষিক হলে original_claim_status='contradicted' দিন।
5. summary-তে কোনো অপরিহার্য তথ্য অনুপস্থিত থাকলে original_claim_status='not_stated' দিন।
6. source-এর qualifier বাদ পড়ে summary যদি অতিরিক্ত নির্দিষ্ট/অতিরঞ্জিত অর্থ দেয়, original_claim_status='not_fully_supported' দিন।
7. proposed correction source-এ স্পষ্টভাবে সমর্থিত না হলে correction_status='insufficient' বা 'uncertain' দিন।
8. factual correction অনুমোদন করবেন কেবল তখনই, যখন original claim 'contradicted', 'not_stated' অথবা 'not_fully_supported', correction 'supported', এবং proposed correction মূল অর্থে একটি বাস্তব সংশোধন/তথ্য সংযোজন। তখনই decision='approve_correction' দিন।
9. original wording ও proposed correction অর্থগতভাবে সমতুল্য হলে decision='preserve_original' দিন, কখনো approve_correction নয়।
10. ambiguity থাকলে 'uncertain' দিন। নিশ্চিত নন এমন correction অনুমোদন করবেন না।
11. exact quote source-এ থাকলেও তা পুরো proposed correction-কে সমর্থন নাও করতে পারে।
12. প্রতিটি issue-এর জন্য সব field দিন; JSON ছাড়া অন্য লেখা নয়।

মূল সংবাদ (sentence IDs সহ):
{numbered}

বর্তমান summary:
{summary}

যে issue-গুলোর source quote exact-match করেছে:
{json.dumps(items, ensure_ascii=False, indent=2)}

JSON কাঠামো:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Verdicts:
"""


def evidence_refine_prompt(article, summary, validated_feedback):
    safe_feedback = json.dumps(validated_feedback, ensure_ascii=False, indent=2)
    return f"""আপনি বাংলা সংবাদ-সারসংক্ষেপ সম্পাদনা করছেন।
মূল সংবাদ, বর্তমান summary এবং অনুমোদিত feedback ব্যবহার করে targeted revision করুন।

কঠোর নিয়ম:
1. factual correction কেবল তখনই অনুসরণ করবেন, যখন evidence_valid=true এবং semantic_support_approved=true।
2. style-only issue-তে source quote প্রয়োজন নেই, তবে অর্থ বদলানো যাবে না।
3. semantic support 'insufficient', 'contradicted', 'uncertain' বা verification failure হলে সেই factual issue ব্যবহার করবেন না।
4. source-এ নেই এমন তথ্য যোগ করবেন না।
5. নাম, সংখ্যা, qualifier, তারিখ, স্থান, negation, attribution এবং temporal/causal relationship সংরক্ষণ করুন।
6. প্রধান সংবাদকে অগ্রাধিকার দিন; গৌণ পটভূমি যোগ করবেন না।
7. সঠিক বাক্য অকারণে rewrite করবেন না।
8. শুধু final revised Bangla summary-এর মূল লেখা দিন; heading/prefix নয়।
9. যদি issue-এর origin 'deterministic_length_review' হয়, Python-এ মাপা word count-কে নির্ভরযোগ্য ধরুন; নিজের word-count অনুমান ব্যবহার করবেন না।
10. ৩১ শব্দের বেশি হলেই অন্ধভাবে ছোট করবেন না। প্রতিটি clause অপরিহার্য কি না যাচাই করুন; redundant/secondary clause থাকলে বাদ দিন। সব clause অপরিহার্য হলে summary অপরিবর্তিত রাখুন।
11. summary-র কেন্দ্রীয় ঘটনা, qualifier, সংখ্যা, তারিখ, attribution ও temporal relationship অক্ষুণ্ণ রাখুন।

{SUMMARY_LENGTH_GUIDANCE}
{QUALIFIER_PRESERVATION_GUIDANCE}

মূল সংবাদ:
{article}

বর্তমান summary:
{summary}

অনুমোদিত feedback:
{safe_feedback}

Revised Bangla summary:
"""


def clean_summary(raw):
    """Remove common model-generated summary labels without changing content."""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    label_pattern = (
        r"^(?:বাংলা\s*সারসংক্ষেপ|বাংলা\s*সারাংশ|"
        r"Revised\s+Bangla\s+summary|Bangla\s+summary|Summary)"
        r"\s*[:：]\s*"
    )
    text = re.sub(label_pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    return text


def parse_json_response(raw):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    raw = str(raw or "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {
        "overall_assessment": "JSON parsing failed; raw feedback retained.",
        "actionable_issues": [],
        "raw_feedback": raw,
        "parse_error": True,
    }


def normalize_quote(text):
    text = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def issue_requires_evidence(issue):
    category = str(issue.get("issue_category", "")).lower()
    factual_terms = (
        "fact", "unsupported", "omission", "entity", "number", "numerical",
        "attribution", "factual", "missing", "inaccur", "wrong", "claim",
        "qualifier", "temporal", "date", "সময়", "সময়", "তারিখ", "ভুল",
        "অসঙ্গতি", "তথ্য", "নাম", "সংখ্যা", "বাদ", "সময়গত",
    )
    return any(term in category for term in factual_terms)


def validate_evidence(parsed_feedback, sentence_map, full_article):
    article_norm = normalize_quote(full_article)
    checked = []
    issues = parsed_feedback.get("actionable_issues", [])
    if not isinstance(issues, list):
        issues = []

    for issue in issues:
        issue = dict(issue)
        needs_evidence = issue_requires_evidence(issue)
        quote = normalize_quote(issue.get("source_quote", ""))
        ids = issue.get("source_sentence_ids", [])
        if isinstance(ids, str):
            ids = [ids]
        ids = [str(x) for x in ids]

        if not needs_evidence:
            issue["requires_evidence"] = False
            issue["evidence_valid"] = True
            issue["evidence_validation_reason"] = "Style-only issue; quotation not required."
            issue["semantic_support_status"] = "not_required"
            issue["semantic_support_approved"] = True
        else:
            known_ids = sorted(
                {sid for sid in ids if sid in sentence_map},
                key=lambda sid: int(re.search(r"\d+", sid).group()) if re.search(r"\d+", sid) else 10**9,
            )
            quote_in_article = bool(quote) and quote in article_norm
            cited_span = normalize_quote(" ".join(sentence_map[sid] for sid in known_ids))
            quote_in_cited_sentence = bool(quote) and quote in cited_span
            valid = quote_in_article and quote_in_cited_sentence and bool(known_ids)
            issue["requires_evidence"] = True
            issue["evidence_valid"] = bool(valid)
            if valid:
                reason = "Exact quote found in the cited source sentence span."
                issue["semantic_support_status"] = "pending"
            elif not quote:
                reason = "No exact source quotation supplied."
                issue["semantic_support_status"] = "blocked_invalid_quote"
            elif not known_ids:
                reason = "No valid source sentence ID supplied."
                issue["semantic_support_status"] = "blocked_invalid_sentence_id"
            elif not quote_in_article:
                reason = "Quotation does not exactly occur in the article."
                issue["semantic_support_status"] = "blocked_quote_not_in_article"
            else:
                reason = "Quotation does not occur in the cited sentence span."
                issue["semantic_support_status"] = "blocked_quote_not_in_cited_span"
            issue["semantic_support_approved"] = False
            issue["evidence_validation_reason"] = reason
        checked.append(issue)

    parsed_feedback = dict(parsed_feedback)
    parsed_feedback["actionable_issues"] = checked
    parsed_feedback["evidence_validation_scope"] = (
        "Deterministic exact-quotation and cited-sentence presence only; this does not establish semantic support."
    )
    return parsed_feedback


def apply_semantic_verdicts(checked_feedback, verifier_result):
    """Attach support and equivalence verdicts; approve only genuine supported corrections."""
    feedback = dict(checked_feedback)
    issues = [dict(x) for x in feedback.get("actionable_issues", [])]
    verdicts = verifier_result.get("verdicts", []) if isinstance(verifier_result, dict) else []
    verdict_map = {}
    for verdict in verdicts if isinstance(verdicts, list) else []:
        try:
            verdict_map[int(verdict.get("issue_index"))] = verdict
        except (TypeError, ValueError):
            continue

    for index, issue in enumerate(issues):
        if not issue.get("requires_evidence"):
            issue["semantic_support_status"] = "not_required"
            issue["semantic_support_approved"] = True
            issue["semantic_equivalence_status"] = "not_applicable"
            continue

        if not issue.get("evidence_valid"):
            issue["semantic_support_status"] = "blocked_invalid_evidence"
            issue["semantic_support_approved"] = False
            issue["semantic_equivalence_status"] = "not_checked"
            continue

        verdict = verdict_map.get(index)
        if verdict is None:
            issue["semantic_support_status"] = "no_verdict"
            issue["semantic_support_approved"] = False
            issue["semantic_equivalence_status"] = "unknown"
            issue["semantic_support_reason"] = "Verifier did not return a verdict for this issue."
            continue

        original_status = str(
            verdict.get("original_claim_status", "uncertain")
        ).strip().lower()
        correction_status = str(
            verdict.get("correction_status", verdict.get("support_status", "uncertain"))
        ).strip().lower()
        equivalence = str(
            verdict.get("semantic_equivalence", "uncertain")
        ).strip().lower()
        decision = str(verdict.get("decision", "uncertain")).strip().lower()

        if original_status not in {"supported", "contradicted", "not_stated", "not_fully_supported", "uncertain"}:
            original_status = "uncertain"
        if correction_status not in {"supported", "contradicted", "insufficient", "uncertain"}:
            correction_status = "uncertain"
        if equivalence not in {"equivalent", "materially_different", "not_applicable", "uncertain"}:
            equivalence = "uncertain"
        if decision not in {"approve_correction", "preserve_original", "reject_issue", "uncertain"}:
            decision = "uncertain"

        # A correction is eligible only when the original is demonstrably wrong
        # or missing, the replacement is supported, and the verifier explicitly
        # distinguishes it from an equivalent paraphrase.
        genuine_problem = original_status in {"contradicted", "not_stated", "not_fully_supported"}
        non_equivalent = equivalence in {"materially_different", "not_applicable"}
        approved = (
            genuine_problem
            and correction_status == "supported"
            and non_equivalent
            and decision == "approve_correction"
        )

        issue["original_claim_status"] = original_status
        issue["correction_status"] = correction_status
        issue["semantic_equivalence_status"] = equivalence
        issue["semantic_verifier_decision"] = decision
        issue["semantic_support_status"] = (
            "supported" if approved else
            "blocked_equivalent_or_original_supported"
            if original_status == "supported" or equivalence == "equivalent" else
            "not_approved"
        )
        issue["semantic_support_approved"] = bool(approved)
        issue["semantic_support_reason"] = str(verdict.get("reason", ""))

    feedback["actionable_issues"] = issues
    feedback["semantic_support_scope"] = (
        "Second-pass assessment by the same configured LLM; checks original-claim status, "
        "correction support, and semantic equivalence. It is a screening signal, not "
        "independent ground truth or a formal entailment guarantee."
    )
    feedback["semantic_verifier_parse_error"] = bool(
        isinstance(verifier_result, dict) and verifier_result.get("parse_error")
    )
    return feedback

