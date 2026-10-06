"""Deterministic evidence selection; no model can add factual text."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ABSTENTION = "I could not find an answer to this question in the provided documents."
GATE_VERSION = "verbatim-cuad-evidence-v2"
MAX_EVIDENCE_SENTENCES = 6
MAX_QUESTION_LENGTH = 2000

_STOP_WORDS = frozenset("""
a an the of to in on at from by with for and or is are was were be been being
do does did have has had can could may might will would shall should must
what which who whom whose when where how why me my i you your we our they their
it its this that these those there here please tell explain describe summarize
summary identify list show find provide about apply applies applicable according
contract contracts agreement agreements document documents provided indexed
clause clauses provision provisions section sections audit auditing under
requirements requirement obligations obligation rights right terms term detail
details specify specifies specified state states stated say says said include
includes included contain contains contained any all other related regarding
much many long often happen happens occur occurs get set period periods
condition conditions procedure process processes permitted allowed possible
require requires required name names named
""".split())

# Explicit vocabulary normalization preserves entities, quantities and modifiers.
_ALIASES = {
    "termination": "terminat", "terminate": "terminat", "terminated": "terminat",
    "terminating": "terminat", "terminates": "terminat",
    "renewal": "renew", "renewable": "renew", "renewed": "renew", "renew": "renew",
    "payment": "pay", "payments": "pay", "paid": "pay", "payable": "pay", "pay": "pay",
    "governing": "law", "governed": "law", "law": "law", "laws": "law",
    "limitation": "limit", "limitations": "limit", "limited": "limit", "limits": "limit",
    "cap": "limit", "caps": "limit", "capped": "limit", "liable": "liability",
    "liabilities": "liability", "parties": "party",
    "confidentiality": "confidential", "confidential": "confidential",
    "exclusivity": "exclusive", "exclusive": "exclusive",
    "assignment": "assign", "assigned": "assign", "assigns": "assign", "assign": "assign",
    "indemnification": "indemn", "indemnify": "indemn", "indemnities": "indemn",
    "indemnity": "indemn", "insurance": "insur", "insured": "insur",
    "licensing": "license", "licence": "license", "licenses": "license",
    "warranties": "warranty", "warranted": "warranty", "warrants": "warranty", "warrant": "warranty",
    "restrictions": "restriction", "restricted": "restriction",
    "days": "day", "years": "year", "months": "month", "weeks": "week", "hours": "hour",
    "competition": "compet", "competitive": "compet", "compete": "compet",
    "solicitation": "solicit", "soliciting": "solicit", "solicit": "solicit",
    "renewals": "renew", "employees": "employee", "customers": "customer",
    "beneficiaries": "beneficiary", "restrictions": "restriction",
    "commitments": "commitment", "profits": "profit", "revenues": "revenue",
    "sharing": "share", "disparagement": "disparage", "disparaging": "disparage",
    "irrevocability": "irrevocable", "perpetuity": "perpetual",
}
_LEGAL_TOPICS = frozenset("""
terminat renew pay law limit liability party confidential exclusive assign indemn
insur license warranty compet solicit notice breach damages fees royalty royalties
price prices pricing duration expiration expiry effective indemnitor indemnified
distributor distribution supplier customer service services product products
intellectual property copyright trademark patent patents ownership consent
arbitration dispute disputes jurisdiction venue amendment amendments transfer
restriction restrictions covenant noncompete nonsolicit commitment commitments
revenue revenues profit profits purchase purchases sale sales delivery deliver
interest penalty penalties deposit deposits guarantee guarantees guarantor
survival survive surviving escrow security securities affiliate affiliates
subcontract subcontractor audits access approval approvals consent
""".split())
# Dominant cues for the complete CUAD taxonomy. Generic words such as party,
# assignment, and license do not select a narrower, unrelated classification.
_CUAD_TOPIC_CUES = {
    "documentname": {"document", "name"},
    "parties": {"party"},
    "agreementdate": {"date"},
    "effectivedate": {"effective"},
    "expirationdate": {"expiration", "expiry", "expires"},
    "renewalterm": {"renew"},
    "noticeperiodtoterminaterenewal": {"notice"},
    "governinglaw": {"law"},
    "mostfavorednation": {"favored", "mfn"},
    "noncompete": {"compet", "noncompete"},
    "exclusivity": {"exclusive"},
    "nosolicitofcustomers": {"solicit", "nonsolicit"},
    "competitiverestrictionexception": {"compet", "exception"},
    "nosolicitofemployees": {"solicit", "nonsolicit"},
    "nondisparagement": {"disparage", "nondisparagement"},
    "terminationforconvenience": {"terminat"},
    "rofrroforofn": {"rofr", "rofo", "rofn", "refusal", "offer", "negotiation"},
    "changeofcontrol": {"control"},
    "antiassignment": {"assign"},
    "revenueprofitsharing": {"revenue", "profit", "share"},
    "pricerestrictions": {"price", "pricing"},
    "minimumcommitment": {"commitment", "minimum"},
    "volumerestriction": {"volume"},
    "ipownershipassignment": {"ip", "intellectual", "ownership"},
    "jointipownership": {"ip", "intellectual", "ownership", "joint"},
    "licensegrant": {"license"},
    "nontransferablelicense": {"nontransferable", "transferable"},
    "affiliatelicenselicensor": {"affiliate", "licensor"},
    "affiliatelicenselicensee": {"affiliate", "licensee"},
    "unlimitedallyoucaneatlicense": {"unlimited"},
    "irrevocableorperpetuallicense": {"irrevocable", "perpetual"},
    "sourcecodeescrow": {"escrow", "code"},
    "postterminationservices": {"post"},
    "auditrights": {"audit", "audits"},
    "uncappedliability": {"uncapped", "unlimited"},
    "caponliability": {"liability", "limit"},
    "liquidateddamages": {"liquidated", "damages"},
    "warrantyduration": {"warranty"},
    "insurance": {"insur"},
    "covenantnottosue": {"sue", "covenant"},
    "thirdpartybeneficiary": {"beneficiary"},
}
_EXPLICIT_ANNOTATION_REQUEST = re.compile(
    r"\b(?:(?:document|contract|agreement)\s+name|audit\s+rights?)\b", re.I
)
# A topic title is not a statement of a contractual requirement. These cues
# identify a proposition conservatively; omitted table fragments can abstain.
_ASSERTION_CUES = re.compile(
    r"\b(?:shall|must|may|will|is|are|was|were|has|have|had|not|no|"
    r"agrees?|agreed|grants?|granted|appoints?|accepts?|provides?|provided|providing|"
    r"requires?|required|requiring|permitted|prohibited|governed|construed|"
    r"expires?|survives?|continues?|maintains?|maintained|remain(?:s|ed)?|"
    r"includes?|including|excludes?|excluded|covers?|covering|payable|"
    r"applies|applied|renews?|renewable|entitled|retains?|owns?|obligated)\b", re.I
)


_DOCUMENT_WORDS = re.compile(r"\b(?:contracts?|agreements?|documents?|clauses?|provisions?)\b", re.I)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_RESERVED_CITATION = re.compile(r"\[\s*Source\b[^\]]*\]", re.I)


def normalize_question(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def query_terms(query: str) -> set[str]:
    tokens = (_ALIASES.get(token, token) for token in _TOKEN.findall(query.casefold()))
    return {token for token in tokens
            if token not in _STOP_WORDS and (len(token) > 1 or token.isdecimal())}


def _annotation_query_terms(query: str) -> set[str]:
    terms = query_terms(query)
    if _EXPLICIT_ANNOTATION_REQUEST.search(query):
        raw = set(_TOKEN.findall(query.casefold()))
        if "name" in raw:
            terms.update({"document", "name"})
        if "audit" in raw or "audits" in raw:
            terms.add("audit")
    return terms


def canonical_embedding_query(query: str) -> str:
    """Use the same explicit vocabulary for geometry and evidence coverage.

    Question verbs and legal synonyms should not move identical intents across
    the cosine cutoff. Entities, quantities and modifiers remain present.
    """
    natural_words = {
        "terminat": "termination", "renew": "renewal", "pay": "payment",
        "law": "governing law", "limit": "limitation", "insur": "insurance",
        "exclusive": "exclusivity", "assign": "assignment", "indemn": "indemnification",
        "compet": "competition", "solicit": "solicitation",
    }
    return " ".join(natural_words.get(term, term)
                    for term in sorted(_annotation_query_terms(query))) or normalize_question(query)


def in_contract_scope(query: str) -> bool:
    terms = query_terms(query)
    annotation_cues = set().union(*_CUAD_TOPIC_CUES.values())
    return bool((terms and (terms & (_LEGAL_TOPICS | annotation_cues)
                           or _DOCUMENT_WORDS.search(query)))
                or _EXPLICIT_ANNOTATION_REQUEST.search(query))


def _annotation_key(clause_type: str) -> str:
    return "".join(_TOKEN.findall(clause_type.casefold()))


def applicable_annotations(query: str, chunk: dict) -> list[dict] | None:
    """Expose independent CUAD type filtering before semantic top-k selection.

    None preserves unannotated legacy retrieval; an empty list fails closed for
    prepared chunks without applicable, well-formed, exact original evidence.
    """
    if "annotated_evidence" not in chunk:
        return None
    encoded = chunk["annotated_evidence"]
    text = chunk.get("text")
    if not isinstance(encoded, str) or not isinstance(text, str) or _RESERVED_CITATION.search(text):
        return []
    try:
        annotations = json.loads(encoded)
    except (ValueError, TypeError):
        return []
    if not isinstance(annotations, list):
        return []
    terms = _annotation_query_terms(query)
    accepted = []
    seen = set()
    for annotation in annotations:
        if (not isinstance(annotation, dict)
                or not isinstance(annotation.get("clause_type"), str)
                or not isinstance(annotation.get("text"), str)
                or not annotation["text"].strip()
                or annotation["text"] not in text):
            return []
        key = _annotation_key(annotation["clause_type"])
        cues = _CUAD_TOPIC_CUES.get(key)
        if cues is None or not terms & cues:
            continue
        identity = (annotation["clause_type"], annotation["text"])
        if identity not in seen:
            seen.add(identity)
            accepted.append({"clause_type": identity[0], "text": identity[1]})
    accepted.sort(key=lambda item: (text.find(item["text"]), item["clause_type"], item["text"]))
    return accepted


def evidence_sentences(query: str, chunk: dict) -> list[str]:
    """Use complete original clauses; labels never supply entities or quantities."""
    annotations = applicable_annotations(query, chunk)
    if annotations is None:
        return _legacy_evidence_sentences(query, chunk)
    if not annotations:
        return []
    text = chunk["text"]
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(
        r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n\s*\n", text
    ))
    boundaries.append(len(text))
    spans = list(zip(boundaries, boundaries[1:]))
    emitted = []
    covered_topics = set()
    for annotation in annotations:
        selected = []
        for index, (start, end) in enumerate(spans):
            if index == 0 and not chunk.get("starts_at_boundary", False):
                continue
            if index == len(spans) - 1 and not chunk.get("ends_at_boundary", False):
                continue
            sentence = text[start:end].strip()
            if sentence and sentence in annotation["text"]:
                selected.append(sentence)
        if not selected or not any(_ASSERTION_CUES.search(sentence) for sentence in selected):
            continue
        covered_topics.update(query_terms(annotation["clause_type"]) & _LEGAL_TOPICS)
        if _annotation_key(annotation["clause_type"]) == "documentname":
            covered_topics.update({"document", "name"})
        if _annotation_key(annotation["clause_type"]) == "auditrights":
            covered_topics.add("audit")
        for sentence in selected:
            if sentence not in emitted:
                emitted.append(sentence)
    terms = _annotation_query_terms(query)
    source_terms = {term for term in query_terms(str(chunk.get("source", "")))
                    if not term.isdecimal()}
    # Classification supports implicit topic vocabulary only. All remaining
    # modifiers, entities and quantities must occur in the exact emitted text.
    covered = covered_topics | source_terms | set().union(*(query_terms(s) for s in emitted))
    if not emitted or not terms or not terms.issubset(covered):
        return []
    # Do not truncate an independently labeled clause: a later sentence can
    # contain an exception, qualification, or negation necessary to its meaning.
    return emitted


def _legacy_evidence_sentences(query: str, chunk: dict) -> list[str]:
    """Return complete exact substrings only when all query terms have evidence."""
    text = chunk.get("text", "")
    if not isinstance(text, str) or not text.strip() or _RESERVED_CITATION.search(text):
        return []
    terms = query_terms(query)
    # A quantity in a filename must never support an amount/date in a question.
    source_terms = {term for term in query_terms(str(chunk.get("source", "")))
                    if not term.isdecimal()}
    if not terms or not terms.issubset(query_terms(text) | source_terms):
        return []
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(
        r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n\s*\n", text
    ))
    boundaries.append(len(text))
    evidence_terms = terms - source_terms or terms
    sentences = []
    spans = list(zip(boundaries, boundaries[1:]))
    for index, (start, end) in enumerate(spans):
        # Partial window edges can omit a negation or condition from a clause.
        # Unknown legacy boundaries are treated conservatively as incomplete.
        if index == 0 and not chunk.get("starts_at_boundary", False):
            continue
        if index == len(spans) - 1 and not chunk.get("ends_at_boundary", False):
            continue
        sentence = text[start:end].strip()
        if len(sentence) >= 12 and query_terms(sentence) & evidence_terms:
            sentences.append(sentence)
    sentences = sentences[:MAX_EVIDENCE_SENTENCES]
    covered = source_terms | set().union(*(query_terms(s) for s in sentences))
    return sentences if sentences and terms.issubset(covered) else []


def eligible_chunks(query: str, chunks: list[dict], threshold: float) -> list[dict]:
    if not in_contract_scope(query) or not math.isfinite(threshold) or not 0 < threshold < 2:
        return []
    eligible = []
    seen_ids: set[str] = set()
    seen_sentences: set[str] = set()

    def sort_key(chunk: dict) -> tuple[float, str]:
        try:
            distance = float(chunk.get("cosine_distance", math.inf))
        except (TypeError, ValueError):
            distance = math.inf
        return (distance if math.isfinite(distance) else math.inf,
                str(chunk.get("chunk_id", "")))

    for chunk in sorted(chunks, key=sort_key):
        distance, _ = sort_key(chunk)
        chunk_id = chunk.get("chunk_id")
        # Equality, missing metrics, NaN, bad IDs and negative distances fail closed.
        if not isinstance(chunk_id, str) or not chunk_id or not 0 <= distance < threshold:
            continue
        sentences = evidence_sentences(query, chunk)
        if not sentences or chunk_id in seen_ids:
            continue
        fresh = [s for s in sentences if s not in seen_sentences]
        if not fresh:
            continue
        if "annotated_evidence" not in chunk:
            sentences = fresh
        seen_ids.add(chunk_id)
        seen_sentences.update(sentences)
        eligible.append({**chunk, "cosine_distance": distance,
                         "evidence_sentences": sentences})
    return eligible


def render_evidence(chunks: list[dict]) -> tuple[str, list[dict]]:
    paragraphs = []
    sources = []
    for number, chunk in enumerate(chunks, 1):
        label = f"[Source {number}]"
        sources.append({
            "label": label, "citation_id": f"source-{number}",
            "chunk_id": chunk["chunk_id"], "source": chunk.get("source", "unknown"),
            "chunk_index": chunk.get("chunk_index", -1),
            "contract_category": chunk.get("contract_category", "Unknown"),
            "cosine_distance": chunk["cosine_distance"],
            "cosine_similarity": 1.0 - chunk["cosine_distance"],
            "rerank_score": 1.0 - chunk["cosine_distance"], "text": chunk["text"],
        })
        for sentence in chunk["evidence_sentences"]:
            if sentence not in chunk["text"] or _RESERVED_CITATION.search(sentence):
                raise ValueError("Evidence must be verbatim and contain no reserved citations")
            paragraphs.append(f"{sentence} {label}")
    return ("\n\n".join(paragraphs), sources) if sources else (ABSTENTION, [])


def policy_signature(top_k: int, max_sources: int) -> dict:
    vocabulary = json.dumps({"stop": sorted(_STOP_WORDS), "aliases": _ALIASES,
                             "topics": sorted(_LEGAL_TOPICS),
                             "cuad_topics": {key: sorted(value) for key, value in _CUAD_TOPIC_CUES.items()}}, sort_keys=True)
    return {
        "gate_version": GATE_VERSION,
        "gate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pipeline_source_sha256": hashlib.sha256(
            Path(__file__).with_name("rag_pipeline.py").read_bytes()
        ).hexdigest(),
        "vocabulary_sha256": hashlib.sha256(vocabulary.encode()).hexdigest(),
        "metric": "cosine", "top_k": top_k, "max_sources": max_sources,
        "max_evidence_sentences": MAX_EVIDENCE_SENTENCES,
        "annotated_evidence_truncation": "never", "answer_mode": "extractive",
    }


def load_calibration(path: Path, corpus_fingerprint: str, embedding_signature: object,
                     policy: dict) -> tuple[str, float | None]:
    if not path.is_file():
        return "required", None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(artifact["max_cosine_distance"])
        if (artifact.get("schema_version") != 1 or not math.isfinite(threshold)
                or not 0 < threshold < 2
                or artifact.get("corpus_fingerprint") != corpus_fingerprint
                or artifact.get("embedding_signature") != embedding_signature
                or artifact.get("policy") != policy
                or artifact.get("validation_passed") is not True
                or not artifact.get("calibration_case_count", 0)
                or not artifact.get("validation_case_count", 0)):
            return "stale", None
        return "calibrated", threshold
    except (OSError, ValueError, KeyError, TypeError):
        return "stale", None

