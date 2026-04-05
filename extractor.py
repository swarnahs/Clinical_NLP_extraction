"""
extractor.py
------------
Clinical NLP extractor using only traditional NLP / rule-based methods.
No LLMs, no OpenAI, no transformers inference.

Techniques used:
  - Regex pattern matching for codes (ICD-10, CPT, HCPCS, modifiers)
  - Curated medical lexicons for entity recognition
  - Section-boundary detection via keyword anchors
  - Multi-word phrase matching with sliding n-gram windows
  - spaCy (optional, falls back gracefully if not installed)
"""

import re
import json
from typing import Any

# ── Optional spaCy import (graceful fallback) ─────────────────────────────────
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _NLP = None
    _SPACY_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# LEXICONS  — curated medical term dictionaries
# ═════════════════════════════════════════════════════════════════════════════

CLINICAL_TERMS_LEXICON: list[str] = [
    # GI conditions
    "diverticulosis", "diverticulitis", "sigmoid diverticulosis", "moderate sigmoid diverticulosis",
    "internal hemorrhoids", "external hemorrhoids", "hemorrhoids", "hemorrhage",
    "rectal bleeding", "rectal hemorrhage",
    "melanosis coli", "colon polyps", "colonic polyps", "sessile polyp", "polyp",
    "proctitis", "localized erosion", "erosion",
    "ulcer of anus and rectum", "ulcer", "gastritis", "antral gastritis", "body gastritis",
    "mild antral and body gastritis", "barrett's esophagus", "barrett esophagus",
    "colon cancer screening", "cancer screening",
    "polyp removal", "biopsy", "biopsy taken", "biopsies",
    "h.pylori", "helicobacter pylori", "h. pylori",
    # Findings / observations
    "irregular z-line", "z-line", "narrow band imaging",
    "no polyps found", "no immediate complications", "no immediate complication",
    "good bowel preparation", "good bowel prep", "bowel prep",
    "minimal estimated blood loss", "estimated blood loss",
    "no ulcers or masses", "no ulcers", "no masses",
    "sessile polyp", "polyp completely removed",
    # Symptoms
    "atypical chest pain", "chest pain",
    "right upper quadrant abdominal pain", "right upper quadrant pain", "ruq pain",
    "abdominal pain",
    # Procedure-related findings
    "rectal exam", "cold forceps", "cold snare", "retroflexion",
    "boston bowel preparation scale", "bbps",
    "monitored anesthesia care", "mac",
    "intravenous medication administration",
    "informed consent", "no immediate complications",
]

ANATOMICAL_LOCATIONS_LEXICON: list[str] = [
    "rectum", "distal rectum", "anal canal", "anal verge", "anus",
    "sigmoid colon", "descending colon", "transverse colon",
    "ascending colon", "proximal colon", "cecum",
    "ileocecal valve", "appendiceal orifice", "terminal ileum",
    "splenic flexure", "hepatic flexure",
    "esophagus", "distal esophagus",
    "stomach", "antrum", "stomach body", "gastric body",
    "duodenum", "duodenal bulb", "2nd portion of duodenum", "second portion of duodenum",
    "right upper quadrant", "ruq",
    "colon", "large intestine", "small intestine", "ileum",
]

PROCEDURES_LEXICON: list[str] = [
    "colonoscopy", "flexible colonoscopy",
    "egd", "esophagogastroduodenoscopy", "egd with biopsy", "egd w/biopsy",
    "upper endoscopy", "upper gi endoscopy",
    "rectal examination", "rectal exam",
    "scope passage to cecum",
    "retroflexion in rectum", "retroflexion in the rectum",
    "retroflexion in stomach", "retroflexion in the stomach",
    "monitored anesthesia care", "mac anesthesia",
    "biopsy", "cold forceps biopsy", "biopsy using cold forceps",
    "cold snare polypectomy", "polypectomy", "snare polypectomy",
    "boston bowel preparation scoring", "bbps scoring",
    "intravenous medication administration",
    "narrow band imaging",
]

# ── Code patterns ─────────────────────────────────────────────────────────────

ICD10_PATTERN   = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,4})?)\b')
CPT_PATTERN     = re.compile(r'\b(4[0-9]{4}|9[0-9]{4}|0[0-9]{4})\b')
HCPCS_PATTERN   = re.compile(r'\b([A-HJKMNPQRSTV]\d{4})\b')
MODIFIER_PATTERN = re.compile(r'\b(25|26|50|51|52|53|59|76|77|78|79|80|81|82|AS|GC|QW|TC|LT|RT|GZ|KX)\b')

# ── Section boundary detectors ─────────────────────────────────────────────────

SECTION_HEADERS = re.compile(
    r'(?i)(diagnosis|procedure|impression|findings?|indication|plan|'
    r'post.?operative|pre.?operative|complications?|procedure codes?)',
    re.IGNORECASE,
)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation at edges."""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def extract_codes(text: str, pattern: re.Pattern) -> list[str]:
    return sorted(set(pattern.findall(text)))


def match_lexicon(text: str, lexicon: list[str]) -> list[str]:
    """
    Sliding window phrase match against a lexicon.
    Returns matched phrases (original casing preserved from lexicon).
    """
    norm = normalise(text)
    found: set[str] = set()
    for phrase in lexicon:
        ph_norm = normalise(phrase)
        if ph_norm in norm:
            found.add(phrase)
    return sorted(found, key=lambda x: -len(x))   # longest match first


def deduplicate_substrings(items: list[str]) -> list[str]:
    """Remove items that are strict substrings of another item in the list."""
    result = []
    norms = [normalise(i) for i in items]
    for i, item in enumerate(items):
        ni = norms[i]
        dominated = any(
            ni != norms[j] and ni in norms[j]
            for j in range(len(items)) if j != i
        )
        if not dominated:
            result.append(item)
    return result


# ── Sentence tokeniser (no external deps) ────────────────────────────────────

def sent_tokenize(text: str) -> list[str]:
    """Simple rule-based sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


# ── Section extraction ────────────────────────────────────────────────────────

def split_sections(text: str) -> dict[str, str]:
    """
    Split report text into named sections using keyword anchors.
    Returns dict of section_name -> section_text.
    """
    # Split on known section headers
    parts = re.split(
        r'(?i)\n?\s*(Diagnosis|Procedure(?:\s+Codes?)?|Impression|Findings?|'
        r'Indication|Plan|Post.?operative[^:]*|Pre.?operative[^:]*|'
        r'Complications?)\s*[:\n]',
        text,
    )

    sections: dict[str, str] = {"full": text}
    if len(parts) > 1:
        # Odd-indexed items are section names, even are content
        for i in range(1, len(parts) - 1, 2):
            key   = normalise(parts[i])
            value = parts[i + 1] if i + 1 < len(parts) else ""
            sections[key] = value

    return sections


# ═════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTORS
# ═════════════════════════════════════════════════════════════════════════════

def extract_clinical_terms(text: str) -> list[str]:
    raw = match_lexicon(text, CLINICAL_TERMS_LEXICON)

    # Also catch explicit observation patterns
    observation_patterns = [
        r'(no (?:immediate )?complications?)',
        r'(good (?:bowel )?prep(?:aration)?)',
        r'(no polyps? (?:found|seen|identified))',
        r'(no ulcers? or masses?)',
        r'(minimal (?:estimated )?blood loss)',
        r'(patient tolerated the procedure)',
        r'(biops(?:y|ies)(?: (?:taken|obtained|were taken|were obtained))?)',
    ]
    for pat in observation_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw.append(m.group(1).strip())

    # Capitalise first letter for display
    cleaned = list({t.strip().title() if t[0].islower() else t.strip() for t in raw if t.strip()})
    return deduplicate_substrings(sorted(set(cleaned)))


def extract_anatomical_locations(text: str) -> list[str]:
    raw = match_lexicon(text, ANATOMICAL_LOCATIONS_LEXICON)
    cleaned = list({t.strip().title() if t[0].islower() else t.strip() for t in raw if t.strip()})
    return deduplicate_substrings(sorted(set(cleaned)))


def extract_diagnoses(text: str) -> list[str]:
    """
    Pull diagnoses from:
      1. Lines near 'Diagnosis' / 'Impression' section headers
      2. ICD-10 code + label pairs (e.g. "K64.8 - Internal hemorrhoids")
      3. Impression sentences
    """
    diagnoses: set[str] = set()

    # Pattern: CODE - Description  (e.g. "K64.8 - Internal hemorrhoids")
    code_label = re.finditer(
        r'[A-Z]\d{2}(?:\.\d{1,4})?\s*[-–]\s*([^\n,;]+)',
        text,
    )
    for m in code_label:
        label = m.group(1).strip().rstrip('.')
        if 3 < len(label) < 120:
            diagnoses.add(label)

    # Impression block sentences
    imp_match = re.search(r'(?i)impression\s*[:\n](.*?)(?=\n[A-Z]{2,}|\Z)', text, re.DOTALL)
    if imp_match:
        for sent in sent_tokenize(imp_match.group(1)):
            sent = sent.strip(' -•\n')
            if 5 < len(sent) < 200:
                diagnoses.add(sent)

    # Diagnosis block (first sentence per line after the header)
    diag_match = re.search(r'(?i)(?:pre.?operative\s+)?diagnosis\s*[:\n](.*?)(?=procedure|impression|\Z)', text, re.DOTALL | re.IGNORECASE)
    if diag_match:
        for line in diag_match.group(1).split('\n'):
            line = re.sub(r'[A-Z]\d{2}(?:\.\d{1,4})?', '', line).strip(' -–[]·•\t')
            line = re.sub(r'\[.*?\]', '', line).strip()
            if 4 < len(line) < 160:
                diagnoses.add(line)

    result = sorted(diagnoses, key=lambda x: len(x))
    return deduplicate_substrings([d for d in result if d.strip()])


def extract_procedures(text: str) -> list[str]:
    raw = match_lexicon(text, PROCEDURES_LEXICON)

    # Also extract from "Procedure:" lines
    proc_block = re.search(r'(?i)procedure(?:\s+codes?)?\s*[:\n](.*?)(?=diagnosis|impression|\Z)', text, re.DOTALL)
    if proc_block:
        for line in proc_block.group(1).split('\n'):
            line = re.sub(r'\d{5}', '', line)
            line = re.sub(r'[A-Z]\d{2}(?:\.\d{1,4})?', '', line).strip(' -,.:;')
            if 3 < len(line) < 120 and not re.match(r'^\d+$', line):
                raw.append(line)

    cleaned = list({t.strip() for t in raw if t.strip()})
    return deduplicate_substrings(sorted(set(cleaned)))


def extract_icd10(text: str) -> list[str]:
    """Extract ICD-10 codes, filtering false positives."""
    codes = ICD10_PATTERN.findall(text)
    # Must have letter + digits (e.g. K64.8, Z86.0100, R07.89)
    valid = [c for c in codes if re.match(r'^[A-Z]\d', c)]
    return sorted(set(valid))


def extract_cpt(text: str) -> list[str]:
    codes = CPT_PATTERN.findall(text)
    # Standard CPT range: 00100-99607
    valid = [c for c in codes if 100 <= int(c) <= 99607]
    return sorted(set(valid))


def extract_hcpcs(text: str) -> list[str]:
    """HCPCS Level II codes: letter + 4 digits, excluding CPT-like patterns."""
    codes = HCPCS_PATTERN.findall(text)
    # HCPCS Level II start with A-V (excl. CPT which are pure digits)
    valid = [c for c in codes if re.match(r'^[A-HJKMNPQRSTV]\d{4}$', c)]
    return sorted(set(valid))


def extract_modifiers(text: str) -> list[str]:
    return sorted(set(MODIFIER_PATTERN.findall(text)))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def extract_report(text: str, report_id: str = "Report") -> dict[str, Any]:
    """
    Full extraction pipeline for a single report.
    Returns the required JSON-serialisable dict.
    """
    return {
        "ReportID":             report_id,
        "Clinical Terms":       extract_clinical_terms(text),
        "Anatomical Locations": extract_anatomical_locations(text),
        "Diagnosis":            extract_diagnoses(text),
        "Procedures":           extract_procedures(text),
        "ICD-10":               extract_icd10(text),
        "CPT":                  extract_cpt(text),
        "HCPCS":                extract_hcpcs(text),
        "Modifiers":            extract_modifiers(text),
    }


def extract_all_reports(reports: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Process multiple reports.
    Each input: {"id": "Report 1", "text": "..."}
    """
    return [extract_report(r["text"], r.get("id", f"Report {i+1}"))
            for i, r in enumerate(reports)]


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, textwrap

    sample = textwrap.dedent("""
        Diagnosis:
        Z86.0100 - Personal history of colonic polyps
        K64.8 - Internal hemorrhoids
        K57.90 - Diverticulosis

        Procedure: Colonoscopy
        Anesthesia Type: Monitored Anesthesia Care
        Propofol 500 MG/50ML Emulsion, Intravenous - 240 00
        Lidocaine HCI 2% Solution, IV - 20 00

        The pediatric colonoscope was inserted into the rectum and advanced to the cecum.
        The cecum was identified by the ileocecal valve, the triradiate fold and appendiceal orifice.
        Retroflexion in the rectum performed.
        There was melanosis coli in the proximal colon.
        There was moderate sigmoid diverticulosis.
        Internal hemorrhoids were seen.
        No immediate complications.

        Impression:
        - Personal history of colonic polyps
        - Internal hemorrhoids
        - Diverticulosis (sigmoid)
        - Melanosis coli
        - No new polyps seen in this examination

        Procedure Codes: 45378
    """)

    result = extract_report(sample, "Report 1")
    print(json.dumps(result, indent=2))