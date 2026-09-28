"""Lightweight, dependency-free text analysis used by the verifier and consistency checker."""

from __future__ import annotations

import re

# ---------------------------------------------------------------- numbers

# (?<![\w.]) keeps digits inside identifiers (C10, ResNet50, v2.1) from being read as data
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s?%)?")
# numbers that are structural references, not data ("Table 2", "Fig. 3", "Section 4")
_STRUCTURAL_RE = re.compile(r"\b(?:Table|Tab\.|Figure|Fig\.|Section|Sec\.|Eq\.|Equation|Appendix|Step|Phase)\s*(\d+)", re.I)

_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_OTHER = {"hundred": 100, "thousand": 1000, "half": 0.5, "twice": 2, "double": 2, "triple": 3, "dozen": 12,
          "million": 1_000_000, "billion": 1_000_000_000}
# "one" alone is usually a pronoun ("one of the"), so it only counts inside a compound ("twenty-one")
WORD_NUMBERS = {**{k: v for k, v in _UNITS.items() if k != "one"}, **_TEENS, **_TENS, **_OTHER}
_COMPOUND_RE = re.compile(r"\b(" + "|".join(_TENS) + r")[- ](" + "|".join(_UNITS) + r")\b", re.I)
_WORD_NUM_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", re.I)


def _word_numbers(text: str) -> list[float]:
    out = [float(_TENS[m.group(1).lower()] + _UNITS[m.group(2).lower()]) for m in _COMPOUND_RE.finditer(text)]
    rest = _COMPOUND_RE.sub(" ", text)             # "twenty-four" is 24, not 20 and 4
    out.extend(float(WORD_NUMBERS[w.lower()]) for w in _WORD_NUM_RE.findall(rest))
    return out


def extract_numbers(text: str, include_words: bool = True) -> list[float]:
    """Data numbers in text (structural refs like 'Table 2' are skipped)."""
    structural_spans = [m.span(1) for m in _STRUCTURAL_RE.finditer(text)]
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        if any(s <= m.start(1) < e for s, e in structural_spans):
            continue
        whole = m.group(1).replace(",", "")
        frac = m.group(2)
        out.append(float(f"{whole}.{frac}") if frac else float(whole))
    if include_words:
        out.extend(_word_numbers(text))
    return out


def number_supported(value: float, source_numbers: list[float]) -> bool:
    for s in source_numbers:
        if abs(s - value) < 1e-9:
            return True
        # 94.2% written as 0.942 (or the reverse) is the same quantity
        if abs(s * 100 - value) < 1e-6 or abs(s - value * 100) < 1e-6:
            return True
    return False


# ---------------------------------------------------------------- words / stems

STOPWORDS = set("""
a an the and or but nor so yet for of in on at to from by with without within into onto upon over under
between among across through during before after above below about against along around as than then
this that these those there here it its itself they them their theirs we us our ours i me my you your
he she his her is are was were be been being am do does did done doing have has had having can
will would shall should must not no only also very more most less least such same other another each
every all any both either neither some many much few several one ones which who whom whose what when
where why how whether while if because although though since unless until whereas via per etc
""".split())

# Generic academic vocabulary: allowed anywhere without source support.
GENERIC = set("""
study studies paper work works approach approaches method methods methodology result results finding findings
analysis analyses data section sections present presents presented propose proposes proposed describe describes
described report reports reported show shows showed shown observe observed obtain obtained achieve achieved
achieves use uses used using apply applied applies perform performed performs performance evaluate evaluated
evaluation evaluations experiment experiments experimental setup set sets model models process procedure
procedures step steps conduct conducted investigate investigated aim aims objective objectives goal goals
contribute contributes contribution contributions limitation limitations direction directions remain remains
main key primary specific specifically several various different respectively overall total number numbers
value values measure measured measures metric metrics table tables figure figures summarize summarized
summarizes detail details detailed following follows previous prior existing current recent first second third
finally additionally furthermore moreover however therefore thus hence addition further particular particularly
include includes included including consist consists consisting based basis compared comparison relative
respect regarding terms case cases example examples instance order provide provides provided given give gives
make makes made take takes taken consider considered consideration focus focuses focused address addresses
addressed discuss discussed discussion introduction conclusion conclusions abstract background related work
outline organized organization remainder rest end part parts along out new article manuscript here below above
lastly next subsequently prior initially briefly namely whereas illustrate illustrated illustrates list listed
""".split())

# Markers of content the writer must not add on its own (checked against the sentence's sources).
MARKER_CLASSES: dict[str, re.Pattern] = {
    "interpretation": re.compile(
        r"\b(suggest\w*|indicat\w*|impl(?:y|ies|ied|ication\w*)|demonstrat\w*|reveal\w*|confirm\w*|"
        r"highlight\w*|underscor\w*|evidenc\w*|prov(?:e|es|ed|en))\b", re.I),
    "causal": re.compile(
        r"\b(because|due to|owing to|caus(?:e|es|ed|ing|al)|leads? to|led to|results? in|resulted in|"
        r"attribut\w*|explain\w*|driven by|thanks to)\b", re.I),
    "hedge": re.compile(
        r"\b(likely|unlikely|probabl\w*|possibl\w*|may|might|could|potential\w*|perhaps|presumabl\w*|plausibl\w*)\b", re.I),
    "evaluative": re.compile(
        r"\b(novel|state[- ]of[- ]the[- ]art|superior|remarkabl\w*|crucial|critical\w*|important\w*|promising|"
        r"excellent|impressive|robust\w*|significant\w*|substantial\w*|outstanding|breakthrough|unprecedented|"
        r"pioneer\w*|the first|first to|valuable|powerful|effective\w*|efficient\w*|strong\w*|notabl\w*|"
        r"rigorous\w*|thorough\w*|comprehensive\w*|high degree|highly|exceptional\w*|superb|reliabl\w*|"
        r"fast|faster|quick\w*|rapid\w*|only|merely|impressive\w*)\b", re.I),
    "comparative": re.compile(
        r"\b(outperform\w*|surpass\w*|better|worse|inferior|improv\w*|exceed\w*|higher|lower|beats?)\b", re.I),
    "inference": re.compile(
        r"\b(therefore|thus|hence|consequently|thereby|this means|which means|as a consequence|"
        r"generali[sz]\w*|limits? the|restricts? the|affects? the (?:validity|reliability))\b", re.I),
    "future": re.compile(
        r"\b(future (?:work|research|stud\w*|direction\w*)|further (?:work|research|stud\w*|investigation\w*)|"
        r"we plan|will (?:be|explore|investigate|extend))\b", re.I),
}

_INLINE_CITATION_RES = [
    re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]"),                                   # [3], [1, 4-6]
    re.compile(r"\((?:[A-Z][A-Za-z'\-]+(?: (?:et al\.?|and|&) ?[A-Z]?[A-Za-z'\-]*)?,? \d{4}[a-z]?;?\s?)+\)"),  # (Smith et al., 2020)
    re.compile(r"\b[A-Z][a-z]+ et al\.?"),                                              # Smith et al.
    re.compile(r"\\cite\w*\{"),
]


def inline_citations(text: str) -> list[str]:
    hits: list[str] = []
    for rx in _INLINE_CITATION_RES:
        hits.extend(m.group(0) for m in rx.finditer(text))
    return hits


def markers(text: str) -> dict[str, list[str]]:
    found = {}
    for cls, rx in MARKER_CLASSES.items():
        hits = [m.group(0) for m in rx.finditer(text)]
        if hits:
            found[cls] = hits
    return found


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*[A-Za-z]|[A-Za-z]")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def stem(word: str) -> str:
    """Crude suffix stripper; good enough to match inflections within one paper."""
    w = word.lower().strip("'-")
    for suffix in ("ational", "ization", "isation", "fulness", "ousness", "iveness", "ations", "ation", "ments",
                   "ment", "ness", "ities", "ity", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "ly", "s"):
        if len(w) - len(suffix) >= 3 and w.endswith(suffix):
            w = w[: -len(suffix)]
            if suffix in ("ies", "ied"):
                w += "y"
            break
    if len(w) > 4 and w.endswith("e"):       # "locate" and "located" share a stem
        w = w[:-1]
    return w


def content_stems(text: str) -> set[str]:
    return {stem(w) for w in words(text) if w.lower() not in STOPWORDS}


def novel_content_words(sentence: str, source: str) -> list[str]:
    """Content words in sentence whose stem never appears in the source."""
    src = content_stems(source) | {stem(w) for w in GENERIC}
    out = []
    for w in words(sentence):
        lw = w.lower()
        if lw in STOPWORDS or lw in GENERIC or len(lw) <= 2:
            continue
        if stem(lw) not in src and lw not in out:
            out.append(lw)
    return out


_ACRONYM_RE = re.compile(
    r"\b(?:(?=\w*\d)(?=\w*[A-Za-z])\w+"      # mixed letters and digits: YOLOv8, GPT4, 3D
    r"|[A-Za-z]+-\d+\w*"                     # ResNet-50
    r"|[A-Z]{2,}[a-z]?s?"                    # CNN, CNNs, AUC
    r"|[A-Z][a-z]+[A-Z]\w*)\b"               # CamelCase: ResNet, PyTorch
)


def technical_terms(text: str) -> list[str]:
    """Acronyms, CamelCase names, alphanumeric identifiers (ResNet-50, GPT4, BERT)."""
    terms = []
    for m in _ACRONYM_RE.finditer(text):
        t = m.group(0)
        if t.isdigit() or re.fullmatch(r"\d+(?:st|nd|rd|th|s)?", t):
            continue
        if t not in terms:
            terms.append(t)
    return terms


def proper_nouns(text: str) -> list[str]:
    """Capitalised words that are not sentence-initial (names, places, products)."""
    out = []
    for sent in split_sentences(text):
        toks = words(sent)
        for i, tok in enumerate(toks):
            if i == 0 or not tok[0].isupper() or tok.isupper():
                continue
            if tok.lower() in STOPWORDS or tok.lower() in GENERIC:
                continue
            if tok not in out:
                out.append(tok)
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text.strip())
    return [p for p in parts if p]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def term_in_source(term: str, source: str) -> bool:
    t = term.lower()
    s = source.lower()
    if t in s:
        return True
    # plural / hyphen variants
    return t.rstrip("s") in s or t.replace("-", " ") in s or t.replace("-", "") in s.replace("-", "")
