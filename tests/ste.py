"""ASD-STE100 (Simplified Technical English, Issue 9) checks for the library's own text.

Covers the README, docs, docstrings, exception messages, CLI help, and comments. It checks the
rules a program can check: sentence length (5.1, 6.3), paragraph length (6.6), semicolons
(8.1), contractions (4.2), "-ing" verb forms (3.5), passive voice (3.6), Latin abbreviations
(GR-6), American spelling (1.14), and unapproved dictionary words (1.1). Run it for a report:

    uv run python -m tests.ste
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "DECISIONS.md",
    ROOT / "docs" / "concepts.md",
    ROOT / "docs" / "RESEARCH.md",
    ROOT / "docs" / "extending.md",
    ROOT / "docs" / "integrations.md",
    ROOT / "docs" / "version-policy.md",
    ROOT / "docs" / "white-paper.md",
]
SOURCE = ROOT / "src" / "meldr"

PROCEDURAL_LIMIT = 20
DESCRIPTIVE_LIMIT = 25
PARAGRAPH_LIMIT = 6

# Unapproved STE words used in this project's text, with the approved alternative (Issue 9).
UNAPPROVED: dict[str, str] = {
    # Unapproved in the Issue 9 dictionary (alternative from the dictionary).
    "achieve": "get, do",
    "acquire": "get",
    "additional": "more, other",
    "adopt": "use",
    "allow": "let",
    "alter": "change",
    "another": "one more, a different",
    "appropriate": "applicable",
    "ask": "tell",
    "attempt": "try",
    "begin": "start",
    "both": "the two",
    "carry": "transmit, have",
    "cease": "stop",
    "commence": "start",
    "confirm": "make sure",
    "create": "make",
    "determine": "find",
    "display": "show",
    "eliminate": "remove",
    "enable": "let",
    "ensure": "make sure",
    "entire": "full, all",
    "exactly": "accurately",
    "execute": "run, do",
    "exist": "be",
    "facilitate": "help",
    "final": "last",
    "generate": "make",
    "handle": "use, do",
    "however": "but",
    "identical": "same",
    "indicate": "show",
    "initiate": "start",
    "later": "subsequently, after",
    "locate": "find",
    "maintain": "keep",
    "may": "can",
    "modify": "change",
    "need": "necessary, must",
    "notify": "tell",
    "now": "at this time",
    "obtain": "get",
    "once": "when, after",
    "per": "for each",
    "perform": "do",
    "prior": "before",
    "produce": "make, cause",
    "proper": "correct",
    "provide": "give",
    "require": "necessary, must",
    "retain": "keep",
    "reuse": "use again",
    "should": "must",
    "since": "because",
    "submit": "send",
    "terminate": "stop",
    "utilize": "use",
    "valid": "correct",
    "various": "different",
    "verify": "make sure",
    "via": "through",
    "whether": "if",
    "whole": "full",
    "would": "can",
    "never": "do not, not",
    "whose": "(write the sentence differently)",
    "under": "below, less than",
    "over": "above, more than",
    "real": "(write the sentence differently)",
    "exact": "accurate",
    "reason": "cause",
    "any": "(write the sentence differently)",
    "yet": "but, at this time",
    # Not in the dictionary and not technical words (rule 1.1).
    "able": "can",
    "actually": "(remove)",
    "basically": "(remove)",
    "essentially": "(remove)",
    "furthermore": "also",
    "just": "only",
    "moreover": "also",
    "multiple": "many, more than one",
    "particular": "specified",
    "really": "(remove)",
    "several": "some, many",
    "simply": "(remove)",
    "specific": "specified",
    "though": "but",
    "typically": "usually",
}
# Technical nouns of this project that contain unapproved words (rule 1.5 allows them).
TECHNICAL_NOUNS = re.compile(
    r"\b(required (field|target|targets|fields)|valid (row|rows|version|versions)|values? lost|"
    r"review reason|run_approved_only|schema-only|final confidence)\b",
    re.IGNORECASE,
)
LATIN = re.compile(r"\b(e\.g\.|i\.e\.|etc\.|vs\.|cf\.|et al\.)", re.IGNORECASE)
CONTRACTION = re.compile(
    r"\b(\w+n't|it's|that's|there's|here's|what's|let's|\w+'re|\w+'ve|\w+'ll|\w+'d|I'm)\b",
    re.IGNORECASE,
)
BRITISH = re.compile(
    r"\b\w*(behaviour|colour|favour|honour|licence|catalogue|centre|analys(e|ed|es|ing)"
    r"|[a-z]+is(e|ed|es|ing|ation)s?)\b",
    re.IGNORECASE,
)
# -ise words that are spelled -ise in American English too.
ISE_OK = {
    "advise",
    "advertise",
    "comprise",
    "compromise",
    "concise",
    "devise",
    "exercise",
    "expertise",
    "franchise",
    "improvise",
    "precise",
    "premise",
    "promise",
    "revise",
    "supervise",
    "surprise",
    "televise",
    "wise",
    "otherwise",
    "likewise",
    "noise",
    "raise",
    "praise",
    "poise",
    "paradise",
    "merchandise",
    "rise",
    "arise",
    "enterprise",
    "disguise",
    "cruise",
    "bruise",
    "chemise",
    "demise",
    "excise",
    "incise",
    "clockwise",
    "stepwise",
    "pairwise",
    "elementwise",
    "rowwise",
    "columnwise",
    "listwise",
    "precisely",
    "promised",
    "revised",
    "raised",
}
PASSIVE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+(\w+ly\s+)?"
    r"(used|called|written|sent|stored|returned|created|added|removed|built|made|saved|loaded|"
    r"read|done|given|taken|passed|kept|converted|computed|resolved|applied|recorded|skipped|"
    r"treated|replaced|dropped|logged|generated|produced|raised|checked|validated|rejected|"
    r"accepted|ignored|handled|counted|filtered|matched|mapped|sent|shown|parsed|cast|cached|"
    r"defined|derived|evaluated|exported|imported|indexed|joined|marked|merged|named|published|"
    r"queried|reused|run|scored|served|split|tagged|tested|upserted|wrapped)\b(?!\s+(?:as|to)\b)",
    re.IGNORECASE,
)
# "-ing" words that are technical nouns, noun modifiers, or approved words.
ING_OK = {
    "string",
    "strings",
    "during",
    "thing",
    "things",
    "nothing",
    "something",
    "anything",
    "everything",
    "missing",
    "mapping",
    "mappings",
    "embedding",
    "embeddings",
    "indexing",
    "profiling",
    "matching",
    "logging",
    "encoding",
    "routing",
    "pricing",
    "staging",
    "padding",
    "sampling",
    "spelling",
    "ring",
    "king",
    "bring",
    "pending",
    "warning",
    "warnings",
    "setting",
    "settings",
    "heading",
    "headings",
    "timestamping",
    "ceiling",
    "morning",
    "evening",
    "building",
    "listing",
    "parsing",
    "caching",
    "streaming",
    "testing",
    "upsert",
    "batching",
    "chunking",
    "tracing",
    "versioning",
    "harmonizing",
    "onboarding",
    "spring",
    "wiring",
    "drawing",
    "shipping",
    "packaging",
    "rendering",
    "polling",
    "hashing",
    "tokenizing",
    "masking",
    "redacting",
    "scoring",
    "ranking",
    "reranking",
    "labeling",
    "learning",
    "training",
    "programming",
    "engineering",
    "accounting",
    "funding",
    "meaning",
    "meanings",
    "opening",
    "closing",
    "offering",
    "ending",
    "sibling",
    "billing",
    "booking",
    "tooling",
    "thinking",
    "boring",
    "nursing",
    "clothing",
    "plumbing",
    "roofing",
    "zing",
    "ingesting",
    "leading",
    "preprocessing",
    "scaling",
}
IMPERATIVE = {
    "add",
    "call",
    "change",
    "check",
    "commit",
    "configure",
    "declare",
    "do",
    "export",
    "give",
    "import",
    "install",
    "keep",
    "let",
    "list",
    "load",
    "make",
    "map",
    "pass",
    "put",
    "read",
    "register",
    "remove",
    "review",
    "run",
    "see",
    "send",
    "set",
    "start",
    "stop",
    "tell",
    "try",
    "use",
    "write",
    "approve",
    "reject",
    "edit",
    "pick",
    "point",
    "clone",
    "implement",
    "plug",
    "define",
    "choose",
}


@dataclass
class Sentence:
    text: str
    where: str
    words: int
    procedural: bool


@dataclass
class Report:
    sentences: int = 0
    passed: int = 0
    violations: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.passed / self.sentences if self.sentences else 1.0

    def add(self, sentence: Sentence, problems: list[str]) -> None:
        self.sentences += 1
        if not problems:
            self.passed += 1
        for p in problems:
            self.violations[p] += 1
            self.examples.setdefault(p, [])
            if len(self.examples[p]) < 5:
                self.examples[p].append(f"{sentence.where}: {sentence.text[:140]}")


_CODE_SPAN = re.compile(r"`[^`]*`")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_PAREN = re.compile(r"\(([^()]*)\)")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"`*(\[])")


def _clean(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("URL", text)
    text = _CODE_SPAN.sub("CODE", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return re.sub(
        r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", "TITLE", text
    )  # 8.6: a title counts as one word


def _count(sentence: str) -> int:
    """STE word count (8.4-8.7): a parenthetical or a hyphenated group counts as one word."""
    flat = _PAREN.sub(" PAREN ", sentence)
    return len([w for w in re.split(r"\s+", flat) if re.search(r"\w", w)])


def _sentences(paragraph: str, where: str) -> list[Sentence]:
    text = " ".join(paragraph.split())
    out = []
    for part in _SPLIT.split(text):
        part = part.strip()
        if not part or not re.search(r"[A-Za-z]{2}", part):
            continue
        first = re.sub(r"^[^A-Za-z]*", "", part).split(" ")[0].lower()
        out.append(Sentence(part, where, _count(part), first in IMPERATIVE))
        out.extend(  # STE 8.5: the text in parentheses is also a sentence of its own
            Sentence(inner, where, _count(inner), False)
            for inner in _PAREN.findall(part)
            if len(inner.split()) > 3
        )
    return out


def markdown_paragraphs(path: Path) -> Iterator[tuple[str, str]]:
    """(paragraph, location); code blocks, tables, and headings are skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    block: list[str] = []
    fenced = False
    start = 0
    for n, line in enumerate([*lines, ""], start=1):
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.lstrip().startswith(("|", "#", "<")):
            continue
        item = re.match(r"^\s*([-*]|\d+\.)\s+", line)
        if not line.strip() or item:
            if block:
                yield " ".join(block), f"{path.relative_to(ROOT)}:{start}"
            block = [line[item.end() :]] if item else []
            start = n
            continue
        block.append(line.strip())


def python_texts(path: Path) -> Iterator[tuple[str, str, str]]:
    """(kind, text, location) for docstrings, exception messages, CLI help, and comments."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=True)
            if doc:
                for para in re.split(r"\n\s*\n", doc):
                    if not para.lstrip().startswith((">>>", "uv ", "```")):
                        yield "docstring", para, f"{rel}:{getattr(node, 'lineno', 1)}"
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
            text = _literal(node.exc.args[0])
            if text:
                yield "message", text, f"{rel}:{node.lineno}"
        if isinstance(node, ast.keyword) and node.arg == "help":
            text = _literal(node.value)
            if text:
                yield "cli", text, f"{rel}:{node.value.lineno}"
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            text = tok.string.lstrip("#").strip()
            if text and not re.match(r"(noqa|type:|pragma|S\d{3})", text):
                yield (
                    "comment",
                    re.sub(r"\s*#\s*(noqa|type:).*$", "", text),
                    f"{rel}:{tok.start[0]}",
                )


def _literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [v.value if isinstance(v, ast.Constant) else "X" for v in node.values]
        return "".join(str(p) for p in parts)
    if isinstance(node, ast.BinOp):
        left, right = _literal(node.left), _literal(node.right)
        return (left or "") + (right or "") if left or right else None
    return None


def problems(s: Sentence, paragraph_sentences: int) -> list[str]:
    text = s.text
    found = []
    limit = PROCEDURAL_LIMIT if s.procedural else DESCRIPTIVE_LIMIT
    if s.words > limit:
        found.append("5.1/6.3 sentence too long")
    if paragraph_sentences > PARAGRAPH_LIMIT:
        found.append("6.6 paragraph has more than six sentences")
    if ";" in text:
        found.append("8.1 semicolon")
    if CONTRACTION.search(text):
        found.append("4.2 contraction")
    if LATIN.search(text):
        found.append("GR-6 Latin abbreviation")
    for m in BRITISH.finditer(text):
        word = m.group(0).lower()
        base = re.sub(r"(s|d|ing)$", "", word)
        if not {word, base, base + "e"} & ISE_OK:
            found.append(f"1.14 spelling: {word}")
    if PASSIVE.search(text):
        found.append("3.6 passive voice")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", TECHNICAL_NOUNS.sub("TN", text))
    for w in words:
        low = w.lower()
        if low.endswith("ing") and len(low) > 4 and low not in ING_OK and not w[0].isupper():
            found.append(f"3.5 -ing form: {low}")
        lemma = low[:-3] + "y" if low.endswith("ies") else re.sub(r"(es|s|ed|d)$", "", low)
        for form in (low, lemma, lemma + "e"):
            if form in UNAPPROVED:
                found.append(f"1.1 unapproved: {form} -> {UNAPPROVED[form]}")
                break
    return found


def collect() -> dict[str, list[tuple[str, str]]]:
    """Paragraphs by kind: docs, docstring, message, cli, comment."""
    out: dict[str, list[tuple[str, str]]] = {
        k: [] for k in ("docs", "docstring", "message", "cli", "comment")
    }
    for path in DOCS:
        if path.exists():  # docs/DECISIONS.md is local only and absent from a clone
            out["docs"].extend(markdown_paragraphs(path))
    for path in sorted(SOURCE.rglob("*.py")):
        for kind, text, where in python_texts(path):
            out[kind].append((_clean(text), where))
    out["docs"] = [(_clean(t), w) for t, w in out["docs"]]
    return out


def check() -> dict[str, Report]:
    reports: dict[str, Report] = {}
    for kind, paragraphs in collect().items():
        report = reports.setdefault(kind, Report())
        for para, where in paragraphs:
            sentences = _sentences(para, where)
            for s in sentences:
                report.add(s, problems(s, len(sentences)))
    return reports


def check_files(prefixes: list[str]) -> list[tuple[str, str, list[str]]]:
    """Every failing sentence in files whose path starts with one of `prefixes`."""
    out = []
    for paragraphs in collect().values():
        for para, where in paragraphs:
            if not any(where.startswith(p) for p in prefixes):
                continue
            sentences = _sentences(para, where)
            for s in sentences:
                found = problems(s, len(sentences))
                if found:
                    out.append((where, s.text, found))
    return out


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        failing = check_files(sys.argv[1:])
        for where, text, found in failing:
            print(f"{where}: {text}\n    -> {'; '.join(found)}")
        print(f"{len(failing)} failing sentences")
        return
    total = Report()
    for kind, r in check().items():
        total.sentences += r.sentences
        total.passed += r.passed
        total.violations.update(r.violations)
        print(f"{kind:10} {r.passed:5}/{r.sentences:<5} sentences pass ({r.rate:.0%})")
        for rule, n in r.violations.most_common(12):
            print(f"    {n:4}  {rule}")
            for ex in r.examples[rule][:2]:
                print(f"            {ex}")
    print(f"TOTAL      {total.passed}/{total.sentences} ({total.rate:.0%})")


if __name__ == "__main__":
    main()
