"""PII scrub — REPLACE, never delete (privacy hard constraint #3). Host: mac | spark.

Engines:
  presidio  MANDATORY for real corpus builds (`--pii-engine presidio`, the default
            in build.py). Catches names/locations regex can't.
  regex     dependency-free fallback used by unit tests and smoke runs; a real
            build refuses it unless --allow-regex-pii is passed.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(
    r"(?<![\w.])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?![\w])"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD = re.compile(r"\b(?:\d{4}[\s-]){3}\d{4}\b")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

_REGEX_ENTITIES = [
    ("EMAIL_ADDRESS", _EMAIL),
    ("US_SSN", _SSN),
    ("CREDIT_CARD", _CREDIT_CARD),
    ("PHONE_NUMBER", _PHONE),
    ("IP_ADDRESS", _IP),
]

# Residual pass, applied AFTER the engine (presidio or regex): shared secrets
# presidio has no recognizers for, plus email/phone leftovers presidio misses
# (typo TLDs like .con/.calm, Teams GUID addresses). Real-corpus audit findings.
# [^\S\n] = whitespace that is not a newline (covers \xa0; never crosses lines,
# so "choose a new password:\n<prose>" stays untouched)
_PASSWORD_VALUE = re.compile(
    r"(?i)\b(password|passwd|pwd)([^\S\n]*[:=][^\S\n]*)(?!<CREDENTIAL>)(\S+)"
)
_KEY_ASSIGN = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)"
    r"([^\S\n]*[:=][^\S\n]*)(?!<CREDENTIAL>)(\S+)"
)
_TEL_URI = re.compile(r"(?i)\btel:[+%;0-9.()-]{7,}")  # URL-encoded dial-ins
_AWS_ACCESS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}\b")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)


def scrub_residuals(text: str) -> tuple[str, dict[str, int]]:
    """Replace shared secrets + leftover email/phone. Replace, never delete."""
    counts: dict[str, int] = {}

    def count(entity: str, n: int) -> None:
        if n:
            counts[entity] = counts.get(entity, 0) + n

    text, n = _PRIVATE_KEY_BLOCK.subn("<PRIVATE_KEY>", text)
    count("CREDENTIAL", n)
    for pattern in (_PASSWORD_VALUE, _KEY_ASSIGN):
        text, n = pattern.subn(r"\1\2<CREDENTIAL>", text)
        count("CREDENTIAL", n)
    text, n = _AWS_ACCESS_KEY.subn("<CREDENTIAL>", text)
    count("CREDENTIAL", n)
    text, n = _BEARER_TOKEN.subn("Bearer <CREDENTIAL>", text)
    count("CREDENTIAL", n)
    text, n = _EMAIL.subn("<EMAIL_ADDRESS>", text)
    count("EMAIL_ADDRESS", n)
    for pattern in (_PHONE, _TEL_URI):
        text, n = pattern.subn("<PHONE_NUMBER>", text)
        count("PHONE_NUMBER", n)
    return text, counts


def _merge(counts: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    for k, v in extra.items():
        counts[k] = counts.get(k, 0) + v
    return counts


class RegexScrubber:
    name = "regex"

    def scrub(self, text: str) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = {}
        for entity, pattern in _REGEX_ENTITIES:
            text, n = pattern.subn(f"<{entity}>", text)
            if n:
                counts[entity] = counts.get(entity, 0) + n
        text, residual = scrub_residuals(text)
        return text, _merge(counts, residual)


class PresidioScrubber:
    """presidio-analyzer + anonymizer; entities replaced with <ENTITY_TYPE>."""

    name = "presidio"
    ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "LOCATION",
    ]

    def __init__(self) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {
            e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in self.ENTITIES
        }

    def scrub(self, text: str) -> tuple[str, dict[str, int]]:
        results = self._analyzer.analyze(text=text, entities=self.ENTITIES, language="en")
        counts: dict[str, int] = {}
        for r in results:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        scrubbed = self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self._operators
        ).text
        scrubbed, residual = scrub_residuals(scrubbed)
        return scrubbed, _merge(counts, residual)


def get_scrubber(engine: str = "presidio"):
    """engine: presidio | regex | auto (presidio if importable, else regex)."""
    if engine == "regex":
        return RegexScrubber()
    try:
        return PresidioScrubber()
    except ImportError:
        if engine == "auto":
            return RegexScrubber()
        raise RuntimeError(
            "presidio not installed — `pip install -e '.[corpus]'`. "
            "PII scrubbing with presidio is mandatory for a real corpus build; "
            "use --pii-engine regex only for smoke tests."
        ) from None
