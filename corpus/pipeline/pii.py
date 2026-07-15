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


class RegexScrubber:
    name = "regex"

    def scrub(self, text: str) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = {}
        for entity, pattern in _REGEX_ENTITIES:
            text, n = pattern.subn(f"<{entity}>", text)
            if n:
                counts[entity] = counts.get(entity, 0) + n
        return text, counts


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
        return scrubbed, counts


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
