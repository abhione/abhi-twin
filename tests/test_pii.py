import pytest

from corpus.pipeline.pii import RegexScrubber, get_scrubber


@pytest.fixture
def scrub():
    return RegexScrubber()


def test_email_replaced_not_deleted(scrub):
    text, counts = scrub.scrub("Ping abhi@sequoiadigital.io when the run finishes.")
    assert "<EMAIL_ADDRESS>" in text
    assert "abhi@sequoiadigital.io" not in text
    assert "when the run finishes" in text  # replace, don't delete
    assert counts["EMAIL_ADDRESS"] == 1


def test_phone_and_ssn(scrub):
    text, counts = scrub.scrub("Call 415-555-0123. SSN 123-45-6789 for the form.")
    assert "<PHONE_NUMBER>" in text and "<US_SSN>" in text
    assert counts["PHONE_NUMBER"] == 1 and counts["US_SSN"] == 1


def test_credit_card(scrub):
    text, counts = scrub.scrub("Card: 4111 1111 1111 1111 exp 09/27")
    assert "<CREDIT_CARD>" in text
    assert counts["CREDIT_CARD"] == 1


def test_clean_text_untouched(scrub):
    original = "The Spark arrives tomorrow and the harness must be ready."
    text, counts = scrub.scrub(original)
    assert text == original and counts == {}


def test_get_scrubber_regex():
    assert get_scrubber("regex").name == "regex"


def test_get_scrubber_auto_falls_back():
    # presidio may or may not be installed; auto must always return a scrubber
    assert get_scrubber("auto").name in ("presidio", "regex")
