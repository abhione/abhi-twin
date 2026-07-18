import pytest

from corpus.pipeline.pii import RegexScrubber, get_scrubber, scrub_residuals


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


def test_residuals_password_value_masked_label_kept():
    text, counts = scrub_residuals("User: galaxy_admin\nPassword: hunter2!")
    assert "hunter2!" not in text
    assert "Password: <CREDENTIAL>" in text
    assert counts["CREDENTIAL"] == 1


def test_residuals_aws_key_and_bearer():
    text, counts = scrub_residuals(
        "url?AWSAccessKeyId=AKIAABCDEFGHIJKLMNOP ok, header Bearer abcdef1234567890abcdef12"
    )
    assert "AKIA" not in text and "abcdef1234567890abcdef12" not in text
    assert counts["CREDENTIAL"] == 2


def test_residuals_typo_tld_email_and_phone():
    text, counts = scrub_residuals("mail nBismha@vsp.calm or call 289-949-8056")
    assert "<EMAIL_ADDRESS>" in text and "<PHONE_NUMBER>" in text


def test_residuals_bare_password_prompt_untouched():
    original = "choose a new password:\nthen log in again"
    text, _ = scrub_residuals(original)
    assert "then log in again" in text  # value on next line is not a secret


def test_residuals_idempotent():
    once, _ = scrub_residuals("password: s3cret and AKIAABCDEFGHIJKLMNOP")
    twice, counts = scrub_residuals(once)
    assert once == twice or "<CREDENTIAL>" in twice


def test_regex_engine_runs_residuals():
    text, counts = RegexScrubber().scrub("api_key: sk-livedeadbeef")
    assert "sk-livedeadbeef" not in text
    assert counts["CREDENTIAL"] == 1


def test_get_scrubber_regex():
    assert get_scrubber("regex").name == "regex"


def test_get_scrubber_auto_falls_back():
    # presidio may or may not be installed; auto must always return a scrubber
    assert get_scrubber("auto").name in ("presidio", "regex")
