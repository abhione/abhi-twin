from corpus.pipeline.signatures import clean_email_body, strip_quoted, strip_signature

EMAIL = """Hey Sarah,

Sounds good — let's move the demo to Thursday. I'll bring the Spark benchmarks
and we can walk through the memory budget together.

Best,
Abhi

On Mon, Jan 5, 2026 at 9:00 AM Sarah Lee <sarah@example.com> wrote:
> Can we move the demo?
> Thursday works better for us.
"""


def test_strip_quoted_removes_reply_thread():
    out = strip_quoted(EMAIL)
    assert "Can we move the demo?" not in out
    assert "wrote:" not in out
    assert "Sounds good" in out


def test_strip_signature_cuts_signoff_near_tail():
    out = strip_signature(strip_quoted(EMAIL))
    assert "Best," not in out
    assert "Abhi" not in out.splitlines()[-1:]  # name line went with the sign-off
    assert "memory budget" in out


def test_dash_dash_signature_cuts_anywhere():
    text = "Real content here.\n-- \nAbhi Bhattacharya\nVSP Vision\nlinkedin.com/abhi"
    assert strip_signature(text) == "Real content here."


def test_signoff_mid_message_is_kept():
    # "Thanks," deep in a long body (not near the tail) must not truncate content
    lines = ["Thanks,"] + [f"line {i} of real content" for i in range(20)]
    out = strip_signature("\n".join(lines))
    assert "line 19 of real content" in out


def test_clean_email_body_collapses_blank_runs():
    out = clean_email_body("a\n\n\n\n\nb")
    assert out == "a\n\nb"


def test_original_message_block_removed():
    text = "My reply.\n-----Original Message-----\nFrom: someone\nold stuff"
    assert strip_quoted(text).strip() == "My reply."
