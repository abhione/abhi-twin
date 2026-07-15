from corpus.pipeline.formatting import to_messages
from corpus.pipeline.records import Record


def test_real_prompt_kept():
    rec = Record(id="a", source="imessage", reply="On my way.", prompt="Where are you?")
    out = to_messages(rec)
    assert out["messages"] == [
        {"role": "user", "content": "Where are you?"},
        {"role": "assistant", "content": "On my way."},
    ]
    assert out["id"] == "a" and out["source"] == "imessage"


def test_promptless_uses_reconstructor():
    rec = Record(id="b", source="apple_notes", reply="Thoughts on pricing AI consulting…")
    out = to_messages(rec, reconstructor=lambda reply: "What's your take on pricing?")
    assert out["messages"][0] == {"role": "user", "content": "What's your take on pricing?"}


def test_promptless_without_reconstructor_dropped():
    rec = Record(id="c", source="github", reply="PR body")
    assert to_messages(rec) is None


def test_system_prompt_prepended():
    rec = Record(id="d", source="slack", reply="ship it", prompt="ready?")
    out = to_messages(rec, system="You are Abhi.")
    assert out["messages"][0] == {"role": "system", "content": "You are Abhi."}
    assert len(out["messages"]) == 3
