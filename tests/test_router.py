from serving.orchestrator.router import classify


def test_sensitive_stays_local():
    for text in (
        "summarize my email from the VSP board",
        "draft an iMessage reply to Sunny",
        "what did my bank statement look like last month",
        "is my salary negotiation email too aggressive?",
    ):
        d = classify(text)
        assert d.route == "sensitive", text
        assert d.local_only


def test_sensitive_beats_research():
    # sensitive content in a research-shaped ask must never leave the box
    d = classify("research my medical diagnosis options")
    assert d.route == "sensitive" and d.local_only


def test_tool_route():
    d = classify("schedule a meeting with the ProSource team on Friday")
    assert d.route == "tool" and d.local_only


def test_research_escalates():
    d = classify("deep research the state of the art in talking-head models")
    assert d.route == "research" and not d.local_only


def test_casual_default():
    d = classify("hey, how was the demo yesterday?")
    assert d.route == "casual" and d.local_only
