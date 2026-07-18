import json
import mailbox
import sqlite3
from email.message import EmailMessage

from corpus.extractors import (
    agent_memory,
    apple_notes,
    common,
    github_prs,
    gmail_takeout,
    hermes_sessions,
    imessage,
    openclaw_sessions,
    slack,
)

ME = "abhi@sequoiadigital.io"
LONG_BODY = (
    "Quick update on the twin build: the corpus extractors are done and the cleaning "
    "pipeline runs end to end on the Mac now. Dedup, PII scrubbing, length filtering "
    "and the frozen holdout split all have unit tests, so the Spark work next week is "
    "just wiring the perplexity stage to the local Qwen model and re-running the build."
)


# ------------------------------------------------------------------ gmail


def _mbox_with(tmp_path, messages):
    path = tmp_path / "sent.mbox"
    box = mailbox.mbox(str(path))
    for frm, subject, body in messages:
        msg = EmailMessage()
        msg["From"] = frm
        msg["To"] = "someone@example.com"
        msg["Subject"] = subject
        msg["Message-ID"] = f"<{subject}@test>"
        msg.set_content(body)
        box.add(msg)
    box.flush()
    return path


def test_gmail_extracts_only_my_long_mail(tmp_path):
    path = _mbox_with(
        tmp_path,
        [
            (f"Abhi <{ME}>", "update", LONG_BODY + "\n\nBest,\nAbhi"),
            (f"Abhi <{ME}>", "short", "Sounds good."),
            ("Other <other@example.com>", "theirs", LONG_BODY),
        ],
    )
    records = gmail_takeout.extract(path, me=ME)
    assert len(records) == 1
    assert records[0].source == "gmail"
    assert "Best," not in records[0].reply  # signature stripped
    assert records[0].prompt is None


# ------------------------------------------------------------------ imessage


def _chat_db(tmp_path, rows):
    path = tmp_path / "chat.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT,
            attributedBody BLOB, is_from_me INTEGER, date INTEGER);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        """
    )
    for msg_id, chat_id, text, is_from_me, date in rows:
        conn.execute(
            "INSERT INTO message (ROWID, text, attributedBody, is_from_me, date) "
            "VALUES (?, ?, NULL, ?, ?)",
            (msg_id, text, is_from_me, date),
        )
        conn.execute("INSERT OR IGNORE INTO chat (ROWID, chat_identifier) VALUES (?, ?)",
                     (chat_id, f"chat{chat_id}"))
        conn.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat_id, msg_id))
    conn.commit()
    conn.close()
    return path


def test_imessage_coalesces_turns(tmp_path):
    db = _chat_db(
        tmp_path,
        [
            (1, 1, "hey, dinner friday?", 0, 100),
            (2, 1, "yes!", 1, 200),
            (3, 1, "I'll book the usual place", 1, 300),
            (4, 1, "great see you then", 0, 400),
            (5, 1, "bring the spark benchmarks", 0, 500),
            (6, 1, "ha, will do", 1, 600),
        ],
    )
    records, skipped = imessage.extract(db)
    assert skipped == 0
    assert len(records) == 2
    assert records[0].prompt == "hey, dinner friday?"
    assert records[0].reply == "yes!\nI'll book the usual place"
    assert records[1].prompt == "great see you then\nbring the spark benchmarks"
    assert records[1].reply == "ha, will do"


def test_imessage_attributed_body_decode():
    blob = b"\x04\x0bstreamtyped\x81NSString\x01\x94\x84\x01+\x0bhello world\x86"
    assert imessage.decode_attributed_body(blob) == "hello world"
    assert imessage.decode_attributed_body(None) is None
    assert imessage.decode_attributed_body(b"garbage") is None


# ------------------------------------------------------------------ slack


def test_slack_dms_and_my_threads(tmp_path):
    export = tmp_path / "export"
    (export / "D123").mkdir(parents=True)
    (export / "C456").mkdir()
    (export / "dms.json").write_text(json.dumps([{"id": "D123"}]))
    (export / "D123" / "2026-01-01.json").write_text(json.dumps([
        {"user": "UOTHER", "text": "can you review my SOW draft?", "ts": "1.0"},
        {"user": "UME", "text": "on it — send the doc over", "ts": "2.0"},
    ]))
    (export / "C456" / "2026-01-02.json").write_text(json.dumps([
        {"user": "UME", "text": "proposal: we cloud-burst all fine-tunes", "ts": "10.0",
         "thread_ts": "10.0", "reply_count": 2},
        {"user": "UOTHER", "text": "what does that cost?", "ts": "11.0", "thread_ts": "10.0"},
        {"user": "UME", "text": "about $40-80 per fine-tune on an H100", "ts": "12.0",
         "thread_ts": "10.0"},
        {"user": "UME", "text": "unrelated channel chatter", "ts": "13.0"},
    ]))
    records = slack.extract(export, me="UME")
    replies = {r.reply for r in records}
    assert "on it — send the doc over" in replies
    assert "about $40-80 per fine-tune on an H100" in replies
    assert "proposal: we cloud-burst all fine-tunes" in replies  # thread I started
    assert "unrelated channel chatter" not in replies
    dm = next(r for r in records if r.reply.startswith("on it"))
    assert dm.prompt == "can you review my SOW draft?"


# ------------------------------------------------------------------ apple notes


def test_apple_notes(tmp_path):
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps([
        {"title": "Pricing philosophy", "plainText": LONG_BODY, "folder": "Work"},
        {"title": "tiny", "plainText": "too short"},
        {"title": "HTML note", "body": "<div>" + LONG_BODY.replace(" ", "&nbsp;") + "</div>"},
    ]))
    records = apple_notes.extract(notes)
    assert len(records) == 2
    assert records[0].source == "apple_notes"
    assert records[0].prompt is None
    assert "<div>" not in records[1].reply


def _note_proto(text: str) -> bytes:
    """Synthetic NoteStore blob: gzipped proto with text at field path 2.3.2."""
    import gzip

    def ld(field_no: int, payload: bytes) -> bytes:
        assert len(payload) < 128 * 128
        key = bytes([(field_no << 3) | 2])
        n = len(payload)
        length = bytes([n]) if n < 128 else bytes([(n & 0x7F) | 0x80, n >> 7])
        return key + length + payload

    return gzip.compress(ld(2, ld(3, ld(2, text.encode()))))


def test_apple_notes_from_db(tmp_path):
    import sqlite3

    db = tmp_path / "NoteStore.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE ZICNOTEDATA (Z_PK INTEGER PRIMARY KEY, ZNOTE INTEGER,
            ZCRYPTOINITIALIZATIONVECTOR BLOB, ZDATA BLOB);
        CREATE TABLE ZICCLOUDSYNCINGOBJECT (Z_PK INTEGER PRIMARY KEY,
            ZNOTEDATA INTEGER, ZFOLDER INTEGER, ZMARKEDFORDELETION INTEGER,
            ZTITLE1 TEXT, ZTITLE2 TEXT);
    """)
    conn.execute("INSERT INTO ZICCLOUDSYNCINGOBJECT VALUES (10, NULL, NULL, 0, NULL, 'Work')")
    rows = [
        (1, _note_proto("Pricing philosophy\n" + LONG_BODY), "Pricing philosophy", 0),
        (2, _note_proto("too short"), "tiny", 0),
        (3, _note_proto(LONG_BODY), "deleted note", 1),
        (4, b"not gzip at all", "corrupt", 0),
    ]
    for pk, blob, title, deleted in rows:
        conn.execute("INSERT INTO ZICNOTEDATA VALUES (?, NULL, NULL, ?)", (pk, blob))
        conn.execute(
            "INSERT INTO ZICCLOUDSYNCINGOBJECT VALUES (?, ?, 10, ?, ?, NULL)",
            (100 + pk, pk, deleted, title),
        )
    conn.execute(  # encrypted note: must be skipped without decode attempts
        "INSERT INTO ZICNOTEDATA VALUES (5, NULL, X'AABB', X'00')")
    conn.commit()
    conn.close()

    records, skipped = apple_notes.extract_db(db)
    assert [r.meta["title"] for r in records] == ["Pricing philosophy"]
    assert records[0].meta["folder"] == "Work"
    assert LONG_BODY in records[0].reply
    assert records[0].reply.startswith("Pricing philosophy")
    assert skipped == 1  # the corrupt blob; encrypted + deleted are filtered in SQL


# ------------------------------------------------------------------ github


def test_github_prs_and_commit_bodies():
    def fake_runner(cmd):
        if "prs" in cmd:
            return json.dumps([
                {"title": "Add corpus pipeline", "body": LONG_BODY, "url": "u1"},
                {"title": "typo fix", "body": "fix", "url": "u2"},
            ])
        return json.dumps([
            {"sha": "abc123", "commit": {"message": "feat: subject line\n\n" + LONG_BODY}},
            {"sha": "def456", "commit": {"message": "chore: subject only"}},
        ])

    records = github_prs.extract("abhione", run=fake_runner)
    kinds = [(r.meta["kind"], r.id) for r in records]
    assert len(records) == 2
    assert {k for k, _ in kinds} == {"pr", "commit"}
    commit = next(r for r in records if r.meta["kind"] == "commit")
    assert "subject line" not in commit.reply  # body only, subject dropped


# ------------------------------------------------------------------ hermes


def _hermes_db(tmp_path, rows):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            content TEXT, timestamp REAL, active INTEGER DEFAULT 1);
        """
    )
    conn.executemany(
        "INSERT INTO messages (id, session_id, role, content, timestamp, active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return path


def test_hermes_pairs_user_reply_with_assistant_context(tmp_path):
    db = _hermes_db(
        tmp_path,
        [
            (1, "s1", "assistant", "Deployed the demo app, want me to continue?", 1.0, 1),
            (2, "s1", "user", "Yes. And then continue the build", 2.0, 1),
            (3, "s1", "tool", "{}", 2.5, 1),
            (4, "s1", "assistant", "Done. Anything else?", 3.0, 1),
            (5, "s1", "user", "[IMPORTANT: Background process proc_1 completed]", 4.0, 1),
            (6, "s2", "user", "Use pnpm not npm for this project", 5.0, 1),
            (7, "s2", "user", "x" * 3000, 6.0, 1),
            (8, "s2", "user", "inactive row", 7.0, 0),
        ],
    )
    records, filtered = hermes_sessions.extract(db)
    assert len(records) == 2
    assert records[0].source == "hermes"
    assert records[0].reply == "Yes. And then continue the build"
    assert "Deployed the demo app" in records[0].prompt
    assert records[0].meta["session_id"] == "s1"
    assert records[1].reply == "Use pnpm not npm for this project"
    assert records[1].prompt is None
    assert filtered == 2  # the [IMPORTANT:...] harness turn + the 3000-char dump


def test_looks_like_pasted_heuristics():
    assert common.looks_like_pasted("```python\nprint('hi')\n```")
    assert common.looks_like_pasted("A" * 2001)
    assert common.looks_like_pasted('{"key": "value", ' + '"x": 1, ' * 20 + '"end": 0}')
    assert common.looks_like_pasted("QUJD" * 100)  # base64 run
    assert common.looks_like_pasted("[Subagent Context] do the thing")
    assert not common.looks_like_pasted("Ship it — but rename the flag to --local first.")


# ------------------------------------------------------------------ openclaw


def _openclaw_session(tmp_path, agent, name, lines):
    d = tmp_path / "agents" / agent / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def _msg(role, text, ts="2026-07-11T14:32:30Z"):
    return json.dumps(
        {"type": "message", "id": "x", "timestamp": ts,
         "message": {"role": role, "timestamp": ts,
                     "content": [{"type": "text", "text": text}]}}
    )


def test_openclaw_walks_agents_and_skips_malformed(tmp_path):
    _openclaw_session(
        tmp_path, "main", "sess-a",
        [
            json.dumps({"type": "session", "version": 3, "id": "sess-a"}),
            _msg("assistant", "Here is the plan for the demo."),
            _msg("user", "Looks good, but use Fly.io instead of Vercel"),
            "this is not json {{{",
            _msg("user", "[Subagent Context] synthetic harness turn"),
        ],
    )
    _openclaw_session(
        tmp_path, "meg", "sess-b",
        [_msg("user", "Summarize my week from the meeting notes")],
    )
    # trajectory sidecars must be ignored
    _openclaw_session(tmp_path, "main", "sess-a.trajectory", ["not even json"])

    records, filtered, malformed = openclaw_sessions.extract(tmp_path / "agents")
    assert len(records) == 2
    by_agent = {r.meta["agent"]: r for r in records}
    assert by_agent["main"].reply == "Looks good, but use Fly.io instead of Vercel"
    assert "plan for the demo" in by_agent["main"].prompt
    assert by_agent["meg"].reply == "Summarize my week from the meeting notes"
    assert all(r.source == "openclaw" for r in records)
    assert filtered == 1 and malformed == 1


def test_openclaw_string_content_and_tool_blocks(tmp_path):
    _openclaw_session(
        tmp_path, "beta", "sess-c",
        [
            json.dumps({"type": "message", "message": {
                "role": "user", "content": "plain string content works too"}}),
            json.dumps({"type": "message", "message": {
                "role": "user",
                "content": [{"type": "tool_result", "output": "ignored"}]}}),
        ],
    )
    records, _filtered, malformed = openclaw_sessions.extract(tmp_path / "agents")
    assert [r.reply for r in records] == ["plain string content works too"]
    assert malformed == 0


# ------------------------------------------------------------------ agent memory


def _mem_db(tmp_path):
    path = tmp_path / "hermes-mem.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE observations (id TEXT PRIMARY KEY, session_id TEXT,
            timestamp TEXT, type TEXT, title TEXT, narrative TEXT DEFAULT '',
            facts TEXT DEFAULT '[]', concepts TEXT DEFAULT '[]');
        CREATE TABLE session_summaries (id TEXT PRIMARY KEY, session_id TEXT,
            timestamp TEXT, request TEXT DEFAULT '', investigated TEXT DEFAULT '',
            learned TEXT DEFAULT '', completed TEXT DEFAULT '',
            next_steps TEXT DEFAULT '', notes TEXT DEFAULT '');
        """
    )
    conn.execute(
        "INSERT INTO observations VALUES ('o1', 's1', '2026-07-01T00:00:00Z', "
        "'feature', 'Abhi prefers pnpm', 'Confirmed across projects.', "
        "'[\"uses pnpm everywhere\"]', '[\"tooling\"]')"
    )
    conn.execute(
        "INSERT INTO observations VALUES ('o2', 's1', '2026-07-01T00:01:00Z', "
        "'feature', '', '', '[]', '[]')"
    )
    conn.execute(
        "INSERT INTO session_summaries VALUES ('m1', 's1', '2026-07-01T01:00:00Z', "
        "'build the twin', '', 'sm_121 needs source builds', 'phase0 done', '', '')"
    )
    conn.commit()
    conn.close()
    return path


def test_agent_memory_routes_to_rag_stream(tmp_path):
    facts = agent_memory.extract(_mem_db(tmp_path))
    assert len(facts) == 2  # empty observation dropped
    obs = next(f for f in facts if f["kind"] == "observation")
    summ = next(f for f in facts if f["kind"] == "session_summary")
    assert all(f["stream"] == "rag" and f["source"] == "hermes-mem" for f in facts)
    assert "Abhi prefers pnpm" in obs["text"]
    assert "uses pnpm everywhere" in obs["text"]
    assert obs["meta"]["concepts"] == ["tooling"]
    assert "learned: sm_121 needs source builds" in summ["text"]

    out = tmp_path / "rag" / "rag_facts.jsonl"
    assert agent_memory.write_facts(out, facts) == 2
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [f["kind"] for f in lines] == ["observation", "session_summary"]


def test_openclaw_unwraps_gateway_metadata_and_drops_harness_turns(tmp_path):
    wrapped = (
        'Conversation info (untrusted metadata):\n```json\n{\n  "sender": "Abhi B"\n}\n```\n\n'
        'Sender (untrusted metadata):\n```json\n{\n  "id": "833"\n}\n```\n\n'
        "Spawn a subagent to audit the gateway disconnects."
    )
    _openclaw_session(
        tmp_path, "main", "sess-d",
        [
            _msg("user", wrapped),
            _msg("user", "Pre-compaction memory flush. Store durable memories now."),
            _msg("user", "HEARTBEAT"),
            _msg("user", "System: node restarted"),
        ],
    )
    records, filtered, _malformed = openclaw_sessions.extract(tmp_path / "agents")
    assert [r.reply for r in records] == ["Spawn a subagent to audit the gateway disconnects."]
    assert filtered == 3


def test_agent_turns_with_credentials_are_dropped(tmp_path):
    db = _hermes_db(
        tmp_path,
        [
            (1, "s1", "user", "login with abhi@x.io and password Hunter2@x then continue", 1.0, 1),
            (2, "s1", "user", "Now wire the extractor into the CLI like the others", 2.0, 1),
        ],
    )
    records, filtered = hermes_sessions.extract(db)
    assert [r.reply for r in records] == ["Now wire the extractor into the CLI like the others"]
    assert filtered == 1
    assert common.contains_secret("here's the key: -----BEGIN RSA PRIVATE KEY-----")
    assert common.contains_secret("use Bearer abcdef1234567890abcdef")
    assert not common.contains_secret("reset your password via the normal flow")
