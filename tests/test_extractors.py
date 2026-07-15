import json
import mailbox
import sqlite3
from email.message import EmailMessage

from corpus.extractors import apple_notes, github_prs, gmail_takeout, imessage, slack

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
