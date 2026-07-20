"""The eval harnesses run on the Spark, but their scoring math is pure —
loaded by file path (eval/ shadows a builtin name, so no package import)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_eval", REPO / "eval" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


persona = load("persona")
voice = load("voice")
video = load("video")


def test_ab_pairs_order_swapped_and_tracked():
    twin = [f"twin {i}" for i in range(20)]
    real = [f"real {i}" for i in range(20)]
    pairs = persona.ab_pairs(twin, real, seed=7)
    assert len(pairs) == 20
    for p in pairs:
        assert {p["A"], p["B"]} == {p["A"], p["B"]}
        assert p[p["real_slot"]].startswith("real")
    slots = {p["real_slot"] for p in pairs}
    assert slots == {"A", "B"}  # bias control actually swaps


def test_indistinguishable_rate():
    pairs = [{"real_slot": "A"}, {"real_slot": "B"}, {"real_slot": "A"}, {"real_slot": "B"}]
    judgments = ["indistinguishable", "B", "B", "A"]
    # fooled: indistinguishable, correct? "B"==real -> not fooled, "B" vs real A -> fooled,
    # "A" vs real B -> fooled  => 3/4
    assert persona.indistinguishable_rate(judgments, pairs) == 0.75
    assert persona.indistinguishable_rate([], []) == 0.0


def test_indistinguishable_rate_verbose_judges():
    # external judges answer with formatting/prose; the verdict is the first token
    pairs = [{"real_slot": "A"}, {"real_slot": "B"}, {"real_slot": "A"}, {"real_slot": "B"}]
    judgments = ["**B**", "B. The phrasing in B is clearly human.", "Indistinguishable.", "A"]
    # B vs real A -> fooled, B==real -> not, indistinguishable -> fooled,
    # A vs real B -> fooled  => 3/4
    assert persona.indistinguishable_rate(judgments, pairs) == 0.75


def test_within_pct():
    assert persona.within_pct(11.0, 10.0, 15.0)
    assert not persona.within_pct(12.0, 10.0, 15.0)


def test_boilerplate_rate_catches_signoffs():
    texts = ["Great idea!\nBest,\nAbhi", "pure content, no signoff", "Sent from my iPhone"]
    assert persona.boilerplate_rate(texts) == 2 / 3


def test_style_cosine_identical_is_one():
    vecs = [[1.0, 0.0], [0.0, 2.0]]
    assert abs(persona.style_cosine(vecs, vecs) - 1.0) < 1e-9
    assert persona.style_cosine([[1.0, 0.0]], [[0.0, 1.0]]) == 0.0


def test_measure_rtf():
    rtf, audio_s = voice.measure_rtf(lambda t: [0] * 24000, "x")  # 1 s of audio
    assert audio_s == 1.0
    assert rtf < 1.0  # instant fake synth


def test_fps_from_frame_times():
    times = [0.0, 0.1, 0.2, 0.3, 0.4]  # 10 fps
    assert abs(video.fps_from_frame_times(times) - 10.0) < 1e-9
    assert video.fps_from_frame_times([1.0]) == 0.0


def test_e2e_wav_builder():
    spec = importlib.util.spec_from_file_location(
        "e2e", REPO / "scripts" / "e2e_roundtrip.py"
    )
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    wav = e2e.make_test_wav(seconds=0.5)
    assert wav[:4] == b"RIFF" and b"WAVE" in wav[:16]
    assert len(wav) == 44 + 16000  # header + 0.5s of 16k mono s16le
