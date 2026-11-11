from pathlib import Path

import numpy as np
import pytest

from noetica.chat import ChatSession, sample
from noetica.model import Qwen2, apply_rope, rms_norm, rope_tables, softmax
from noetica.tokenizer import Tokenizer
from noetica.weights import bf16_to_f32

MODEL_DIR = Path(__file__).parent.parent / "models" / "qwen2.5-1.5b-instruct"
needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(), reason="run scripts/download.sh first"
)


# --- pure math -------------------------------------------------------------


def test_bf16_to_f32():
    vals = np.array([1.0, -2.5, 0.0, 3.140625], dtype=np.float32)
    raw = (vals.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(bf16_to_f32(raw), vals)


def test_rms_norm_unit_rms():
    x = np.random.default_rng(0).normal(size=(3, 16)).astype(np.float32)
    y = rms_norm(x, np.ones(16, np.float32), 1e-6)
    np.testing.assert_allclose(np.sqrt((y * y).mean(-1)), 1.0, rtol=1e-4)


def test_softmax_sums_to_one():
    p = softmax(np.array([[1.0, 2.0, -np.inf]]))
    np.testing.assert_allclose(p.sum(), 1.0)
    assert p[0, 2] == 0


def test_rope_preserves_norm_and_relative_position():
    cos, sin = rope_tables(8, 32, 10000.0)
    rng = np.random.default_rng(1)
    q, k = rng.normal(size=(2, 1, 1, 8)).astype(np.float32)
    rq = lambda p: apply_rope(q, cos[p : p + 1], sin[p : p + 1])
    rk = lambda p: apply_rope(k, cos[p : p + 1], sin[p : p + 1])
    np.testing.assert_allclose(np.linalg.norm(rq(5)), np.linalg.norm(q), rtol=1e-5)
    # Dot product depends only on the position difference.
    np.testing.assert_allclose((rq(7) * rk(3)).sum(), (rq(14) * rk(10)).sum(), rtol=1e-4)


def test_sample_greedy_and_repetition_penalty():
    rng = np.random.default_rng(0)
    logits = np.array([1.0, 3.0, 2.9])
    assert sample(logits, rng, temperature=0) == 1
    assert sample(logits, rng, temperature=0, repetition_penalty=1.5, seen=np.array([1])) == 2


def test_sample_top_p_excludes_tail():
    rng = np.random.default_rng(0)
    logits = np.array([10.0, 0.0, 0.0, 0.0])
    assert {sample(logits, rng, temperature=1.0, top_p=0.5) for _ in range(50)} == {0}


# --- tokenizer -------------------------------------------------------------


@pytest.fixture(scope="module")
def tok():
    if not (MODEL_DIR / "tokenizer.json").exists():
        pytest.skip("run scripts/download.sh first")
    return Tokenizer(MODEL_DIR / "tokenizer.json")


def test_known_ids(tok):
    assert tok.encode("Hello, world!") == [9707, 11, 1879, 0]
    assert tok.encode("<|im_start|>user\n") == [151644, 872, 198]
    assert tok.special["<|im_end|>"] == 151645


@pytest.mark.parametrize(
    "text",
    ["plain text", "I'm   spaced\n\n\tout", "café naïve 日本語 😀", "x = f(1_000) + y**2", ""],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_stream_decoder_buffers_partial_utf8(tok):
    ids = tok.encode("😀日本")
    dec = tok.stream_decoder()
    assert "".join(dec(i) for i in ids) == "😀日本"


# --- model -----------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return Qwen2(MODEL_DIR, max_seq=256)


@needs_model
def test_kv_cache_matches_full_prefill(model, tok):
    ids = tok.encode("The quick brown fox jumps over the lazy dog")
    model.reset()
    full = model.forward(ids)
    model.reset()
    model.forward(ids[:4])
    for i in ids[4:]:
        inc = model.forward([i])
    np.testing.assert_allclose(inc, full, rtol=1e-3, atol=1e-3)


@needs_model
def test_multi_turn_continuity(model, tok):
    s = ChatSession(model, tok, temperature=0, max_new_tokens=30)
    s.reply("My name is Priya. Please remember it.")
    assert "Priya" in s.reply("What is my name?")
    s.reset()
    assert model.pos == 0 and s.messages == []


@needs_model
def test_overflow_drops_oldest_turns(model, tok):
    s = ChatSession(model, tok, temperature=0, max_new_tokens=20)
    for i in range(8):
        s.reply(f"Count to three, attempt {i}. " + "padding " * 20)
    assert model.pos <= model.max_seq
    assert s.messages[-1][0] == "assistant"
