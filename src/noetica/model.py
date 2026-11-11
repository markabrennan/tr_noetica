"""Qwen2 decoder-only transformer forward pass in pure numpy, with a KV cache."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from noetica.weights import load_safetensors


@dataclass
class Config:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    intermediate: int
    vocab: int
    eps: float
    rope_theta: float
    tied: bool

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @classmethod
    def load(cls, path: str | Path) -> Config:
        c = json.loads(Path(path).read_text())
        return cls(
            hidden=c["hidden_size"],
            n_layers=c["num_hidden_layers"],
            n_heads=c["num_attention_heads"],
            n_kv_heads=c["num_key_value_heads"],
            intermediate=c["intermediate_size"],
            vocab=c["vocab_size"],
            eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            tied=c.get("tie_word_embeddings", False),
        )


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rope_tables(head_dim: int, max_pos: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables of shape (max_pos, head_dim), HF 'rotate_half' layout."""
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    freqs = np.outer(np.arange(max_pos, dtype=np.float64), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: (n_heads, T, head_dim); cos/sin: (T, head_dim)."""
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


@dataclass
class Layer:
    attn_norm: np.ndarray
    w_qkv: np.ndarray  # (q+k+v out, hidden)
    b_qkv: np.ndarray
    w_o: np.ndarray
    mlp_norm: np.ndarray
    w_gate_up: np.ndarray  # (2*intermediate, hidden)
    w_down: np.ndarray


class Qwen2:
    def __init__(self, model_dir: str | Path, max_seq: int = 4096):
        model_dir = Path(model_dir)
        self.cfg = cfg = Config.load(model_dir / "config.json")
        self.max_seq = max_seq
        w = load_safetensors(model_dir / "model.safetensors")

        self.embed = w.pop("model.embed_tokens.weight")
        self.final_norm = w.pop("model.norm.weight")
        self.lm_head = self.embed if cfg.tied else w.pop("lm_head.weight")
        self.layers: list[Layer] = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            a, m = p + "self_attn.", p + "mlp."
            self.layers.append(
                Layer(
                    attn_norm=w.pop(p + "input_layernorm.weight"),
                    w_qkv=np.concatenate([w.pop(a + f"{n}_proj.weight") for n in "qkv"], axis=0),
                    b_qkv=np.concatenate([w.pop(a + f"{n}_proj.bias") for n in "qkv"]),
                    w_o=w.pop(a + "o_proj.weight"),
                    mlp_norm=w.pop(p + "post_attention_layernorm.weight"),
                    w_gate_up=np.concatenate(
                        [w.pop(m + "gate_proj.weight"), w.pop(m + "up_proj.weight")], axis=0
                    ),
                    w_down=w.pop(m + "down_proj.weight"),
                )
            )
        self.cos, self.sin = rope_tables(cfg.head_dim, max_seq, cfg.rope_theta)
        shape = (cfg.n_layers, cfg.n_kv_heads, max_seq, cfg.head_dim)
        self.k_cache = np.zeros(shape, dtype=np.float32)
        self.v_cache = np.zeros(shape, dtype=np.float32)
        self.pos = 0  # number of tokens currently in the cache

    def reset(self) -> None:
        self.pos = 0

    def forward(self, tokens: list[int]) -> np.ndarray:
        """Append `tokens` to the cache and return logits for the last one."""
        cfg = self.cfg
        T = len(tokens)
        start, end = self.pos, self.pos + T
        if end > self.max_seq:
            raise ValueError(f"context overflow: {end} > {self.max_seq}")
        H, KV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        G = H // KV
        cos, sin = self.cos[start:end], self.sin[start:end]
        # Causal mask over [cached + new] keys for the new queries.
        q_pos = np.arange(start, end)[:, None]
        k_pos = np.arange(end)[None, :]
        mask = np.where(k_pos <= q_pos, 0.0, -np.inf).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(D))

        x = self.embed[tokens]  # (T, hidden)
        for li, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, cfg.eps)
            qkv = h @ L.w_qkv.T + L.b_qkv
            q, k, v = np.split(qkv, [H * D, H * D + KV * D], axis=-1)
            q = apply_rope(q.reshape(T, H, D).transpose(1, 0, 2), cos, sin)
            k = apply_rope(k.reshape(T, KV, D).transpose(1, 0, 2), cos, sin)
            self.k_cache[li, :, start:end] = k
            self.v_cache[li, :, start:end] = v.reshape(T, KV, D).transpose(1, 0, 2)
            keys = self.k_cache[li, :, :end]  # (KV, S, D)
            vals = self.v_cache[li, :, :end]
            # Grouped-query attention: query head h uses kv head h // G.
            qg = q.reshape(KV, G * T, D)
            scores = (qg @ keys.transpose(0, 2, 1)) * scale  # (KV, G*T, S)
            scores = scores.reshape(KV, G, T, end) + mask
            attn = softmax(scores).reshape(KV, G * T, end) @ vals  # (KV, G*T, D)
            attn = attn.reshape(H, T, D).transpose(1, 0, 2).reshape(T, H * D)
            x = x + attn @ L.w_o.T

            h = rms_norm(x, L.mlp_norm, cfg.eps)
            gate, up = np.split(h @ L.w_gate_up.T, 2, axis=-1)
            x = x + (silu(gate) * up) @ L.w_down.T

        self.pos = end
        last = rms_norm(x[-1], self.final_norm, cfg.eps)
        return last @ self.lm_head.T
