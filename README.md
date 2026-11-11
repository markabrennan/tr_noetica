# noetica: numpy-only chat CLI

A minimal chat assistant that runs Qwen2.5-1.5B-Instruct with **only numpy** for the math.
The weights loader (safetensors), byte-level BPE tokenizer, transformer forward pass and
sampler are all hand-written.

```sh
pdm install -G test -G lint
scripts/download.sh 1.5B          # ~3 GB into models/ (also accepts 0.5B, 3B)
pdm run chat --stats              # chat; /reset clears history, /quit exits
pdm run chat --model models/qwen2.5-3b-instruct   # bigger model, same code
pdm run pytest -q
```

| File | What it does |
|---|---|
| `src/noetica/weights.py` | safetensors parser; bf16 → fp32 |
| `src/noetica/tokenizer.py` | byte-level BPE from `tokenizer.json`, streaming UTF-8 decode |
| `src/noetica/model.py` | Qwen2 forward pass: RMSNorm, RoPE, GQA attention, SwiGLU, KV cache |
| `src/noetica/chat.py` | ChatML template, temperature/top-p sampling, multi-turn REPL |

Conversation continuity: the KV cache persists across turns, so each turn feeds in only the
new tokens. When the 4096-token context fills up, the oldest exchanges are dropped and the
cache is rebuilt.
