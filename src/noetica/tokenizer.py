"""Hand-written byte-level BPE tokenizer that reads a Hugging Face tokenizer.json."""

import codecs
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# Qwen2 pre-tokenizer pattern, with \p{L} / \p{N} approximated for stdlib `re`:
#   letter          -> [^\W\d_]
#   letter-or-digit -> [^\W_]   (so "not letter/digit" -> [^\w] | _ )
#   number          -> \d
_PRETOKENIZE = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\r\n\w]|_)?[^\W\d_]+"
    r"|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible mapping from bytes to printable unicode characters."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


class Tokenizer:
    def __init__(self, path: str | Path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        model = spec["model"]
        self.vocab: dict[str, int] = model["vocab"]
        merges = [m.split(" ") if isinstance(m, str) else m for m in model["merges"]]
        self.ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.special = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.id_to_token.update({i: t for t, i in self.special.items()})
        self.byte_enc = _bytes_to_unicode()
        self.byte_dec = {c: b for b, c in self.byte_enc.items()}
        alts = sorted(self.special, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(map(re.escape, alts)) + ")")
        self._bpe = lru_cache(maxsize=65536)(self._bpe_uncached)

    def _bpe_uncached(self, word: str) -> tuple[int, ...]:
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            merged = parts[best] + parts[best + 1]
            # Merge every occurrence of this pair, left to right.
            pair = (parts[best], parts[best + 1])
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                    out.append(merged)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        return tuple(self.vocab[p] for p in parts)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            if chunk in self.special:
                ids.append(self.special[chunk])
                continue
            chunk = unicodedata.normalize("NFC", chunk)
            for piece in _PRETOKENIZE.findall(chunk):
                mapped = "".join(self.byte_enc[b] for b in piece.encode("utf-8"))
                ids.extend(self._bpe(mapped))
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        tok = self.id_to_token[token_id]
        if tok in self.special:
            return tok.encode("utf-8")
        return bytes(self.byte_dec[c] for c in tok)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def stream_decoder(self):
        """Return a function token_id -> text that buffers partial UTF-8 sequences."""
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        return lambda token_id: dec.decode(self.token_bytes(token_id))
