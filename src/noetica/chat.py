"""Terminal chat loop: ChatML template, sampling, streaming, KV-cache continuity."""

import argparse
import readline  # noqa: F401  (enables line editing / history for input())
import sys
import time
from pathlib import Path

import numpy as np

from noetica.model import Qwen2
from noetica.tokenizer import Tokenizer

DEFAULT_MODEL = Path("models/qwen2.5-1.5b-instruct")
SYSTEM_PROMPT = "You are a helpful, knowledgeable assistant. Answer clearly and concisely."
IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"


def sample(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.05,
    seen: np.ndarray | None = None,
) -> int:
    logits = logits.astype(np.float64)
    if seen is not None and len(seen) and repetition_penalty != 1.0:
        s = logits[seen]
        logits[seen] = np.where(s > 0, s / repetition_penalty, s * repetition_penalty)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    keep = order[: int(np.searchsorted(cum, top_p)) + 1]
    p = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=p))


def format_turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


class ChatSession:
    """Holds the message history and keeps the model's KV cache in sync with it."""

    def __init__(
        self,
        model: Qwen2,
        tok: Tokenizer,
        system: str = SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        seed: int | None = None,
    ):
        self.model, self.tok = model, tok
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.stop_ids = {tok.special[IM_END], tok.special[EOT]}
        self.reset()

    def reset(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.model.reset()
        self.seen: list[int] = []
        # Tokens that belong in the cache but haven't been fed yet.
        self.pending = self.tok.encode(format_turn("system", self.system))

    def _rebuild(self, reserve: int) -> None:
        """Drop the oldest exchanges until history + `reserve` tokens fit, then re-prefill."""
        budget = self.model.max_seq - reserve
        while True:
            text = format_turn("system", self.system) + "".join(
                format_turn(r, c) for r, c in self.messages
            )
            ids = self.tok.encode(text)
            if len(ids) <= budget or not self.messages:
                break
            self.messages = self.messages[2:]  # one (user, assistant) exchange
        self.model.reset()
        self.seen = []
        self.pending = ids

    def reply(self, user_text: str, on_text=None):
        """Generate and return the assistant reply, streaming pieces to `on_text`."""
        prompt = self.tok.encode(format_turn("user", user_text) + f"{IM_START}assistant\n")
        reserve = len(prompt) + self.max_new_tokens
        if self.model.pos + len(self.pending) + reserve > self.model.max_seq:
            self._rebuild(reserve)
        self.messages.append(("user", user_text))
        feed = self.pending + prompt
        self.pending = []
        self.seen.extend(feed)

        decode = self.tok.stream_decoder()
        out_ids: list[int] = []
        text = []
        t_prefill = 0.0
        t0 = time.perf_counter()
        try:
            logits = self.model.forward(feed)
            t_prefill = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(self.max_new_tokens):
                nxt = sample(
                    logits,
                    self.rng,
                    temperature=self.temperature,
                    seen=np.fromiter(set(self.seen), dtype=np.int64),
                )
                if nxt in self.stop_ids:
                    break
                out_ids.append(nxt)
                self.seen.append(nxt)
                piece = decode(nxt)
                text.append(piece)
                if on_text:
                    on_text(piece)
                logits = self.model.forward([nxt])
        except KeyboardInterrupt:
            if on_text:
                on_text(" [interrupted]")
        reply = "".join(text)
        self.messages.append(("assistant", reply))
        # Close the assistant turn in the cache before the next user turn.
        self.pending = self.tok.encode(f"{IM_END}\n")
        gen_time = time.perf_counter() - t0
        self.last_stats = {
            "prefill_tokens": len(feed),
            "prefill_s": t_prefill,
            "gen_tokens": len(out_ids),
            "tok_per_s": len(out_ids) / gen_time if gen_time > 0 else 0.0,
            "context": self.model.pos,
        }
        return reply


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="numpy-only chat with Qwen2.5-Instruct")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    ap.add_argument("--stats", action="store_true", help="print speed after each reply")
    args = ap.parse_args(argv)

    print(f"Loading {args.model} ...", end="", flush=True)
    t0 = time.perf_counter()
    tok = Tokenizer(args.model / "tokenizer.json")
    model = Qwen2(args.model, max_seq=args.max_seq)
    print(f" done in {time.perf_counter() - t0:.1f}s")
    session = ChatSession(
        model,
        tok,
        system=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print("Type a message. Commands: /reset  /quit   (Ctrl-C interrupts a reply)\n")

    def emit(piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()

    while True:
        try:
            user = input("\033[1myou>\033[0m ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            session.reset()
            print("(conversation cleared)\n")
            continue
        sys.stdout.write("\033[1massistant>\033[0m ")
        session.reply(user, on_text=emit)
        print()
        if args.stats:
            s = session.last_stats
            print(
                f"\033[2m[prefill {s['prefill_tokens']} tok in {s['prefill_s']:.2f}s | "
                f"{s['gen_tokens']} tok @ {s['tok_per_s']:.1f} tok/s | "
                f"context {s['context']}/{args.max_seq}]\033[0m"
            )
        print()


if __name__ == "__main__":
    main()
