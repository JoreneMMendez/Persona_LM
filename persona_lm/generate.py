"""Sampling text from a trained model, and a plain chat loop."""
import sys

import numpy as np

from .bpe import ASSISTANT, END, EOS, ROLE_ID
from .model import forward


def pick_token(lg, rng, temperature=0.8, top_k=40, top_p=0.95):
    """Chooses the next token id from one row of logits. Shared by the plain and the cached generator so both draw the same token for the same seed."""
    lg = lg.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(lg))
    lg = lg / temperature
    if top_k and top_k < len(lg):
        kth = np.partition(lg, -top_k)[-top_k]
        lg = np.where(lg < kth, -np.inf, lg)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p)) + 1]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(len(p), p=p))


def sample_ids(params, cfg, prompt_ids, max_new=100, temperature=0.8, top_k=40, top_p=0.95, seed=0, stop_ids=(EOS, END)):
    """Returns only the newly generated ids. temperature 0 = always the most likely token. (Simple version: re-reads the whole window for every
    new token. `fastgen.sample_ids_cached` gives the same answer much faster.)"""
    rng = np.random.Generator(np.random.PCG64(seed))
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new):
        window = np.array([ids[-cfg.ctx:]], dtype=np.int64)
        logits, _ = forward(params, cfg, window, last_only=True)
        nxt = pick_token(logits[0, -1], rng, temperature, top_k, top_p)
        if nxt in stop_ids:
            break
        ids.append(nxt)
        out.append(nxt)
    return out


def complete(params, cfg, tok, text, **kw):
    return tok.decode(sample_ids(params, cfg, tok.encode(text), **kw))


def chat_prompt_ids(tok, messages):
    ids = []
    for role, content in messages:
        ids += [ROLE_ID[role]] + tok.encode(content, allow_special=False) + [END]
    return ids + [ASSISTANT]


def chat_loop(params, cfg, tok, system=None, **kw):
    history = [("system", system)] if system else []
    print("Type a message. /reset starts over, /quit leaves.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line == "/quit":
            return
        if line == "/reset":
            history = [("system", system)] if system else []
            continue
        if not line:
            continue
        history.append(("user", line))
        ids = chat_prompt_ids(tok, history)
        if len(ids) > cfg.ctx - 8:                         # keep the newest part that fits
            ids = ids[-(cfg.ctx - 8):]
        reply = tok.decode(sample_ids(params, cfg, ids, **kw))
        print("ai> " + reply)
        sys.stdout.flush()
        history.append(("assistant", reply))
