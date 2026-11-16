"""Sampling text from a trained model, and a plain chat loop."""

import numpy as np

from .bpe import ASSISTANT, END, EOS, ROLE_ID
from .model import forward


REP_WINDOW = 128      # how many of the newest tokens the repetition penalty looks at


def is_looping(out, times=3, min_len=3, max_len=40):
    """Returns the length L when the end of `out` is one block of L tokens (min_len <= L <= max_len) repeated at least `times` times in a row, else 0."""
    n = len(out)
    for size in range(min_len, min(max_len, n // times) + 1):
        block = out[n - size:]
        if all(out[n - size * (k + 1): n - size * k] == block for k in range(1, times)):
            return size
    return 0


def pick_token(lg, rng, temperature=0.8, top_k=40, top_p=0.95, rep_penalty=1.0, recent=()):
    """Chooses the next token id from one row of logits. Shared by the plain and the cached generator so both draw the same token for the same seed.
    rep_penalty > 1 makes the tokens in `recent` less likely (1.0 = off, nothing changes)."""
    lg = lg.astype(np.float64)
    if rep_penalty and rep_penalty != 1.0 and len(recent):
        seen = np.unique(np.asarray(list(recent)[-REP_WINDOW:], dtype=np.int64))
        seen = seen[(seen >= 0) & (seen < len(lg))]
        lg[seen] = np.where(lg[seen] > 0, lg[seen] / rep_penalty, lg[seen] * rep_penalty)
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


def sample_ids(params, cfg, prompt_ids, max_new=100, temperature=0.8, top_k=40, top_p=0.95, seed=0, stop_ids=(EOS, END),
               rep_penalty=1.0, stop_repeat=False):
    """Returns only the newly generated ids. temperature 0 = always the most likely token. (Simple version: re-reads the whole window for every
    new token. `fastgen.sample_ids_cached` gives the same answer much faster.)
    rep_penalty (1.0 = off) and stop_repeat (False = off): see pick_token and is_looping; stop_repeat ends the reply when it starts repeating one block 3 times and keeps one copy."""
    rng = np.random.Generator(np.random.PCG64(seed))
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new):
        window = np.array([ids[-cfg.ctx:]], dtype=np.int64)
        logits, _ = forward(params, cfg, window, last_only=True)
        nxt = pick_token(logits[0, -1], rng, temperature, top_k, top_p, rep_penalty, ids[-REP_WINDOW:])
        if nxt in stop_ids:
            break
        ids.append(nxt)
        out.append(nxt)
        if stop_repeat and is_looping(out):
            return out[:len(out) - 2 * is_looping(out)]
    return out


def complete(params, cfg, tok, text, **kw):
    return tok.decode(sample_ids(params, cfg, tok.encode(text), **kw))


def chat_prompt_ids(tok, messages):
    ids = []
    for role, content in messages:
        ids += [ROLE_ID[role]] + tok.encode(content, allow_special=False) + [END]
    return ids + [ASSISTANT]
