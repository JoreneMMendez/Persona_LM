"""Faster text generation with a KV cache (Stage 9a). Pure NumPy: no PyTorch, no training code.

Plain generation (generate.sample_ids) re-reads the whole window for every new token. Here the keys and values of the tokens already read are
kept (the "KV cache"), so each new token costs one small step instead of a full pass. The maths is the same as model.forward, so the answers match
(tests/test_stage9a.py checks the logits and the sampled tokens against the plain version).

One honest limit: RoPE (the position code) counts positions from the start of the window. When prompt + reply grow past the window (ctx), the plain
generator drops the oldest token and counts again from 0, which changes every key. The cache cannot be patched for that, so in that case it is
rebuilt from the newest `ctx` tokens each step (exactly what the plain version does, so same speed and same answer, just no gain).
"""
import math

import numpy as np

from .bpe import EOS, END
from .generate import REP_WINDOW, is_looping, pick_token
from .model import _rms_fwd, _rope_tables, _softmax


class KVCache:
    """Keys/values for the tokens read so far. `n` = how many tokens are in it."""

    def __init__(self, cfg, dtype=np.float32):
        self.cfg = cfg
        shape = (cfg.n_head, cfg.ctx, cfg.head_dim)
        self.k = [np.zeros(shape, dtype) for _ in range(cfg.n_layer)]
        self.v = [np.zeros(shape, dtype) for _ in range(cfg.n_layer)]
        self.n = 0

    def reset(self):
        self.n = 0


def _rope_at(x, cos, sin):
    """x: (H, T, hd); cos/sin: (T, hd/2) for the positions of these T tokens. Rotates adjacent pairs, like model._rope."""
    H, T, hd = x.shape
    xr = x.reshape(H, T, hd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    return np.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1).reshape(H, T, hd)


def step(params, cfg, new_ids, cache):
    """Reads `new_ids` (a list of token ids) after what the cache already holds. Returns the logits (vocab,) for the LAST new token."""
    T = len(new_ids)
    n0 = cache.n
    if T == 0:
        raise ValueError("nothing to read")
    if n0 + T > cfg.ctx:
        raise ValueError("sequence longer than the context window")
    H, hd, d = cfg.n_head, cfg.head_dim, cfg.d_model
    dt = params["tok"].dtype
    cos_all, sin_all = _rope_tables(cfg, cfg.ctx, dt)
    cos, sin = cos_all[n0:n0 + T], sin_all[n0:n0 + T]
    neg = dt.type(-1e30)
    scale = 1.0 / math.sqrt(hd)
    # causal mask: new token j (absolute position n0+j) may look at keys 0..n0+j
    allowed = np.arange(n0 + T)[None, :] <= (n0 + np.arange(T))[:, None]
    h = params["tok"][np.asarray(new_ids, dtype=np.int64)]                  # (T, d)
    for i in range(cfg.n_layer):
        P = lambda name: params["l%d.%s" % (i, name)]
        n1, _ = _rms_fwd(h, P("attn_norm"), cfg.eps)
        q = (n1 @ P("wq").T).reshape(T, H, hd).transpose(1, 0, 2)
        k = (n1 @ P("wk").T).reshape(T, H, hd).transpose(1, 0, 2)
        v = (n1 @ P("wv").T).reshape(T, H, hd).transpose(1, 0, 2)
        cache.k[i][:, n0:n0 + T] = _rope_at(k, cos, sin)
        cache.v[i][:, n0:n0 + T] = v
        qr = _rope_at(q, cos, sin)
        K, V = cache.k[i][:, :n0 + T], cache.v[i][:, :n0 + T]
        sc = np.where(allowed[None], (qr @ K.transpose(0, 2, 1)) * scale, neg)
        am = (_softmax(sc) @ V).transpose(1, 0, 2).reshape(T, d)
        h = h + am @ P("wo").T
        n2, _ = _rms_fwd(h, P("ffn_norm"), cfg.eps)
        ga = n2 @ P("w_gate").T
        up = n2 @ P("w_up").T
        sg = ga * (1.0 / (1.0 + np.exp(-np.clip(ga, -60, 60))))
        h = h + (sg * up) @ P("w_down").T
    cache.n = n0 + T
    nf, _ = _rms_fwd(h[-1:], params["out_norm"], cfg.eps)
    return (nf @ params["tok"].T)[0]


def sample_ids_cached(params, cfg, prompt_ids, max_new=100, temperature=0.8, top_k=40, top_p=0.95, seed=0, stop_ids=(EOS, END),
                      should_stop=None, on_token=None, rep_penalty=1.0, stop_repeat=False):
    """Same contract and same answers as generate.sample_ids, but with a KV cache. Returns only the newly generated ids.
    should_stop() (optional) is asked before every new token; True ends the reply early with what exists so far (the web UI's Stop button).
    on_token(id) (optional) is told about each new token (the web UI shows the reply while it grows). Both default to off: nothing changes.
    rep_penalty and stop_repeat work exactly as in generate.sample_ids (off by default)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    ids = list(prompt_ids)
    if not ids:
        raise ValueError("the prompt is empty")
    out = []
    cache = KVCache(cfg, params["tok"].dtype)
    for _ in range(max_new):
        if should_stop is not None and should_stop():
            break
        if cache.n == 0 or cache.n + 1 > cfg.ctx:
            window = ids[-cfg.ctx:]                                         # first step, or the window slid: read the newest ctx tokens afresh
            cache.reset()
            logits = step(params, cfg, window, cache)
        else:
            logits = step(params, cfg, [ids[-1]], cache)
        nxt = pick_token(logits, rng, temperature, top_k, top_p, rep_penalty, ids[-REP_WINDOW:])
        if nxt in stop_ids:
            break
        ids.append(nxt)
        out.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if stop_repeat and is_looping(out):
            return out[:len(out) - 2 * is_looping(out)]
    return out
