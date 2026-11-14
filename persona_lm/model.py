"""Llama-style decoder-only transformer in plain NumPy, with the backward pass written by hand.

Architecture (matches llama.cpp's "llama" architecture so it can be exported to GGUF):
  token embedding -> N x [ RMSNorm -> causal self-attention with RoPE -> residual -> RMSNorm -> SwiGLU MLP -> residual ]
  -> RMSNorm -> logits through the SAME embedding matrix (tied weights).
RoPE rotates adjacent pairs (x[2i], x[2i+1]), which is the layout llama.cpp expects for the llama architecture.
All weight matrices are stored (out_features, in_features), y = x @ W.T.
"""
import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Config:
    vocab_size: int = 4096
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 352
    ctx: int = 128
    rope_base: float = 10000.0
    eps: float = 1e-5

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("head size must be even (RoPE)")

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    def n_params(self):
        d, F, L = self.d_model, self.d_ff, self.n_layer
        return self.vocab_size * d + L * (2 * d + 4 * d * d + 3 * d * F) + d

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# every size keeps d_model and d_ff multiples of 32 (needed for Q8_0 export)
PRESETS = {
    "tiny": dict(n_layer=2, d_model=64, n_head=4, d_ff=192, ctx=64),
    "s10m": dict(n_layer=6, d_model=384, n_head=6, d_ff=1024, ctx=512),
    "s30m": dict(n_layer=8, d_model=512, n_head=8, d_ff=1376, ctx=512),
    "s125m": dict(n_layer=12, d_model=768, n_head=12, d_ff=2048, ctx=512),
}


def preset(name, vocab_size):
    return Config(vocab_size=vocab_size, **PRESETS[name])


def init_params(cfg, seed=0, dtype=np.float32):
    rng = np.random.Generator(np.random.PCG64(seed))
    d, F = cfg.d_model, cfg.d_ff
    std = 0.02
    out_std = std / math.sqrt(2 * cfg.n_layer)      # residual-branch outputs start small

    def w(shape, s):
        return (rng.standard_normal(shape) * s).astype(dtype)

    p = {"tok": w((cfg.vocab_size, d), std)}
    for i in range(cfg.n_layer):
        p[f"l{i}.attn_norm"] = np.ones(d, dtype)
        p[f"l{i}.wq"] = w((d, d), std)
        p[f"l{i}.wk"] = w((d, d), std)
        p[f"l{i}.wv"] = w((d, d), std)
        p[f"l{i}.wo"] = w((d, d), out_std)
        p[f"l{i}.ffn_norm"] = np.ones(d, dtype)
        p[f"l{i}.w_gate"] = w((F, d), std)
        p[f"l{i}.w_up"] = w((F, d), std)
        p[f"l{i}.w_down"] = w((d, F), out_std)
    p["out_norm"] = np.ones(d, dtype)
    return p


# ------------------------------------------------------------------ building blocks
_ROPE = {}


def _rope_tables(cfg, T, dtype):
    key = (cfg.head_dim, cfg.rope_base, T, np.dtype(dtype).str)
    hit = _ROPE.get(key)
    if hit is None:
        hd = cfg.head_dim
        inv = cfg.rope_base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        hit = (np.cos(ang).astype(dtype), np.sin(ang).astype(dtype))
        if len(_ROPE) > 64:
            _ROPE.clear()
        _ROPE[key] = hit
    return hit


def _rope(x, cos, sin, inverse=False):
    B, H, T, hd = x.shape
    xr = x.reshape(B, H, T, hd // 2, 2)
    x0 = xr[..., 0]
    x1 = xr[..., 1]
    c = cos[None, None]
    s = -sin[None, None] if inverse else sin[None, None]
    return np.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1).reshape(B, H, T, hd)


def _rms_fwd(x, g, eps):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * r * g, (x, g, r)


def _rms_bwd(dy, cache):
    x, g, r = cache
    dg = np.sum(dy * x * r, axis=tuple(range(dy.ndim - 1)))
    dxh = dy * g
    dx = r * dxh - x * (r ** 3 / x.shape[-1]) * np.sum(dxh * x, axis=-1, keepdims=True)
    return dx, dg


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _mm_grad(dout, x):
    """Gradient of y = x @ W.T with respect to W (summed over batch and time)."""
    return dout.reshape(-1, dout.shape[-1]).T @ x.reshape(-1, x.shape[-1])


# ------------------------------------------------------------------ forward / backward
def forward(params, cfg, ids, keep=False, last_only=False):
    """ids: (B,T) int array. Returns (logits, cache). logits is (B,T,V), or (B,1,V) with last_only."""
    B, T = ids.shape
    if T > cfg.ctx:
        raise ValueError("sequence longer than the context window")
    d, H, hd = cfg.d_model, cfg.n_head, cfg.head_dim
    dt = params["tok"].dtype
    cos, sin = _rope_tables(cfg, T, dt)
    causal = np.tril(np.ones((T, T), dtype=bool))
    neg = dt.type(-1e30)
    scale = 1.0 / math.sqrt(hd)
    h = params["tok"][ids]
    layers = []
    for i in range(cfg.n_layer):
        P = lambda n: params[f"l{i}.{n}"]
        n1, c1 = _rms_fwd(h, P("attn_norm"), cfg.eps)
        q = (n1 @ P("wq").T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = (n1 @ P("wk").T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = (n1 @ P("wv").T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        qr = _rope(q, cos, sin)
        kr = _rope(k, cos, sin)
        sc = np.where(causal, (qr @ kr.transpose(0, 1, 3, 2)) * scale, neg)
        p = _softmax(sc)
        am = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        h = h + am @ P("wo").T
        n2, c2 = _rms_fwd(h, P("ffn_norm"), cfg.eps)
        ga = n2 @ P("w_gate").T
        up = n2 @ P("w_up").T
        sig = 1.0 / (1.0 + np.exp(-np.clip(ga, -60, 60)))
        sg = ga * sig
        s = sg * up
        h = h + s @ P("w_down").T
        if keep:
            layers.append(dict(n1=n1, c1=c1, qr=qr, kr=kr, v=v, p=p, am=am, n2=n2, c2=c2, ga=ga, sig=sig, sg=sg, up=up, s=s))
    hs = h[:, -1:, :] if last_only else h
    nf, cf = _rms_fwd(hs, params["out_norm"], cfg.eps)
    logits = nf @ params["tok"].T
    return logits, (dict(layers=layers, nf=nf, cf=cf) if keep else None)


def loss_and_grads(params, cfg, x, y, mask=None, grads=None, scale=1.0, denom=None):
    """Cross-entropy summed over the positions where mask is 1, divided by `denom`. Gradients are ADDED (times `scale`) into `grads`.

    By default denom is the number of masked positions in THIS batch (a plain mean). When accumulating several micro-batches
    into one step, pass denom = the masked positions of the WHOLE step: then the summed gradients equal those of one big
    batch exactly. (Dividing each micro-batch by its own count is only right when every micro-batch has the same count,
    which is true for pretraining but not for chat data where the mask covers only some tokens.)"""
    B, T = x.shape
    d, H, hd, V = cfg.d_model, cfg.n_head, cfg.head_dim, cfg.vocab_size
    logits, cache = forward(params, cfg, x, keep=True)
    dt = logits.dtype
    mx = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - mx)
    z = e.sum(axis=-1, keepdims=True)
    logp = (logits - mx) - np.log(z)
    tgt = np.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    m = np.ones((B, T), dtype=dt) if mask is None else mask.astype(dt)
    cnt = max(float(m.sum()), 1.0) if denom is None else max(float(denom), 1.0)
    loss = float(-(tgt.astype(np.float64) * m).sum() / cnt)
    if grads is None:
        grads = {k: np.zeros_like(v) for k, v in params.items()}

    dl = e / z
    dl[np.arange(B)[:, None], np.arange(T)[None, :], y] -= 1.0
    dl *= (m * (scale / cnt))[..., None]

    grads["tok"] += _mm_grad(dl, cache["nf"])     # tied output projection
    dh, dg = _rms_bwd(dl @ params["tok"], cache["cf"])
    grads["out_norm"] += dg
    cos, sin = _rope_tables(cfg, T, dt)
    att_scale = 1.0 / math.sqrt(hd)
    for i in reversed(range(cfg.n_layer)):
        L = cache["layers"][i]
        P = lambda n: params[f"l{i}.{n}"]
        G = lambda n: grads[f"l{i}.{n}"]
        # MLP block
        G("w_down")[...] += _mm_grad(dh, L["s"])
        ds = dh @ P("w_down")
        dup = ds * L["sg"]
        dga = ds * L["up"] * (L["sig"] * (1.0 + L["ga"] * (1.0 - L["sig"])))
        G("w_up")[...] += _mm_grad(dup, L["n2"])
        G("w_gate")[...] += _mm_grad(dga, L["n2"])
        dx, dg = _rms_bwd(dga @ P("w_gate") + dup @ P("w_up"), L["c2"])
        G("ffn_norm")[...] += dg
        dh = dh + dx
        # attention block
        G("wo")[...] += _mm_grad(dh, L["am"])
        da = (dh @ P("wo")).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        p = L["p"]
        dp = da @ L["v"].transpose(0, 1, 3, 2)
        dv = p.transpose(0, 1, 3, 2) @ da
        dsc = p * (dp - (dp * p).sum(axis=-1, keepdims=True)) * att_scale
        dq = _rope(dsc @ L["kr"], cos, sin, inverse=True)
        dk = _rope(dsc.transpose(0, 1, 3, 2) @ L["qr"], cos, sin, inverse=True)
        merge = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, d)
        dq, dk, dv = merge(dq), merge(dk), merge(dv)
        G("wq")[...] += _mm_grad(dq, L["n1"])
        G("wk")[...] += _mm_grad(dk, L["n1"])
        G("wv")[...] += _mm_grad(dv, L["n1"])
        dx, dg = _rms_bwd(dq @ P("wq") + dk @ P("wk") + dv @ P("wv"), L["c1"])
        G("attn_norm")[...] += dg
        dh = dh + dx
    np.add.at(grads["tok"], x, dh)
    return loss, grads


def zero_grads(params):
    return {k: np.zeros_like(v) for k, v in params.items()}
