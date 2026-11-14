"""GGUF v3 writer (and a small reader for checking), from scratch.

Writes the "llama" architecture with a GPT-2 style byte-level BPE vocabulary, so llama.cpp (and the Android engine built on it)
can load the file. Tensor types: F32, F16, Q8_0. (Q4_K_M: run llama.cpp's own `llama-quantize` on the F16 file.)

GGUF stores tensor dimensions fastest-first, so a NumPy matrix of shape (out, in) is written with dims [in, out].
"""
import struct

import numpy as np

MAGIC = b"GGUF"
VERSION = 3
ALIGN = 32

# value types
U32, F32_, STR, ARR = 4, 6, 8, 9
I32 = 5
BOOL = 7
# tensor types
T_F32, T_F16, T_Q8_0 = 0, 1, 8
QK8_0 = 32


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _val(kind, v):
    if kind == "u32":
        return struct.pack("<I", U32) + struct.pack("<I", v)
    if kind == "f32":
        return struct.pack("<I", F32_) + struct.pack("<f", v)
    if kind == "bool":
        return struct.pack("<I", BOOL) + struct.pack("<B", 1 if v else 0)
    if kind == "str":
        return struct.pack("<I", STR) + _str(v)
    if kind == "arr_str":
        return struct.pack("<I", ARR) + struct.pack("<IQ", STR, len(v)) + b"".join(_str(s) for s in v)
    if kind == "arr_i32":
        return struct.pack("<I", ARR) + struct.pack("<IQ", I32, len(v)) + np.asarray(v, dtype="<i4").tobytes()
    raise ValueError(kind)


def quantize_q8_0(a):
    flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, QK8_0)
    amax = np.abs(flat).max(axis=1)
    d = (amax / 127.0).astype(np.float16)
    d32 = d.astype(np.float32)
    inv = np.where(d32 > 0, 1.0 / np.where(d32 > 0, d32, 1.0), 0.0)
    q = np.rint(flat * inv[:, None]).clip(-127, 127).astype(np.int8)
    rec = np.empty(len(flat), dtype=np.dtype([("d", "<f2"), ("q", "i1", (QK8_0,))]))
    rec["d"] = d
    rec["q"] = q
    return rec.tobytes()


def dequantize_q8_0(buf, n):
    rec = np.frombuffer(buf, dtype=np.dtype([("d", "<f2"), ("q", "i1", (QK8_0,))]), count=n // QK8_0)
    return (rec["d"].astype(np.float32)[:, None] * rec["q"].astype(np.float32)).reshape(-1)


def tensor_bytes(arr, ttype):
    if ttype == T_F32:
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if ttype == T_F16:
        return np.ascontiguousarray(arr, dtype="<f2").tobytes()
    if ttype == T_Q8_0:
        if arr.shape[-1] % QK8_0:
            raise ValueError("Q8_0 needs the row length to be a multiple of 32, got %s" % (arr.shape,))
        return quantize_q8_0(arr)
    raise ValueError(ttype)


def write_gguf(path, meta, tensors):
    """meta: list of (key, kind, value). tensors: list of (name, ndarray, ggml_type)."""
    header = MAGIC + struct.pack("<IQQ", VERSION, len(tensors), len(meta) + 1)
    kv = _str("general.alignment") + _val("u32", ALIGN)
    for key, kind, v in meta:
        kv += _str(key) + _val(kind, v)
    infos = b""
    blobs = []
    offset = 0
    for name, arr, ttype in tensors:
        data = tensor_bytes(arr, ttype)
        dims = list(reversed(arr.shape))
        infos += _str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims)
        infos += struct.pack("<IQ", ttype, offset)
        blobs.append(data)
        offset += (len(data) + ALIGN - 1) // ALIGN * ALIGN
    head = header + kv + infos
    pad = (-len(head)) % ALIGN
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad)
        for data in blobs:
            f.write(data)
            f.write(b"\0" * ((-len(data)) % ALIGN))


# ------------------------------------------------------------------ reader (for verification)
class _R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.i)
        self.i += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def string(self):
        n = self.take("Q")
        s = self.b[self.i:self.i + n].decode("utf-8")
        self.i += n
        return s

    def value(self, t):
        if t == U32:
            return self.take("I")
        if t == I32:
            return self.take("i")
        if t == F32_:
            return self.take("f")
        if t == BOOL:
            return bool(self.take("B"))
        if t == STR:
            return self.string()
        if t == 10:
            return self.take("Q")
        if t == ARR:
            et = self.take("I")
            n = self.take("Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError("unsupported value type %d" % t)


def read_gguf(path):
    """Returns (metadata dict, {tensor name: float32 ndarray in the original NumPy shape})."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != MAGIC:
        raise ValueError("not a GGUF file")
    r = _R(b)
    r.i = 4
    version = r.take("I")
    n_t = r.take("Q")
    n_kv = r.take("Q")
    meta = {"_version": version}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = r.value(r.take("I"))
    infos = []
    for _ in range(n_t):
        name = r.string()
        nd = r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        ttype = r.take("I")
        off = r.take("Q")
        infos.append((name, dims, ttype, off))
    align = meta.get("general.alignment", ALIGN)
    base = (r.i + align - 1) // align * align
    out = {}
    for name, dims, ttype, off in infos:
        shape = tuple(reversed(dims))
        n = int(np.prod(shape))
        s = base + off
        if ttype == T_F32:
            arr = np.frombuffer(b, dtype="<f4", count=n, offset=s).astype(np.float32)
        elif ttype == T_F16:
            arr = np.frombuffer(b, dtype="<f2", count=n, offset=s).astype(np.float32)
        elif ttype == T_Q8_0:
            arr = dequantize_q8_0(b[s:s + n // QK8_0 * 34], n)
        else:
            raise ValueError("unsupported tensor type %d" % ttype)
        out[name] = arr.reshape(shape)
    return meta, out


# ------------------------------------------------------------------ model -> GGUF
CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def export_model(path, params, cfg, tok, kind="f16", name="persona-lm"):
    """kind: f32, f16 or q8_0. Norm vectors are always F32 (llama.cpp requires it)."""
    mat = {"f32": T_F32, "f16": T_F16, "q8_0": T_Q8_0}[kind]
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError("tokenizer has %d tokens but the model expects %d" % (tok.vocab_size, cfg.vocab_size))
    types = [3 if i < tok.n_special else 1 for i in range(tok.vocab_size)]      # 3 = control, 1 = normal
    meta = [
        ("general.architecture", "str", "llama"),
        ("general.name", "str", name),
        ("general.file_type", "u32", {"f32": 0, "f16": 1, "q8_0": 7}[kind]),
        ("general.quantization_version", "u32", 2),
        ("llama.context_length", "u32", cfg.ctx),
        ("llama.embedding_length", "u32", cfg.d_model),
        ("llama.block_count", "u32", cfg.n_layer),
        ("llama.feed_forward_length", "u32", cfg.d_ff),
        ("llama.attention.head_count", "u32", cfg.n_head),
        ("llama.attention.head_count_kv", "u32", cfg.n_head),
        ("llama.rope.dimension_count", "u32", cfg.head_dim),
        ("llama.rope.freq_base", "f32", float(cfg.rope_base)),
        ("llama.attention.layer_norm_rms_epsilon", "f32", float(cfg.eps)),
        ("llama.vocab_size", "u32", cfg.vocab_size),
        ("tokenizer.ggml.model", "str", "gpt2"),
        ("tokenizer.ggml.pre", "str", "gpt-2"),
        ("tokenizer.ggml.tokens", "arr_str", list(tok.id_to_tok)),
        ("tokenizer.ggml.token_type", "arr_i32", types),
        ("tokenizer.ggml.merges", "arr_str", [a + " " + b for a, b in tok.merges]),
        ("tokenizer.ggml.bos_token_id", "u32", 0),
        ("tokenizer.ggml.eos_token_id", "u32", 0),
        ("tokenizer.ggml.add_bos_token", "bool", False),
        ("tokenizer.chat_template", "str", CHAT_TEMPLATE),
    ]
    t = [("token_embd.weight", params["tok"], mat)]
    for i in range(cfg.n_layer):
        g = lambda n: params[f"l{i}.{n}"]
        t += [
            (f"blk.{i}.attn_norm.weight", g("attn_norm"), T_F32),
            (f"blk.{i}.attn_q.weight", g("wq"), mat),
            (f"blk.{i}.attn_k.weight", g("wk"), mat),
            (f"blk.{i}.attn_v.weight", g("wv"), mat),
            (f"blk.{i}.attn_output.weight", g("wo"), mat),
            (f"blk.{i}.ffn_norm.weight", g("ffn_norm"), T_F32),
            (f"blk.{i}.ffn_gate.weight", g("w_gate"), mat),
            (f"blk.{i}.ffn_up.weight", g("w_up"), mat),
            (f"blk.{i}.ffn_down.weight", g("w_down"), mat),
        ]
    t += [("output_norm.weight", params["out_norm"], T_F32), ("output.weight", params["tok"], mat)]   # tied: same matrix
    write_gguf(path, meta, t)


def params_from_gguf(tensors, cfg):
    """Rebuild a params dict from a GGUF read back (used to check the exported file)."""
    p = {"tok": tensors["token_embd.weight"], "out_norm": tensors["output_norm.weight"]}
    names = dict(attn_norm="attn_norm", wq="attn_q", wk="attn_k", wv="attn_v", wo="attn_output",
                 ffn_norm="ffn_norm", w_gate="ffn_gate", w_up="ffn_up", w_down="ffn_down")
    for i in range(cfg.n_layer):
        for mine, theirs in names.items():
            p[f"l{i}.{mine}"] = tensors[f"blk.{i}.{theirs}.weight"]
    return p
