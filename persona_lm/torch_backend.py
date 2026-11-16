"""PyTorch backend: the same model, optimizer and checkpoint format as the NumPy code, running on a CUDA GPU (or on CPU).

Why PyTorch rather than CuPy: PyTorch ships tested GPU kernels, mixed precision (fp16/bf16) and fused attention, and it is
what Colab, Kaggle, RunPod, Lambda and Vast.ai all have ready. A CuPy engine would keep the hand-written backprop but you would
have to write and debug mixed precision and numerics yourself. The NumPy model stays in the project as the REFERENCE: this file
must produce the same numbers (run `python -m persona_lm check-gpu`).

What is shared with the NumPy code (so a run can move between backends and machines):
  * the checkpoint file (float32 master weights + AdamW state + RNG state + step),
  * the data sampler (NumPy, on the CPU, seeded): the batches are identical on every machine,
  * the learning-rate schedule, clipping rule, AdamW formula and loss normalization.
What differs: where the arithmetic happens. GPU arithmetic is not bit-identical to CPU arithmetic (floating-point addition is not
associative), so across machines expect agreement to ~1e-6 relative, not exact equality.

Importing this file is safe without PyTorch installed; the functions then raise a clear error.
"""
import math
import os
import threading
import time
import json

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:                      # allowed: the NumPy backend still works
    torch = None
    F = None

from .backend import NO_TORCH_MSG
from .ckpt import save_ckpt
from .safety import backup_before_training, is_new_best, save_best, should_early_stop
from .model import Config, init_params
from .optim import lr_at
from .train import (_meta, install_stop_flag, open_streams, start_or_resume, step_denominators)


def _need_torch():
    if torch is None:
        raise RuntimeError(NO_TORCH_MSG)


def _no_grad(fn):
    """Like @torch.no_grad(), but only looks at torch when the function is CALLED, so this file imports without PyTorch."""
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ------------------------------------------------------------------ devices and precision
def pick_device(want="auto"):
    """"auto" = the first CUDA GPU if there is one, else the CPU."""
    _need_torch()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU is visible to PyTorch (is the NVIDIA driver installed, and PyTorch's CUDA build?). Use --device cpu.")
    return torch.device(want)


def bf16_ok(device):
    # bfloat16 is fast only on NVIDIA Ampere (compute capability 8.x) or newer; older cards emulate it slowly
    return device.type == "cpu" or torch.cuda.get_device_capability(device)[0] >= 8


def pick_precision(device, want="auto"):
    """fp32 | bf16 | fp16. "auto": bf16 on Ampere+ GPUs, fp16 on older GPUs, fp32 on CPU."""
    _need_torch()
    if want == "auto":
        if device.type != "cuda":
            return "fp32"
        return "bf16" if bf16_ok(device) else "fp16"
    if want not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be auto, fp32, bf16 or fp16")
    if want == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 needs a CUDA GPU; on the CPU use fp32 or bf16")
    if want == "bf16" and device.type == "cuda" and not bf16_ok(device):
        raise RuntimeError("this GPU is older than NVIDIA Ampere and has no fast bf16: use fp16 or fp32")
    return want


_AMP_DTYPE = {"bf16": "bfloat16", "fp16": "float16"}


def autocast(device, precision):
    """Matrix multiplications run in low precision; the weights, residual stream, norms, softmax statistics and loss stay float32."""
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=getattr(torch, _AMP_DTYPE[precision]))


# ------------------------------------------------------------------ the model (same maths as model.py, written with torch ops)
_ROPE = {}


def _rope_tables(cfg, T, device):
    key = (cfg.head_dim, cfg.rope_base, T, str(device))
    hit = _ROPE.get(key)
    if hit is None:
        hd = cfg.head_dim
        inv = cfg.rope_base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)       # computed in float64 exactly like model.py
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        hit = (torch.tensor(np.cos(ang), dtype=torch.float32, device=device), torch.tensor(np.sin(ang), dtype=torch.float32, device=device))
        if len(_ROPE) > 64:
            _ROPE.clear()
        _ROPE[key] = hit
    return hit


def _rope(x, cos, sin):
    """Rotate ADJACENT pairs (x[2i], x[2i+1]), the layout llama.cpp expects. x: (B, H, T, hd)."""
    B, H, T, hd = x.shape
    xr = x.float().reshape(B, H, T, hd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape(B, H, T, hd)
    return out.to(x.dtype)


def _rms(x, g, eps):
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * r * g).to(x.dtype)


def forward(p, cfg, ids):
    """p: dict name -> tensor (same names as the NumPy params). ids: (B, T) int64 tensor. Returns logits (B, T, V)."""
    B, T = ids.shape
    if T > cfg.ctx:
        raise ValueError("sequence longer than the context window")
    d, H, hd = cfg.d_model, cfg.n_head, cfg.head_dim
    cos, sin = _rope_tables(cfg, T, ids.device)
    h = p["tok"][ids]
    for i in range(cfg.n_layer):
        P = lambda n: p[f"l{i}.{n}"]
        n1 = _rms(h, P("attn_norm"), cfg.eps)
        heads = lambda t: t.reshape(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)
        q = _rope(heads(F.linear(n1, P("wq"))), cos, sin)
        k = _rope(heads(F.linear(n1, P("wk"))), cos, sin)
        v = heads(F.linear(n1, P("wv")))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)         # softmax(q k^T / sqrt(hd)) v with a causal mask
        h = h + F.linear(a.transpose(1, 2).reshape(B, T, d), P("wo"))
        n2 = _rms(h, P("ffn_norm"), cfg.eps)
        h = h + F.linear(F.silu(F.linear(n2, P("w_gate"))) * F.linear(n2, P("w_up")), P("w_down"))
    return F.linear(_rms(h, p["out_norm"], cfg.eps), p["tok"])              # tied embedding


def loss_sum(p, cfg, x, y, m):
    """Cross-entropy SUMMED over the masked positions (the caller divides by the step's denominator)."""
    logits = forward(p, cfg, x)
    per_tok = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none").reshape(y.shape)
    return per_tok.sum() if m is None else (per_tok * m).sum()


# ------------------------------------------------------------------ parameters and optimizer
def params_to_torch(params_np, device):
    return {k: torch.tensor(np.asarray(v, dtype=np.float32), device=device, requires_grad=True) for k, v in params_np.items()}


def params_to_numpy(p):
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in p.items()}


class TorchAdamW:
    """The same AdamW as optim.AdamW (decoupled decay on 2-D matrices except the embedding, eps outside the square root)."""

    def __init__(self, p, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.m = {k: torch.zeros_like(v) for k, v in p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in p.items()}
        self.t = 0

    @_no_grad
    def step(self, p, lr):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, w in p.items():
            g = w.grad
            m, v = self.m[k], self.v[k]
            m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            if w.ndim == 2 and k != "tok":
                w.mul_(1.0 - lr * self.wd)
            w.sub_(lr * upd)

    def state(self):
        out = {"opt_t": np.array(self.t, dtype=np.int64)}
        for k in self.m:
            out["m." + k] = self.m[k].cpu().numpy()
            out["v." + k] = self.v[k].cpu().numpy()
        return out

    def load_state(self, arrays):
        self.t = int(arrays["opt_t"])
        for k in self.m:
            self.m[k].copy_(torch.from_numpy(np.ascontiguousarray(arrays["m." + k])))
            self.v[k].copy_(torch.from_numpy(np.ascontiguousarray(arrays["v." + k])))


class LossScaler:
    """Dynamic loss scaling for fp16 (fp16 gradients underflow to zero unless the loss is multiplied up first).
    Gradients are divided by the scale again before clipping and the optimizer. If any gradient is inf/NaN the step is skipped
    and the scale is halved; after `interval` good steps it doubles. Disabled (scale 1) for fp32 and bf16."""

    def __init__(self, enabled, scale=2.0 ** 14, interval=200):
        self.enabled = enabled
        self.scale = float(scale) if enabled else 1.0
        self.interval = interval
        self.good = 0

    def update(self, finite):
        if not self.enabled:
            return
        if finite:
            self.good += 1
            if self.good >= self.interval:
                self.scale *= 2.0
                self.good = 0
        else:
            self.scale = max(self.scale / 2.0, 1.0)
            self.good = 0


def grad_norm_and_check(p):
    """Global L2 norm of the gradients (float64 accumulation), and whether it is finite. One GPU->CPU sync."""
    sq = torch.stack([w.grad.double().pow(2).sum() for w in p.values()]).sum()
    n = float(sq.sqrt().item())
    return n, math.isfinite(n)


def to_device_batch(x, y, m, device):
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    mt = None if m is None else torch.from_numpy(m.astype(np.float32)).to(device)
    return xt, yt, mt


def accumulate_step(p, cfg, batches, denoms, device, precision, scale=1.0):
    """Forward+backward over the micro-batches of one optimizer step, adding gradients into p[...].grad (float32).
    Returns the step's loss (a float). Each micro-batch's summed loss is divided by its denominator, so the gradients add up to
    the right mean (see train.step_denominators)."""
    total = torch.zeros((), dtype=torch.float64, device=device)
    for (x, y, m), dn in zip(batches, denoms):
        xt, yt, mt = to_device_batch(x, y, m, device)
        with autocast(device, precision):
            s = loss_sum(p, cfg, xt, yt, mt)
        (s / dn * scale).backward()
        total += s.detach().double() / dn
    return float(total.item())


@_no_grad
def evaluate(p, cfg, stream, hp, device, precision):
    rng = np.random.Generator(np.random.PCG64(4242))      # the same fixed batches as the NumPy evaluate()
    tot = 0.0
    for _ in range(hp["eval_batches"]):
        x, y, m = stream.batch(rng, hp["batch"], cfg.ctx)
        xt, yt, mt = to_device_batch(x, y, m, device)
        with autocast(device, precision):
            s = loss_sum(p, cfg, xt, yt, mt)
        tot += float(s.item()) / max(micro_count(x, m), 1.0)
    return tot / hp["eval_batches"]


def micro_count(x, m):
    return float(x.size) if m is None else float(m.sum())


# ------------------------------------------------------------------ CPU and GPU together (hybrid)
# One optimizer step is `accum` micro-batches. The GPU does most of them; a worker thread does the last `n_cpu` of them on the
# CPU at the same time (PyTorch releases the GIL inside its maths, so both really run in parallel). The CPU keeps its own
# float32 copy of the weights, refreshed from the GPU at the start of every step; its gradients are added to the GPU's gradients
# before clipping and AdamW. Every micro-batch is still divided by its own denominator, so the update is the same one the GPU
# alone would make (up to floating-point rounding). The checkpoint, the data order and the optimizer are unchanged.
# The gain is real only if the CPU is not much slower than the GPU: see choose_cpu_share() and `--cpu-share auto`.
def choose_cpu_share(t_gpu, t_cpu, accum, overhead=0.0, min_gain=0.05):
    """Pure arithmetic. t_gpu / t_cpu: seconds for ONE micro-batch on each device. Returns (n_cpu, predicted_step_seconds,
    gpu_only_step_seconds). n_cpu is 0 unless some split beats the GPU alone by at least min_gain (5 %)."""
    base = accum * t_gpu
    best_n, best_t = 0, base
    for n in range(1, accum):
        t = max((accum - n) * t_gpu, n * t_cpu) + overhead
        if t < best_t:
            best_n, best_t = n, t
    if best_n and best_t > base * (1.0 - min_gain):
        return 0, base, base
    return best_n, best_t, base


def make_cpu_params(p):
    """A float32 CPU copy of the weights (its own tensors, with gradients)."""
    return {k: w.detach().to("cpu", copy=True).requires_grad_(True) for k, w in p.items()}


def sync_cpu_params(p_cpu, p):
    with torch.no_grad():
        for k, w in p.items():
            p_cpu[k].copy_(w.detach())


def hybrid_accumulate_step(p, p_cpu, cfg, batches, denoms, device, precision, n_cpu, scale=1.0):
    """Like accumulate_step(), but the last n_cpu micro-batches run on the CPU in a worker thread while the GPU does the rest.
    Gradients end up in p[...].grad exactly as accumulate_step would leave them. Returns the step's loss."""
    n_cpu = max(0, min(int(n_cpu), len(batches) - 1))      # the main device always keeps at least one micro-batch
    if n_cpu == 0 or p_cpu is None:
        return accumulate_step(p, cfg, batches, denoms, device, precision, scale)
    main_b, cpu_b = batches[:-n_cpu], batches[-n_cpu:]
    main_d, cpu_d = denoms[:-n_cpu], denoms[-n_cpu:]
    sync_cpu_params(p_cpu, p)
    for w in p_cpu.values():
        w.grad = None
    out = {}

    def work():
        try:
            out["loss"] = accumulate_step(p_cpu, cfg, cpu_b, cpu_d, torch.device("cpu"), "fp32", scale)
        except BaseException as e:           # handed back to the main thread below
            out["error"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        main_loss = accumulate_step(p, cfg, main_b, main_d, device, precision, scale)
    finally:
        th.join()
    if "error" in out:
        raise out["error"]
    with torch.no_grad():
        for k, w in p.items():
            g = p_cpu[k].grad
            if g is None:
                continue
            g = g.to(w.device)
            if w.grad is None:
                w.grad = g.clone()
            else:
                w.grad.add_(g)
    return main_loss + out["loss"]


def probe_cpu_share(p, p_cpu, cfg, batch, device, precision, accum, log=print):
    """Times one micro-batch on the GPU and on the CPU (plus the weight/gradient copying) and returns the best n_cpu."""
    x, y, m = batch
    one, dn = [(x, y, m)], [micro_count(x, m)]
    cpu = torch.device("cpu")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def clear():
        for w in list(p.values()) + list(p_cpu.values()):
            w.grad = None

    accumulate_step(p, cfg, one, dn, device, precision)                  # warm-up (kernel selection, memory allocation)
    sync()
    t = time.time()
    accumulate_step(p, cfg, one, dn, device, precision)
    sync()
    t_gpu = time.time() - t
    t = time.time()
    sync_cpu_params(p_cpu, p)
    t_copy = time.time() - t
    t = time.time()
    accumulate_step(p_cpu, cfg, one, dn, cpu, "fp32")
    t_cpu = time.time() - t
    t = time.time()
    with torch.no_grad():
        for k, w in p.items():
            w.grad = p_cpu[k].grad.to(w.device).clone() if w.grad is None else w.grad.add_(p_cpu[k].grad.to(w.device))
    sync()
    t_back = time.time() - t
    clear()
    n, t_hyb, t_base = choose_cpu_share(t_gpu, t_cpu, accum, overhead=t_copy + t_back)
    log("cpu-share probe: one micro-batch takes %.2f s on the GPU and %.2f s on the CPU (copying %.2f s). GPU only: %.2f s per step; %s" % (
        t_gpu, t_cpu, t_copy + t_back, t_base,
        ("%d of %d micro-batches on the CPU: about %.2f s per step (%.0f %% faster)" % (n, accum, t_hyb, (t_base / t_hyb - 1) * 100)) if n
        else "the CPU is too slow to help, so the GPU works alone"))
    return n


# ------------------------------------------------------------------ training loop
def train_torch(run_dir, data_dir, cfg=None, hp=None, init_from=None, log=print, max_steps_this_call=None,
                device="auto", precision="auto", deterministic=False, cpu_share="0", cpu_threads=None, tokenizer=None, backup=True):
    """Same contract as train.train(), on PyTorch. Reads and writes the same ckpt.npz, so a run can be started on one machine
    (or backend) and continued on another."""
    _need_torch()
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)      # warn_only: some GPU kernels have no deterministic version
    dev = pick_device(device)
    prec = pick_precision(dev, precision)
    ck_path = os.path.join(run_dir, "ckpt.npz")
    stop = install_stop_flag()
    if backup:
        backup_before_training(run_dir, tokenizer, log)          # before anything can change the files
    st = start_or_resume(run_dir, data_dir, cfg, hp, init_from, log)
    cfg, hp, step, history, rng = st.cfg, st.hp, st.step, st.history, st.rng
    p = params_to_torch(st.params, dev)
    opt = TorchAdamW(p, weight_decay=hp["weight_decay"])
    if st.opt_arrays is not None:
        opt.load_state(st.opt_arrays)
    saved_scale = ((st.meta or {}).get("backend") or {}).get("loss_scale")
    scaler = LossScaler(prec == "fp16", scale=saved_scale or 2.0 ** 14)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    log("backend: PyTorch %s on %s (%s), precision %s%s" % (torch.__version__, dev, name, prec, ", deterministic" if deterministic else ""))

    train_s, val_s = open_streams(data_dir, cfg)
    last_ck, t0, done_here, skipped = time.time(), time.time(), 0, 0
    early = False
    tokens_per_step = hp["batch"] * hp["accum"] * cfg.ctx

    # CPU + GPU together: cpu_share = "0" (off), "auto" (time both and choose) or a number of micro-batches per step for the CPU
    n_cpu, p_cpu = 0, None
    want_cpu = str(cpu_share if cpu_share is not None else "0").strip().lower()
    if want_cpu != "0":
        if dev.type == "cpu":
            log("cpu-share ignored: training already runs on the CPU")
        elif hp["accum"] < 2:
            log("cpu-share ignored: it needs --accum 2 or more (each step must have at least 2 micro-batches to share)")
        else:
            torch.set_num_threads(max(1, int(cpu_threads) if cpu_threads else (os.cpu_count() or 2) - 2))     # leave cores for the GPU's own work
            p_cpu = make_cpu_params(p)
            if want_cpu == "auto":
                probe = train_s.batch(np.random.Generator(np.random.PCG64(777)), hp["batch"], cfg.ctx)         # own RNG: the data order is not touched
                n_cpu = probe_cpu_share(p, p_cpu, cfg, probe, dev, prec, hp["accum"], log)
            else:
                n_cpu = max(0, min(int(want_cpu), hp["accum"] - 1))
            if n_cpu:
                log("hybrid training: %d of %d micro-batches per step on the CPU (%d threads), the rest on %s" % (n_cpu, hp["accum"], torch.get_num_threads(), dev))
            else:
                p_cpu = None

    def checkpoint():
        backend = {"name": "torch", "device": str(dev), "precision": prec, "loss_scale": scaler.scale}
        save_ckpt(ck_path, params_to_numpy(p), opt.state(), _meta(cfg, hp, step, rng, history, st.tok_hash, backend))

    while step < hp["steps"] and not stop["flag"] and (max_steps_this_call is None or done_here < max_steps_this_call):
        lr = lr_at(step, hp["steps"], hp["lr"], hp["min_lr"], hp["warmup"])
        for w in p.values():
            w.grad = None
        batches = [train_s.batch(rng, hp["batch"], cfg.ctx) for _ in range(hp["accum"])]     # identical data order on every machine
        denoms = step_denominators(batches, hp["loss_norm"])
        used_scale = scaler.scale                          # the factor backward() is about to multiply the gradients by
        loss = hybrid_accumulate_step(p, p_cpu, cfg, batches, denoms, dev, prec, n_cpu, scale=used_scale)     # returned loss is the true (unscaled) one
        gn, finite = grad_norm_and_check(p)
        gn /= used_scale
        scaler.update(finite)                              # may change scaler.scale for the NEXT step; this step uses used_scale
        if not finite and not scaler.enabled:
            raise FloatingPointError("gradients became inf/NaN at step %d (loss %s): lower the learning rate" % (step + 1, loss))
        if finite:
            with torch.no_grad():
                if used_scale != 1.0:
                    for w in p.values():
                        w.grad.mul_(1.0 / used_scale)
                if hp["clip"] and gn > hp["clip"]:
                    f = hp["clip"] / (gn + 1e-6)
                    for w in p.values():
                        w.grad.mul_(f)
            opt.step(p, lr)
        else:
            skipped += 1                       # fp16 overflow: skip this update (the loss scale was just halved)
            log("step %d: fp16 overflow, update skipped, loss scale now %g" % (step + 1, scaler.scale))
        step += 1
        done_here += 1
        rec = {"step": step, "loss": round(loss, 5), "lr": lr, "grad_norm": round(gn, 4)}
        if val_s is not None and (step % hp["eval_every"] == 0 or step == hp["steps"]):
            rec["val_loss"] = round(evaluate(p, cfg, val_s, hp, dev, prec), 5)
        history.append(rec)
        if step % hp["log_every"] == 0 or "val_loss" in rec:
            el = time.time() - t0
            tps = done_here * tokens_per_step / max(el, 1e-9)
            eta = (hp["steps"] - step) * tokens_per_step / max(tps, 1e-9)
            log("step %d/%d loss %.4f%s lr %.2e  %.0f tok/s  ETA %.1f h" % (
                step, hp["steps"], loss, (" val %.4f" % rec["val_loss"]) if "val_loss" in rec else "", lr, tps, eta / 3600))
            with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        if "val_loss" in rec:
            if hp["save_best"] and is_new_best(history):
                save_best(run_dir, params_to_numpy(p), _meta(cfg, hp, step, rng, history, st.tok_hash,
                                                             {"name": "torch", "device": str(dev), "precision": prec, "best": True}))
                log("new best validation loss %.4f at step %d, saved to ckpt.best.npz" % (rec["val_loss"], step))
            if should_early_stop(history, hp["early_stop"]):
                log("early stop: validation loss has not improved for %d checks in a row. The best weights are in ckpt.best.npz." % hp["early_stop"])
                early = True
                break
        if step % hp["ckpt_every"] == 0 or time.time() - last_ck > hp["ckpt_minutes"] * 60:
            checkpoint()
            last_ck = time.time()
    checkpoint()
    log("saved checkpoint at step %d%s" % (step, " (stopped early)" if stop["flag"] or early else ""))
    return params_to_numpy(p)


# ------------------------------------------------------------------ benchmark
def benchmark_torch(cfg, batch, device="auto", precision="auto", steps=5, log=print):
    """Returns (tokens_per_second, peak_gpu_memory_bytes or None, device, precision). Raises torch.cuda.OutOfMemoryError if batch does not fit."""
    _need_torch()
    dev = pick_device(device)
    prec = pick_precision(dev, precision)
    p = params_to_torch(init_params(cfg, 0), dev)
    rng = np.random.Generator(np.random.PCG64(0))
    x = rng.integers(0, cfg.vocab_size, size=(batch, cfg.ctx)).astype(np.int64)
    y = np.roll(x, -1, axis=1)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    one = [(x, y, None)]
    accumulate_step(p, cfg, one, [float(x.size)], dev, prec)                    # warm-up (kernel selection, allocator)
    sync()
    t0 = time.time()
    for _ in range(steps):
        for w in p.values():
            w.grad = None
        accumulate_step(p, cfg, one, [float(x.size)], dev, prec)
    sync()
    tps = steps * batch * cfg.ctx / (time.time() - t0)
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
    return tps, peak, str(dev), prec


# ------------------------------------------------------------------ self-check against the NumPy reference
def environment_report():
    _need_torch()
    lines = ["PyTorch %s" % torch.__version__, "CUDA available: %s" % torch.cuda.is_available()]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            lines.append("GPU %d: %s, %.1f GB VRAM, compute capability %d.%d, bf16 %s" % (
                i, pr.name, pr.total_memory / 2 ** 30, pr.major, pr.minor, "fast" if pr.major >= 8 else "NOT fast (use fp16)"))
    return lines


def _rel(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def selfcheck(device="auto", log=print):
    """Compares this backend with the NumPy reference. Returns a list of (name, passed, detail). Used by `python -m persona_lm
    check-gpu` and by the unit tests."""
    import tempfile
    from .model import loss_and_grads, forward as np_forward, zero_grads
    from .optim import AdamW
    from .train import train as train_numpy

    _need_torch()
    dev = pick_device(device)
    results = []

    def add(name, ok, detail):
        results.append((name, bool(ok), detail))
        log("  [%s] %s  %s" % ("PASS" if ok else "FAIL", name, detail))

    cfg = Config(vocab_size=128, n_layer=2, d_model=64, n_head=4, d_ff=128, ctx=32)
    params = init_params(cfg, 3)
    rng = np.random.Generator(np.random.PCG64(5))
    B, T = 4, 32
    x = rng.integers(0, cfg.vocab_size, (B, T)).astype(np.int64)
    y = rng.integers(0, cfg.vocab_size, (B, T)).astype(np.int64)
    m = np.zeros((B, T), np.uint8)
    for i, k in enumerate([2, 5, 17, 32]):               # uneven masks, like chat data
        m[i, -k:] = 1

    # 1. forward logits
    p = params_to_torch(params, dev)
    with torch.no_grad():
        lt = forward(p, cfg, torch.from_numpy(x).to(dev)).cpu().numpy()
    ln, _ = np_forward(params, cfg, x)
    err = float(np.abs(lt - ln).max())
    add("forward logits match NumPy (fp32)", err < 1e-3, "max abs difference %.2e" % err)

    # 2. loss and gradients
    g_np = zero_grads(params)
    l_np, _ = loss_and_grads(params, cfg, x, y, m, grads=g_np)
    p = params_to_torch(params, dev)
    cnt = float(m.sum())
    l_t = accumulate_step(p, cfg, [(x, y, m)], [cnt], dev, "fp32")
    worst = max(_rel(p[k].grad.cpu().numpy(), g_np[k]) for k in p)
    add("loss matches NumPy", abs(l_t - l_np) < 1e-4, "torch %.6f numpy %.6f" % (l_t, l_np))
    add("all gradients match NumPy", worst < 1e-3, "worst relative error %.2e over %d tensors" % (worst, len(p)))

    # 3. accumulation == one big batch, with an uneven mask
    p1 = params_to_torch(params, dev)
    accumulate_step(p1, cfg, [(x, y, m)], [cnt], dev, "fp32")
    p2 = params_to_torch(params, dev)
    halves = [(x[:2], y[:2], m[:2]), (x[2:], y[2:], m[2:])]
    accumulate_step(p2, cfg, halves, step_denominators(halves, "batch"), dev, "fp32")
    worst = max(_rel(p2[k].grad.cpu().numpy(), p1[k].grad.cpu().numpy()) for k in p1)
    add("gradient accumulation == one big batch (masked)", worst < 1e-4, "worst relative error %.2e" % worst)

    # 4. AdamW step
    g = {k: np.random.default_rng(i).standard_normal(v.shape).astype(np.float32) for i, (k, v) in enumerate(params.items())}
    pn = {k: v.copy() for k, v in params.items()}
    an = AdamW(pn)
    pt = params_to_torch(params, dev)
    at = TorchAdamW(pt)
    for _ in range(3):
        an.step(pn, g, 1e-3)
        for k in pt:
            pt[k].grad = torch.from_numpy(g[k]).to(dev)
        at.step(pt, 1e-3)
    worst = max(float(np.abs(pt[k].detach().cpu().numpy() - pn[k]).max()) for k in pn)
    add("AdamW matches NumPy after 3 steps", worst < 1e-6, "max abs difference %.2e" % worst)

    # 5. stop-and-resume gives the same weights as never stopping; and a NumPy-started run continues on this backend
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data")
        os.makedirs(data)
        np.random.default_rng(0).integers(0, cfg.vocab_size, 6000).astype(np.uint16).tofile(os.path.join(data, "train.bin"))
        np.random.default_rng(1).integers(0, cfg.vocab_size, 600).astype(np.uint16).tofile(os.path.join(data, "val.bin"))
        hpx = dict(batch=2, accum=2, steps=4, lr=1e-3, min_lr=1e-4, warmup=1, eval_every=1000, ckpt_every=1000, log_every=1000)
        quiet = lambda *_: None
        dev_s = str(dev)
        a = train_torch(os.path.join(tmp, "a"), data, cfg, hpx, log=quiet, device=dev_s, precision="fp32")
        train_torch(os.path.join(tmp, "b"), data, cfg, hpx, log=quiet, device=dev_s, precision="fp32", max_steps_this_call=2)
        b = train_torch(os.path.join(tmp, "b"), data, None, None, log=quiet, device=dev_s, precision="fp32")
        worst = max(float(np.abs(a[k] - b[k]).max()) for k in a)
        add("stop + resume == uninterrupted run", worst == 0.0 if dev.type == "cpu" else worst < 1e-5,
            "max abs weight difference %.2e (%s)" % (worst, "must be exactly 0 on CPU" if dev.type == "cpu" else "GPU: tiny differences are allowed"))
        train_numpy(os.path.join(tmp, "c"), data, cfg, hpx, log=quiet, max_steps_this_call=2)       # started with NumPy ...
        c = train_torch(os.path.join(tmp, "c"), data, None, None, log=quiet, device=dev_s, precision="fp32")   # ... finished with torch
        worst = max(float(np.abs(a[k] - c[k]).max()) for k in a)
        add("NumPy-started run continues correctly on this backend", worst < 1e-4, "max abs weight difference vs all-torch run %.2e" % worst)

        # 6. low precision runs and stays close to fp32 (only the precisions this hardware supports)
        l32 = benchmark_loss(cfg, params, x, y, dev, "fp32")
        for prec in ("bf16", "fp16"):
            try:
                pick_precision(dev, prec)
            except RuntimeError as e:
                log("  [skip] %s: %s" % (prec, e))
                continue
            lp = benchmark_loss(cfg, params, x, y, dev, prec)
            add("%s loss is finite and close to fp32" % prec, math.isfinite(lp) and abs(lp - l32) < 0.05, "fp32 %.4f %s %.4f" % (l32, prec, lp))
    return results


def benchmark_loss(cfg, params, x, y, dev, prec):
    p = params_to_torch(params, dev)
    with torch.no_grad(), autocast(dev, prec):
        s = loss_sum(p, cfg, torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), None)
    return float(s.item()) / x.size
