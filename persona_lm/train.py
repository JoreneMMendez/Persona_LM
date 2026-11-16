"""Training loop (NumPy backend): pretraining and chat tuning use the same code (chat data simply carries a loss mask).

Everything needed to continue is in the checkpoint (weights, optimizer, random-number state, step), so stopping with Ctrl+C,
a crash or a reboot and starting again gives exactly the same result as never stopping.

The "start a new run or resume one" logic lives in start_or_resume(), shared with the PyTorch backend (torch_backend.py),
so both backends read and write the same checkpoint file and resume the same way.
"""
import json
import math
import os
import signal
import time
from dataclasses import dataclass

import numpy as np

from .ckpt import load_ckpt, save_ckpt
from .data import Stream
from .model import Config, init_params, loss_and_grads, zero_grads, forward
from .optim import AdamW, clip_grad_norm, lr_at
from .safety import backup_before_training, is_new_best, save_best, should_early_stop

DEFAULT_HP = dict(
    batch=8, accum=4, lr=6e-4, min_lr=6e-5, warmup=100, steps=10000, weight_decay=0.1, clip=1.0,
    eval_every=200, eval_batches=10, ckpt_every=100, ckpt_minutes=10, log_every=10, seed=1337,
    early_stop=0,           # stop after this many validation checks in a row without a new best (0 = off)
    save_best=1,            # 1 = keep the best-validation weights in ckpt.best.npz (0 = off)
    loss_norm="batch",      # "batch": divide by ALL trained tokens of the step (correct). "micro": legacy, per micro-batch (old chat runs)
)


def evaluate(params, cfg, stream, hp):
    rng = np.random.Generator(np.random.PCG64(4242))      # fixed, so validation numbers are comparable over time
    tot = 0.0
    for _ in range(hp["eval_batches"]):
        x, y, m = stream.batch(rng, hp["batch"], cfg.ctx)
        logits, _ = forward(params, cfg, x)
        mx = logits.max(-1, keepdims=True)
        lp = (logits - mx) - np.log(np.exp(logits - mx).sum(-1, keepdims=True))
        t = np.take_along_axis(lp, y[..., None], -1)[..., 0].astype(np.float64)
        mm = np.ones_like(t) if m is None else m.astype(np.float64)
        tot += float(-(t * mm).sum() / max(mm.sum(), 1.0))
    return tot / hp["eval_batches"]


def _meta(cfg, hp, step, rng, history, tok_hash=None, backend=None):
    meta = {"cfg": cfg.to_dict(), "hp": hp, "step": step, "rng": rng.bit_generator.state, "history": history}
    if tok_hash:
        meta["tokenizer_sha256"] = tok_hash
    if backend:
        meta["backend"] = backend
    return meta


def data_tokenizer_hash(data_dir):
    """The tokenizer fingerprint that `prepare` wrote next to the token files (None for data made before fingerprints existed)."""
    try:
        with open(os.path.join(data_dir, "meta.json")) as f:
            return json.load(f).get("tokenizer_sha256")
    except (OSError, ValueError):
        return None


def check_tokenizer(found, expected, what):
    if found and expected and found != expected:
        raise ValueError("tokenizer mismatch: the data was made with a different tokenizer than %s. Mixing tokenizers makes a model "
                         "read garbage. Re-run `prepare` with the tokenizer this model was trained with." % what)


def micro_batch_count(x, m):
    """How many positions of this micro-batch carry loss."""
    return float(x.size) if m is None else float(m.sum())


def step_denominators(batches, loss_norm):
    """One loss divisor per micro-batch of a step. Dividing the summed loss of micro-batch i by denoms[i] and adding everything up gives:
      "batch": the mean over all trained tokens of the step (identical to one big batch);
      "micro": the mean of per-micro-batch means (legacy behaviour, kept so old chat runs resume unchanged)."""
    counts = [max(micro_batch_count(x, m), 1.0) for x, _, m in batches]
    if loss_norm == "micro":
        return [c * len(batches) for c in counts]
    return [sum(counts)] * len(batches)


def install_stop_flag():
    """Ctrl+C / SIGTERM set a flag instead of killing the process, so the loop can write a checkpoint first."""
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except (ValueError, OSError):
            pass                                          # not the main thread (tests)
    return stop


@dataclass
class RunState:
    params: dict
    opt_arrays: dict          # None for a brand-new run
    cfg: Config
    hp: dict
    step: int
    history: list
    rng: np.random.Generator
    tok_hash: str
    meta: dict                # the stored checkpoint meta (None for a new run)


def start_or_resume(run_dir, data_dir, cfg, hp, init_from, log):
    """Shared by both backends. Resumes run_dir's checkpoint if there is one, otherwise starts a new run
    (optionally from the weights of init_from)."""
    os.makedirs(run_dir, exist_ok=True)
    overrides = dict(hp or {})          # only what the caller explicitly set
    data_tok = data_tokenizer_hash(data_dir)
    loaded = load_ckpt(os.path.join(run_dir, "ckpt.npz"))
    if loaded is not None:
        params, opt_arrays, meta = loaded
        cfg = Config.from_dict(meta["cfg"])
        stored = dict(meta["hp"])
        stored.setdefault("loss_norm", "micro")             # checkpoints written before the fix keep their old behaviour
        hp = {**DEFAULT_HP, **stored, **overrides}          # stored values, unless the caller changed them now
        check_tokenizer(data_tok, meta.get("tokenizer_sha256"), "the one this run started with")
        if stored.get("batch", 0) * stored.get("accum", 0) != hp["batch"] * hp["accum"]:
            log("note: effective batch changed from %d to %d sequences per step (batch x accum). Fine for fitting a smaller GPU as long as "
                "batch x accum stays the same." % (stored["batch"] * stored["accum"], hp["batch"] * hp["accum"]))
        rng = np.random.Generator(np.random.PCG64(0))
        rng.bit_generator.state = meta["rng"]
        log("resumed at step %d" % meta["step"])
        return RunState(params, opt_arrays, cfg, hp, meta["step"], meta["history"], rng, meta.get("tokenizer_sha256") or data_tok, meta)
    hp = {**DEFAULT_HP, **overrides}
    if cfg is None:
        raise ValueError("cfg is required for a new run")
    if init_from:
        got = load_ckpt(init_from)
        if got is None:
            raise ValueError("cannot read " + init_from)
        params = got[0]
        if Config.from_dict(got[2]["cfg"]) != cfg:
            raise ValueError("init_from model has a different size than cfg")
        check_tokenizer(data_tok, got[2].get("tokenizer_sha256"), "the model in --init-from")
    else:
        params = init_params(cfg, hp["seed"])
    rng = np.random.Generator(np.random.PCG64(hp["seed"]))
    log("new run: %.2fM parameters" % (cfg.n_params() / 1e6))
    return RunState(params, None, cfg, hp, 0, [], rng, data_tok, None)


def open_streams(data_dir, cfg):
    train_s = Stream(data_dir, "train")
    val_path = os.path.join(data_dir, "val.bin")
    val_s = Stream(data_dir, "val") if os.path.exists(val_path) and os.path.getsize(val_path) > 2 * cfg.ctx * 2 else None
    return train_s, val_s


def train(run_dir, data_dir, cfg=None, hp=None, init_from=None, log=print, max_steps_this_call=None, tokenizer=None, backup=True):
    """Train (or continue) the run in run_dir. Returns the final params.

    cfg is needed for a new run. If run_dir already has a checkpoint it is resumed and cfg/hp come from it
    (hp values passed now override the stored ones, e.g. to extend `steps`).
    init_from: path of a checkpoint whose WEIGHTS start a new run (used to chat-tune a pretrained model).
    max_steps_this_call: stop after this many steps (mainly for tests); the checkpoint is still written.
    """
    ck_path = os.path.join(run_dir, "ckpt.npz")
    stop = install_stop_flag()
    if backup:
        backup_before_training(run_dir, tokenizer, log)          # before anything can change the files
    st = start_or_resume(run_dir, data_dir, cfg, hp, init_from, log)
    params, cfg, hp, step, history, rng = st.params, st.cfg, st.hp, st.step, st.history, st.rng
    opt = AdamW(params, weight_decay=hp["weight_decay"])
    if st.opt_arrays is not None:
        opt.load_state(st.opt_arrays)

    train_s, val_s = open_streams(data_dir, cfg)
    grads = zero_grads(params)
    last_ck = time.time()
    t0 = time.time()
    done_here = 0
    early = False
    tokens_per_step = hp["batch"] * hp["accum"] * cfg.ctx

    def checkpoint():
        save_ckpt(ck_path, params, opt.state(), _meta(cfg, hp, step, rng, history, st.tok_hash, {"name": "numpy"}))

    while step < hp["steps"] and not stop["flag"] and (max_steps_this_call is None or done_here < max_steps_this_call):
        lr = lr_at(step, hp["steps"], hp["lr"], hp["min_lr"], hp["warmup"])
        for g in grads.values():
            g.fill(0)
        batches = [train_s.batch(rng, hp["batch"], cfg.ctx) for _ in range(hp["accum"])]     # same draws, in the same order, as ever
        denoms = step_denominators(batches, hp["loss_norm"])
        loss = 0.0
        for (x, y, m), dn in zip(batches, denoms):
            l, _ = loss_and_grads(params, cfg, x, y, m, grads=grads, denom=dn)
            loss += l
        if not math.isfinite(loss):
            raise FloatingPointError("loss became %s at step %d: lower the learning rate" % (loss, step))
        gn = clip_grad_norm(grads, hp["clip"])
        opt.step(params, grads, lr)
        step += 1
        done_here += 1
        rec = {"step": step, "loss": round(loss, 5), "lr": lr, "grad_norm": round(gn, 4)}
        if val_s is not None and (step % hp["eval_every"] == 0 or step == hp["steps"]):
            rec["val_loss"] = round(evaluate(params, cfg, val_s, hp), 5)
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
                save_best(run_dir, params, _meta(cfg, hp, step, rng, history, st.tok_hash, {"name": "numpy", "best": True}))
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
    return params
