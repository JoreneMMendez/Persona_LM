"""`python -m persona_lm plan-steps`: how many training steps, and a ready-to-run train command that stops at the best point.

  python -m persona_lm plan-steps --data-dir data_chat4096 --size s30m --run work3\\run_chat4 --init-from work3\\run_pre\\ckpt.npz
  python -m persona_lm plan-steps --data-dir data_chat4096 --size s30m --passes 2          # fewer passes over the data
  python -m persona_lm plan-steps --from-run work3\\run_chat2 --data-dir data_chat4096      # where did an earlier run peak?

Chat tuning on a small personal data set overfits after about 3 passes (your 3000-step run peaked at step 800 and then answered
every message with a stock reply). This command does the arithmetic first: steps = passes x training tokens / tokens per step,
then it warns about what usually goes wrong and prints the command with --early-stop and --save-best switched on, so the run
stops by itself when validation loss stops improving and keeps the best weights in ckpt.best.npz.
It only reads token-folder sizes and checkpoint metadata. It changes nothing and never prints chat text.
"""
import math
import os

from .model import PRESETS, preset
from .report import data_summary, read_ckpt_meta

DEFAULT_PASSES = {"chat": 3.0, "pretrain": 1.0}
OVERFIT_PASSES = 4.0            # your own runs: val loss rose after about 3 passes on ~1,440 messages


def tokens_per_step(batch, accum, ctx):
    return batch * accum * ctx


def steps_for(passes, train_tokens, per_step):
    return max(1, int(round(passes * train_tokens / per_step)))


def passes_for(steps, train_tokens, per_step):
    return steps * per_step / max(train_tokens, 1)


def pick_eval_every(steps):
    """About 8 validation checks per run, rounded to a multiple of 5 (800 steps -> 100, 105 steps -> 15)."""
    return max(5, int(round(steps / 8 / 5.0)) * 5)


def pick_warmup(steps):
    return max(8, min(100, int(round(steps * 0.025)))) if steps >= 32 else max(1, steps // 4)


def make_plan(summary, ctx, batch=1, accum=4, passes=None, size="s30m", tokens_per_sec=None):
    kind = summary["kind"] if summary["kind"] in DEFAULT_PASSES else "chat"
    passes = passes if passes is not None else DEFAULT_PASSES[kind]
    per_step = tokens_per_step(batch, accum, ctx)
    train = summary["train_tokens"]
    steps = steps_for(passes, train, per_step)
    plan = {"kind": kind, "passes": passes, "per_step": per_step, "train_tokens": train, "val_tokens": summary["val_tokens"], "steps": steps,
            "steps_per_pass": train / per_step, "eval_every": pick_eval_every(steps), "warmup": pick_warmup(steps), "ctx": ctx,
            "batch": batch, "accum": accum, "warnings": []}
    plan["lr"], plan["min_lr"] = (1e-4, 1e-5) if kind == "chat" else (6e-4, 6e-5)
    w = plan["warnings"]
    if passes > OVERFIT_PASSES and kind == "chat":
        w.append("%.1f passes is a lot for chat tuning on a small personal data set: your 3000-step run (about 12 passes) overfit and answers every message with a stock reply. Stay near 3." % passes)
    if summary["val_tokens"] <= 2 * ctx:
        w.append("the validation data is smaller than 2 windows of %d tokens, so validation loss will NOT be computed and --early-stop / ckpt.best.npz cannot work. Make more validation data with `prepare`." % ctx)
    if steps < 8:
        w.append("only %d step(s): the data is smaller than one batch of %d tokens. Use a smaller --batch/--accum or more data." % (steps, per_step))
    elif steps // plan["eval_every"] < 3:
        w.append("fewer than 3 validation checks in the run: early stopping has little to go on.")
    if kind == "chat" and summary.get("loss_tokens") is not None and summary["loss_tokens"] < 50_000:
        w.append("only %s tokens count for learning (your replies): expect a rough hint of your style at best." % "{:,}".format(summary["loss_tokens"]))
    if tokens_per_sec:
        plan["minutes"] = steps * per_step / tokens_per_sec / 60.0
    return plan


def command(plan, run, data_dir, size, init_from=None):
    parts = ["python -m persona_lm train --run %s --data-dir %s --size %s" % (run, data_dir, size)]
    if init_from:
        parts.append("--init-from %s" % init_from)
    parts.append("--steps %d --lr %g --min-lr %g --warmup %d --eval-every %d --batch %d --accum %d --early-stop 3 --save-best 1" % (
        plan["steps"], plan["lr"], plan["min_lr"], plan["warmup"], plan["eval_every"], plan["batch"], plan["accum"]))
    return " ".join(parts)


def history_of(run_dir):
    """(history list, per_step, cfg dict) from a run folder's checkpoint metadata; (None, None, None) when unreadable."""
    meta = read_ckpt_meta(os.path.join(run_dir, "ckpt.npz"))
    if not meta:
        return None, None, None
    hp, cfg = meta.get("hp") or {}, meta.get("cfg") or {}
    per_step = tokens_per_step(hp.get("batch", 1), hp.get("accum", 1), cfg.get("ctx", 0)) if cfg.get("ctx") else None
    return meta.get("history", []), per_step, meta


def where_it_peaked(history, per_step, train_tokens):
    vals = [(r["step"], r["val_loss"]) for r in history if "val_loss" in r]
    if not vals:
        return None
    best = min(vals, key=lambda t: t[1])
    last = vals[-1]
    return {"best_step": best[0], "best_val": best[1], "last_step": last[0], "last_val": last[1],
            "best_passes": passes_for(best[0], train_tokens, per_step) if per_step else None,
            "last_passes": passes_for(last[0], train_tokens, per_step) if per_step else None}


def add_arguments(s):
    s.add_argument("--data-dir", required=True, help="the token folder (made by `prepare`)")
    s.add_argument("--size", choices=sorted(PRESETS), default="s30m")
    s.add_argument("--ctx", type=int, default=None, help="override the preset's context length")
    s.add_argument("--batch", type=int, default=1, help="default 1 (fits a 6 GB GPU at ctx 4096)")
    s.add_argument("--accum", type=int, default=4, help="default 4: batch x accum = sequences per step")
    s.add_argument("--passes", type=float, default=None, help="how many times to read the data (default 3 for chat data, 1 for pretraining text)")
    s.add_argument("--run", default="work3\\run_chatN", help="the run folder to put in the printed command")
    s.add_argument("--init-from", default=None, help="the pretrained base to start from (chat tuning)")
    s.add_argument("--tokens-per-sec", type=float, default=None, help="your speed from `benchmark`, to estimate the time")
    s.add_argument("--from-run", default=None, help="an earlier run folder: say at which step and pass its validation loss was best")


def cmd(a, say=print):
    import sys
    try:
        summary = data_summary(a.data_dir)
    except (OSError, ValueError) as e:
        sys.exit("error: %s" % e)
    cfg = preset(a.size, summary["vocab_size"])
    ctx = a.ctx or cfg.ctx
    plan = make_plan(summary, ctx, a.batch, a.accum, a.passes, a.size, a.tokens_per_sec)
    say("Step plan for %s  (kind: %s, model %s, window %d tokens)" % (a.data_dir, plan["kind"], a.size, ctx))
    say("  training tokens: %s   validation tokens: %s" % ("{:,}".format(plan["train_tokens"]), "{:,}".format(plan["val_tokens"])))
    say("  one step reads %d x %d x %d = %s tokens, so one pass over the data = %.0f steps" % (a.batch, a.accum, ctx, "{:,}".format(plan["per_step"]), plan["steps_per_pass"]))
    say("  %.1f passes  ->  %d steps   (validation every %d, warm-up %d, learning rate %g down to %g)" % (plan["passes"], plan["steps"], plan["eval_every"], plan["warmup"], plan["lr"], plan["min_lr"]))
    if "minutes" in plan:
        say("  at %.0f tokens/s that is about %.0f minutes" % (a.tokens_per_sec, plan["minutes"]))
    for w in plan["warnings"]:
        say("  WARNING: " + w)
    if a.from_run:
        hist, per_step, _ = history_of(a.from_run)
        peak = where_it_peaked(hist, per_step or plan["per_step"], plan["train_tokens"]) if hist else None
        if peak is None:
            say("  %s: no validation numbers found in its checkpoint (or no checkpoint)." % a.from_run)
        else:
            say("  %s: validation loss was best (%.3f) at step %d%s; the latest check (step %d) was %.3f." % (
                a.from_run, peak["best_val"], peak["best_step"], " = about %.1f passes" % peak["best_passes"] if peak["best_passes"] else "",
                peak["last_step"], peak["last_val"]))
            if peak["last_val"] > peak["best_val"] * 1.05 and peak["last_step"] > peak["best_step"]:
                say("  It overfit after that peak. For a repeat run use about %d steps (the plan above) with --early-stop 3." % max(plan["steps"], 1))
    say("")
    say("Run this (early stop and best-checkpoint saving are on; chat with ckpt.best.npz afterwards):")
    say("  " + command(plan, a.run, a.data_dir, a.size, a.init_from))
    say("Then: python -m persona_lm chat-probe --ckpt %s\\ckpt.best.npz   (does it still give the same reply to everything?)" % a.run)
