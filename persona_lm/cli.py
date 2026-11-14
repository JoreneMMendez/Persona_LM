"""Command line: python -m persona_lm <command> ...   (see README.md for the whole workflow)"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import data as D
from . import messenger as M
from .backend import NO_TORCH_MSG, choose_backend, torch_installed
from .bpe import Tokenizer
from .ckpt import load_ckpt
from .generate import chat_loop, complete
from .gguf import export_model, params_from_gguf, read_gguf
from .model import PRESETS, Config, forward, init_params, loss_and_grads, preset, zero_grads
from .train import DEFAULT_HP, train


def _texts_for_tokenizer(paths, limit_mb):
    used = 0
    seen = set()          # chat messages repeat (in Messenger "window" data one message is context for ~8 examples): count each text once
    for path in D._files(paths):
        ext = os.path.splitext(path)[1].lower()
        if ext in D.TEXT_EXT:
            for text, _ in D._text_blocks(path):
                used += len(text)
                yield text
        elif ext in D.JSON_EXT:
            for rec in D._json_records(path):
                msgs = D.as_messages(rec)
                if msgs:
                    for _, c in msgs:
                        if c in seen:
                            continue
                        seen.add(c)
                        used += len(c)
                        yield c
                else:
                    t = rec if isinstance(rec, str) else (rec.get("text") if isinstance(rec, dict) else None)
                    if isinstance(t, str):
                        used += len(t)
                        yield t
        if used > limit_mb * 1e6:
            return


def cmd_tokenizer(a):
    t0 = time.time()
    tok = Tokenizer.train(_texts_for_tokenizer(a.data, a.max_mb), a.vocab, log=print)
    tok.save(a.out)
    print("saved %s: %d tokens in %.0f s" % (a.out, tok.vocab_size, time.time() - t0))


def cmd_prepare(a):
    tok = Tokenizer.load(a.tokenizer)
    chat = a.kind == "chat"
    make = D.chat_docs if chat else D.pretrain_docs
    val_docs = make(a.val_data, tok) if a.val_data else None      # explicit validation files: no automatic split
    D.prepare(tok, make(a.data, tok), a.out, with_mask=chat, val_every=a.val_every, val_docs=val_docs)


def cmd_messenger_scan(a):
    threads = M.load_table(a.export)
    me = {n.casefold() for n in (a.me or [])} or None
    print(M.format_table(threads, me, a.top))
    print("\nBusiest senders (one of them is probably you):")
    for name, n in M.rank_senders(threads).most_common(8):
        print("  %-32s %d messages" % (name, n))
    print("\nNext: python -m persona_lm messenger-import %s --out data_msgr --me \"YOUR NAME\" --chats 1,2,3   (or leave out --me/--chats to be asked)" % a.export)


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def cmd_messenger_import(a):
    interactive = sys.stdin.isatty()
    threads = M.load_table(a.export)
    senders = M.rank_senders(threads)
    me = list(a.me or [])
    if not me:
        if not interactive:
            raise SystemExit("--me \"Your Name\" is required when this is not an interactive terminal (see `messenger-scan` for the names)")
        me = M.choose_me(senders)
    lowered = {n.casefold() for n in senders}
    unknown = [n for n in me if n.casefold() not in lowered]
    if unknown:
        raise SystemExit("not a sender name in this export: %s. Run `messenger-scan` to see the exact spelling." % unknown)
    chats = a.chats
    if chats is None and interactive:
        print(M.format_table(threads, {n.casefold() for n in me}, a.top))
        chats = input("\nWhich chats should the AI learn from? (all, or numbers like 1,3,5-9): ")
    chats = "all" if chats is None else (chats if chats.strip().lower() == "all" else M.parse_selection(chats, len(threads)))
    if a.no_anonymize:
        print("WARNING: anonymization is OFF. Use this data only on your own computer. NEVER upload it to a cloud GPU.")
    aliases = dict(x.split("=", 1) for x in (a.alias or []) if "=" in x)
    extra = _read_lines(a.extra_names) if a.extra_names else []
    stats = M.import_export(a.export, a.out, me, chats, a.mode, a.context, a.session_gap_hours, a.max_chars, a.val_fraction, a.system,
                            not a.no_anonymize, extra, aliases, a.paranoid, a.min_my_messages, not a.no_groups, a.max_participants,
                            log=print, threads=threads)
    print("\n=== import finished: %s ===" % a.out)
    print("%d chats, %d sessions, %d of your messages (%d turns), period %s" % (
        stats["chats_imported"], stats["sessions"], stats["my_messages"], stats["my_turns"], " to ".join(stats["period"] or ["?"])))
    print("examples: %d train, %d validation (validation = the most recent %d sessions)" % (stats["examples_train"], stats["examples_val"], stats["val_sessions"]))
    print("dropped message kinds: %s" % stats["dropped_message_kinds"])
    print("\nIS MY DATA ENOUGH?\n  " + stats["data_size_verdict"])
    if stats["anonymized"]:
        print("\nanonymizer replaced: %s" % stats["anonymizer_counts"])
        if stats["audit"]["leaks"]:
            print("!! LEAKS FOUND after anonymizing: %s -- details in %s. Do NOT upload this data." % (stats["audit"]["leaks"], os.path.join(a.out, "private", "audit.txt")))
        else:
            print("audit: no known name, e-mail, link or phone number left in the output.")
        if stats["audit"]["suspicious"]:
            print("audit: %s look suspicious (long digit runs, @ signs ...): read %s" % (stats["audit"]["suspicious"], os.path.join(a.out, "private", "audit.txt")))
        print("Names of people who are NOT in the chats cannot be detected. Read %s and put real names in an --extra-names file, then re-run." % os.path.join(a.out, "private", "capitalised_words.txt"))
        print("Addresses are only found by pattern and some will be missed: read a sample of train.jsonl yourself before uploading anything.")
    print("\nNEVER upload the folder %s. Only train.jsonl / val.jsonl (or the token files made from them) may leave your computer." % os.path.join(a.out, "private"))
    print("Next: python -m persona_lm prepare --kind chat --tokenizer tok.json --data %s --val-data %s --out data_chat" % (
        os.path.join(a.out, "train.jsonl"), os.path.join(a.out, "val.jsonl")))
    if stats["anonymized"] and stats["audit"]["leaks"]:
        sys.exit(2)


def _cfg_for(a, vocab):
    cfg = preset(a.size, vocab)
    if a.ctx:
        cfg.ctx = a.ctx
    return cfg


def cmd_train(a):
    hp = {k: getattr(a, k) for k in DEFAULT_HP if getattr(a, k) is not None}
    with open(os.path.join(a.data_dir, "meta.json")) as f:
        meta = json.load(f)
    cfg = _cfg_for(a, meta["vocab_size"])
    if choose_backend(a.backend) == "torch":
        from .torch_backend import train_torch
        train_torch(a.run, a.data_dir, cfg, hp, init_from=a.init_from, device=a.device, precision=a.precision, deterministic=a.deterministic)
    else:
        train(a.run, a.data_dir, cfg, hp, init_from=a.init_from)


def _load_model(a):
    got = load_ckpt(a.ckpt)
    if got is None:
        sys.exit("cannot read checkpoint " + a.ckpt)
    return got[0], Config.from_dict(got[2]["cfg"]), Tokenizer.load(a.tokenizer)


def cmd_generate(a):
    params, cfg, tok = _load_model(a)
    print(complete(params, cfg, tok, a.prompt, max_new=a.max_new, temperature=a.temperature, top_k=a.top_k, seed=a.seed))


def cmd_chat(a):
    params, cfg, tok = _load_model(a)
    chat_loop(params, cfg, tok, system=a.system, max_new=a.max_new, temperature=a.temperature, top_k=a.top_k)


def cmd_export(a):
    params, cfg, tok = _load_model(a)
    export_model(a.out, params, cfg, tok, kind=a.type, name=a.name)
    print("wrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))
    if a.verify:
        _, tensors = read_gguf(a.out)
        p2 = params_from_gguf(tensors, cfg)
        ids = np.array([tok.encode("Hello world")[: cfg.ctx] or [0]], dtype=np.int64)
        l1, _ = forward(params, cfg, ids)
        l2, _ = forward(p2, cfg, ids)
        print("verify: max logit difference vs the checkpoint = %.5f" % float(np.abs(l1 - l2).max()))


def _days(tokens, tps):
    return "%.1f days" % (tokens / tps / 86400) if tps * 86400 < tokens * 3 else "%.1f hours" % (tokens / tps / 3600)


def cmd_benchmark(a):
    cfg = _cfg_for(a, a.vocab)
    if choose_backend(a.backend) == "torch":
        from . import torch_backend as tb
        for line in tb.environment_report():
            print(line)
        sizes = [int(b) for b in a.try_batches.split(",")] if a.try_batches else [a.batch]
        results = []
        for b in sizes:
            try:
                tps, peak, dev, prec = tb.benchmark_torch(cfg, b, a.device, a.precision)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("  micro-batch %d: out of GPU memory (use a smaller --batch and a larger --accum)" % b)
                    tb.torch.cuda.empty_cache()
                    break
                raise
            results.append((b, tps))
            print("  micro-batch %d: %.0f tokens/s on %s, precision %s%s" % (
                b, tps, dev, prec, (", peak GPU memory %.2f GB" % (peak / 2 ** 30)) if peak else ""))
        if not results:
            sys.exit("nothing fit in memory: try --size tiny or a smaller --ctx")
        best = max(results, key=lambda r: r[1])
        print("%s: %.1fM parameters (vocab %d, context %d). Best measured speed: %.0f tokens/s at micro-batch %d" % (
            a.size, cfg.n_params() / 1e6, cfg.vocab_size, cfg.ctx, best[1], best[0]))
        for toks in (1e8, 1e9, 3e9):
            print("  %.0e training tokens would take about %s (one step of the training loop is slower than this raw forward+backward)" % (toks, _days(toks, best[1])))
        return
    params = init_params(cfg, 0)
    rng = np.random.Generator(np.random.PCG64(0))
    x = rng.integers(0, cfg.vocab_size, size=(a.batch, cfg.ctx)).astype(np.int64)
    y = np.roll(x, -1, axis=1)
    g = zero_grads(params)
    loss_and_grads(params, cfg, x, y, grads=g)                  # warm-up
    t0 = time.time()
    n = 3
    for _ in range(n):
        loss_and_grads(params, cfg, x, y, grads=g)
    tps = n * a.batch * cfg.ctx / (time.time() - t0)
    print("%s: %.1fM parameters, %.0f tokens/s on this machine (no GPU, NumPy)" % (a.size, cfg.n_params() / 1e6, tps))
    for toks in (1e8, 1e9, 3e9):
        print("  %.0e training tokens would take about %.1f days" % (toks, toks / tps / 86400))


def cmd_check_gpu(a):
    if not torch_installed():
        sys.exit(NO_TORCH_MSG)
    from . import torch_backend as tb
    for line in tb.environment_report():
        print(line)
    print("Checking the PyTorch backend against the NumPy reference (this takes a few seconds)...")
    results = tb.selfcheck(a.device)
    failed = [r for r in results if not r[1]]
    print("\n%d checks passed, %d failed." % (len(results) - len(failed), len(failed)))
    if failed:
        print("Please copy everything above and send it back: the failing lines say exactly what differs.")
    sys.exit(1 if failed else 0)


def _add_backend_args(s):
    s.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto", help="auto = PyTorch if installed, else slow NumPy CPU")
    s.add_argument("--device", default="auto", help="(PyTorch) auto, cpu, cuda or cuda:1; auto picks the GPU if there is one")
    s.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                   help="(PyTorch) auto = bf16 on Ampere+ GPUs, fp16 on older GPUs, fp32 on CPU")


def build_parser():
    p = argparse.ArgumentParser(prog="persona_lm", description="Train a small language model from scratch (pure Python + NumPy).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tokenizer", help="learn a BPE vocabulary from text")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--vocab", type=int, default=8000)
    s.add_argument("--max-mb", type=float, default=200, help="read at most this much text")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_tokenizer)

    s = sub.add_parser("prepare", help="turn text/JSON into token files")
    s.add_argument("--kind", choices=["pretrain", "chat"], required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-every", type=int, default=50)
    s.add_argument("--val-data", nargs="+", default=None, help="validation files; if given, --data is all training and no automatic split is made")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("messenger-scan", help="list the conversations in a Messenger JSON export (read-only, local)")
    s.add_argument("export", help="the unzipped export folder (or its messages/ folder)")
    s.add_argument("--me", action="append", help="your name, to show how many messages in each chat are yours")
    s.add_argument("--top", type=int, default=60)
    s.set_defaults(fn=cmd_messenger_scan)

    s = sub.add_parser("messenger-import", help="turn a Messenger JSON export into chat training data (anonymized by default)")
    s.add_argument("export")
    s.add_argument("--out", required=True, help="output folder: train.jsonl, val.jsonl, stats.json, private/")
    s.add_argument("--me", action="append", help="the sender name that is YOU (repeat the option if you appear under several names); asked if missing")
    s.add_argument("--chats", default=None, help="which chats: 'all' or the numbers from `messenger-scan`, e.g. 1,3,5-9; asked if missing")
    s.add_argument("--mode", choices=["window", "session"], default="window", help="window: one example per reply of yours with the previous --context turns; session: whole conversations")
    s.add_argument("--context", type=int, default=8, help="window mode: how many previous turns the AI sees")
    s.add_argument("--session-gap-hours", type=float, default=6.0, help="a silence longer than this starts a new session (context never crosses it)")
    s.add_argument("--max-chars", type=int, default=1400, help="size limit of one example in characters (about 400 tokens; the model's window is 512 tokens)")
    s.add_argument("--val-fraction", type=float, default=0.05, help="share of the most recent sessions held out for validation")
    s.add_argument("--system", default=None, help="optional system line put in front of every example")
    s.add_argument("--min-my-messages", type=int, default=20, help="skip chats where you wrote fewer messages than this")
    s.add_argument("--no-groups", action="store_true", help="skip group chats")
    s.add_argument("--max-participants", type=int, default=12, help="skip chats with more people than this")
    s.add_argument("--no-anonymize", action="store_true", help="keep real names/numbers (ONLY for data that never leaves your computer)")
    s.add_argument("--paranoid", action="store_true", help="drop whole messages that contain an address-like text or a long number instead of masking it")
    s.add_argument("--extra-names", default=None, help="text file, one name per line: people who are mentioned but are not in the chats")
    s.add_argument("--alias", action="append", help="nickname=Real Name, e.g. --alias \"Mikee=Michael Santos\" (repeatable)")
    s.add_argument("--top", type=int, default=60, help="rows of the chat table to show")
    s.set_defaults(fn=cmd_messenger_import)

    s = sub.add_parser("train", help="train or continue a run (Ctrl+C is safe; run the same command to continue)")
    s.add_argument("--run", required=True, help="folder for checkpoints and logs")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--size", choices=sorted(PRESETS), default="s10m")
    s.add_argument("--ctx", type=int, default=None, help="override the preset's context length")
    s.add_argument("--init-from", default=None, help="start from this checkpoint's weights (chat tuning)")
    for k, v in DEFAULT_HP.items():
        s.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=None, help="default %s" % (v,))
    _add_backend_args(s)
    s.add_argument("--deterministic", action="store_true", help="(PyTorch) ask for deterministic GPU kernels; slower, and not guaranteed bit-exact on every GPU")
    s.set_defaults(fn=cmd_train)

    for name, fn in (("generate", cmd_generate), ("chat", cmd_chat), ("export-gguf", cmd_export)):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True)
        s.add_argument("--tokenizer", required=True)
        if name != "export-gguf":
            s.add_argument("--max-new", type=int, default=120)
            s.add_argument("--temperature", type=float, default=0.8)
            s.add_argument("--top-k", type=int, default=40)
        if name == "generate":
            s.add_argument("--prompt", required=True)
            s.add_argument("--seed", type=int, default=0)
        if name == "chat":
            s.add_argument("--system", default=None)
        if name == "export-gguf":
            s.add_argument("--out", required=True)
            s.add_argument("--type", choices=["f32", "f16", "q8_0"], default="f16")
            s.add_argument("--name", default="persona-lm")
            s.add_argument("--verify", action="store_true", help="read the file back and compare with the checkpoint")
        s.set_defaults(fn=fn)

    s = sub.add_parser("benchmark", help="measure speed on this computer and estimate training time")
    s.add_argument("--size", choices=sorted(PRESETS), default="s10m")
    s.add_argument("--ctx", type=int, default=None)
    s.add_argument("--vocab", type=int, default=8000)
    s.add_argument("--batch", type=int, default=4, help="micro-batch size to measure")
    s.add_argument("--try-batches", default=None, help="comma list, e.g. 1,2,4,8,16: measure each and stop at the first that does not fit in GPU memory")
    _add_backend_args(s)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("check-gpu", help="check the PyTorch/GPU backend against the NumPy reference and print a report")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_check_gpu)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):                      # a Windows console can fail to print emoji in chat titles and names
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    a.fn(a)
