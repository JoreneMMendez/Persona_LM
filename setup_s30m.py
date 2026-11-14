"""setup_s30m.py - prepare a NEW work folder (work2) for the bigger s30m model (about 33M parameters), without touching your s10m work folder.

Usage (from the project folder):
    python setup_s30m.py                  # makes work2 from work, fine-tune data = data_msgr_clean (the merged set from clean_chat.py)
    python setup_s30m.py --dry-run        # shows what it would do, changes nothing

What it does, in this order (the order matters: the project settings are written BEFORE the frozen tokenizer is copied in,
because a folder that already holds a tokenizer refuses to change its settings):
  1. reads work\\project.json (languages, vocabulary, window, public text folder, system line) and reuses those values;
  2. creates work2\\project.json with size s30m and your chat data = data_msgr_clean\\train.jsonl / val.jsonl;
  3. copies work\\tok.json (the frozen tokenizer: s30m MUST use the same one) and work\\data_pre (the public-text token files) into work2.
It does NOT copy work\\data_chat or work\\run_pre/run_chat: the chat tokens are made again from the merged set, and an s10m checkpoint
cannot be loaded into an s30m model.

After it: put the pretrained s30m checkpoint (from the Kaggle notebook) in work2\\run_pre\\ckpt.npz, or let `pipeline run` pretrain it locally (slow),
then run   python -m persona_lm pipeline run --dir work2   once per step (it prints what is next).
"""
import argparse
import os
import shutil
import sys

from persona_lm import pipeline as PL
from persona_lm.bpe import Tokenizer

DEFAULT_STEPS = 18000                # the Kaggle notebook's setting (about 295M tokens at 16,384 tokens per step)


def plan(args):
    """Works out every setting from the old project; returns (settings dict for init_project, list of problems)."""
    problems = []
    old_dir, chat_dir = args.from_dir, args.chat_dir
    try:
        old = PL.load_project(old_dir)
    except PL.ProjectError as e:
        problems.append(str(e))
        old = dict(PL.DEFAULTS)
    if not os.path.exists(os.path.join(old_dir, "tok.json")):
        problems.append("%s\\tok.json not found: the s30m model must use the tokenizer that was frozen in the s10m folder." % old_dir)
    for name in ("train.jsonl", "val.jsonl"):
        if not os.path.exists(os.path.join(chat_dir, name)):
            problems.append("%s not found. Make it first with:  python clean_chat.py" % os.path.join(chat_dir, name))
    settings = dict(languages=",".join(old["languages"]) if old["languages"] else None, vocab=old["vocab"], size="s30m", ctx=old["ctx"],
                    pretrain_text=old["pretrain_text"] or None, chat_train=os.path.join(chat_dir, "train.jsonl"),
                    chat_val=os.path.join(chat_dir, "val.jsonl"), pretrain_steps=args.pretrain_steps, batch=args.batch, accum=args.accum,
                    system=old["system"])
    if settings["languages"]:
        settings["languages"] = settings["languages"].split(",")
    return settings, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="prepare work2 for the s30m model")
    ap.add_argument("--from-dir", default="work", help="the existing (s10m) work folder with the frozen tok.json and data_pre (default: work)")
    ap.add_argument("--dir", default="work2", help="the new work folder (default: work2)")
    ap.add_argument("--chat-dir", default="data_msgr_clean", help="folder with the merged train.jsonl / val.jsonl from clean_chat.py (default: data_msgr_clean)")
    ap.add_argument("--pretrain-steps", type=int, default=DEFAULT_STEPS, help="default %d: the Kaggle notebook's STEPS" % DEFAULT_STEPS)
    ap.add_argument("--batch", type=int, default=4, help="micro-batch; 4 should fit the 6 GB laptop GPU (not measured for s30m)")
    ap.add_argument("--accum", type=int, default=8, help="4 x 8 x 512 = 16,384 tokens per step, same as the Kaggle run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if os.path.abspath(a.dir) == os.path.abspath(a.from_dir):
        sys.exit("--dir and --from-dir are the same folder. work2 must be a NEW folder (s10m and s30m checkpoints cannot share one).")
    settings, problems = plan(a)
    if problems:
        for p in problems:
            print("Problem: " + p)
        sys.exit(1)
    exists = os.path.exists(PL.project_path(a.dir))
    print("Old folder: %s   New folder: %s%s" % (a.from_dir, a.dir, "   (already has a project.json: only changed settings are updated)" if exists else ""))
    for k in ("size", "ctx", "vocab", "languages", "pretrain_text", "chat_train", "chat_val", "pretrain_steps", "batch", "accum"):
        print("   %-15s %s" % (k, settings[k]))
    if a.dry_run:
        print("\nDry run: nothing was changed.")
        return 0
    try:
        PL.init_project(a.dir, **settings)                                   # BEFORE tok.json is copied in (see the top of this file)
    except PL.ProjectError as e:
        sys.exit("error: %s" % e)
    for name in ("tok.json", "data_pre"):
        src, dst = os.path.join(a.from_dir, name), os.path.join(a.dir, name)
        if not os.path.exists(src):
            print("Note: %s not found, not copied (pretraining needs data_pre: run `pipeline run --dir %s` to make it)." % (src, a.from_dir))
        elif os.path.exists(dst):
            print("%s already there, left as it is." % dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
            print("Copied %s" % dst)
        else:
            shutil.copy2(src, dst)
            print("Copied %s" % dst)
    tok_path = os.path.join(a.dir, "tok.json")
    if os.path.exists(tok_path):
        proj = PL.load_project(a.dir)
        proj["tokenizer_sha256"] = Tokenizer.load(tok_path).fingerprint()
        proj["decisions_confirmed"] = True                                   # the tokenizer already exists, so there is nothing left to confirm
        PL.save_project(a.dir, proj)
    print("\nNext:\n  python -m persona_lm pipeline plan --dir %s       (shows every step; the pretrain step is done once run_pre\\ckpt.npz has %d steps)" % (a.dir, a.pretrain_steps))
    print("  python -m persona_lm pipeline run --dir %s        (one step at a time; chat-data is made from your merged set)" % a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
