"""Command line: python -m persona_lm <command> ...   (see README.md for the whole workflow)"""
import argparse
from collections import Counter
import json
import os
import shlex
import sys
import time

import numpy as np

from . import data as D
from . import messenger as M
from .backend import NO_TORCH_MSG, choose_backend, torch_installed
from .bpe import Tokenizer
from .ckpt import load_ckpt
from . import pipeline as PL
from . import report as R
from .chatrepl import ChatSession, run_repl
from . import memory as MEM
from . import archive as ARC
from . import memory_export as MEX
from . import emotion as EMO
from . import person as PER
from . import identity as ID
from . import lineage as LIN
from . import proposals as PRO
from . import personas as PERS
from .generate import complete, sample_ids
from .fastgen import sample_ids_cached
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


def _settings_path():
    """persona_settings.json in the folder you run commands from (holds your real name and export path: private, never uploaded, see .gitignore).
    The environment variable PERSONA_SETTINGS can point somewhere else (the tests use it)."""
    return os.environ.get("PERSONA_SETTINGS") or "persona_settings.json"


def _settings_load():
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _settings_save(**changes):
    d = _settings_load()
    d.update({k: v for k, v in changes.items() if v})
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def _export_folder(a):
    """The export folder typed now, else the one remembered from last time."""
    folder = a.export or _settings_load().get("export")
    if not folder:
        raise SystemExit("Give the export folder once, e.g.  python -m persona_lm messenger-scan \"C:\\path\\to\\your_facebook_activity\"  (it is remembered after that).")
    return folder


def cmd_messenger_scan(a):
    folder = _export_folder(a)
    threads = M.load_table(folder)
    _settings_save(export=folder, me=list(a.me or []))
    me = {n.casefold() for n in (a.me or _settings_load().get("me") or [])} or None
    print(M.format_table(threads, me, a.top))
    print("\nBusiest senders (one of them is probably you):")
    for name, n in M.rank_senders(threads).most_common(8):
        print("  %-32s %d messages" % (name, n))
    print("\nNext, just run:  python -m persona_lm messenger-import      (it asks which name is you; the folder is remembered)")


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def cmd_messenger_import(a):
    interactive = sys.stdin.isatty()
    folder = _export_folder(a)
    threads = M.load_table(folder)
    senders = M.rank_senders(threads)
    me = list(a.me or _settings_load().get("me") or [])
    if not me:
        if not interactive:
            raise SystemExit("--me \"Your Name\" is required when this is not an interactive terminal (see `messenger-scan` for the names)")
        me = M.choose_me(senders)
    lowered = {n.casefold() for n in senders}
    unknown = [n for n in me if n.casefold() not in lowered]
    if unknown:
        raise SystemExit("not a sender name in this export: %s. Run `messenger-scan` to see the exact spelling." % unknown)
    everyone = not a.only_mine                               # learning from everyone: take every chat, no size limits, unless the owner sets them
    min_mine = a.min_my_messages if a.min_my_messages is not None else (0 if everyone else 20)
    max_people = a.max_participants if a.max_participants is not None else (None if everyone else 12)
    chats = a.chats
    if chats is None and interactive:
        print(M.format_table(threads, {n.casefold() for n in me}, a.top))
        chats = input("\nWhich chats should the AI learn from? (press Enter for ALL, or type numbers like 1,3,5-9): ")
    chats = "all" if (chats is None or not chats.strip()) else (chats if chats.strip().lower() == "all" else M.parse_selection(chats, len(threads)))
    _settings_save(export=folder, me=me)
    if a.no_anonymize:
        print("WARNING: anonymization is OFF. Use this data only on your own computer. NEVER upload it to a cloud GPU.")
    aliases = dict(x.split("=", 1) for x in (a.alias or []) if "=" in x)
    extra = _read_lines(a.extra_names) if a.extra_names else []
    stats = M.import_export(folder, a.out, me, chats, a.mode, a.context, a.session_gap_hours, a.max_chars, a.val_fraction, a.system,
                            not a.no_anonymize, extra, aliases, a.paranoid, min_mine, not a.no_groups, max_people,
                            log=print, threads=threads, learn_others=not a.only_mine)
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
    core = personas = None
    if getattr(a, "core", None):                        # Stage 6: the identity core fills in what was not typed; checked BEFORE the (slow) model load
        try:
            core = ID.Core(a.core)
            personas = PERS.Personas(core)                  # Stage 8a/8b: checked BEFORE the (slow) model load
            if personas.active and not a.version and not a.ckpt:
                a.version = personas.version_for(personas.active)       # a version fine-tuned for the mask in use (None = the current version)
            _fill_from_core(a, core)
        except (ID.IdentityProblem, LIN.LineageProblem, PERS.PersonaProblem, OSError) as e:
            raise SystemExit("error: %s" % e)
    elif getattr(a, "version", None):
        raise SystemExit("error: --version only works together with --core DIR")
    if not a.ckpt or not a.tokenizer:
        raise SystemExit("error: chat needs --ckpt and --tokenizer (or --core DIR with a registered version)")
    store = None
    if a.memory:                                        # checked BEFORE the (slow) model load, so a broken memory file fails at once
        try:
            store = MEM.MemoryStore(a.memory)
        except MEM.MemoryProblem as e:
            raise SystemExit("error: %s" % e)
    archive = None
    if getattr(a, "archive", None):                     # also checked before the slow model load
        try:
            archive = ARC.ChatArchive(a.archive)
        except ARC.ArchiveProblem as e:
            raise SystemExit("error: %s" % e)
    mood = tagger = card = None
    try:                                                # Stage 5 files are checked BEFORE the model loads, too
        if getattr(a, "mood", None):
            model = EMO.NaiveBayes.load(a.emotion_model) if getattr(a, "emotion_model", None) else None
            extra = EMO.load_lexicon_file(a.lexicon) if getattr(a, "lexicon", None) else None
            tagger = EMO.Tagger(model, extra)
            mood = EMO.Mood(a.mood, baseline=core.baseline() if core else (0.1, 0.0))
            if core:
                mood.baseline = list(core.baseline())       # the identity's temperament wins over a baseline stored in an old mood file
        elif getattr(a, "emotion_model", None) or getattr(a, "lexicon", None):
            raise SystemExit("error: --emotion-model and --lexicon only work together with --mood FILE")
        if getattr(a, "person", None):
            card = PER.load(a.person)
    except (EMO.EmotionProblem, PER.PersonProblem) as e:
        raise SystemExit("error: %s" % e)
    params, cfg, tok = _load_model(a)
    system, left_out = a.system, []
    if core and system is None:
        count = lambda text: len(tok.encode(text, allow_special=False))
        system, left_out = ID.system_line(core.sections, count, max(((cfg.ctx - a.max_new) // 2) - 4, 8))
    session = ChatSession(tok, cfg.ctx, lambda ids, **kw: sample_ids_cached(params, cfg, ids, **kw), system=system, save_dir=a.save_dir,
                          settings={"max_new": a.max_new, "temperature": a.temperature, "top_k": a.top_k},
                          memory=store, memory_tokens=a.memory_tokens, mood=mood, tagger=tagger, person=card,
                          person_tokens=getattr(a, "person_tokens", None), identity=core, personas=personas, archive=archive,
                          proposals=PRO.Proposals(core, memory_path=getattr(a, "memory", None), person_path=getattr(a, "person", None)) if core is not None else None)
    if core is not None:
        name = getattr(a, "_version_name", None)
        print("Identity core: %s (%s). Version: %s%s." % (core.sections["Name"], core.dir, name or "(a checkpoint typed by hand)",
                                                          " [an OLDER version, not the current one]" if name and name != core.current else ""))
        if left_out:
            print("Note: the AI's window is small, so these identity sections were left out of its prompt: %s." % ", ".join(left_out))
        if personas is not None and personas.active:
            print("Persona (a mask): %s. Underneath, still %s. /persona none takes it off." % (personas.active, core.sections["Name"]))
            if personas.d["personas"][personas.active].get("version"):
                print("Using the version fine-tuned for this mask. Note: /persona NAME switches only the prompt text, not the weights; restart the chat to change the version.%s" % (
                    " WARNING: the mask text changed after that version was trained." if personas.stale(personas.active) else ""))
        if core.pending():
            print("A protected identity change is waiting (identity pending). It does not affect this chat until applied.")
        if core.hand_edits():
            print("self.md was edited by hand; the official identity text is in force. Use `identity adopt` to make the edit official.")
    if store is not None:
        print("Memory file: %s (%s)" % (a.memory, store.overview().splitlines()[0]))
    if archive is not None:
        print(archive.overview())
    if mood is not None:
        print("Mood file: %s (%s)" % (a.mood, mood.describe().split(" (")[0]))
    if card is not None:
        print("Person card: %s" % a.person)
    run_repl(session)


def _fill_from_core(a, core):
    """Whatever was not typed on the command line comes from the identity core. The chosen version's checkpoint is checked against its checksum."""
    name = a.version or (core.current if not a.ckpt else None)
    if name:
        core.verify_version_files(name)
        a._version_name = name
        if not a.ckpt:
            a.ckpt = core.abs(core.version(name)["ckpt"])
    elif not a.ckpt:
        raise ID.IdentityProblem("The core has no current version yet. Register and promote one (identity add-version / record / promote), or give --ckpt.")
    if not a.tokenizer:
        a.tokenizer = core.resolve("tokenizer")
    for key, attr in (("memory", "memory"), ("mood", "mood"), ("person", "person")):
        if not getattr(a, attr, None) and core.resolve(key) and (key == "mood" or os.path.exists(core.resolve(key))):
            setattr(a, attr, core.resolve(key))


def cmd_memory(a):
    """python -m persona_lm memory [--file memory.json] <action> ... : the long-term memory graph (Stage 4). Local files only."""
    try:
        _memory_action(a)
    except (MEM.MemoryProblem, ValueError) as e:
        raise SystemExit("error: %s" % e)


def _memory_action(a):
    store = MEM.MemoryStore(a.file)
    act = a.action
    if act == "add":
        text, opts = MEM.parse_remember(" ".join(a.text))
        for k, v in (("emotion", a.emotion), ("importance", a.importance)):
            if v is not None:
                opts[k] = v
        if a.pin:
            opts["pinned"] = True
        if a.when:
            opts["when"] = a.when
        mem = store.add(text, **opts)
        store.log("add", [mem["id"]])
        store.save()
        print("Remembered. " + store.format_memory(mem))
    elif act == "list":
        mems = sorted(store.memories.values(), key=lambda m: (m["timestamp"], m["id"]))
        if a.node:
            node = store.find_node(a.node)
            if not node:
                raise MEM.MemoryProblem("There is no node called %r. See: python -m persona_lm memory nodes" % a.node)
            mems = [m for m in mems if node["id"] in m["nodes"]]
        if a.status != "all":
            mems = [m for m in mems if m["status"] == a.status]
        for m in mems[-a.limit:]:
            print(store.format_memory(m))
        print("(%d shown of %d matching)" % (min(len(mems), a.limit), len(mems)))
    elif act == "show":
        print(store.format_memory(store.get(a.id)))
    elif act == "nodes":
        scores = store.node_scores()
        for n in sorted(store.nodes.values(), key=lambda n: (-scores[n["id"]], n["id"])):
            count = sum(1 for m in store.active() if n["id"] in m["nodes"])
            print("%-9s %-30s size %.2f  %d active memor%s" % (n["kind"], n["name"], scores[n["id"]], count, "y" if count == 1 else "ies"))
        print(store.overview())
    elif act == "search":
        found = store.retrieve(" ".join(a.query), k=a.k)
        for m, sc, why in found:
            print("%s   (score %.2f: %s)" % (store.format_memory(m), sc, why))
        if not found:
            print("Nothing in memory matches those words (matching is by words and names, not by meaning).")
    elif act == "set":
        if a.importance is None and not (a.pin or a.unpin):
            raise MEM.MemoryProblem("Give --importance and/or --pin / --unpin.")
        mem = store.get(a.id)
        store.set_importance(mem["id"], mem["importance"] if a.importance is None else a.importance, True if a.pin else False if a.unpin else None)
        store.log("set", [mem["id"]])
        store.save()
        print(store.format_memory(mem))
    elif act == "forget":
        mem = store.get(a.id)
        if not a.yes:
            print("This would delete for good:\n  " + store.format_memory(mem) + "\nRun again with --yes to do it.")
            return
        store.forget(mem["id"])
        store.log("forget", [mem["id"]])
        store.save(keep_backup=False)
        print("Forgot %s for good (the backup copy was removed too)." % mem["id"])
    elif act == "maintain":
        plan = store.plan_maintenance(half_life=a.half_life, dormant_below=a.dormant_below, dup_threshold=a.dup_threshold, cluster_min=a.cluster_min)
        print("Maintenance plan (nothing changes unless you add --apply):")
        print("  fade to dormant : %s" % (", ".join(plan["fade"]) or "none"))
        print("  bring back      : %s" % (", ".join(plan["revive"]) or "none"))
        print("  duplicates      : %s" % (", ".join("%s absorbs %s (%.0f%% same words)" % (k, d, 100 * sim) for k, d, sim in plan["duplicates"]) or "none"))
        for s in plan["summaries"]:
            print("  summary of %d memories about %s:\n      %s" % (len(s["ids"]), store.nodes[s["node"]]["name"], s["text"]))
        if not plan["summaries"]:
            print("  summaries       : none")
        print("(Summaries are EXTRACTIVE: they copy the first sentences of the most important memories. They are not written by the AI.)")
        if a.apply:
            if not any(plan[k] for k in plan):
                print("Nothing to apply.")
                return
            done = store.apply_maintenance(plan)
            store.log("maintain", [], note=json.dumps(done))
            store.save()
            print("Applied: %s. (Previous version kept as %s)" % (json.dumps(done), store._backup_path()))
    elif act == "export-vault":
        print("Warning: the vault contains your personal memories as plain text. Keep it on this computer.")
        r = MEX.export_vault(store, a.out)
        print("Wrote %d memories and %d nodes to %s. Open that folder as a vault in Obsidian." % (r["memories"], r["nodes"], r["folder"]))
    elif act == "export-graph":
        print("Warning: the graph page contains your personal memories as plain text. Keep it on this computer.")
        r = MEX.export_graph(store, a.out)
        print("Wrote %s (%d nodes, %d links). Open it by double-clicking; no internet needed." % (r["file"], r["nodes"], r["edges"]))


def cmd_emotion(a):
    """python -m persona_lm emotion <tag|label|review|train|report|tag-memories> ... (Stage 5)."""
    try:
        _emotion_action(a)
    except (EMO.EmotionProblem, MEM.MemoryProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


def _tagger_from(a):
    model = EMO.NaiveBayes.load(a.model) if getattr(a, "model", None) else None
    extra = EMO.load_lexicon_file(a.lexicon) if getattr(a, "lexicon", None) else None
    return EMO.Tagger(model, extra)


def _emotion_action(a):
    act = a.action
    if act == "tag":
        print(_tagger_from(a).tag(" ".join(a.text)).describe())
    elif act == "label":
        texts = EMO.owner_messages(a.data)
        if not texts:
            raise EMO.EmotionProblem("No messages of yours (assistant turns with loss on) found in %s. Use the train.jsonl made by messenger-import." % ", ".join(a.data))
        rows = EMO.make_samples(texts, _tagger_from(a), neutral_ratio=a.neutral_ratio, limit=a.limit, seed=a.seed)
        EMO.write_samples(a.out, rows)
        counts = Counter(r["label"] for r in rows)
        print("Wrote %s: %d of your messages with a guessed feeling (from %d distinct messages)." % (a.out, len(rows), len(texts)))
        print("  " + ", ".join("%s %d" % (l, counts[l]) for l in EMO.LABEL_ORDER if counts[l]))
        print("The guesses come from a word list, so they are weak. Correct some with:  python -m persona_lm emotion review --samples %s" % a.out)
        print("This file contains your messages. Keep it on this computer.")
    elif act == "review":
        EMO.review(a.samples, input, count=a.count, seed=a.seed, include_neutral=a.include_neutral)
    elif act == "train":
        rows = EMO.read_samples(a.samples)
        if len(rows) < 30:
            raise EMO.EmotionProblem("Only %d samples: too few to learn from. Make more with `emotion label` (from a bigger train.jsonl)." % len(rows))
        model, rep = EMO.train_from_samples(rows, owner_weight=a.owner_weight, min_count=a.min_count)
        model.save(a.out)
        print("Wrote %s (%d words and word pieces learned from %d messages, %d corrected by you)." % (a.out, model.data["vocab"], rep["messages"], rep["fixed"]))
        if "agrees_with_word_list" in rep:
            print("On %d held-out messages the learner agrees with the guessed/corrected labels %.0f%% of the time." % (rep["held_out"], 100 * rep["agrees_with_word_list"]))
            print("  (This is NOT accuracy: most labels are the word list's own guesses. Only your corrections are real answers.)")
        if "fixed_right_with_model" in rep:
            print("On %d held-out messages YOU corrected: word list alone right %.0f%%, word list + learner right %.0f%%." % (
                rep["fixed_held_out"], 100 * rep["fixed_right_words_only"], 100 * rep["fixed_right_with_model"]))
        elif rep["fixed"] < 50:
            print("You have corrected %d messages. Correct at least 50-100 (emotion review) before trusting any number here." % rep["fixed"])
    elif act == "report":
        rows = EMO.read_samples(a.samples)
        guess, fixed = Counter(r["label"] for r in rows), Counter(r["fixed"] for r in rows if r["fixed"])
        print("%d samples, %d corrected by you." % (len(rows), sum(fixed.values())))
        print("%-10s %8s %10s" % ("feeling", "guessed", "corrected"))
        for l in EMO.LABEL_ORDER:
            print("%-10s %8d %10d" % (l, guess[l], fixed[l]))
        changed = sum(1 for r in rows if r["fixed"] and r["fixed"] != r["label"])
        print("You disagreed with the word list on %d of %d corrected messages." % (changed, sum(fixed.values())))
    elif act == "tag-memories":
        store = MEM.MemoryStore(a.memory)
        tagger = _tagger_from(a)
        plan = []
        for m in sorted(store.memories.values(), key=lambda m: m["id"]):
            if m["status"] == "active" and m["emotion"] == "neutral":
                tag = tagger.tag(m["text"])
                if tag.label != "neutral" and tag.confidence >= a.min_confidence:
                    plan.append((m, tag))
        print("Plan (nothing changes unless you add --apply): %d memor%s still tagged neutral would get a feeling." % (len(plan), "y" if len(plan) == 1 else "ies"))
        for m, tag in plan:
            print("  %s  neutral -> %s (sure %.0f%%; %s)   %s" % (m["id"], tag.label, 100 * tag.confidence, ", ".join(tag.evidence) or "learned words", m["text"][:70]))
        if a.apply and plan:
            for m, tag in plan:
                m["emotion"] = tag.label
            store.log("tag-emotions", [m["id"] for m, _ in plan])
            store.save()
            print("Applied. (Previous version kept as %s)" % store._backup_path())


def cmd_mood(a):
    try:
        mood = EMO.Mood(a.file)
    except EMO.EmotionProblem as e:
        raise SystemExit("error: %s" % e)
    if a.action == "show":
        print(mood.describe() + ("" if os.path.exists(a.file) else "  (no mood file yet: it is created by `chat --mood`)"))
        for h in mood.history[-5:]:
            print("  recent: %s (sure %.0f%%)" % (h["label"], 100 * h["confidence"]))
    elif a.action == "reset":
        mood.reset()
        mood.save()
        print("Mood reset to the baseline. " + mood.describe())


def cmd_person(a):
    try:
        _person_action(a)
    except (PER.PersonProblem, EMO.EmotionProblem, MEM.MemoryProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


def _person_action(a):
    path = a.file
    if a.action == "init":
        if os.path.exists(path):
            raise PER.PersonProblem("%s already exists; nothing was overwritten." % path)
        PER.save(PER.new_card(a.name), path, keep_backup=False)
        print("Made %s. Open it with Notepad and fill in Habits and Values yourself; `person build` fills Speaking style, Frequent words and Relationships from your data." % path)
    elif a.action == "show":
        print(PER.load(path).render(), end="")
    elif a.action == "add":
        card = PER.load(path)
        card.add_owner(a.section, " ".join(a.text))
        PER.save(card, path)
        print("Added to %s." % PER.Card._canonical(a.section))
    elif a.action == "build":
        card = PER.load(path)
        texts = EMO.owner_messages(a.data) if a.data else None
        store = MEM.MemoryStore(a.memory) if a.memory else None
        if texts is None and store is None:
            raise PER.PersonProblem("Give --data train.jsonl (your messages) and/or --memory memory.json (people you remember).")
        if texts is not None and len(texts) < 20:
            print("Only %d of your messages found: too few to describe your style, so Speaking style and Frequent words are left alone." % len(texts))
            texts = None
        plan = PER.plan_build(card, texts, store)
        print("Build plan (nothing changes unless you add --apply):")
        for title, text in plan["remove"]:
            print("  - [%s] %s" % (title, text))
        for title, text in plan["add"]:
            print("  + [%s] %s" % (title, text))
        if not plan["add"] and not plan["remove"]:
            print("  nothing to change.")
        elif a.apply:
            PER.apply_plan(card, plan)
            PER.save(card, path)
            print("Applied. Lines you wrote yourself were not touched. (Previous version kept as %s)" % (path[:-3] + ".prev.md" if path.endswith(".md") else path + ".prev"))
        else:
            print("Lines marked [auto] are only suggestions. To keep one for good delete the word [auto]; to ban it change [auto] to [no].")


def cmd_proposal(a):
    try:
        _proposal_action(a)
    except (PRO.ProposalProblem, ID.IdentityProblem, LIN.LineageProblem, MEM.MemoryProblem, PER.PersonProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


def _proposal_action(a):
    """python -m persona_lm proposal [--dir identity_core] <action> (Stage 7a). Filing a proposal changes nothing; approve/undo need typing at the keyboard."""
    pr = PRO.Proposals(ID.Core(a.dir), memory_path=a.memory, person_path=a.person)
    act = a.action
    if act in ("memory-add", "memory-forget", "memory-importance", "card-add", "config", "finetune"):
        if act == "config":
            args = {"key": a.key, "value": a.value}
        elif act == "finetune":
            args = {"run": a.run, "data_dir": a.data_dir, "size": a.size, "steps": a.steps, "lr": a.lr, "init_from": a.init_from}
        elif act == "memory-add":
            args = {"text": " ".join(a.text), "emotion": a.emotion, "importance": a.importance, "pinned": a.pinned, "nodes": a.node or []}
        elif act == "memory-forget":
            args = {"id": a.id}
        elif act == "memory-importance":
            args = {"id": a.id, "importance": a.importance, "pinned": a.pinned}
        else:
            args = {"section": a.section, "text": " ".join(a.text)}
        p = pr.submit(act, args, a.why)
        print(pr.describe(p))
        print("Filed as %s. Nothing was changed. To apply it: python -m persona_lm proposal approve %s" % (p["id"], p["id"]))
    elif act == "list":
        rows = pr.listing(None if a.all else "pending")
        if not rows:
            print("No %sproposals." % ("" if a.all else "waiting "))
        for p in rows:
            print("%s  %-9s %-18s %s" % (p["id"], p["status"], p["kind"], p["why"][:70]))
    elif act == "show":
        print(pr.describe(pr.get(a.id)))
    elif act == "approve":
        print(pr.approve(a.id, ask=ID.keyboard_ask))
    elif act == "reject":
        print(pr.reject(a.id, a.note))
    elif act == "undo":
        print(pr.undo(a.id, ask=ID.keyboard_ask))
    elif act == "purge-undo":
        print(pr.purge_undo(a.id, ask=ID.keyboard_ask))


def cmd_persona(a):
    try:
        _persona_action(a)
    except (PERS.PersonaProblem, ID.IdentityProblem, LIN.LineageProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


def _persona_action(a):
    """python -m persona_lm persona [--dir identity_core] <list|show|add|edit|remove|use> (Stage 8a). Masks only: the identity core's sections are never touched."""
    ps = PERS.Personas(ID.Core(a.dir))
    act = a.action
    if act == "list":
        print(ps.describe())
    elif act == "show":
        name, p = ps.get(a.name)
        print("%s%s\n  text: %s\n  what the AI sees:  <|persona|> %s <|end|>" % (name, "  (on now)" if name == ps.active else "", p["text"],
                                                                                 PERS.block_text(ps.core.sections["Name"], name, p["text"])))
    elif act == "add":
        print(ps.add(a.name, " ".join(a.text)))
    elif act == "edit":
        print(ps.edit(a.name, " ".join(a.text)))
    elif act == "remove":
        print(ps.remove(a.name))
    elif act == "use":
        print(ps.use(a.name))
    elif act == "example":
        print(ps.add_example(a.name, a.user, a.reply))
    elif act == "make-data":
        tok_path = a.tokenizer or ps.core.resolve("tokenizer")
        if not tok_path:
            raise PERS.PersonaProblem("No tokenizer: give --tokenizer tok.json (or register one: identity set-file tokenizer tok.json).")
        n = ps.make_data(a.name, a.out, Tokenizer.load(tok_path), a.ctx)
        print("Wrote %d example(s) to %s. Each has the identity line, the persona block and the message with NO loss, and your reply WITH loss." % (n, a.out))
        print("Next (your own steps): prepare --kind chat on this file, train with --init-from your current checkpoint into a NEW run folder, then:\n"
              "  identity add-version NAME --ckpt <new ckpt>   ->   identity record / score NAME   ->   persona attach \"%s\" NAME   ->   persona check \"%s\"" % (a.name, a.name))
        print("This makes a full fine-tuned COPY of the weights (not a small adapter: LoRA is not built). It is optional; the mask also works as prompt text alone.")
    elif act == "attach":
        print(ps.attach(a.name, a.version))
    elif act == "detach":
        print(ps.detach(a.name))
    elif act == "check":
        name, _ = ps.get(a.name)
        version = a.version or ps.version_for(name) or ps.core.current
        if not version:
            raise PERS.PersonaProblem("There is no version to test. Register and promote one first (identity add-version / record / promote).")
        generate, tok, ctx = ID.load_runner(ps.core, version)
        print("Asking version %s the probes with the mask %r on (greedy, no randomness):" % (version, name))
        res = PERS.check_mask(ps.core, ps, name, generate, tok, ctx, version=version, against=a.against, out=print)
        print(PERS.describe_check(name, res))


def cmd_identity(a):
    try:
        _identity_action(a)
    except (ID.IdentityProblem, LIN.LineageProblem, MEM.MemoryProblem, PER.PersonProblem, EMO.EmotionProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


def _identity_action(a):
    """python -m persona_lm identity [--dir identity_core] <action> ... (Stage 6). Every change is typed by the owner; protected ones also need a typed confirmation."""
    act = a.action
    if act == "init":
        core = ID.Core.create(a.dir, a.name, a.owner, personality=a.personality, values=a.values, temperament=a.temperament, person=a.person,
                              memory=a.memory, mood=a.mood, tokenizer=a.tokenizer)
        print("Made the identity core in %s." % core.dir)
        print(core.describe(last=1))
        print("Next: edit nothing by hand; use `identity set Personality ...` (protected: typed confirmation + cooldown) or `identity set \"Speaking style\" ...`.")
        print("Then: identity add-version v1 --ckpt run/ckpt.npz   ->  identity make-probes   ->  identity record v1   ->  identity promote v1")
        return
    core = ID.Core(a.dir)
    if act == "show":
        print(core.describe(last=8))
    elif act == "set":
        print(core.propose({a.section: " ".join(a.text)}, ask=ID.keyboard_ask))
    elif act == "adopt":
        print(core.adopt(ask=ID.keyboard_ask))
    elif act == "pending":
        p = core.pending()
        if not p:
            print("No protected change is waiting.")
        else:
            print("Waiting (staged %s UTC, can be applied after %s UTC):" % (LIN.iso(p["staged_t"]), LIN.iso(p["earliest_t"])))
            print("\n".join(ID.diff_text(s, core.sections[s], t) for s, t in p["changes"].items()))
            print("Apply with `identity apply`, or cancel with `identity cancel`.")
    elif act == "apply":
        print(core.apply_pending(ask=ID.keyboard_ask))
    elif act == "cancel":
        print(core.cancel_pending())
    elif act == "settings":
        given = [(k, getattr(a, k)) for k in ("threshold", "warn_drift", "cooldown_hours") if getattr(a, k) is not None]
        for k, v in given:
            print(core.set_setting(k, v))
        if not given:
            print("\n".join("%s = %s" % (k, v) for k, v in core.settings.items()))
    elif act == "set-file":
        core.paths_set(a.kind, a.path)
        print("Saved: %s file = %s" % (a.kind, core.d["paths"][a.kind]))
    elif act == "add-version":
        digest = core.add_version(a.name, a.ckpt, parent=a.parent, note=a.note or "")
        print("Registered version %s (checkpoint checksum %s...). Next: identity record %s  (the first version) or identity score %s  (a new one)." % (a.name, digest[:16], a.name, a.name))
    elif act == "versions":
        print(core.describe(last=0))
    elif act == "make-probes":
        if os.path.exists(core.probes_path()):
            raise ID.IdentityProblem("probes.json already exists. The probes are FIXED on purpose (every version is compared on the same questions). To start over, move the file away and record every baseline again.")
        store = MEM.MemoryStore(core.resolve("memory")) if core.resolve("memory") and os.path.exists(core.resolve("memory")) else None
        doc = ID.make_probes(core, store, max_new=a.max_new)
        with open(core.probes_path(), "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("Wrote %s with %d probes:" % (core.probes_path(), len(doc["probes"])))
        for p in doc["probes"]:
            print("  %s %-6s %s%s" % (p["id"], p["kind"], p["prompt"], "   (expects: %s)" % ", ".join(p["expect"]) if p["expect"] else ""))
    elif act == "show-probes":
        for p in ID.load_probes(core)["probes"]:
            print("%s %-6s %s%s" % (p["id"], p["kind"], p["prompt"], "   (expects: %s)" % ", ".join(p["expect"]) if p["expect"] else ""))
    elif act == "record":
        generate, tok, ctx = ID.load_runner(core, a.name)
        print("Asking version %s the probes (greedy, no randomness):" % a.name)
        ID.record_answers(core, a.name, generate, tok, ctx, out=print)
        print("Saved %s. This is now the baseline other versions are scored against." % core.answers_path(a.name))
    elif act == "score":
        against = a.against or core.current
        generate, tok, ctx = ID.load_runner(core, a.name)
        print("Asking version %s the probes:" % a.name)
        res = ID.score_version(core, a.name, generate, tok, ctx, against=against, out=print)
        print(ID.describe_score(a.name, against, res, core.settings["threshold"], core.settings["warn_drift"]))
    elif act == "promote":
        print(core.promote(a.name))
    elif act == "rollback":
        print(core.rollback(a.name))
    elif act == "lineage":
        ok, msg = core.lineage.verify()
        print(core.lineage.describe(a.last))
        print("Chain check: %s%s" % ("OK, " if ok else "PROBLEM: ", msg))
    elif act == "verify":
        results = core.verify()
        for ok, text in results:
            print("%s  %s" % ("OK  " if ok else "FAIL", text))
        bad = [r for r in results if not r[0]]
        print("\n%d check(s) passed, %d failed." % (len(results) - len(bad), len(bad)))
        if bad:
            sys.exit(1)
    elif act == "distill":
        core.version(a.teacher)
        generate, tok, ctx = ID.load_runner(core, a.teacher)
        prompts = []
        if os.path.exists(core.probes_path()):
            prompts += [p["prompt"] for p in ID.load_probes(core)["probes"]]
        if a.prompts:
            with open(a.prompts, encoding="utf-8") as f:
                prompts += [l.strip() for l in f if l.strip()]
        if a.data:
            prompts += ID.prompts_from_data(a.data, limit=a.limit)
        n = ID.distill(core, generate, tok, ctx, prompts, a.out, a.teacher, max_new=a.max_new, temperature=a.temperature, seed=a.seed, limit=a.limit)
        print("Wrote %d teacher answers to %s." % (n, a.out))
        print("Use it as extra chat-tuning data for a NEW model (same tokenizer): prepare --kind chat ... --data %s. The file contains the AI's words; keep it on this computer." % a.out)


def cmd_status(a):
    for line in R.format_status(a.dir):
        print(line)


def cmd_data_report(a):
    try:
        lines = R.format_data_report(a.data_dir, a.size, a.ctx, a.batch, a.accum, a.steps, a.tokenizer)
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    for line in lines:
        print(line)


def cmd_pipeline(a):
    try:
        if a.pcmd == "init":
            fields = dict(languages=[x.strip() for x in a.languages.split(",") if x.strip()] if a.languages else None, vocab=a.vocab, size=a.size, ctx=a.ctx,
                          pretrain_text=a.pretrain_text, chat_train=a.chat_train, chat_val=a.chat_val, pretrain_steps=a.pretrain_steps,
                          finetune_epochs=a.finetune_epochs, batch=a.batch, accum=a.accum, system=a.system,
                          train_args=shlex.split(a.train_args) if a.train_args is not None else None)
            proj, changed = PL.init_project(a.dir, **fields)
            print("Saved %s (%s)." % (PL.project_path(a.dir), ("changed: " + ", ".join(changed)) if changed else "nothing changed"))
            if not proj["decisions_confirmed"]:
                print("Decisions are NOT confirmed yet. Read `pipeline plan`, then `pipeline confirm`.")
        elif a.pcmd == "confirm":
            if not a.yes:
                sys.exit("This locks languages, vocabulary size and window length for the tokenizer. Add --yes once you have said OK to them.")
            proj = PL.confirm_decisions(a.dir)
            print("Confirmed: languages %s, vocabulary %d, window %d, model %s. The tokenizer step may run now." % (",".join(proj["languages"]), proj["vocab"], proj["ctx"], proj["size"]))
        elif a.pcmd == "plan":
            for line in PL.plan_lines(a.dir):
                print(line)
        elif a.pcmd == "run":
            PL.run_step(a.dir, a.step, main)
    except PL.ProjectError as e:
        sys.exit(str(e))


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


def cmd_export_package(a):
    """python -m persona_lm export-package --dir identity_core --out NEWFOLDER --yes : one folder with weights, tokenizer, identity, memory ... (Stage 9b)."""
    from . import export_pkg as EX
    try:
        core = ID.Core(a.dir)
        EX.export_package(core, a.out, version=a.version, fmt=a.format, gguf_type=a.gguf_type, weights_only=a.weights_only, memory=not a.no_memory,
                          person=not a.no_person, mood=not a.no_mood, personas=not a.no_personas, confirmed=a.yes)
    except (EX.ExportProblem, ID.IdentityProblem, LIN.LineageProblem, OSError, ValueError) as e:
        raise SystemExit("error: %s" % e)


def cmd_verify_package(a):
    from . import export_pkg as EX
    try:
        problems, notes = EX.verify_package(a.package)
    except EX.ExportProblem as e:
        raise SystemExit("error: %s" % e)
    for n in notes:
        print("note: " + n)
    for p in problems:
        print("PROBLEM: " + p)
    if problems:
        raise SystemExit("The package is damaged or was changed (%d problem(s))." % len(problems))
    print("OK: every file that should not change matches its checksum.")


def cmd_serve(a):
    from . import runtime as RT
    from . import server as SV
    if bool(a.package) == bool(a.core):
        raise SystemExit("error: give exactly one of --package FOLDER or --core FOLDER")
    try:
        kw = dict(max_new=a.max_new, temperature=a.temperature, seed=a.seed)
        ai = RT.load_package(a.package, **kw) if a.package else RT.load(a.core, version=a.version, **kw)
        SV.serve(ai, host=a.host, port=a.port, token=a.token, allow_persona_switch=a.allow_persona_switch)
    except (RT.RuntimeProblem, SV.ServerProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


def cmd_ui(a):
    from . import ui_server as UI
    try:
        UI.serve(a.root, port=a.port, open_browser=not a.no_open)
    except (UI.UIProblem, OSError) as e:
        raise SystemExit("error: %s" % e)


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
    s.add_argument("export", nargs="?", default=None, help="the unzipped export folder (or its messages/ folder); remembered after the first time")
    s.add_argument("--me", action="append", help="your name, to show how many messages in each chat are yours")
    s.add_argument("--top", type=int, default=60)
    s.set_defaults(fn=cmd_messenger_scan)

    s = sub.add_parser("messenger-import", help="turn a Messenger JSON export into chat training data (anonymized by default)")
    s.add_argument("export", nargs="?", default=None, help="the export folder; leave out to use the one remembered by messenger-scan")
    s.add_argument("--out", default="data_msgr", help="output folder: train.jsonl, val.jsonl, stats.json, private/ (default data_msgr)")
    s.add_argument("--me", action="append", help="the sender name that is YOU (repeat the option if you appear under several names); asked if missing")
    s.add_argument("--chats", default=None, help="which chats: 'all' or the numbers from `messenger-scan`, e.g. 1,3,5-9; asked if missing")
    s.add_argument("--mode", choices=["window", "session"], default="window", help="window: one example per reply of yours with the previous --context turns; session: whole conversations")
    s.add_argument("--context", type=int, default=8, help="window mode: how many previous turns the AI sees")
    s.add_argument("--session-gap-hours", type=float, default=6.0, help="a silence longer than this starts a new session (context never crosses it)")
    s.add_argument("--max-chars", type=int, default=1400, help="size limit of one example in characters (about 400 tokens; keep an example inside the model's window, 4096 tokens by default)")
    s.add_argument("--val-fraction", type=float, default=0.05, help="share of the most recent sessions held out for validation")
    s.add_argument("--system", default=None, help="optional system line put in front of every example")
    s.add_argument("--min-my-messages", type=int, default=None, help="skip chats where you wrote fewer messages than this (default: 0 when learning from everyone, 20 with --only-mine)")
    s.add_argument("--no-groups", action="store_true", help="skip group chats")
    s.add_argument("--only-mine", action="store_true", help="learn ONLY from your own messages (default: the AI learns from everyone's messages and from all chats and groups)")
    s.add_argument("--learn-from-others", action="store_true", help=argparse.SUPPRESS)      # kept so the old command still works: this is now the default
    s.add_argument("--max-participants", type=int, default=None, help="skip chats with more people than this (default: no limit when learning from everyone, 12 with --only-mine)")
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
        s.add_argument("--ckpt", required=name != "chat", default=None)
        s.add_argument("--tokenizer", required=name != "chat", default=None)
        if name != "export-gguf":
            s.add_argument("--max-new", type=int, default=120)
            s.add_argument("--temperature", type=float, default=0.8)
            s.add_argument("--top-k", type=int, default=40)
        if name == "generate":
            s.add_argument("--prompt", required=True)
            s.add_argument("--seed", type=int, default=0)
        if name == "chat":
            s.add_argument("--system", default=None)
            s.add_argument("--save-dir", default=".", help="where /save writes conversations (this computer only)")
            s.add_argument("--memory", default=None, help="memory file (Stage 4): turns on /memory /remember /forget and recall into the prompt")
            s.add_argument("--memory-tokens", type=int, default=None, help="most tokens of the window that recalled memories may use (default: a fifth of the window)")
            s.add_argument("--archive", default=None, help="chat archive file, e.g. chat_archive.jsonl: every exchange is kept and the old ones that match your message are shown to the AI (even from 200 chats ago); /recall works")
            s.add_argument("--mood", default=None, help="mood file (Stage 5): turns on /mood, a mood block in the prompt and a small temperature effect (created on first save)")
            s.add_argument("--emotion-model", default=None, help="a model made by `emotion train` (needs --mood); without it only the word list reads your feelings")
            s.add_argument("--lexicon", default=None, help="JSON of extra feeling words, e.g. {\"anger\": [\"inis\"]} (needs --mood)")
            s.add_argument("--person", default=None, help="your Person card (Stage 5; made by `person init`): put into the prompt as the speaker block")
            s.add_argument("--person-tokens", type=int, default=None, help="most tokens of the window the Person card may use (default: a sixth of the window)")
            s.add_argument("--core", default=None, help="identity core folder (Stage 6): its current version, tokenizer, self-description, Person card, memory and mood are used unless you type your own")
            s.add_argument("--version", default=None, help="(needs --core) chat with this registered version instead of the current one")
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

    s = sub.add_parser("status", help="what exists in a work folder (tokenizers, token files, runs) and what to do next; read-only")
    s.add_argument("dir", nargs="?", default=".")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("data-report", help="how much data a token folder holds and whether it suits a model size; read-only")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--size", choices=sorted(PRESETS), default="s10m")
    s.add_argument("--ctx", type=int, default=None)
    s.add_argument("--batch", type=int, default=None)
    s.add_argument("--accum", type=int, default=None)
    s.add_argument("--steps", type=int, default=None, help="how many passes over the data would this many steps make?")
    s.add_argument("--tokenizer", default=None, help="check that the data was made with this tokenizer")
    s.set_defaults(fn=cmd_data_report)

    s = sub.add_parser("pipeline", help="guided steps: tokenizer -> pretrain -> chat fine-tune (decisions are written to project.json)")
    ps = s.add_subparsers(dest="pcmd", required=True)
    q = ps.add_parser("init", help="create or update project.json (only the options you give change)")
    q.add_argument("--dir", required=True, help="work folder (created if missing)")
    q.add_argument("--languages", default=None, help="language codes you want, comma separated, e.g. en,tl")
    q.add_argument("--vocab", type=int, default=None, help="vocabulary size (recommended 16000)")
    q.add_argument("--size", choices=sorted(PRESETS), default=None)
    q.add_argument("--ctx", type=int, default=None, help="window length in tokens (default 4096: room for about 150 chat messages)")
    q.add_argument("--pretrain-text", nargs="+", default=None, help="files/folders of PUBLIC text in all wanted languages")
    q.add_argument("--chat-train", default=None, help="train.jsonl from messenger-import")
    q.add_argument("--chat-val", default=None, help="val.jsonl from messenger-import")
    q.add_argument("--pretrain-steps", type=int, default=None)
    q.add_argument("--finetune-epochs", type=float, default=None, help="about how many passes over your chats the fine-tune makes (default 3)")
    q.add_argument("--batch", type=int, default=None)
    q.add_argument("--accum", type=int, default=None)
    q.add_argument("--train-args", default=None, help="extra flags for train inside quotes, e.g. \"--precision bf16\"")
    q.add_argument("--system", default=None, help="system line used by the chat command")
    q = ps.add_parser("confirm", help="say OK to the decisions: needed before the tokenizer can be made")
    q.add_argument("--dir", required=True)
    q.add_argument("--yes", action="store_true")
    q = ps.add_parser("plan", help="show every step with its exact command and whether it is done")
    q.add_argument("--dir", required=True)
    q = ps.add_parser("run", help="run ONE step (default: the next one that is not done)")
    q.add_argument("--dir", required=True)
    q.add_argument("--step", default="next", choices=["next"] + PL.STEP_ORDER)
    s.set_defaults(fn=cmd_pipeline)

    s = sub.add_parser("memory", help="the long-term memory graph (Stage 4): add, search, list, forget, maintain, export")
    s.add_argument("--file", default="memory.json", help="the memory file (default memory.json in the current folder)")
    ms = s.add_subparsers(dest="action", required=True)
    q = ms.add_parser("add", help="remember something: [[Name]] or [[kind:Name]] makes a node; #emotion *0.8 !pin at the end are options")
    q.add_argument("text", nargs="+")
    q.add_argument("--emotion", default=None)
    q.add_argument("--importance", type=float, default=None, help="0 to 1 (default 0.5)")
    q.add_argument("--pin", action="store_true", help="never fades")
    q.add_argument("--when", default=None, help="when it happened: 2026-11-14 or 2026-11-14T10:30:00Z (default now)")
    q = ms.add_parser("list", help="list memories, oldest first")
    q.add_argument("--node", default=None)
    q.add_argument("--status", default="active", choices=list(MEM.STATUSES) + ["all"])
    q.add_argument("--limit", type=int, default=30)
    q = ms.add_parser("show"); q.add_argument("id")
    ms.add_parser("nodes", help="people, topics, events, feelings with their size")
    q = ms.add_parser("search", help="what would be recalled for these words (changes nothing)")
    q.add_argument("query", nargs="+"); q.add_argument("-k", type=int, default=8)
    q = ms.add_parser("set", help="change a memory's importance or pin it")
    q.add_argument("id"); q.add_argument("--importance", type=float, default=None)
    q.add_argument("--pin", action="store_true"); q.add_argument("--unpin", action="store_true")
    q = ms.add_parser("forget", help="delete one memory for good (shows it first; needs --yes)")
    q.add_argument("id"); q.add_argument("--yes", action="store_true")
    q = ms.add_parser("maintain", help="show a plan: fade old memories, merge duplicates, summarise big clusters. --apply does it")
    q.add_argument("--apply", action="store_true")
    q.add_argument("--half-life", type=float, default=MEM.HALF_LIFE_DAYS, help="days until an unused memory's strength halves (default 90)")
    q.add_argument("--dormant-below", type=float, default=MEM.DORMANT_BELOW)
    q.add_argument("--dup-threshold", type=float, default=0.8, help="share of the same words that counts as a duplicate (default 0.8)")
    q.add_argument("--cluster-min", type=int, default=6, help="memories about one node before a summary is proposed (default 6)")
    q = ms.add_parser("export-vault", help="Obsidian folder with [[wiki-links]] (a new empty folder; nothing is overwritten)")
    q.add_argument("out")
    q = ms.add_parser("export-graph", help="one HTML file with the force-directed graph (needs no internet)")
    q.add_argument("out")
    s.set_defaults(fn=cmd_memory)

    s = sub.add_parser("emotion", help="Stage 5: read the feeling in a message, learn it from your chats, tag memories")
    es = s.add_subparsers(dest="action", required=True)
    def _tag_args(q):
        q.add_argument("--model", default=None, help="a model made by `emotion train`")
        q.add_argument("--lexicon", default=None, help="JSON of extra feeling words, e.g. {\"anger\": [\"inis\"]}")
    q = es.add_parser("tag", help="how would this text be read? (changes nothing)")
    q.add_argument("text", nargs="+"); _tag_args(q)
    q = es.add_parser("label", help="make a samples file: your messages with a guessed feeling (a new file; never overwrites)")
    q.add_argument("--data", nargs="+", required=True, help="train.jsonl made by messenger-import")
    q.add_argument("--out", required=True)
    q.add_argument("--neutral-ratio", type=float, default=3.0, help="keep this many times as many no-feeling messages as feeling ones (default 3)")
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--lexicon", default=None, help="JSON of extra feeling words, e.g. {\"anger\": [\"inis\"]}")
    q = es.add_parser("review", help="correct the guesses by hand, one message at a time (answers are saved at once)")
    q.add_argument("--samples", required=True)
    q.add_argument("--count", type=int, default=20)
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--include-neutral", action="store_true")
    q = es.add_parser("train", help="learn from the samples file and write the emotion model")
    q.add_argument("--samples", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--owner-weight", type=float, default=5.0, help="how much more your corrections count than the word list's guesses (default 5)")
    q.add_argument("--min-count", type=int, default=2, help="drop words seen fewer times than this (default 2)")
    q = es.add_parser("report", help="how many guesses and corrections per feeling")
    q.add_argument("--samples", required=True)
    q = es.add_parser("tag-memories", help="give memories that are still tagged neutral a feeling (a plan; --apply does it)")
    q.add_argument("--memory", default="memory.json")
    q.add_argument("--min-confidence", type=float, default=0.5)
    q.add_argument("--apply", action="store_true"); _tag_args(q)
    s.set_defaults(fn=cmd_emotion)

    s = sub.add_parser("mood", help="Stage 5: look at or reset the saved mood")
    s.add_argument("--file", default="mood.json")
    ms2 = s.add_subparsers(dest="action", required=True)
    ms2.add_parser("show"); ms2.add_parser("reset")
    s.set_defaults(fn=cmd_mood)

    s = sub.add_parser("person", help="Stage 5: your Person card, a plain file built partly from your data")
    s.add_argument("--file", default="person_card.md")
    ps2 = s.add_subparsers(dest="action", required=True)
    q = ps2.add_parser("init", help="make a new card (never overwrites)")
    q.add_argument("--name", default=None)
    ps2.add_parser("show")
    q = ps2.add_parser("add", help="add a line you write yourself: person add Values I care about my family first")
    q.add_argument("section"); q.add_argument("text", nargs="+")
    q = ps2.add_parser("build", help="show what could be filled in from your data (a plan; --apply writes it)")
    q.add_argument("--data", nargs="+", default=None, help="train.jsonl from messenger-import: your messages")
    q.add_argument("--memory", default=None, help="memory.json: the people you remember")
    q.add_argument("--apply", action="store_true")
    s.set_defaults(fn=cmd_person)

    s = sub.add_parser("proposal", help="Stage 7a: changes the AI may only PROPOSE (memory, Person card); you approve at the keyboard; every change can be undone")
    s.add_argument("--dir", default="identity_core"); s.add_argument("--memory", default=None, help="memory file (default: the one registered in the core)")
    s.add_argument("--person", default=None, help="Person card file (default: the one registered in the core)")
    pa = s.add_subparsers(dest="action", required=True)
    q = pa.add_parser("memory-add", help="propose a new memory"); q.add_argument("text", nargs="+"); q.add_argument("--why", required=True)
    q.add_argument("--emotion", default="neutral"); q.add_argument("--importance", type=float, default=0.5); q.add_argument("--pinned", action="store_true"); q.add_argument("--node", action="append")
    q = pa.add_parser("memory-forget", help="propose forgetting one memory"); q.add_argument("id"); q.add_argument("--why", required=True)
    q = pa.add_parser("memory-importance", help="propose a new importance (0-1) for one memory"); q.add_argument("id"); q.add_argument("importance", type=float)
    q.add_argument("--why", required=True); q.add_argument("--pinned", action=argparse.BooleanOptionalAction, default=None)
    q = pa.add_parser("card-add", help="propose a line for the Person card"); q.add_argument("section"); q.add_argument("text", nargs="+"); q.add_argument("--why", required=True)
    q = pa.add_parser("config", help="propose a new value for an identity setting (only warn_drift; the safeguards are yours alone)"); q.add_argument("key"); q.add_argument("value", type=float)
    q.add_argument("--why", required=True)
    q = pa.add_parser("finetune", help="propose a training run (approving only RECORDS the command; you run it yourself)"); q.add_argument("--run", required=True); q.add_argument("--data-dir", required=True)
    q.add_argument("--size", required=True); q.add_argument("--steps", type=int, required=True); q.add_argument("--lr", type=float, required=True); q.add_argument("--init-from", default=None)
    q.add_argument("--why", required=True)
    q = pa.add_parser("list", help="waiting proposals (--all for every one)"); q.add_argument("--all", action="store_true")
    q = pa.add_parser("show", help="what / why / diff / how to undo"); q.add_argument("id")
    q = pa.add_parser("approve", help="apply a proposal (you type its number to confirm; keyboard only)"); q.add_argument("id")
    q = pa.add_parser("reject", help="throw a proposal away"); q.add_argument("id"); q.add_argument("--note", default="")
    q = pa.add_parser("undo", help="undo an approved proposal (typed confirmation)"); q.add_argument("id")
    q = pa.add_parser("purge-undo", help="erase the saved copy of a forgotten memory (then it cannot be undone)"); q.add_argument("id")
    s.set_defaults(fn=cmd_proposal)

    s = sub.add_parser("persona", help="Stage 8a: masks the AI can wear (prompt text only; never changes who it is); every switch is logged")
    s.add_argument("--dir", default="identity_core")
    pe = s.add_subparsers(dest="action", required=True)
    pe.add_parser("list", help="the masks you made and which one is on")
    q = pe.add_parser("show", help="one mask and exactly what the AI is shown"); q.add_argument("name")
    q = pe.add_parser("add", help="make a mask: persona add \"calm mentor\" You speak slowly and kindly."); q.add_argument("name"); q.add_argument("text", nargs="+")
    q = pe.add_parser("edit", help="change a mask's text"); q.add_argument("name"); q.add_argument("text", nargs="+")
    q = pe.add_parser("remove", help="delete a mask"); q.add_argument("name")
    q = pe.add_parser("use", help="put a mask on (or 'none' to take it off)"); q.add_argument("name")
    q = pe.add_parser("example", help="Stage 8b: add one of YOUR example exchanges for a mask: persona example NAME \"what they say\" \"how this mask answers\""); q.add_argument("name"); q.add_argument("user"); q.add_argument("reply")
    q = pe.add_parser("make-data", help="Stage 8b: write the mask's examples as chat training data (never overwrites)"); q.add_argument("name"); q.add_argument("--out", required=True)
    q.add_argument("--tokenizer", default=None, help="default: the one registered in the identity core"); q.add_argument("--ctx", type=int, default=4096)
    q = pe.add_parser("attach", help="Stage 8b: say that a registered version was fine-tuned for this mask (the chat uses it while the mask is on)"); q.add_argument("name"); q.add_argument("version")
    q = pe.add_parser("detach", help="Stage 8b: forget the attached version"); q.add_argument("name")
    q = pe.add_parser("check", help="Stage 8b: identity regression test WITH the mask on (do the facts about who it is survive?)"); q.add_argument("name")
    q.add_argument("--version", default=None, help="which version wears it (default: the attached one, else the current one)"); q.add_argument("--against", default=None, help="baseline version (default: the current one)")
    s.set_defaults(fn=cmd_persona)

    s = sub.add_parser("identity", help="Stage 6: the identity core (who the AI is, outside the weights), version history, regression probes")
    s.add_argument("--dir", default="identity_core")
    ids = s.add_subparsers(dest="action", required=True)
    q = ids.add_parser("init", help="make a new identity core (never overwrites)")
    q.add_argument("--name", required=True, help="the AI's own name")
    q.add_argument("--owner", required=True, help="your name")
    q.add_argument("--personality", default=""); q.add_argument("--values", default=""); q.add_argument("--temperament", default="")
    q.add_argument("--person", default=None, help="your Person card file"); q.add_argument("--memory", default=None, help="the memory file")
    q.add_argument("--mood", default=None, help="the mood file"); q.add_argument("--tokenizer", default=None, help="the tokenizer file (it becomes the frozen one)")
    ids.add_parser("show", help="the identity card, versions and recent history")
    q = ids.add_parser("set", help="change one section: identity set Personality kind, curious. Protected sections need typed confirmations and a cooldown")
    q.add_argument("section"); q.add_argument("text", nargs="+")
    ids.add_parser("adopt", help="make a hand edit of self.md official (protected parts go through the staged process)")
    ids.add_parser("pending", help="show a waiting protected change")
    ids.add_parser("apply", help="apply the waiting protected change (after the cooldown, with a second typed confirmation)")
    ids.add_parser("cancel", help="throw the waiting protected change away")
    q = ids.add_parser("settings", help="show or change threshold / warn-drift / cooldown-hours")
    q.add_argument("--threshold", type=float, default=None); q.add_argument("--warn-drift", dest="warn_drift", type=float, default=None)
    q.add_argument("--cooldown-hours", dest="cooldown_hours", type=float, default=None)
    q = ids.add_parser("set-file", help="tell the core where a file is: identity set-file memory memory.json")
    q.add_argument("kind", choices=["person", "memory", "mood", "tokenizer"]); q.add_argument("path")
    q = ids.add_parser("add-version", help="register a checkpoint as a version (stores its checksum)")
    q.add_argument("name"); q.add_argument("--ckpt", required=True); q.add_argument("--parent", default=None); q.add_argument("--note", default=None)
    ids.add_parser("versions", help="list the versions")
    q = ids.add_parser("make-probes", help="write the fixed regression questions (once)")
    q.add_argument("--max-new", type=int, default=60)
    ids.add_parser("show-probes")
    q = ids.add_parser("record", help="ask a version the probes and save its answers (the baseline)")
    q.add_argument("name")
    q = ids.add_parser("score", help="the regression test: ask a candidate the probes and compare with the current version's answers")
    q.add_argument("name"); q.add_argument("--against", default=None)
    q = ids.add_parser("promote", help="make a version current (it must have passed the test)"); q.add_argument("name")
    q = ids.add_parser("rollback", help="go back to an older version"); q.add_argument("name")
    q = ids.add_parser("lineage", help="the history log and a check that it was not tampered with"); q.add_argument("--last", type=int, default=20)
    ids.add_parser("verify", help="check the log chain, self.md, and every checkpoint checksum")
    q = ids.add_parser("distill", help="let an old version answer prompts and write them as training data for a new model")
    q.add_argument("--teacher", required=True); q.add_argument("--out", required=True)
    q.add_argument("--prompts", default=None, help="a text file, one prompt per line"); q.add_argument("--data", nargs="+", default=None, help="train.jsonl: use what your friends said as prompts")
    q.add_argument("--limit", type=int, default=None); q.add_argument("--max-new", type=int, default=80)
    q.add_argument("--temperature", type=float, default=0.7); q.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_identity)

    s = sub.add_parser("export-package", help="Stage 9b: put the AI in ONE new folder (weights, tokenizer, identity, memory, ...) with checksums, to run it elsewhere")
    s.add_argument("--dir", default="identity_core"); s.add_argument("--out", required=True, help="a NEW folder (nothing is overwritten)")
    s.add_argument("--version", default=None, help="default: the current version"); s.add_argument("--format", choices=("npz", "gguf", "both"), default="npz")
    s.add_argument("--gguf-type", choices=("f32", "f16", "q8_0"), default="f16")
    s.add_argument("--weights-only", action="store_true", help="only the model and the tokenizer (no identity, memory, card, mood or masks)")
    for flag in ("memory", "person", "mood", "personas"):
        s.add_argument("--no-" + flag, action="store_true", help="leave the %s out" % flag)
    s.add_argument("--yes", action="store_true", help="I read the privacy warning (without it the command only prints the warning)")
    s.set_defaults(fn=cmd_export_package)

    s = sub.add_parser("verify-package", help="Stage 9b: check a package folder against its checksums")
    s.add_argument("package")
    s.set_defaults(fn=cmd_verify_package)

    s = sub.add_parser("serve", help="Stage 9b: a small local web server (JSON over HTTP) for your own app; see docs/EXPORT_GUIDE.md")
    s.add_argument("--package", default=None); s.add_argument("--core", default=None); s.add_argument("--version", default=None)
    s.add_argument("--host", default="127.0.0.1", help="default: this computer only; another host needs --token")
    s.add_argument("--port", type=int, default=8765); s.add_argument("--token", default=None)
    s.add_argument("--allow-persona-switch", action="store_true", help="let POST /persona switch the mask (off by default)")
    s.add_argument("--max-new", type=int, default=120); s.add_argument("--temperature", type=float, default=0.8); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("ui", help="Web UI (stage U0): a local, offline web page for the whole system; opens your browser; see docs/UI_PROMPT.md")
    s.add_argument("--port", type=int, default=8777, help="default 8777 (0 = any free port)")
    s.add_argument("--root", default=".", help="the project folder (default: the folder you run the command from)")
    s.add_argument("--no-open", dest="no_open", action="store_true", help="do not open the browser; just print the link")
    s.set_defaults(fn=cmd_ui)

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
