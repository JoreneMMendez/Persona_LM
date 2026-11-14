"""The export package (Stage 9b): one NEW folder that holds everything needed to run your AI somewhere else, with a manifest of checksums.

    python -m persona_lm export-package --dir identity_core --out my_ai_package --yes

Layout (what is inside depends on the options):
    package.json               the manifest: what is inside, sizes, SHA-256 checksums, which files change when the AI is used
    model/ckpt.npz             the weights ONLY (no optimizer state); read by persona_lm.runtime.load_package
    model/model.gguf           (optional, --format gguf|both) for llama.cpp and similar
    tokenizer.json             the frozen tokenizer
    identity/identity.json     the identity sections (who the AI is) + the mood baseline
    memory/memory.json         the memory graph, plus memory/vault/ = the same as Markdown notes (Obsidian style)
    person/person.md           the Person card
    mood/mood.json             the current mood
    personas/personas.json     the masks (name + text only)

NEVER included: the lineage log, proposals, the persona examples (your own words), the optimizer state, probes, Messenger data, other versions.
`--weights-only` leaves out everything except model + tokenizer. Read PRIVACY below: even weights can leak what they were trained on.
Pure Python + NumPy (no PyTorch).
"""
import json
import os
import shutil
import time

from . import memory as MEM
from . import memory_export as MEX
from .bpe import Tokenizer
from .ckpt import load_ckpt, save_ckpt
from .emotion import Mood
from .gguf import export_model
from .lineage import iso, sha256_file

FORMAT_VERSION = 1
MANIFEST = "package.json"
KIND = "persona_lm_package"
# files the AI rewrites while it is used (memory recall counts, the mood): a changed checksum there is normal, not damage
MUTABLE = ("memory/memory.json", "mood/mood.json", "personas/personas.json")
NEVER = ["lineage log", "proposals", "persona example texts", "optimizer state", "regression probes", "Messenger data", "other versions"]


class ExportProblem(Exception):
    pass


def privacy_text(contains):
    lines = ["PRIVACY: read this before you share the package.",
             "  - The weights were trained on your messages. A small model can repeat parts of its training data word for word, so the weights themselves can leak private text.",
             "    Treat the package like your messages: only give it to people and apps you would give them to."]
    if contains.get("identity"):
        lines.append("  - The identity file names the AI and says who you are (your name appears in it).")
    if contains.get("memory"):
        lines.append("  - The memory graph and its Markdown notes hold what the AI remembers about you and the people in your life.")
    if contains.get("person"):
        lines.append("  - The Person card holds facts about you.")
    if contains.get("personas"):
        lines.append("  - The persona file holds the text of your masks.")
    if not any(contains.get(k) for k in ("identity", "memory", "person", "personas", "mood")):
        lines.append("  - Weights-only package: no identity, memory, card, mood or masks are included.")
    lines.append("  Never included: " + ", ".join(NEVER) + ".")
    return "\n".join(lines)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)


def export_package(core, out_dir, version=None, fmt="npz", gguf_type="f16", weights_only=False, memory=True, person=True, mood=True, personas=True,
                   confirmed=False, out=print, clock=time.time):
    """Builds the package in `out_dir` (a NEW folder; nothing is overwritten). `core` is an identity.Core. Returns the manifest dict.
    Without confirmed=True it raises ExportProblem carrying the privacy warning (nothing is written)."""
    if fmt not in ("npz", "gguf", "both"):
        raise ExportProblem("--format must be npz, gguf or both.")
    from .personas import Personas
    name = version or core.current
    if not name:
        raise ExportProblem("The identity core has no current version. Say which one: --version NAME.")
    mem_path, person_path, mood_path = core.resolve("memory"), core.resolve("person"), core.resolve("mood")
    contains = {"identity": not weights_only,
                "memory": bool(memory and not weights_only and mem_path and os.path.exists(mem_path)),
                "person": bool(person and not weights_only and person_path and os.path.exists(person_path)),
                "mood": bool(mood and not weights_only),
                "personas": False}
    persona_list = {}
    if personas and not weights_only:
        ps = Personas(core)
        persona_list = {n: {"text": ps.d["personas"][n]["text"]} for n in ps.names()}
        contains["personas"] = bool(persona_list)
    if not confirmed:
        raise ExportProblem(privacy_text(contains) + "\nNothing was written. If you accept this, run the same command again with --yes.")
    out_dir = os.path.abspath(out_dir)
    partial = out_dir + ".partial"
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise ExportProblem("%s already exists and is not empty. Nothing was written. Choose a new folder name." % out_dir)
    if os.path.exists(partial):
        raise ExportProblem("%s exists (an earlier export stopped half way). Delete that folder and try again." % partial)
    core.verify_version_files(name)                                      # checksum of the checkpoint and the frozen tokenizer
    got = load_ckpt(core.abs(core.version(name)["ckpt"]))
    if got is None:
        raise ExportProblem("Cannot read the checkpoint of version %s." % name)
    params, _, meta = got
    tok_path = core.resolve("tokenizer")
    if not tok_path or not os.path.exists(tok_path):
        raise ExportProblem("The identity core has no tokenizer file.")
    from .model import Config
    cfg = Config.from_dict(meta["cfg"])
    tok = Tokenizer.load(tok_path)
    os.makedirs(partial)
    try:
        if fmt in ("npz", "both"):
            keep = {k: meta[k] for k in ("cfg", "tokenizer_sha256") if k in meta}
            os.makedirs(os.path.join(partial, "model"), exist_ok=True)
            save_ckpt(os.path.join(partial, "model", "ckpt.npz"), params, {}, dict(keep, exported_version=name))
        if fmt in ("gguf", "both"):
            os.makedirs(os.path.join(partial, "model"), exist_ok=True)
            export_model(os.path.join(partial, "model", "model.gguf"), params, cfg, tok, kind=gguf_type, name=core.sections["Name"])
        shutil.copyfile(tok_path, os.path.join(partial, "tokenizer.json"))
        if contains["identity"]:
            _write_json(os.path.join(partial, "identity", "identity.json"),
                        {"format_version": 1, "name": core.sections["Name"], "sections": dict(core.sections), "baseline": list(core.baseline())})
        if contains["memory"]:
            store = MEM.MemoryStore(mem_path)
            _write_json(os.path.join(partial, "memory", "memory.json"), store.data)
            MEX.export_vault(store, os.path.join(partial, "memory", "vault"))
        if contains["person"]:
            os.makedirs(os.path.join(partial, "person"))
            shutil.copyfile(person_path, os.path.join(partial, "person", "person.md"))
        if contains["mood"]:
            if mood_path and os.path.exists(mood_path):
                os.makedirs(os.path.join(partial, "mood"))
                shutil.copyfile(mood_path, os.path.join(partial, "mood", "mood.json"))
            else:
                Mood(os.path.join(partial, "mood", "mood.json"), baseline=core.baseline()).save()
        if contains["personas"]:
            _write_json(os.path.join(partial, "personas", "personas.json"), {"format_version": 1, "active": Personas(core).active, "personas": persona_list})
        files = {}
        for root, _, names in os.walk(partial):
            for n in names:
                full = os.path.join(root, n)
                rel = os.path.relpath(full, partial).replace(os.sep, "/")
                files[rel] = {"sha256": sha256_file(full), "bytes": os.path.getsize(full), "mutable": rel in MUTABLE}
        manifest = {"format_version": FORMAT_VERSION, "kind": KIND, "created": iso(clock()), "name": core.sections["Name"], "version": name,
                    "weights": {"npz": fmt in ("npz", "both"), "gguf": fmt in ("gguf", "both"), "gguf_type": gguf_type if fmt != "npz" else None},
                    "weights_only": bool(weights_only), "contains": contains, "never_included": NEVER,
                    "model": {"cfg": cfg.to_dict(), "tokenizer_sha256": tok.fingerprint()}, "files": dict(sorted(files.items()))}
        _write_json(os.path.join(partial, MANIFEST), manifest)
        os.replace(partial, out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    core.lineage.append("exported", "package exported (version %s, %s, %d files)" % (name, "weights only" if weights_only else "full", len(manifest["files"])),
                        "owner (typed command)", {"version": name, "weights_only": bool(weights_only), "contains": contains, "files": len(manifest["files"]),
                                                  "manifest_sha256": sha256_file(os.path.join(out_dir, MANIFEST))})
    core.save()
    out("wrote %s (%d files)" % (out_dir, len(manifest["files"])))
    return manifest


def read_manifest(folder):
    path = os.path.join(folder, MANIFEST)
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("kind") != KIND or m.get("format_version") != FORMAT_VERSION or not isinstance(m.get("files"), dict):
            raise ValueError("not a persona_lm package of this format")
        return m
    except (OSError, ValueError) as e:
        raise ExportProblem("%s is not a readable package manifest (%s)." % (path, e))


def verify_package(folder):
    """Returns (problems, notes). problems = missing or changed files (the package is damaged); notes = normal changes (files the AI rewrites when used)
    and files that are in the folder but not in the manifest."""
    m = read_manifest(folder)
    problems, notes = [], []
    for rel, info in m["files"].items():
        full = os.path.join(folder, *rel.split("/"))
        if not os.path.exists(full):
            problems.append("missing: " + rel)
        elif sha256_file(full) != info["sha256"]:
            (notes if info.get("mutable") else problems).append(("changed since export (normal after use): " if info.get("mutable") else "CHANGED: ") + rel)
    listed = set(m["files"]) | {MANIFEST}
    for root, _, names in os.walk(folder):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), folder).replace(os.sep, "/")
            if rel not in listed:
                notes.append("not in the manifest: " + rel)
    return problems, notes
