"""Safe cleanup. Dry run by default; --apply moves safe items to _to_delete (never deletes)."""
import hashlib
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
TRASH = os.path.join(ROOT, "_to_delete")
PROTECTED = [os.path.join(*p.split("/")) for p in (
    "persona_lm", "tests", "docs", "work/tok.json", "work3/run_pre/ckpt.npz", "work3/run_chat2", "data_chat4096",
    "identity_core", "private", "_to_delete", ".git")]
JUNK_DIRS = {"__pycache__", ".pytest_cache"}
JUNK_SUFFIX = (".pyc", ".tmp.npz", ".tmp")
JUNK_NAMES = {"thumbs.db", "desktop.ini"}
ASK_ME = ["work/run_chat", "work/run_chat2", "work3/run_chat", "work2"]


def rel(p):
    return os.path.relpath(p, ROOT)


def protected(p):
    r = rel(p)
    return any(r == q or r.startswith(q + os.sep) for q in PROTECTED)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def tree_hash(folder):
    h = hashlib.sha256()
    for d, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, folder).encode())
            h.update(sha(p).encode())
    return h.hexdigest()


def find_candidates():
    out = []
    for d, dirs, files in os.walk(ROOT):
        if protected(d):
            dirs[:] = []
            continue
        for name in list(dirs):
            if name in JUNK_DIRS:
                out.append(("A", os.path.join(d, name), "regenerable junk folder"))
                dirs.remove(name)
        for f in files:
            p = os.path.join(d, f)
            if protected(p):
                continue
            if f.lower() in JUNK_NAMES or f.endswith(JUNK_SUFFIX):
                out.append(("A", p, "junk / temporary file"))
            if f == "ckpt.prev.npz":
                main = os.path.join(d, "ckpt.npz")
                if os.path.exists(main) and os.path.getsize(main) == os.path.getsize(p) and sha(main) == sha(p):
                    out.append(("B", p, "identical to ckpt.npz (SHA256 equal)"))
                elif os.path.exists(main):
                    print("  keep (differs from ckpt.npz, it is the real fallback): " + rel(p))
    ref = os.path.join(ROOT, "work", "data_pre")
    if os.path.isdir(ref):
        ref_hash = tree_hash(ref)
        for other in ("kaggle_upload", "work2"):
            p = os.path.join(ROOT, other, "data_pre")
            if os.path.isdir(p) and tree_hash(p) == ref_hash:
                out.append(("C", p, "byte-identical copy of work\\data_pre"))
            elif os.path.isdir(p):
                print("  keep (differs from work\\data_pre): " + rel(p))
    return out


def move(p):
    dest = os.path.join(TRASH, rel(p))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(p, dest)


def main():
    apply = "--apply" in sys.argv
    print("Project folder: %s\nMode: %s\n" % (ROOT, "APPLY (moving to _to_delete)" if apply else "DRY RUN (nothing changes)"))
    cands = find_candidates()
    total = 0
    for cat, p, why in sorted(cands, key=lambda c: -size_of(c[1])):
        s = size_of(p)
        total += s
        print("[%s] %8.1f MB  %s   (%s)" % (cat, s / 1e6, rel(p), why))
    print("\nSafe items: %d, %.1f MB" % (len(cands), total / 1e6))
    print("\nAsk me (NOT moved):")
    for a in ASK_ME:
        p = os.path.join(ROOT, *a.split("/"))
        if os.path.exists(p):
            print("   %8.1f MB  %s" % (size_of(p) / 1e6, a))
    if apply:
        for _, p, _ in cands:
            if os.path.exists(p):
                move(p)
        print("\nMoved %d items into %s. Delete that folder yourself when you are sure." % (len(cands), TRASH))
    else:
        print("\nRun again with --apply to move the safe items.")


if __name__ == "__main__":
    main()
