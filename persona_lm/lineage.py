"""The lineage log (Stage 6): the AI's history, one line per event, in a plain text file (JSON lines).

    {"n": 3, "time": "2026-11-14T10:00:00Z", "t": 1700000000.0, "kind": "promote", "summary": "...", "approved_by": "owner (typed)", "data": {...},
     "prev": "<hash of line 2>", "hash": "<hash of this line>"}

Each line contains the hash of the line before it, so the lines form a chain: if one line is edited, deleted from the middle or reordered,
`verify` finds it. This is TAMPER-EVIDENT, not tamper-proof: someone who can edit files on your computer could rewrite the whole chain.
The identity core also remembers the newest line's number and hash, so cutting lines off the END is noticed as well.
Only code run from a typed command appends here; nothing the AI writes can.
Pure Python (hashlib, json).
"""
import hashlib
import json
import os
import time

GENESIS = "0" * 64
KINDS = ("genesis", "identity-edit", "identity-staged", "identity-applied", "identity-cancelled", "version-added", "scored", "promote", "rollback",
         "settings", "distill", "note",
         "proposal-approved", "proposal-rejected", "proposal-undone",
         "persona-added", "persona-edited", "persona-removed", "persona-switched")


class LineageProblem(Exception):
    pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _hash_of(entry):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return sha256_text(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


class Lineage:
    def __init__(self, path, clock=time.time):
        self.path, self.clock = path, clock

    # -- reading
    def entries(self):
        """All entries (a list of dicts); [] when the file does not exist. A line that is not JSON raises LineageProblem."""
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    if not isinstance(e, dict):
                        raise ValueError("not an object")
                except ValueError as err:
                    raise LineageProblem("Line %d of %s is not a valid log line (%s)." % (i, self.path, err))
                out.append(e)
        return out

    def verify(self):
        """(ok, message). Checks numbering, the chain of hashes and every line's own hash."""
        try:
            es = self.entries()
        except LineageProblem as e:
            return False, str(e)
        prev = GENESIS
        for i, e in enumerate(es, 1):
            if e.get("n") != i:
                return False, "Line %d has the number %r (expected %d): a line was removed, added or reordered." % (i, e.get("n"), i)
            if e.get("prev") != prev:
                return False, "Line %d does not follow line %d: the log was changed before it." % (i, i - 1)
            if e.get("hash") != _hash_of(e):
                return False, "Line %d was edited after it was written." % i
            prev = e["hash"]
        return True, "%d line(s), chain intact." % len(es)

    def head(self):
        """{"n": last number, "hash": last hash} (n = 0 for an empty log)."""
        es = self.entries()
        return {"n": len(es), "hash": es[-1]["hash"] if es else GENESIS}

    def last_of(self, *kinds):
        for e in reversed(self.entries()):
            if e.get("kind") in kinds:
                return e
        return None

    # -- writing
    def append(self, kind, summary, approved_by, data=None):
        """Adds one line (after checking the chain is intact). approved_by says who allowed it, e.g. "owner (typed at the keyboard)" or "system (automatic rule)"."""
        if kind not in KINDS:
            raise LineageProblem("Unknown log kind %r." % kind)
        ok, msg = self.verify()
        if not ok:
            raise LineageProblem("The lineage log is damaged, so nothing was added. %s" % msg)
        head = self.head()
        now = self.clock()
        entry = {"n": head["n"] + 1, "time": iso(now), "t": round(now, 3), "kind": kind, "summary": summary, "approved_by": approved_by,
                 "data": data or {}, "prev": head["hash"]}
        entry["hash"] = _hash_of(entry)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return entry

    def describe(self, last=None):
        es = self.entries()
        if last:
            es = es[-last:]
        return "\n".join("%3d  %s  %-17s %s   [%s]" % (e["n"], e["time"], e["kind"], e["summary"], e["approved_by"]) for e in es) or "(the log is empty)"
