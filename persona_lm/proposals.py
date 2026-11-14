"""Stage 7a: self-modification, the safe way. The AI may only PROPOSE; the owner decides, at the keyboard.

A proposal is a record in `proposals.json` (inside the identity core folder): what kind of change, why, the exact change (a diff), and how to undo it.
Nothing is changed in memory or in the Person card until the owner types the proposal's number at a terminal (`identity.keyboard_ask`, which refuses
pipes and scripts). Neither the AI's words nor anything in the training data can reach `approve`: it is only called from a typed command.

Allowed kinds in this half (7a):
    memory-add         remember something new
    memory-forget      forget one memory (its text is kept in proposals.json so it can be undone; `purge-undo` erases that copy)
    memory-importance  change one memory's importance / pinned flag
    card-add           add a line to a section of the Person card
Not allowed at all: the identity core (name, personality, temperament, values). Those keep the Stage 6 rules (typed phrase, cooldown, second phrase).
Config changes and fine-tune/LoRA proposals come in 7b.

Every approval, rejection and undo is written to the lineage log (hash chain) WITHOUT the memory texts, only the proposal number, kind and a hash.
Pure Python.
"""
import copy
import json
import os
import time

from . import identity as ID
from . import memory as MEM
from . import person as PER
from .lineage import iso, sha256_text

FORMAT_VERSION = 1
KINDS = ("memory-add", "memory-forget", "memory-importance", "card-add")
MAX_PENDING = 25
MAX_WHY = 300
APPROVER = "owner (typed at the keyboard)"


class ProposalProblem(Exception):
    pass


def args_hash(args):
    return sha256_text(json.dumps(args, sort_keys=True, ensure_ascii=False))


def _short(text, n=140):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


class Proposals:
    def __init__(self, core, memory_path=None, person_path=None, clock=None):
        self.core = core
        self.clock = clock or core.clock
        self.path = os.path.join(core.dir, "proposals.json")
        self.memory_path = memory_path or core.resolve("memory")
        self.person_path = person_path or core.resolve("person")
        self.d = {"format_version": FORMAT_VERSION, "next": 1, "items": {}}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    d = json.load(f)
                if not isinstance(d.get("items"), dict) or d.get("format_version") != FORMAT_VERSION:
                    raise ValueError("unexpected content")
                self.d = d
            except (OSError, ValueError, AttributeError) as e:
                raise ProposalProblem("Cannot read %s (%s). Nothing was changed." % (self.path, e))

    # ------------------------------------------------------------------ storage
    @property
    def items(self):
        return self.d["items"]

    def _save(self):
        ID._atomic_write(self.path, json.dumps(self.d, indent=1, sort_keys=True, ensure_ascii=False) + "\n", backup=False)

    def get(self, pid):
        pid = pid.strip().lower()
        if pid.isdigit():
            pid = "p%04d" % int(pid)
        if pid not in self.items:
            raise ProposalProblem("There is no proposal %s." % pid)
        return self.items[pid]

    def listing(self, status=None):
        return [p for p in self.items.values() if status is None or p["status"] == status]

    # ------------------------------------------------------------------ files
    def _store(self):
        if not self.memory_path:
            raise ProposalProblem("No memory file is known. Give --memory FILE (or register one: identity set-file memory FILE).")
        return MEM.MemoryStore(self.memory_path, clock=self.clock)

    def _card(self):
        if not self.person_path:
            raise ProposalProblem("No Person card file is known. Give --person FILE (or register one: identity set-file person FILE).")
        return PER.load(self.person_path)

    # ------------------------------------------------------------------ submitting (the only thing the AI side may do)
    def submit(self, kind, args, why, source="typed command"):
        """Check the proposal and file it as pending. Changes NOTHING except proposals.json."""
        if kind not in KINDS:
            raise ProposalProblem("Unknown kind %r. Allowed: %s. (The identity core cannot be changed by proposals.)" % (kind, ", ".join(KINDS)))
        why = " ".join(str(why or "").split())
        if not why:
            raise ProposalProblem("Say why: a proposal without a reason is not accepted.")
        if len(why) > MAX_WHY:
            raise ProposalProblem("The reason is %d characters; the limit is %d." % (len(why), MAX_WHY))
        if len(self.listing("pending")) >= MAX_PENDING:
            raise ProposalProblem("There are already %d waiting proposals. Approve or reject some first." % MAX_PENDING)
        args = self._check_args(kind, args)
        pid = "p%04d" % self.d["next"]
        self.d["next"] += 1
        self.items[pid] = {"id": pid, "kind": kind, "status": "pending", "why": why, "args": args, "args_sha256": args_hash(args),
                           "source": str(source)[:40], "created": iso(self.clock()), "decided": None, "undo": None, "note": ""}
        self._save()
        return self.items[pid]

    def _check_args(self, kind, args):
        args = dict(args or {})
        if kind == "memory-add":
            keep = {k: args[k] for k in ("text", "emotion", "importance", "pinned", "nodes") if k in args}
            MEM.MemoryStore(None, clock=self.clock).add(keep.get("text", ""), emotion=keep.get("emotion", "neutral"), importance=keep.get("importance", 0.5),
                                                          pinned=keep.get("pinned", False), nodes=keep.get("nodes"))
            keep.setdefault("emotion", "neutral"); keep.setdefault("importance", 0.5)
            keep["importance"] = float(keep["importance"]); keep["pinned"] = bool(keep.get("pinned", False)); keep["nodes"] = list(keep.get("nodes") or [])
            keep["text"] = " ".join(str(keep["text"]).split())
            return keep
        if kind == "memory-forget":
            return {"id": self._store().get(str(args.get("id", ""))).get("id")}
        if kind == "memory-importance":
            mem = self._store().get(str(args.get("id", "")))
            try:
                value = float(args.get("importance"))
            except (TypeError, ValueError):
                raise ProposalProblem("Importance must be a number from 0 to 1.")
            if not 0 <= value <= 1:
                raise ProposalProblem("Importance must be a number from 0 to 1.")
            out = {"id": mem["id"], "importance": round(value, 3)}
            if args.get("pinned") is not None:
                out["pinned"] = bool(args["pinned"])
            return out
        card = self._card()                                                   # card-add
        section = PER.Card._canonical(str(args.get("section", "")))
        text = " ".join(str(args.get("text", "")).split())
        copy.deepcopy(card).add_owner(section, text)                          # same checks as the real add
        return {"section": section, "text": text}

    # ------------------------------------------------------------------ showing
    def preview(self, p):
        """The exact change as it would be made NOW (a diff), plus how to undo it. Raises ProposalProblem if it can no longer be made."""
        a, k = p["args"], p["kind"]
        if k == "memory-add":
            extra = "emotion %s, importance %s%s%s" % (a["emotion"], a["importance"], ", pinned" if a["pinned"] else "",
                                                       (", nodes " + ", ".join(a["nodes"])) if a["nodes"] else "")
            return ["+ new memory: %s" % a["text"], "  (%s)" % extra], "forget that memory again (typed `proposal undo %s`)" % p["id"]
        if k == "memory-forget":
            m = self._store().get(a["id"])
            return ["- memory %s: %s" % (m["id"], m["text"])], "put the memory back with the same number (typed `proposal undo %s`)" % p["id"]
        if k == "memory-importance":
            m = self._store().get(a["id"])
            line = "~ memory %s importance %s -> %s" % (m["id"], m["importance"], a["importance"])
            if "pinned" in a:
                line += ", pinned %s -> %s" % ("yes" if m.get("pinned") else "no", "yes" if a["pinned"] else "no")
            return [line, "  memory: " + _short(m["text"])], "set the old values back (typed `proposal undo %s`)" % p["id"]
        card = self._card()
        return ["+ Person card, section %s: %s" % (a["section"], a["text"])], "remove that line again (typed `proposal undo %s`)" % p["id"]

    def describe(self, p):
        try:
            diff, undo = self.preview(p)
        except (ProposalProblem, MEM.MemoryProblem, PER.PersonProblem) as e:
            diff, undo = ["(cannot be shown now: %s)" % e], "-"
        lines = ["%s  %s  [%s]  from: %s  (%s)" % (p["id"], p["kind"], p["status"], p["source"], p["created"]), "  Why:  " + p["why"]]
        lines += ["  Change:"] + ["    " + d for d in diff] + ["  Undo: " + undo]
        if p.get("note"):
            lines.append("  Note: " + p["note"])
        return "\n".join(lines)

    # ------------------------------------------------------------------ deciding (typed commands only)
    def _confirm(self, p, ask, verb):
        answer = ask("Type the proposal number (%s) to %s it; anything else cancels: " % (p["id"], verb))
        if answer.strip().lower() != p["id"]:
            raise ProposalProblem("Not confirmed. Nothing was changed.")

    def _log(self, kind, p, what):
        self.core.lineage.append(kind, "proposal %s %s: %s" % (p["id"], what, p["kind"]), APPROVER,
                                 {"proposal": p["id"], "kind": p["kind"], "args_sha256": p["args_sha256"]})
        self.core.save()

    def approve(self, pid, ask=ID.keyboard_ask, out=print):
        p = self.get(pid)
        if p["status"] != "pending":
            raise ProposalProblem("%s is %s, not waiting." % (p["id"], p["status"]))
        if args_hash(p["args"]) != p["args_sha256"]:
            raise ProposalProblem("%s was edited by hand after it was filed. It is refused; reject it and file a new one." % p["id"])
        out(self.describe(p))
        self.preview(p)                                                         # fails early if the target is gone
        self._confirm(p, ask, "approve")
        ok, msg = self.core.lineage.verify()
        if not ok:
            raise ProposalProblem("The lineage log is damaged, so nothing was changed. " + msg)
        p["undo"] = self._apply(p)
        p["status"], p["decided"] = "approved", iso(self.clock())
        self._log("proposal-approved", p, "approved")
        self._save()
        return "Approved %s and applied it. Undo with: proposal undo %s" % (p["id"], p["id"])

    def reject(self, pid, note=""):
        p = self.get(pid)
        if p["status"] != "pending":
            raise ProposalProblem("%s is %s, not waiting." % (p["id"], p["status"]))
        p["status"], p["decided"], p["note"] = "rejected", iso(self.clock()), " ".join(str(note).split())[:200]
        self._log("proposal-rejected", p, "rejected")
        self._save()
        return "Rejected %s. Nothing was changed." % p["id"]

    def undo(self, pid, ask=ID.keyboard_ask, out=print):
        p = self.get(pid)
        if p["status"] != "approved":
            raise ProposalProblem("%s is %s; only an approved proposal can be undone." % (p["id"], p["status"]))
        if not p.get("undo"):
            raise ProposalProblem("The undo copy of %s was erased (purge-undo), so it cannot be undone." % p["id"])
        out(self.describe(p))
        out("This will UNDO the change above.")
        self._confirm(p, ask, "undo")
        self._revert(p)
        p["status"], p["decided"] = "undone", iso(self.clock())
        self._log("proposal-undone", p, "undone")
        self._save()
        return "Undone %s." % p["id"]

    def purge_undo(self, pid, ask=ID.keyboard_ask):
        """Erase the saved copy a forgotten memory left in proposals.json (after this the forget can no longer be undone)."""
        p = self.get(pid)
        if not p.get("undo"):
            raise ProposalProblem("%s has no undo copy." % p["id"])
        self._confirm(p, ask, "erase the undo copy of")
        p["undo"] = None
        self._save()
        return "The undo copy of %s was erased." % p["id"]

    # ------------------------------------------------------------------ applying and reverting
    def _apply(self, p):
        a, k = p["args"], p["kind"]
        if k == "card-add":
            card = self._card()
            card.add_owner(a["section"], a["text"])
            PER.save(card, self.person_path)
            return {"section": a["section"], "text": a["text"]}
        store = self._store()
        if k == "memory-add":
            mem = store.add(a["text"], emotion=a["emotion"], importance=a["importance"], pinned=a["pinned"], nodes=a["nodes"], source="ai-proposal " + p["id"])
            undo = {"id": mem["id"], "text": mem["text"]}
            store.log("proposal-add", [mem["id"]], p["id"])
        elif k == "memory-forget":
            mem = copy.deepcopy(store.get(a["id"]))
            nodes = {nid: copy.deepcopy(store.nodes[nid]) for nid in mem["nodes"] if nid in store.nodes}
            store.forget(a["id"])
            undo = {"memory": mem, "nodes": nodes}
            store.log("proposal-forget", [a["id"]], p["id"])
        else:
            mem = store.get(a["id"])
            undo = {"id": mem["id"], "importance": mem["importance"], "pinned": bool(mem.get("pinned")), "new_importance": a["importance"], "new_pinned": a.get("pinned")}
            store.set_importance(a["id"], a["importance"], a.get("pinned"))
            store.log("proposal-importance", [a["id"]], p["id"])
        store.save(keep_backup=(k != "memory-forget"))
        return undo

    def _revert(self, p):
        u, k = p["undo"], p["kind"]
        if k == "card-add":
            card = self._card()
            items = card.section(u["section"])[1]
            for i, it in enumerate(items):
                if it["kind"] == "fact" and it["tag"] == "owner" and it["text"] == u["text"]:
                    del items[i]
                    PER.save(card, self.person_path)
                    return
            raise ProposalProblem("That line is no longer in the Person card (it was changed by hand), so there is nothing to undo.")
        store = self._store()
        if k == "memory-add":
            mem = store.memories.get(u["id"])
            if mem is None or mem["text"] != u["text"]:
                raise ProposalProblem("Memory %s is gone or was changed since, so it was not touched." % u["id"])
            store.forget(u["id"])
        elif k == "memory-forget":
            mem = u["memory"]
            if mem["id"] in store.memories:
                raise ProposalProblem("Memory %s exists again, so it was not overwritten." % mem["id"])
            for nid, node in u["nodes"].items():
                store.nodes.setdefault(nid, node)
            store.memories[mem["id"]] = mem
        else:
            mem = store.get(u["id"])
            if mem["importance"] != u["new_importance"]:
                raise ProposalProblem("Memory %s was changed since, so it was not touched." % u["id"])
            store.set_importance(u["id"], u["importance"], u["pinned"] if u["new_pinned"] is not None else None)
        store.log("proposal-undo", [], p["id"])
        store.save()
