"""Personas, prompt side (Stage 8a): masks the AI can wear. A persona is a name plus a short instruction, kept in `personas.json` inside the identity folder.

Rules (they are what makes a persona a MASK and not a change of who the AI is):
  * Personas NEVER touch the identity core: this module does not read or write the self-description (core.json "sections", self.md). The system line
    built from the identity stays in every prompt; the persona block is added on top, and the block itself says "you are still <name> underneath".
  * Only typed commands change anything: `persona add/edit/remove/use` on the command line, `/persona NAME` in the chat. Nothing the model writes
    can add, edit, remove or switch a persona (the chat never looks at the reply for persona words).
  * Every add / edit / remove / switch is a line in the lineage log. The log holds names and a fingerprint of the text, never the text itself.
  * What a persona is: PROMPT text (the model reads it in a `<|persona|> ... <|end|>` block). A model that was never fine-tuned with persona blocks follows
    it only weakly. Stage 8b adds the OPTIONAL trained side: the owner's own example replies for a persona (`example`), a training file in which every example
    carries the exact same block the chat uses (`make-data`), and `attach` to say "this registered version was fine-tuned for this mask". That version is a
    full fine-tuned COPY of the weights, not a small adapter: LoRA is not built. Only prompt text works without training; training only makes it stick better.
  * `check` asks the identity probes with the mask ON and compares with the version recorded without it: do the facts about who the AI is survive the mask?
Pure Python (json, os, re).
"""
import json
import os
import re
import time

from .lineage import iso, sha256_text

FORMAT_VERSION = 1
FILE_NAME = "personas.json"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,29}$")
RESERVED = ("none", "default", "off", "me-as-identity")      # words that mean "no mask"; they cannot be persona names
MAX_TEXT_CHARS = 300
MAX_PERSONAS = 20
MAX_EXAMPLES = 200
MAX_EXAMPLE_CHARS = 300
OWNER = "owner (typed command)"


class PersonaProblem(Exception):
    pass


def check_name(name):
    name = " ".join(str(name).split())
    if not NAME_RE.match(name):
        raise PersonaProblem("A persona name is 1-30 letters, digits, spaces, dots, dashes or underscores (it must start with a letter or digit): %r." % name)
    if name.casefold() in RESERVED:
        raise PersonaProblem("%r means 'no mask' and cannot be a persona name." % name)
    return name


def check_text(text):
    text = " ".join(str(text).split())
    if not text:
        raise PersonaProblem("A persona needs some text: how it speaks and what it is for.")
    if len(text) > MAX_TEXT_CHARS:
        raise PersonaProblem("The persona text is %d characters; at most %d (it has to fit in the AI's small window). Shorten it." % (len(text), MAX_TEXT_CHARS))
    if "<|" in text or "|>" in text:
        raise PersonaProblem("The persona text may not contain '<|' or '|>' (they would forge special tokens).")
    return text


def check_example(user, reply):
    out = []
    for what, t in (("the message", user), ("the reply", reply)):
        t = " ".join(str(t).split())
        if not t:
            raise PersonaProblem("An example needs both a message and a reply (%s is empty)." % what)
        if len(t) > MAX_EXAMPLE_CHARS:
            raise PersonaProblem("%s is %d characters; at most %d." % (what.capitalize(), len(t), MAX_EXAMPLE_CHARS))
        if "<|" in t or "|>" in t:
            raise PersonaProblem("%s may not contain '<|' or '|>' (they would forge special tokens)." % what.capitalize())
        out.append(t)
    return out[0], out[1]


class Wearing:
    """A persona put on for ONE test only (nothing is switched, nothing is saved). Has the same prompt_text() the chat session asks for."""

    def __init__(self, core_name, name, text):
        self.core_name, self.name, self.text = core_name, name, text

    def prompt_text(self):
        return block_text(self.core_name, self.name, self.text)


def block_text(core_name, persona_name, text):
    """The words inside the <|persona|> block. The last sentence is fixed: it is how the AI keeps knowing it is the same one underneath."""
    return "Persona %s: %s You are still %s underneath; this is only a mask." % (persona_name, text if text.endswith((".", "!", "?")) else text + ".", core_name)


class Personas:
    """The persona file of one identity core. `core` is an identity.Core (used for the folder and the lineage log ONLY)."""

    def __init__(self, core, clock=time.time):
        self.core, self.clock = core, clock
        self.path = os.path.join(core.dir, FILE_NAME)
        self.d = {"format_version": FORMAT_VERSION, "active": None, "personas": {}}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("format_version") != FORMAT_VERSION:
                    raise PersonaProblem("%s has format version %r; this program reads version %d." % (self.path, d.get("format_version"), FORMAT_VERSION))
                personas = {str(k): dict(v) for k, v in d["personas"].items()}
                for k, v in personas.items():
                    check_name(k)
                    check_text(v["text"])
                    v["examples"] = [list(check_example(*e)) for e in v.get("examples", [])]
                    if len(v["examples"]) > MAX_EXAMPLES:
                        raise ValueError("too many examples for %r" % k)
                    if v.get("version") is not None and not isinstance(v["version"], str):
                        raise ValueError("bad version for %r" % k)
                active = d.get("active")
                if active is not None and active not in personas:
                    raise ValueError("the active persona %r does not exist" % active)
                self.d = {"format_version": FORMAT_VERSION, "active": active, "personas": personas}
            except PersonaProblem:
                raise
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                raise PersonaProblem("Cannot read %s (%s). Nothing was changed." % (self.path, e))

    # -- reading
    @property
    def active(self):
        return self.d["active"]

    def names(self):
        return sorted(self.d["personas"], key=str.casefold)

    def get(self, name):
        key = self._find(name)
        return key, self.d["personas"][key]

    def _find(self, name):
        wanted = " ".join(str(name).split()).casefold()
        for k in self.d["personas"]:
            if k.casefold() == wanted:
                return k
        raise PersonaProblem("There is no persona called %r. Personas: %s." % (name, ", ".join(self.names()) or "(none yet; add one with: persona add)"))

    def prompt_text(self):
        """The text for the <|persona|> block, or None when no mask is on."""
        if self.active is None:
            return None
        return block_text(self.core.sections["Name"], self.active, self.d["personas"][self.active]["text"])

    def describe(self):
        if not self.d["personas"]:
            return "No personas yet. Add one with:  python -m persona_lm persona add \"calm mentor\" \"You speak slowly and kindly and ask good questions.\""
        lines = ["Personas (masks: they never change who %s is):" % self.core.sections["Name"]]
        for n in self.names():
            p = self.d["personas"][n]
            extra = ""
            if p.get("examples"):
                extra += "  [%d example(s)]" % len(p["examples"])
            if p.get("version"):
                extra += "  [trained version: %s%s]" % (p["version"], ", STALE: the text changed after training" if self.stale(n) else "")
            lines.append("  %s %-20s %s%s" % ("*" if n == self.active else " ", n, p["text"], extra))
        lines.append("* = on now." if self.active else "No mask is on: the AI speaks as itself.")
        return "\n".join(lines)

    def stale(self, name):
        """True when the mask's text changed after its trained version was attached (the variant learned the OLD text)."""
        _, p = self.get(name)
        return bool(p.get("version")) and p.get("trained_text_sha256") != p["text_sha256"]

    def wearing(self, name):
        key, p = self.get(name)
        return Wearing(self.core.sections["Name"], key, p["text"])

    def version_for(self, name):
        """The registered version attached to this persona, or None. A rejected or vanished version is refused, not silently replaced."""
        key, p = self.get(name)
        v = p.get("version")
        if not v:
            return None
        try:
            status = self.core.version(v)["status"]
        except Exception:
            raise PersonaProblem("Persona %r points at version %r, which is not registered any more. Detach it: persona detach %s" % (key, v, key))
        if status == "rejected":
            raise PersonaProblem("Version %r (attached to persona %r) was REJECTED by the regression test, so it will not be used. Detach it: persona detach %s" % (v, key, key))
        return v

    # -- writing (all typed commands; each is logged)
    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(self.d, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _log(self, kind, summary, data):
        """Lineage line first (it also checks the chain is intact), then core.json learns the new log end. The identity SECTIONS are not touched."""
        self.core.lineage.append(kind, summary, OWNER, data)
        self.core.save()

    def add(self, name, text):
        name, text = check_name(name), check_text(text)
        if any(k.casefold() == name.casefold() for k in self.d["personas"]):
            raise PersonaProblem("A persona called %r already exists (use edit to change it)." % name)
        if len(self.d["personas"]) >= MAX_PERSONAS:
            raise PersonaProblem("There are already %d personas; remove one first." % MAX_PERSONAS)
        sha = sha256_text(text)
        self._log("persona-added", "persona %s added" % name, {"persona": name, "text_sha256": sha})     # log first: a damaged log stops the change
        self.d["personas"][name] = {"text": text, "created": iso(self.clock()), "text_sha256": sha}
        self._save()
        return "Persona %r added (not switched on; use: persona use %s)." % (name, name)

    def edit(self, name, text):
        key, p = self.get(name)
        text = check_text(text)
        if text == p["text"]:
            return "Nothing to change: the text is the same."
        sha = sha256_text(text)
        self._log("persona-edited", "persona %s edited" % key, {"persona": key, "old_sha256": p["text_sha256"], "new_sha256": sha})
        self.d["personas"][key] = dict(p, text=text, text_sha256=sha)
        self._save()
        return "Persona %r changed.%s" % (key, " Its trained version %r learned the OLD text: train again or detach it." % p["version"] if p.get("version") else "")

    def remove(self, name):
        key, p = self.get(name)
        self._log("persona-removed", "persona %s removed%s" % (key, " (it was on; the mask is off now)" if key == self.active else ""),
                  {"persona": key, "text_sha256": p["text_sha256"]})
        del self.d["personas"][key]
        if self.d["active"] == key:
            self.d["active"] = None
        self._save()
        return "Persona %r removed." % key

    def use(self, name):
        """Switch a mask on, or off with none/default/off. Logged. Switching to the mask that is already on changes nothing."""
        if " ".join(str(name).split()).casefold() in RESERVED:
            new = None
        else:
            new, _ = self.get(name)
        if new == self.active:
            return "Already %s." % ("wearing %r" % new if new else "without a mask")
        self._log("persona-switched", "mask %s -> %s" % (self.active or "none", new or "none"), {"old": self.active, "new": new})
        self.d["active"] = new
        self._save()
        return "Now wearing %r. Underneath, still %s." % (new, self.core.sections["Name"]) if new else "Mask off: speaking as %s itself." % self.core.sections["Name"]

    # -- Stage 8b: the owner's examples, the training file, the trained version
    def add_example(self, name, user, reply):
        key, p = self.get(name)
        user, reply = check_example(user, reply)
        examples = list(p.get("examples", []))
        if len(examples) >= MAX_EXAMPLES:
            raise PersonaProblem("Persona %r already has %d examples (the most allowed)." % (key, MAX_EXAMPLES))
        if [user, reply] in examples:
            raise PersonaProblem("That exact example is already there.")
        self._log("persona-edited", "example %d added to persona %s" % (len(examples) + 1, key),
                  {"persona": key, "examples": len(examples) + 1, "example_sha256": sha256_text(user + "\n" + reply)})
        self.d["personas"][key] = dict(p, examples=examples + [[user, reply]])
        self._save()
        return "Example %d added to %r." % (len(examples) + 1, key)

    def make_data(self, name, out_path, tok, ctx, max_new=80):
        """Writes the persona's examples as chat training lines: system line (identity, no loss), persona block (no loss), message (no loss), reply (LOSS).
        The persona message holds exactly the text the chat puts in its <|persona|> block, so training and chatting look the same. Never overwrites."""
        from . import identity as ID
        key, p = self.get(name)
        if not p.get("examples"):
            raise PersonaProblem("Persona %r has no examples yet. Add some: persona example %s \"what they say\" \"how this persona answers\"" % (key, key))
        if os.path.exists(out_path):
            raise PersonaProblem("%s already exists; nothing was overwritten. Pick another name." % out_path)
        count = lambda t: len(tok.encode(t, allow_special=False))
        sysline, _ = ID.system_line(self.core.sections, count, max(((ctx - max_new) // 2) - 4, 8))
        block = block_text(self.core.sections["Name"], key, p["text"])
        budget = max(ctx - max_new, ctx // 2) - 1                  # the same rule the chat uses: a mask over a third of what is left after the system line is refused
        room = (budget - (count(sysline) + 2 if sysline else 0)) // 3
        if count(block) + 2 > room:
            raise PersonaProblem("This persona (%d tokens) does not fit a window of %d tokens next to the identity line (at most %d). Shorten the persona text, "
                                 "or use a bigger --ctx if the model really has a bigger window." % (count(block) + 2, ctx, room))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for user, reply in p["examples"]:
                msgs = ([{"role": "system", "content": sysline, "train": False}] if sysline else []) + [
                    {"role": "persona", "content": block, "train": False}, {"role": "user", "content": user, "train": False},
                    {"role": "assistant", "content": reply}]
                f.write(json.dumps({"messages": msgs, "source": "persona:" + key}, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        from .lineage import sha256_file
        self._log("persona-data", "%d example(s) of persona %s written to %s" % (len(p["examples"]), key, os.path.basename(out_path)),
                  {"persona": key, "examples": len(p["examples"]), "text_sha256": p["text_sha256"], "file_sha256": sha256_file(out_path)})
        return len(p["examples"])

    def attach(self, name, version):
        """Say that registered `version` was fine-tuned for this mask. The chat then uses it while the mask is on. It must exist and must not be rejected."""
        key, p = self.get(name)
        try:
            status = self.core.version(version)["status"]
        except Exception as e:
            raise PersonaProblem(str(e))
        if status == "rejected":
            raise PersonaProblem("Version %r was rejected by the regression test; it cannot be attached." % version)
        if p.get("version") == version and p.get("trained_text_sha256") == p["text_sha256"]:
            return "Already attached."
        self._log("persona-edited", "version %s attached to persona %s" % (version, key), {"persona": key, "version": version, "text_sha256": p["text_sha256"]})
        self.d["personas"][key] = dict(p, version=version, trained_text_sha256=p["text_sha256"])
        self._save()
        return "Version %r is attached to %r: the chat will use it while this mask is on (unless you name a version yourself). Check it with: persona check %s" % (version, key, key)

    def detach(self, name):
        key, p = self.get(name)
        if not p.get("version"):
            return "Persona %r has no version attached." % key
        self._log("persona-edited", "version %s detached from persona %s" % (p["version"], key), {"persona": key, "version": None})
        q = dict(p)
        q.pop("version", None)
        q.pop("trained_text_sha256", None)
        self.d["personas"][key] = q
        self._save()
        return "Detached %r from %r." % (p["version"], key)


def check_mask(core, ps, name, generate, tok, ctx, version=None, against=None, out=print):
    """The identity regression test WITH a mask on. Asks every identity probe while the persona is worn and compares with the answers recorded WITHOUT a mask
    for `against` (default: the current version). Judges only whether the facts about who the AI is survive (the 'recall' category of the probes, against the
    owner's threshold). Overall similarity and tone are SHOWN but are expected to change, since changing the way it speaks is what a mask is for.
    Changes no identity file and promotes nothing; the result is one lineage line (numbers only). Returns a dict."""
    from . import identity as ID
    from .lineage import sha256_file
    key, p = ps.get(name)
    against = against or core.current
    if not against:
        raise ID.IdentityProblem("There is no current version to compare with. Register, record and promote one first (identity add-version / record / promote).")
    base = ID.load_answers(core, against)
    doc = ID.load_probes(core)
    if base["probes_sha256"] != sha256_file(core.probes_path()):
        raise ID.IdentityProblem("probes.json changed after the baseline (%s) was recorded. Record the baseline again:  identity record %s" % (against, against))
    now = ID._state_hashes(core)
    if base["identity_sha256"] != now["identity_sha256"]:
        raise ID.IdentityProblem("The self-description changed after the baseline (%s) was recorded. Record the baseline again:  identity record %s" % (against, against))
    notes = [("note: the %s changed since the baseline was recorded; answers may differ for that reason too." % label) for k, label in
             (("person_sha256", "Person card"), ("memory_sha256", "memory file")) if base[k] != now[k]]
    answers = ID.run_probes(core, generate, tok, ctx, doc, out=out, personas=ps.wearing(key))
    could = {a["id"] for a in base["answers"] if a["reply"]}
    silent = [a["id"] for a in answers if not a["reply"] and a["id"] in could]
    if could and len(silent) == len(could):
        raise PersonaProblem("With this mask on the model gave no answer to any probe. Most likely the mask text does not fit in the model's window (%d tokens) next to the "
                             "identity line, so the chat would refuse every message too. Shorten the persona text (persona edit)." % ctx)
    res = ID.compare(doc, base["answers"], answers)
    if silent:
        notes.append("note: %d probe(s) got no answer with the mask on (%s); they count as facts lost." % (len(silent), ", ".join(silent)))
    recall = res["categories"].get("recall")
    threshold = core.settings["threshold"]
    res["judged"] = recall is not None
    res["passed"] = recall is not None and recall >= threshold
    res["recall"], res["threshold"], res["against"], res["version"], res["notes"] = recall, threshold, against, version or against, notes
    core.lineage.append("persona-checked", "persona %s on version %s vs %s: facts kept %s (threshold %.2f)" % (
        key, res["version"], against, "n/a" if recall is None else "%.2f" % recall, threshold), "owner (typed command)",
        {"persona": key, "version": res["version"], "against": against, "recall": recall, "score": round(res["score"], 4), "passed": res["passed"]})
    core.save()
    return res


def describe_check(name, res):
    lines = ["Persona %r worn by version %s, compared with %s worn WITHOUT a mask:" % (name, res["version"], res["against"])]
    for r in res["rows"]:
        extra = "  facts kept %.2f" % r["recall"] if "recall" in r else ""
        lines.append("  %s %-6s similarity %.2f%s" % (r["id"], r["kind"], r["similarity"], extra))
    if not res["judged"]:
        lines.append("There are no identity probes with expected facts (name, owner, values, memories), so nothing can be judged. See: identity show-probes")
    else:
        lines.append("FACTS ABOUT WHO IT IS KEPT: %.2f   threshold %.2f   ->  %s" % (
            res["recall"], res["threshold"], "OK: the mask does not hide who it is." if res["passed"] else
            "TOO LOW: with this mask on, the AI no longer states the facts about itself. Make the mask text shorter or gentler, or train it with examples that keep them."))
    lines.append("Overall similarity %.2f and tone are expected to differ (that is what a mask does); they are not judged." % res["categories"]["similarity"])
    lines += res["notes"]
    lines.append("This compares words, not minds. Nothing was promoted or changed; read a few answers yourself.")
    return "\n".join(lines)
