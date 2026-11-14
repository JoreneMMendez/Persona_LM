"""Personas, prompt side (Stage 8a): masks the AI can wear. A persona is a name plus a short instruction, kept in `personas.json` inside the identity folder.

Rules (they are what makes a persona a MASK and not a change of who the AI is):
  * Personas NEVER touch the identity core: this module does not read or write the self-description (core.json "sections", self.md). The system line
    built from the identity stays in every prompt; the persona block is added on top, and the block itself says "you are still <name> underneath".
  * Only typed commands change anything: `persona add/edit/remove/use` on the command line, `/persona NAME` in the chat. Nothing the model writes
    can add, edit, remove or switch a persona (the chat never looks at the reply for persona words).
  * Every add / edit / remove / switch is a line in the lineage log. The log holds names and a fingerprint of the text, never the text itself.
  * What a persona is: only PROMPT text (the model reads it in a `<|persona|> ... <|end|>` block). Nothing is trained into the weights yet (that is 8b),
    and a model that was never fine-tuned with persona blocks will follow the text only weakly.
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
            lines.append("  %s %-20s %s" % ("*" if n == self.active else " ", n, self.d["personas"][n]["text"]))
        lines.append("* = on now." if self.active else "No mask is on: the AI speaks as itself.")
        return "\n".join(lines)

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
        return "Persona %r changed." % key

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
