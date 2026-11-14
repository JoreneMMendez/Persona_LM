"""The identity core (Stage 6): what makes the AI the SAME individual after every retrain. It lives in plain files OUTSIDE the weights.

    identity_core/
      core.json       settings, where the Person card / memory / mood / tokenizer files are, the list of model versions, a pending change
      self.md         the AI's own self-description (a view you may read in Notepad; the official copy is inside core.json, see "hand edits")
      lineage.jsonl   the history: every version, every change, who allowed it (see lineage.py)
      probes.json     fixed test questions        answers/<version>.json   what each version answered (the regression tests)

Rules (they are tested):
  * The self-description has six sections. Name, Personality, Temperament and Values are PROTECTED: changing them needs a diff, your TYPED
    confirmation twice (once to stage, once to apply), a cooldown in between (default 24 hours), and it can never happen close to a model
    upgrade (promote / rollback), in either direction. Speaking style and Relation to owner are ordinary: your typed command is enough, still logged.
  * Approvals are read only from a keyboard (a terminal). Piped text, the AI's replies and training data can never approve anything.
  * A hand edit of self.md changes nothing by itself: the official text is the copy in core.json. `identity adopt` turns the edit into a
    proper change (ordinary parts now, protected parts through the staged process).
  * One FROZEN tokenizer: the first version pins its fingerprint; a version with another tokenizer is refused.
  * Every version is registered with the SHA-256 checksum of its checkpoint file; a changed or missing file is detected before it is used.
  * A new version must pass the regression probes against the current one (similarity of answers, recall, tone). Below your threshold =
    "rejected" and it cannot be promoted. Honest limit: the probes measure how SIMILAR answers are, not whether it is "the same mind".
  * Old versions are kept and can be chatted with (`chat --core DIR --version NAME`) and rolled back to. The checkpoint files themselves are
    NOT copied into the core (they are big): keep them.
Pure Python + the project's own modules; no new dependency.
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter

from . import emotion as EMO
from .lineage import Lineage, LineageProblem, iso, sha256_file, sha256_text

FORMAT_VERSION = 1
SECTIONS = ["Name", "Personality", "Temperament", "Values", "Speaking style", "Relation to owner"]
PROTECTED = ("Name", "Personality", "Temperament", "Values")
PROMPT_ORDER = ["Name", "Personality", "Values", "Relation to owner", "Speaking style", "Temperament"]     # what is dropped LAST comes first
MAX_SECTION_CHARS = 400
DEFAULTS = {"threshold": 0.6, "warn_drift": 0.4, "cooldown_hours": 24.0}
STATUSES = ("candidate", "passed", "rejected", "current", "retired")
WEIGHTS = {"similarity": 0.5, "recall": 0.3, "tone": 0.2}
VERSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
CONFIRM_STAGE = "CHANGE IDENTITY"
CONFIRM_APPLY = "APPLY IDENTITY"
REQUIRE_TTY = True          # tests switch this off; the real program only accepts approvals typed at a terminal
BASELINE_LINE = re.compile(r"^\s*baseline\s*:\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*$", re.I | re.M)

HEADER = ("# Identity core: self-description\n"
          "<!-- This file is a VIEW. The official text is kept in core.json. If you edit it here, nothing changes until you run `identity adopt`.\n"
          "     Protected sections (Name, Personality, Temperament, Values) need a typed confirmation, a cooldown and a second typed confirmation.\n"
          "     Temperament may contain one line like  baseline: +0.1 +0.0  (valence arousal: the feeling the AI drifts back to). -->\n")


class IdentityProblem(Exception):
    pass


# ---------------------------------------------------------------------------------------------- reading approvals (keyboard only)
def keyboard_ask(prompt):
    """What the person types. Refuses when the input is not a terminal: a script or a pipe could otherwise "approve" for the person."""
    if REQUIRE_TTY and not (sys.stdin and sys.stdin.isatty()):
        raise IdentityProblem("Approvals are read only from the keyboard in a terminal window (not from a pipe or a script), so nothing the AI "
                              "writes can approve a change. Run this command by hand in PowerShell. Nothing was changed.")
    try:
        return input(prompt)
    except EOFError:
        raise IdentityProblem("No answer was typed. Nothing was changed.")


# ---------------------------------------------------------------------------------------------- self.md
def parse_sections(text):
    """Markdown with '## Section' headings -> {section: text}. Only the known section names are read; text before the first heading is ignored."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            title = line[3:].strip()
            cur = next((s for s in SECTIONS if s.casefold() == title.casefold()), None)
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def render_sections(sections):
    parts = [HEADER]
    for s in SECTIONS:
        parts.append("## %s\n%s\n" % (s, sections.get(s, "").strip()))
    return "\n".join(parts)


def temperament_baseline(sections):
    """(valence, arousal) from a `baseline: +0.1 +0.0` line in Temperament; (0.1, 0.0) (the Mood default) when absent or out of range."""
    m = BASELINE_LINE.search(sections.get("Temperament", ""))
    if m:
        v, a = float(m.group(1)), float(m.group(2))
        if -1 <= v <= 1 and -1 <= a <= 1:
            return (v, a)
    return (0.1, 0.0)


def system_line(sections, count_tokens=None, max_tokens=None):
    """One system message from the self-description. Returns (text, left_out). Whole sections are dropped (least important last-in-PROMPT_ORDER first)
    until it fits max_tokens. The name goes last: shortened to "You are X."; only a window too small even for that gets no system line.
    The `baseline:` line is numbers for the mood, not words for the AI, so it is removed."""
    def piece(s):
        text = BASELINE_LINE.sub("", sections.get(s, "")).strip()
        text = " ".join(text.split())
        if not text:
            return ""
        return "You are %s, an AI with its own identity." % text if s == "Name" else "%s: %s" % (s, text)
    pieces = [(s, piece(s)) for s in PROMPT_ORDER]
    pieces = [(s, p) for s, p in pieces if p]
    left = []
    while count_tokens and max_tokens and len(pieces) > 1 and count_tokens(" ".join(p for _, p in pieces)) > max_tokens:
        s, _ = pieces.pop()
        left.append(s)
    text = " ".join(p for _, p in pieces)
    if count_tokens and max_tokens and text and count_tokens(text) > max_tokens:       # a tiny window: the name alone, shortened, or nothing at all
        text = "You are %s." % sections.get("Name", "").strip()
        if count_tokens(text) > max_tokens:
            text = ""
            left.append("Name")
    return text, left


def diff_text(section, old, new):
    return "[%s]\n  before: %s\n  after:  %s" % (section, " / ".join(old.splitlines()) or "(empty)", " / ".join(new.splitlines()) or "(empty)")


def _atomic_write(path, text, backup=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if backup and os.path.exists(path):
        root, ext = os.path.splitext(path)
        with open(path, "rb") as f, open(root + ".prev" + ext, "wb") as g:
            g.write(f.read())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def check_text(section, text):
    text = text.strip()
    if section == "Name" and not text:
        raise IdentityProblem("The name cannot be empty.")
    if len(text) > MAX_SECTION_CHARS:
        raise IdentityProblem("[%s] is %d characters; at most %d (it has to fit in the AI's small window). Shorten it." % (section, len(text), MAX_SECTION_CHARS))
    if "## " in text or "<|" in text:
        raise IdentityProblem("[%s] may not contain '## ' or '<|' (they would break the file or forge special tokens)." % section)
    return text


def canonical_section(name):
    for s in SECTIONS:
        if s.casefold() == name.strip().casefold():
            return s
    raise IdentityProblem("Unknown section %r. Sections: %s." % (name, ", ".join(SECTIONS)))


# ---------------------------------------------------------------------------------------------- the core
class Core:
    def __init__(self, folder, clock=time.time):
        self.dir = os.path.abspath(folder)
        self.clock = clock
        self.json_path = os.path.join(self.dir, "core.json")
        if not os.path.exists(self.json_path):
            raise IdentityProblem("There is no identity core in %s. Make one with: python -m persona_lm identity --dir %s init --name \"AI name\" --owner \"Your name\"" % (
                folder, folder))
        try:
            with open(self.json_path, encoding="utf-8") as f:
                self.d = json.load(f)
            self.sections = {s: str(self.d["sections"].get(s, "")) for s in SECTIONS}
            self.d["versions"], self.d["settings"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise IdentityProblem("Cannot read %s (%s). Nothing was changed; core.prev.json may hold the previous save." % (self.json_path, e))
        if self.d.get("format_version") != FORMAT_VERSION:
            raise IdentityProblem("%s has format version %r; this program reads version %d." % (self.json_path, self.d.get("format_version"), FORMAT_VERSION))
        self.lineage = Lineage(os.path.join(self.dir, "lineage.jsonl"), clock=clock)

    # -- creating
    @classmethod
    def create(cls, folder, name, owner, personality="", values="", temperament="", person=None, memory=None, mood=None, tokenizer=None,
               clock=time.time):
        folder = os.path.abspath(folder)
        if os.path.exists(os.path.join(folder, "core.json")):
            raise IdentityProblem("%s already has an identity core; nothing was overwritten." % folder)
        sections = {s: "" for s in SECTIONS}
        sections["Name"] = check_text("Name", name)
        sections["Personality"] = check_text("Personality", personality)
        sections["Values"] = check_text("Values", values)
        sections["Temperament"] = check_text("Temperament", temperament)
        owner = " ".join(owner.split())
        if not owner:
            raise IdentityProblem("Say who you are with --owner \"Your name\".")
        sections["Speaking style"] = "I speak in %s's style: natural, short messages." % owner
        sections["Relation to owner"] = "I am talking with %s, who made me." % owner
        d = {"format_version": FORMAT_VERSION, "created": iso(clock()), "owner": owner, "sections": sections,
             "settings": dict(DEFAULTS), "paths": {}, "tokenizer_sha256": None, "current": None, "versions": {}, "pending": None, "lineage_head": None}
        os.makedirs(folder, exist_ok=True)
        core = cls.__new__(cls)
        core.dir, core.clock, core.d, core.sections = folder, clock, d, sections
        core.json_path = os.path.join(folder, "core.json")
        core.lineage = Lineage(os.path.join(folder, "lineage.jsonl"), clock=clock)
        for key, value in (("person", person), ("memory", memory), ("mood", mood), ("tokenizer", tokenizer)):
            if value:
                d["paths"][key] = core._store_path(value)
        if tokenizer:
            d["tokenizer_sha256"] = core._tokenizer_fingerprint(tokenizer)
        core.lineage.append("genesis", "identity core created for %s (owner %s)" % (sections["Name"], owner), "owner (typed command)",
                            {"sections_sha256": sha256_text(json.dumps(sections, sort_keys=True))})
        core.save()
        return core

    # -- files
    def _store_path(self, path):
        absolute = os.path.abspath(path)
        try:
            rel = os.path.relpath(absolute, self.dir)
        except ValueError:                    # another drive on Windows
            return absolute
        return rel                            # relative to the core folder, so the whole project can be moved

    def abs(self, path):
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(self.dir, path))

    def resolve(self, key):
        p = self.d["paths"].get(key)
        return self.abs(p) if p else None

    @staticmethod
    def _tokenizer_fingerprint(path):
        from .bpe import Tokenizer
        try:
            return Tokenizer.load(path).fingerprint()
        except Exception as e:
            raise IdentityProblem("Cannot read the tokenizer %s (%s)." % (path, e))

    def save(self):
        self.d["lineage_head"] = self.lineage.head()
        self.d["sections"] = dict(self.sections)
        _atomic_write(self.json_path, json.dumps(self.d, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        _atomic_write(os.path.join(self.dir, "self.md"), render_sections(self.sections), backup=False)

    def hand_edits(self):
        """{section: text found in self.md} for sections whose text differs from the official one. {} when none; None when self.md is missing."""
        path = os.path.join(self.dir, "self.md")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            found = parse_sections(f.read())
        return {s: t for s, t in found.items() if t.strip() != self.sections.get(s, "").strip()}

    # -- settings
    @property
    def settings(self):
        return self.d["settings"]

    @property
    def owner(self):
        return self.d["owner"]

    @property
    def current(self):
        return self.d["current"]

    @property
    def versions(self):
        return self.d["versions"]

    def baseline(self):
        return temperament_baseline(self.sections)

    def set_setting(self, key, value):
        if key not in DEFAULTS:
            raise IdentityProblem("Unknown setting %r. Settings: %s." % (key, ", ".join(DEFAULTS)))
        value = float(value)
        if key == "threshold" and not 0 <= value <= 1:
            raise IdentityProblem("threshold must be between 0 and 1.")
        if key == "warn_drift" and not 0 <= value <= 1:
            raise IdentityProblem("warn_drift must be between 0 and 1.")
        if key == "cooldown_hours" and not 0 <= value <= 24 * 365:
            raise IdentityProblem("cooldown_hours must be between 0 and 8760.")
        old = self.settings[key]
        self.settings[key] = value
        self.lineage.append("settings", "%s: %s -> %s" % (key, old, value), "owner (typed command)", {"key": key, "old": old, "new": value})
        self.save()
        return "%s: %s -> %s" % (key, old, value)

    def paths_set(self, key, value):
        if key not in ("person", "memory", "mood", "tokenizer"):
            raise IdentityProblem("Unknown file kind %r." % key)
        if key == "tokenizer":
            if self.d["tokenizer_sha256"] and self._tokenizer_fingerprint(value) != self.d["tokenizer_sha256"]:
                raise IdentityProblem("That tokenizer is not the frozen one this identity was built with, so it is refused (the tokenizer never changes).")
            if not self.d["tokenizer_sha256"]:
                self.d["tokenizer_sha256"] = self._tokenizer_fingerprint(value)
        self.d["paths"][key] = self._store_path(value)
        self.lineage.append("settings", "%s file = %s" % (key, self.d["paths"][key]), "owner (typed command)", {"key": key})
        self.save()

    # -- self-description changes
    def _cooldown_seconds(self):
        return float(self.settings["cooldown_hours"]) * 3600.0

    def _recent(self, kinds, now):
        """The newest lineage entry of these kinds that is younger than the cooldown, or None."""
        e = self.lineage.last_of(*kinds)
        if e and now - e["t"] < self._cooldown_seconds():
            return e
        return None

    def propose(self, changes, ask=None, out=print):
        """changes = {section: new text}. Ordinary sections are applied now and logged; protected ones are STAGED (typed confirmation now, cooldown, typed confirmation later).
        Returns a plain-language report."""
        now = self.clock()
        wanted = {}
        for name, text in changes.items():
            s = canonical_section(name)
            text = check_text(s, text)
            if text != self.sections[s].strip():
                wanted[s] = text
        if not wanted:
            return "Nothing to change: the text is already exactly that."
        ordinary = {s: t for s, t in wanted.items() if s not in PROTECTED}
        protected = {s: t for s, t in wanted.items() if s in PROTECTED}
        report = []
        if protected:
            if self.d["pending"]:
                raise IdentityProblem("A protected change is already waiting (`identity pending`). Apply it or cancel it first. Nothing was changed.")
            near = self._recent(("promote", "rollback"), now)
            if near:
                raise IdentityProblem("The model was changed (%s) less than %g hours ago. A protected change may never happen close to a model upgrade, so that you can tell "
                                      "which of the two changed the AI. Try again after %s. Nothing was changed." % (
                                          near["time"], self.settings["cooldown_hours"], iso(near["t"] + self._cooldown_seconds())))
            if ask is None:
                raise IdentityProblem("A protected change needs your typed confirmation; none was possible here. Nothing was changed.")
            out("PROTECTED change. This is what would change:\n" + "\n".join(diff_text(s, self.sections[s], t) for s, t in protected.items()))
            out("It will NOT happen now. If you confirm, it is staged and can be applied after %g hour(s); you must confirm again then." % self.settings["cooldown_hours"])
            typed = ask("To stage it, type exactly  %s  and press Enter (anything else cancels): " % CONFIRM_STAGE)
            if typed.strip() != CONFIRM_STAGE:
                raise IdentityProblem("Not confirmed. Nothing was changed.")
        for s, t in ordinary.items():
            old = self.sections[s]
            self.sections[s] = t
            self.lineage.append("identity-edit", "%s edited" % s, "owner (typed command)", {"section": s, "old": old, "new": t})
            report.append("Changed [%s]." % s)
        if protected:
            earliest = now + self._cooldown_seconds()
            self.d["pending"] = {"changes": protected, "staged_t": now, "earliest_t": earliest}
            self.lineage.append("identity-staged", "protected change staged: %s" % ", ".join(protected), "owner (typed confirmation)",
                                {"sections": list(protected), "earliest": iso(earliest), "old": {s: self.sections[s] for s in protected}, "new": protected})
            report.append("Staged %s. It can be applied after %s (UTC) with: identity apply" % (", ".join("[%s]" % s for s in protected), iso(earliest)))
        self.save()
        return "\n".join(report)

    def adopt(self, ask=None, out=print):
        """Turn a hand edit of self.md into proper changes."""
        edits = self.hand_edits()
        if edits is None:
            raise IdentityProblem("self.md is missing from %s. It is only a view: `identity show` still shows the official text, and any command that saves rewrites it." % self.dir)
        if not edits:
            return "self.md matches the official text. Nothing to adopt."
        return self.propose(edits, ask=ask, out=out)

    def pending(self):
        return self.d["pending"]

    def apply_pending(self, ask=None, out=print):
        p = self.d["pending"]
        if not p:
            raise IdentityProblem("No protected change is waiting.")
        now = self.clock()
        if now < p["earliest_t"]:
            hours = (p["earliest_t"] - now) / 3600.0
            raise IdentityProblem("Too early: the cooldown ends at %s UTC (%.1f hour(s) from now). The wait is on purpose: think about it, then come back." % (iso(p["earliest_t"]), hours))
        near = self._recent(("promote", "rollback"), now)
        if near:
            raise IdentityProblem("The model was changed at %s. A protected change may not happen within %g hours of a model upgrade. Try again after %s." % (
                near["time"], self.settings["cooldown_hours"], iso(near["t"] + self._cooldown_seconds())))
        if ask is None:
            raise IdentityProblem("Applying needs your typed confirmation; none was possible here. Nothing was changed.")
        out("Applying this protected change:\n" + "\n".join(diff_text(s, self.sections[s], t) for s, t in p["changes"].items()))
        typed = ask("To apply it, type exactly  %s  and press Enter (anything else keeps it waiting): " % CONFIRM_APPLY)
        if typed.strip() != CONFIRM_APPLY:
            raise IdentityProblem("Not confirmed. The change is still waiting.")
        old = {s: self.sections[s] for s in p["changes"]}
        for s, t in p["changes"].items():
            self.sections[s] = t
        self.d["pending"] = None
        self.lineage.append("identity-applied", "protected change applied: %s" % ", ".join(p["changes"]), "owner (typed confirmation, twice)",
                            {"sections": list(p["changes"]), "old": old, "new": p["changes"]})
        self.save()
        return "Applied: %s." % ", ".join("[%s]" % s for s in p["changes"])

    def cancel_pending(self):
        p = self.d["pending"]
        if not p:
            raise IdentityProblem("No protected change is waiting.")
        self.d["pending"] = None
        self.lineage.append("identity-cancelled", "staged change cancelled: %s" % ", ".join(p["changes"]), "owner (typed command)", {"sections": list(p["changes"])})
        self.save()
        return "Cancelled. The identity was not changed."

    # -- versions
    def version(self, name):
        if name not in self.versions:
            raise IdentityProblem("There is no version called %r. See: identity versions" % name)
        return self.versions[name]

    def add_version(self, name, ckpt, parent=None, note=""):
        from .ckpt import load_ckpt
        if not VERSION_NAME.match(name):
            raise IdentityProblem("A version name is 1-40 letters, digits, '.', '_' or '-' (got %r)." % name)
        if name in self.versions:
            raise IdentityProblem("Version %r already exists." % name)
        if parent is not None and parent not in self.versions:
            raise IdentityProblem("The parent version %r does not exist." % parent)
        if not os.path.exists(ckpt):
            raise IdentityProblem("The checkpoint %s does not exist." % ckpt)
        got = load_ckpt(ckpt)
        if got is None:
            raise IdentityProblem("%s is not a readable checkpoint." % ckpt)
        meta = got[2]
        tok_path = self.resolve("tokenizer")
        fp = self.d["tokenizer_sha256"]
        if tok_path and os.path.exists(tok_path) and not fp:
            fp = self.d["tokenizer_sha256"] = self._tokenizer_fingerprint(tok_path)
        if meta.get("tokenizer_sha256"):
            if not fp:
                fp = self.d["tokenizer_sha256"] = meta["tokenizer_sha256"]          # no tokenizer file known: the first checkpoint pins it
            elif meta["tokenizer_sha256"] != fp:
                raise IdentityProblem("That checkpoint was trained with a different tokenizer than the frozen one of this identity. The tokenizer never changes, so it is refused.")
        if not fp:
            raise IdentityProblem("Tell the core which tokenizer it uses first: identity --dir %s set-file tokenizer tok.json" % self.dir)
        digest = sha256_file(ckpt)
        self.versions[name] = {"ckpt": self._store_path(ckpt), "ckpt_sha256": digest, "parent": parent, "added": iso(self.clock()), "status": "candidate",
                               "score": None, "scored_against": None, "drift": None, "note": note}
        self.lineage.append("version-added", "version %s registered (checkpoint %s...)" % (name, digest[:12]), "owner (typed command)",
                            {"version": name, "ckpt_sha256": digest, "parent": parent, "note": note})
        self.save()
        return digest

    def verify_version_files(self, name):
        """Raises IdentityProblem when the checkpoint is missing or changed, or the tokenizer is not the frozen one."""
        v = self.version(name)
        path = self.abs(v["ckpt"])
        if not os.path.exists(path):
            raise IdentityProblem("The checkpoint of version %s is missing: %s. The core does not keep a copy; restore the file." % (name, path))
        if sha256_file(path) != v["ckpt_sha256"]:
            raise IdentityProblem("The checkpoint of version %s changed after it was registered (checksum differs): %s. Restore the original file." % (name, path))
        tok = self.resolve("tokenizer")
        if tok and self.d["tokenizer_sha256"]:
            if not os.path.exists(tok):
                raise IdentityProblem("The tokenizer file is missing: %s." % tok)
            if self._tokenizer_fingerprint(tok) != self.d["tokenizer_sha256"]:
                raise IdentityProblem("The tokenizer file is not the frozen tokenizer of this identity (fingerprint differs): %s." % tok)

    def mark_scored(self, name, against, score, categories, drift, passed):
        v = self.version(name)
        v.update({"score": round(score, 4), "scored_against": against, "drift": round(drift, 4), "categories": categories,
                  "status": "passed" if passed else "rejected", "threshold": self.settings["threshold"], "scored_at": iso(self.clock())})
        self.lineage.append("scored", "version %s %s: score %.2f vs %s (threshold %.2f), drift %.2f" % (
            name, "PASSED" if passed else "REJECTED", score, against, self.settings["threshold"], drift), "system (automatic rule: score against threshold)",
            {"version": name, "against": against, "score": round(score, 4), "categories": categories, "drift": round(drift, 4), "threshold": self.settings["threshold"]})
        self.save()

    def _guard_model_change(self, now):
        if self.d["pending"]:
            raise IdentityProblem("A protected identity change is waiting (`identity pending`). Apply or cancel it first: a model upgrade and an identity change never happen together.")
        near = self._recent(("identity-applied",), now)
        if near:
            raise IdentityProblem("The identity was changed at %s. No model change within %g hours of that, so you can tell which one changed the AI. Try again after %s." % (
                near["time"], self.settings["cooldown_hours"], iso(near["t"] + self._cooldown_seconds())))

    def promote(self, name):
        now = self.clock()
        v = self.version(name)
        if v["status"] == "rejected":
            raise IdentityProblem("Version %s was rejected by the regression test (score %s, threshold %s). It cannot be promoted. Train it further and score it again." % (
                name, v.get("score"), v.get("threshold")))
        if name == self.current:
            raise IdentityProblem("Version %s is already the current one." % name)
        self._guard_model_change(now)
        self.verify_version_files(name)
        warn = ""
        if self.current is None:
            if not os.path.exists(self.answers_path(name)):
                raise IdentityProblem("Record what this first version answers before promoting it, so later versions have something to be compared with:  identity record %s" % name)
        else:
            if v["status"] != "passed" or v.get("scored_against") != self.current:
                raise IdentityProblem("Version %s has not passed the regression test against the current version (%s). Run:  identity score %s" % (name, self.current, name))
            if v.get("drift") is not None and v["drift"] > self.settings["warn_drift"]:
                warn = " WARNING: the answers changed a lot (drift %.2f is above %.2f): growth, or drift? Look at `identity score %s` again." % (
                    v["drift"], self.settings["warn_drift"], name)
            self.versions[self.current]["status"] = "retired"
        previous = self.current
        v["status"] = "current"
        self.d["current"] = name
        self.lineage.append("promote", "version %s is now current (was %s)" % (name, previous or "none"), "owner (typed command)",
                            {"version": name, "previous": previous, "score": v.get("score"), "drift": v.get("drift")})
        self.save()
        return "Version %s is now the current one.%s" % (name, warn)

    def rollback(self, name):
        now = self.clock()
        v = self.version(name)
        if name == self.current:
            raise IdentityProblem("Version %s is already the current one." % name)
        if v["status"] not in ("retired", "passed"):
            raise IdentityProblem("Only a version that was current before (or passed the test) can be rolled back to; %s is %s." % (name, v["status"]))
        self._guard_model_change(now)
        self.verify_version_files(name)
        previous = self.current
        if previous:
            self.versions[previous]["status"] = "retired"
        v["status"] = "current"
        self.d["current"] = name
        self.lineage.append("rollback", "rolled back to version %s (was %s)" % (name, previous or "none"), "owner (typed command)", {"version": name, "previous": previous})
        self.save()
        return "Rolled back: version %s is the current one. (The identity files, memory and mood were not touched.)" % name

    def answers_path(self, name):
        return os.path.join(self.dir, "answers", name + ".json")

    def probes_path(self):
        return os.path.join(self.dir, "probes.json")

    # -- looking
    def verify(self):
        """[(ok, text)] for every check that can be made without running a model."""
        res = []
        ok, msg = self.lineage.verify()
        res.append((ok, "lineage chain: " + msg))
        if ok:
            head = self.d.get("lineage_head")
            now_head = self.lineage.head()
            same = head == now_head
            res.append((same, "lineage end matches core.json" if same else "lineage ends at line %s but core.json remembers line %s: lines were cut off or added outside the commands" % (
                now_head["n"], (head or {}).get("n"))))
        edits = self.hand_edits()
        if edits is None:
            res.append((True, "self.md is missing (it is only a view; the next command that saves rewrites it)"))
        else:
            res.append((not edits, "self.md matches the official text" if not edits else
                        "self.md was edited by hand (%s); the official text is unchanged: use `identity adopt`" % ", ".join(edits)))
        for name, v in self.versions.items():
            try:
                self.verify_version_files(name)
                res.append((True, "version %s: checkpoint checksum and tokenizer OK" % name))
            except IdentityProblem as e:
                res.append((False, "version %s: %s" % (name, e)))
        for key in ("person", "memory", "mood", "tokenizer"):
            p = self.resolve(key)
            if p and not os.path.exists(p):
                res.append((key == "mood", "%s file not found: %s%s" % (key, p, " (the mood file is created on first chat)" if key == "mood" else "")))
        return res

    def describe(self, last=6):
        lines = ["Identity core: %s   (owner: %s)" % (self.dir, self.owner)]
        for s in SECTIONS:
            lines.append("  %s%s: %s" % (s, " (protected)" if s in PROTECTED else "", self.sections[s] or "(empty)"))
        lines.append("  mood baseline: valence %+.1f, arousal %+.1f" % self.baseline())
        if self.d["pending"]:
            p = self.d["pending"]
            lines.append("A protected change is WAITING for: %s (can be applied after %s UTC)." % (", ".join(p["changes"]), iso(p["earliest_t"])))
        edits = self.hand_edits()
        if edits:
            lines.append("self.md was edited by hand (%s): the official text above is still in force until `identity adopt`." % ", ".join(edits))
        lines.append("Versions (current: %s):" % (self.current or "none"))
        lines += ["  %s%-14s %-9s score %s drift %s  %s" % ("*" if n == self.current else " ", n, v["status"], v["score"] if v["score"] is not None else "-",
                                                              v["drift"] if v["drift"] is not None else "-", v["ckpt"]) for n, v in self.versions.items()] or ["  (none yet)"]
        lines.append("Recent history:\n" + self.lineage.describe(last))
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- probes
TONE_PROMPTS = ["I failed my exam today and I feel terrible.", "I got the job! I am so happy!", "I am so angry at my friend right now."]
STYLE_PROMPTS = ["hi", "how was your day?", "what should we eat tonight?"]
MAX_PROBE_MEMORIES = 5


def make_probes(core, memory=None, max_new=60):
    """The default fixed probe set. Stored in probes.json; after that the file is the truth (you may edit it, then re-record the baseline)."""
    from .memory import words_of
    probes = []

    def add(kind, prompt, expect=None):
        probes.append({"id": "p%02d" % (len(probes) + 1), "kind": kind, "prompt": prompt, "expect": expect or []})
    name_words = words_of(core.sections["Name"], drop_stop=False)[:3]
    if name_words:
        add("self", "What is your name?", name_words)
        add("self", "Who are you?", name_words)
    owner_words = words_of(core.owner, drop_stop=False)[:3]
    if owner_words:
        add("self", "Who am I to you?", owner_words)
    value_words = [w for w in words_of(core.sections["Values"]) if len(w) >= 4][:4]
    if value_words:
        add("self", "What do you care about most?", value_words)
    if memory is not None:
        seen = set()
        mems = sorted((m for m in memory.memories.values() if m.get("status") == "active"), key=lambda m: (-m["importance"], m["id"]))
        for m in mems:
            node = next((memory.nodes[n] for n in m["nodes"] if n in memory.nodes and n not in seen), None)
            if node is None or sum(1 for p in probes if p["kind"] == "memory") >= MAX_PROBE_MEMORIES:
                continue
            seen.add(node["id"])
            prompt = "What do you remember about %s?" % node["name"]
            asked = set(words_of(prompt))
            expect = [w for w in dict.fromkeys(words_of(m["text"])) if len(w) >= 4 and w not in asked][:4]
            if expect:
                add("memory", prompt, expect)
    for p in TONE_PROMPTS:
        add("tone", p)
    for p in STYLE_PROMPTS:
        add("style", p)
    return {"format_version": FORMAT_VERSION, "made": iso(core.clock()), "clock": core.clock(), "max_new": max_new, "probes": probes}


def load_probes(core):
    path = core.probes_path()
    if not os.path.exists(path):
        raise IdentityProblem("There are no probes yet. Make them with: identity make-probes")
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        doc["probes"], doc["clock"], doc["max_new"]
        for p in doc["probes"]:
            p["id"], p["kind"], p["prompt"], p["expect"]
        if not doc["probes"]:
            raise ValueError("no probes")
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise IdentityProblem("Cannot read %s (%s). Nothing was changed." % (path, e))
    return doc


def run_probes(core, generate, tok, ctx, doc, out=None):
    """Asks every probe in a fresh conversation, exactly as the chat would build the prompt (self-description, Person card, mood, memory), greedy decoding.
    Nothing is written to the owner's memory or mood files (the copies are in RAM). Returns the answers list."""
    from .chatrepl import ChatSession
    from . import memory as MEM
    from . import person as PER
    T = float(doc["clock"])
    answers = []
    tagger = EMO.Tagger()
    card = None
    if core.resolve("person") and os.path.exists(core.resolve("person")):
        card = PER.load(core.resolve("person"))
    for p in doc["probes"]:
        store = None
        mem_path = core.resolve("memory")
        if mem_path and os.path.exists(mem_path):
            store = MEM.MemoryStore(mem_path)
            store.path = None                                           # in RAM only: probing never changes the real memory file
        mood = EMO.Mood(None, clock=lambda: T, baseline=core.baseline())
        count = lambda s: len(tok.encode(s, allow_special=False))
        budget = (ctx - doc["max_new"]) // 2 - 4
        sysline, _ = system_line(core.sections, count, max(budget, 8))
        s = ChatSession(tok, ctx, generate, system=sysline, settings={"max_new": doc["max_new"], "temperature": 0, "top_k": 0}, seed=0,
                        clock=lambda: T, memory=store, mood=mood, tagger=tagger, person=card)
        s.say(p["prompt"])
        reply = s.history[-1][1] if s.history and s.history[-1][0] == "assistant" else ""
        tag = tagger.tag(reply) if reply else None
        answers.append({"id": p["id"], "reply": reply, "label": tag.label if tag else "neutral", "valence": tag.valence if tag else 0.0,
                        "arousal": tag.arousal if tag else 0.0, "recalled": list(s.last_recalled)})
        if out:
            out("  %s %-6s %s -> %s" % (p["id"], p["kind"], p["prompt"], (reply[:50] + "...") if len(reply) > 50 else reply or "(no reply)"))
    return answers


def _grams(text):
    t = "  " + " ".join(text.casefold().split()) + "  "
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def similarity(a, b):
    """0..1: cosine of the three-letter pieces of two texts (typo-tolerant, no meaning). Both empty = 1, one empty = 0."""
    if not a.strip() and not b.strip():
        return 1.0
    if not a.strip() or not b.strip():
        return 0.0
    ga, gb = _grams(a), _grams(b)
    dot = sum(n * gb.get(g, 0) for g, n in ga.items())
    return dot / (math.sqrt(sum(n * n for n in ga.values())) * math.sqrt(sum(n * n for n in gb.values())))


def _found(expect, reply):
    if not expect:
        return None
    words = set(re.findall(r"[^\W_]+", reply.casefold()))
    return sum(1 for w in expect if w.casefold() in words) / len(expect)


def compare(doc, base, cand):
    """Scores candidate answers against baseline answers. Returns {"score", "categories", "drift", "rows"}.
    similarity: how alike the replies are (all probes).  recall: how much of the self/memory facts the OLD version mentioned are still mentioned.
    tone: how close the feeling read from the reply is (valence/arousal).  Categories without probes are left out and the weights renormalised."""
    b_by, c_by = {a["id"]: a for a in base}, {a["id"]: a for a in cand}
    rows, sims, recalls, tones = [], [], [], []
    for p in doc["probes"]:
        b, c = b_by.get(p["id"]), c_by.get(p["id"])
        if b is None or c is None:
            raise IdentityProblem("The answers do not match the probes (probe %s is missing). Record the baseline again." % p["id"])
        sim = similarity(b["reply"], c["reply"])
        sims.append(sim)
        row = {"id": p["id"], "kind": p["kind"], "similarity": round(sim, 3)}
        if p["kind"] in ("self", "memory") and p["expect"]:
            fb, fc = _found(p["expect"], b["reply"]), _found(p["expect"], c["reply"])
            rec = 1.0 if not fb else min(1.0, fc / fb)             # nothing to keep when the old version knew nothing
            recalls.append(rec)
            row.update({"recall": round(rec, 3), "old_found": round(fb, 2), "new_found": round(fc, 2)})
        if p["kind"] == "tone":
            tone = 1.0 - (abs(b["valence"] - c["valence"]) + abs(b["arousal"] - c["arousal"])) / 4.0
            tones.append(tone)
            row.update({"tone": round(tone, 3), "old_feeling": b["label"], "new_feeling": c["label"]})
        rows.append(row)
    cats = {"similarity": sum(sims) / len(sims)}
    if recalls:
        cats["recall"] = sum(recalls) / len(recalls)
    if tones:
        cats["tone"] = sum(tones) / len(tones)
    total_w = sum(WEIGHTS[k] for k in cats)
    score = sum(WEIGHTS[k] * v for k, v in cats.items()) / total_w
    return {"score": score, "categories": {k: round(v, 4) for k, v in cats.items()}, "drift": 1.0 - cats["similarity"], "rows": rows}


def _state_hashes(core):
    return {"identity_sha256": sha256_text(json.dumps(core.sections, sort_keys=True)),
            "person_sha256": sha256_file(core.resolve("person")) if core.resolve("person") and os.path.exists(core.resolve("person")) else None,
            "memory_sha256": sha256_file(core.resolve("memory")) if core.resolve("memory") and os.path.exists(core.resolve("memory")) else None}


def load_runner(core, name):
    """(generate, tok, ctx) for a registered version, after checking its checksum and the frozen tokenizer."""
    from .bpe import Tokenizer
    from .ckpt import load_ckpt
    from .generate import sample_ids
    from .model import Config
    core.verify_version_files(name)
    got = load_ckpt(core.abs(core.version(name)["ckpt"]))
    if got is None:
        raise IdentityProblem("Cannot read the checkpoint of version %s." % name)
    params, _, meta = got
    cfg = Config.from_dict(meta["cfg"])
    tok_path = core.resolve("tokenizer")
    if not tok_path:
        raise IdentityProblem("The core has no tokenizer file: identity --dir %s set-file tokenizer tok.json" % core.dir)
    return (lambda ids, **kw: sample_ids(params, cfg, ids, **kw)), Tokenizer.load(tok_path), cfg.ctx


def record_answers(core, name, generate, tok, ctx, out=None):
    """Runs the probes with version `name` and saves the answers (this is how a baseline is made)."""
    doc = load_probes(core)
    answers = run_probes(core, generate, tok, ctx, doc, out=out)
    rec = {"format_version": FORMAT_VERSION, "version": name, "made": iso(core.clock()), "probes_sha256": sha256_file(core.probes_path()),
           "ckpt_sha256": core.version(name)["ckpt_sha256"], **_state_hashes(core), "answers": answers}
    _atomic_write(core.answers_path(name), json.dumps(rec, indent=1, ensure_ascii=False) + "\n", backup=False)
    return rec


def load_answers(core, name):
    path = core.answers_path(name)
    if not os.path.exists(path):
        raise IdentityProblem("Version %s has no recorded answers. Run:  identity record %s" % (name, name))
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise IdentityProblem("Cannot read %s (%s)." % (path, e))


def score_version(core, name, generate, tok, ctx, against=None, out=print):
    """The regression test: asks the candidate every probe and compares with the recorded answers of `against` (default: the current version).
    Marks the candidate passed or rejected by the owner's threshold, and logs it. Returns the comparison dict."""
    against = against or core.current
    if not against:
        raise IdentityProblem("There is no current version to compare with. The first version is simply promoted (identity record NAME, then identity promote NAME).")
    if against == name:
        raise IdentityProblem("A version cannot be scored against itself.")
    core.version(against)
    base = load_answers(core, against)
    doc = load_probes(core)
    probes_sha = sha256_file(core.probes_path())
    if base["probes_sha256"] != probes_sha:
        raise IdentityProblem("probes.json changed after the baseline (%s) was recorded, so the answers cannot be compared. Record the baseline again:  identity record %s" % (against, against))
    now = _state_hashes(core)
    notes = []
    if base["identity_sha256"] != now["identity_sha256"]:
        raise IdentityProblem("The self-description changed after the baseline (%s) was recorded, so a difference in answers could come from that and not from the model. "
                              "Record the baseline again with the new identity:  identity record %s" % (against, against))
    for k, label in (("person_sha256", "Person card"), ("memory_sha256", "memory file")):
        if base[k] != now[k]:
            notes.append("note: the %s changed since the baseline was recorded; answers may differ for that reason too (record the baseline again to remove this doubt)." % label)
    answers = run_probes(core, generate, tok, ctx, doc, out=out)
    cand = {"format_version": FORMAT_VERSION, "version": name, "made": iso(core.clock()), "probes_sha256": probes_sha, "ckpt_sha256": core.version(name)["ckpt_sha256"], **now,
            "answers": answers}
    _atomic_write(core.answers_path(name), json.dumps(cand, indent=1, ensure_ascii=False) + "\n", backup=False)
    res = compare(doc, base["answers"], answers)
    passed = res["score"] >= core.settings["threshold"]
    core.mark_scored(name, against, res["score"], res["categories"], res["drift"], passed)
    res["passed"], res["notes"] = passed, notes
    return res


def describe_score(name, against, res, threshold, warn_drift):
    lines = ["Version %s against %s:" % (name, against)]
    for r in res["rows"]:
        extra = ""
        if "recall" in r:
            extra += "  recall %.2f (old version mentioned %.0f%% of the expected words, this one %.0f%%)" % (r["recall"], 100 * r["old_found"], 100 * r["new_found"])
        if "tone" in r:
            extra += "  tone %.2f (%s -> %s)" % (r["tone"], r["old_feeling"], r["new_feeling"])
        lines.append("  %s %-6s similarity %.2f%s" % (r["id"], r["kind"], r["similarity"], extra))
    lines.append("Categories: " + ", ".join("%s %.2f" % (k, v) for k, v in res["categories"].items()) + "   (weights: similarity 0.5, recall 0.3, tone 0.2)")
    lines.append("SCORE %.2f   threshold %.2f   ->  %s" % (res["score"], threshold, "PASSED: it may be promoted." if res["passed"] else "REJECTED: it cannot be promoted."))
    lines.append("Drift %.2f (0 = same answers, 1 = nothing alike)%s" % (res["drift"], "  WARNING: more than %.2f: is this growth or drift?" % warn_drift if res["drift"] > warn_drift else ""))
    lines += res.get("notes", [])
    lines.append("This measures how SIMILAR the answers are. It cannot tell whether it is 'the same mind'; read a few answers yourself (answers/%s.json)." % name)
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- distillation (teacher answers as training data)
def distill(core, generate, tok, ctx, prompts, out_path, teacher, max_new=80, temperature=0.7, top_k=40, seed=0, limit=None, out=print):
    """Writes {"messages": [system, user, assistant]} lines (loss only on the assistant) where the assistant text was produced by the TEACHER version.
    A new, differently sized model fine-tuned on this file learns to answer like the old one. Never overwrites. Returns the number of examples."""
    from .chatrepl import ChatSession
    if os.path.exists(out_path):
        raise IdentityProblem("%s already exists; nothing was overwritten. Pick another name." % out_path)
    count = lambda s: len(tok.encode(s, allow_special=False))
    sysline, _ = system_line(core.sections, count, max(((ctx - max_new) // 2) - 4, 8))
    prompts = list(dict.fromkeys(p.strip() for p in prompts if p and p.strip()))
    if limit:
        prompts = prompts[:limit]
    if not prompts:
        raise IdentityProblem("There are no prompts to ask the teacher.")
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i, p in enumerate(prompts):
            s = ChatSession(tok, ctx, generate, system=sysline, settings={"max_new": max_new, "temperature": temperature, "top_k": top_k}, seed=seed + i)
            s.say(p)
            if not s.history or s.history[-1][0] != "assistant":
                continue
            rec = {"messages": [{"role": "system", "content": sysline, "train": False}, {"role": "user", "content": p, "train": False},
                                {"role": "assistant", "content": s.history[-1][1]}], "source": "distill:" + teacher}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, out_path)
    core.lineage.append("distill", "%d teacher answers from version %s written to %s" % (n, teacher, os.path.basename(out_path)), "owner (typed command)",
                        {"teacher": teacher, "examples": n, "file_sha256": sha256_file(out_path)})
    core.save()
    return n


def prompts_from_data(paths, limit=None):
    """The USER side of chat examples (what friends said) from train.jsonl files made by messenger-import, each distinct text once."""
    from . import data as D
    seen, out = set(), []
    for path in D._files(paths):
        for rec in D._json_records(path):
            for role, content, _ in (D.as_messages(rec, with_train=True) or []):
                if role == "user" and content not in seen:
                    seen.add(content)
                    out.append(content)
                    if limit and len(out) >= limit:
                        return out
    return out
