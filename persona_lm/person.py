"""The Person card (Stage 5): a plain Markdown file about the owner, built partly from data and always editable by hand.

    # Person card
    ## Name
    - [owner] Jaymar
    ## Relationships
    - [auto] Maria: Maria is my sister (3 memories, mostly tagged love)
    - [no] Jun: ...                        <- you wrote [no]: the builder will never suggest a line with the key "Jun" again

Rules (they are tested):
  * Lines you write yourself (no tag, or [owner]) are NEVER touched by the builder. Lines tagged [auto] are the builder's: a rebuild replaces them.
    To keep an [auto] line for good, delete the word [auto] (or change it to [owner]). To get rid of one for good, change [auto] to [no].
  * `person build` only shows a PLAN (what would be added and removed). Nothing is written without --apply, and the old file is kept as <file>.prev.md.
  * What is built from data: Speaking style and Frequent words (counted from your own messages), Relationships (from the memory graph's people).
    NOT built, written by you only: Name, Habits (the chat files carry no times), Values (nothing in text counts as proof of what you value).
  * The card is about the OWNER. Counting is not understanding: "Frequent words" lists words, not what you care about.
Pure Python.
"""
import os
import re
import shutil
from collections import Counter

SECTIONS = ["Name", "Relationships", "Habits", "Speaking style", "Frequent words", "Values"]
AUTO_SECTIONS = ["Relationships", "Speaking style", "Frequent words"]
PROMPT_ORDER = ["Name", "Relationships", "Values", "Habits", "Frequent words", "Speaking style"]
TAGS = ("owner", "auto", "no")

HEADER = ("# Person card\n"
          "<!-- A plain text file about you. Edit it with Notepad. Lines starting with '- ' are facts.\n"
          "     [auto] = built from your data by `person build` (a rebuild replaces it).  Delete the word [auto] to keep a line for good.\n"
          "     [no]   = never suggest a line with this beginning again.  Lines with no tag are yours and are never changed by the program. -->\n")

_FACT = re.compile(r"^-\s*(?:\[(owner|auto|no)\]\s*)?(.*\S)\s*$", re.I)


class PersonProblem(Exception):
    pass


class Card:
    """sections: ordered list of [title, items]; item = {"kind": "fact", "tag", "text"} or {"kind": "raw", "text"} (comments and blank lines are kept)."""
    def __init__(self, title="# Person card"):
        self.title = title
        self.preamble = []                       # raw lines before the first section
        self.sections = []

    # -- parse / render
    @classmethod
    def parse(cls, text):
        card, cur = cls(), None
        for line in text.splitlines():
            if line.startswith("# ") and cur is None and not card.sections:
                card.title = line.strip()
                continue
            if line.startswith("## "):
                cur = [line[3:].strip(), []]
                card.sections.append(cur)
                continue
            target = card.preamble if cur is None else cur[1]
            m = _FACT.match(line)
            if m and line.lstrip().startswith("-") and cur is not None:
                target.append({"kind": "fact", "tag": (m.group(1) or "owner").lower(), "text": m.group(2)})
            else:
                target.append({"kind": "raw", "text": line})
        return card

    def render(self):
        out = [self.title] + [r for r in self.preamble_lines()]
        for title, items in self.sections:
            out.append("## " + title)
            for it in items:
                out.append(it["text"] if it["kind"] == "raw" else "- %s%s" % ("" if it["tag"] == "owner" else "[%s] " % it["tag"], it["text"]))
        return "\n".join(out).rstrip("\n") + "\n"

    def preamble_lines(self):
        return [it["text"] for it in self.preamble]

    # -- access
    def section(self, title, create=True):
        for s in self.sections:
            if s[0].casefold() == title.casefold():
                return s
        if not create:
            return None
        s = [title, []]
        self.sections.append(s)
        return s

    def facts(self, title, tags=None):
        s = self.section(title, create=False)
        return [it for it in (s[1] if s else []) if it["kind"] == "fact" and (tags is None or it["tag"] in tags)]

    def add_owner(self, title, text):
        text = " ".join(text.split())
        if not text:
            raise PersonProblem("There is nothing to add: the text is empty.")
        if len(text) > 400:
            raise PersonProblem("A card line should be short (%d characters; at most 400). Split it into several lines." % len(text))
        _put(self.section(self._canonical(title))[1], {"kind": "fact", "tag": "owner", "text": text})

    @staticmethod
    def _canonical(title):
        for t in SECTIONS:
            if t.casefold() == title.strip().casefold():
                return t
        raise PersonProblem("Unknown section %r. Sections: %s." % (title, ", ".join(SECTIONS)))


def _put(items, fact):
    """New facts go right after the last fact of the section (before any blank line or comment that follows), so the file stays tidy."""
    pos = max([i + 1 for i, it in enumerate(items) if it["kind"] == "fact"] or [0])
    items.insert(pos, fact)


def new_card(name=None):
    card = Card()
    card.preamble = [{"kind": "raw", "text": l} for l in HEADER.splitlines()[1:]] + [{"kind": "raw", "text": ""}]
    for t in SECTIONS:
        card.sections.append([t, [{"kind": "raw", "text": ""}]])
    if name:
        card.add_owner("Name", name)
    return card


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return Card.parse(f.read())
    except OSError as e:
        raise PersonProblem("Cannot read the Person card %s (%s). Make one with: python -m persona_lm person init --name \"Your name\"" % (path, e))


def save(card, path, keep_backup=True):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    if keep_backup and os.path.exists(path):
        shutil.copy2(path, path[:-3] + ".prev.md" if path.endswith(".md") else path + ".prev")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(card.render())
    os.replace(tmp, path)


def key_of(text):
    """'Maria: Maria is my sister' -> 'maria' (the part before the first colon, or the whole line)."""
    return text.split(":", 1)[0].strip().casefold()


# ---------------------------------------------------------------------------------------------- what is counted from the owner's messages
_EN = frozenset("the and is are was were to of you it that for with my me this have just not but what so im dont i a an in on at be do if or as we he she they your".split())
_TL = frozenset("ng mga hindi yung kasi talaga siya ikaw tayo po opo namin natin nila pero sana daw raw ito iyon iyan nasa huwag wag naman".split())
_BS = frozenset("kaayo gyud unsa asa ngano kinsa pila mao pud bitaw diay gani naa dili dayon karon gamay dako lagi baya nimo nako namo atong ayaw unya ra".split())
_FUNCTION = _EN | _TL | _BS | frozenset("ang sa na ko ka mo lang din rin pa ba ay may wala ni si kay ug og nga man ra eh oh uy oy".split())
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_LAUGH = re.compile(r"\b(?:ha|he|hi|ho){2,}\b", re.I)
_STRETCH = re.compile(r"(\w)\1{2,}")
_PLACEHOLDER = re.compile(r"\[[A-Z]+\]|Person_\d+|\[[A-Za-z_ ]+\]")


def _pct(n, total):
    return "%.0f%%" % (100.0 * n / total) if total else "0%"


def _emoji_chars(text):
    from .emotion import _is_symbol_char
    return [c for c in text if _is_symbol_char(c) and c not in "\uFE0F\u200d"]


def style_facts(texts):
    """[(key, text)] counted from the owner's own messages. Every number is a plain count; nothing is guessed except the language mix."""
    n = len(texts)
    if n < 20:
        return []
    facts = []
    lengths = sorted(len(t.split()) for t in texts)
    facts.append(("Length", "%d messages analysed; typically %d word(s) (median), %d on average; 90%% are %d words or fewer"
                  % (n, lengths[n // 2], round(sum(lengths) / n), lengths[min(n - 1, int(0.9 * n))])))
    alpha = [t for t in texts if any(c.isalpha() for c in t)]
    if alpha:
        lower_start = sum(1 for t in alpha if next(c for c in t if c.isalpha()).islower())
        facts.append(("Capital letters", "starts with a lower-case letter in %s of messages" % _pct(lower_start, len(alpha))))
    ends_dot = sum(1 for t in texts if t.rstrip().endswith("."))
    facts.append(("Punctuation", "ends with a full stop in %s, has a question mark in %s, an exclamation mark in %s, '...' in %s"
                  % (_pct(ends_dot, n), _pct(sum(1 for t in texts if "?" in t), n), _pct(sum(1 for t in texts if "!" in t), n),
                     _pct(sum(1 for t in texts if "..." in t or "\u2026" in t), n))))
    laugh = Counter()
    for t in texts:
        for m in _LAUGH.findall(t):
            laugh[m.lower()] += 1
    laugh_msgs = sum(1 for t in texts if _LAUGH.search(t))
    if laugh_msgs:
        facts.append(("Laughter", "'haha'-style laughter in %s of messages (most common: %s)" % (_pct(laugh_msgs, n), ", ".join(w for w, _ in laugh.most_common(3)))))
    emo = Counter()
    for t in texts:
        emo.update(_emoji_chars(t))
    emo_msgs = sum(1 for t in texts if _emoji_chars(t))
    if emo_msgs:
        facts.append(("Emoji", "in %s of messages; favourites: %s" % (_pct(emo_msgs, n), ", ".join("%s (%d)" % (e, c) for e, c in emo.most_common(5)))))
    stretch = sum(1 for t in texts if _STRETCH.search(t))
    if stretch:
        facts.append(("Stretched words", "letters stretched, like 'sooo', in %s of messages" % _pct(stretch, n)))
    en = tl = bs = 0
    for t in texts:
        for w in _WORD.findall(_PLACEHOLDER.sub(" ", t).lower()):
            en += w in _EN
            tl += w in _TL
            bs += w in _BS
    hits = en + tl + bs
    if hits >= 30:
        facts.append(("Language mix", "rough guess from short word lists, not a language detector: English %s, Tagalog %s, Bisaya %s of the %d recognised words"
                      % (_pct(en, hits), _pct(tl, hits), _pct(bs, hits), hits)))
    return facts


def frequent_words(texts, top=10, min_count=5):
    c = Counter()
    for t in texts:
        for w in set(_WORD.findall(_PLACEHOLDER.sub(" ", t).lower())):
            if len(w) >= 4 and w not in _FUNCTION and not _LAUGH.fullmatch(w) and not _STRETCH.search(w):     # laughter and stretched spellings are style, not topics
                c[w] += 1
    common = [(w, k) for w, k in c.most_common(top * 3) if k >= min_count][:top]
    if not common:
        return []
    return [("Top words", ", ".join("%s (%d)" % wk for wk in common) + " -- counted in how many messages each appears; frequency is not meaning")]


def relationship_facts(store, top=8):
    """People from the memory graph: the first sentence of each person's strongest memory, how many memories, the most common feeling tag."""
    from .memory import first_sentence
    scores = store.node_scores()
    people = sorted((n for n in store.nodes.values() if n["kind"] == "person"), key=lambda n: (-scores[n["id"]], n["id"]))[:top]
    out = []
    for n in people:
        mems = [m for m in store.active() if n["id"] in m["nodes"]]
        if not mems:
            continue
        best = max(mems, key=lambda m: (store.strength(m), m["id"]))
        moods = Counter(m["emotion"] for m in mems if m["emotion"] != "neutral")
        note = "%d memor%s%s" % (len(mems), "y" if len(mems) == 1 else "ies", (", mostly tagged " + moods.most_common(1)[0][0]) if moods else "")
        out.append((n["name"], "%s (%s)" % (first_sentence(best["text"]), note)))
    return out


def plan_build(card, owner_texts=None, store=None):
    """-> {"add": [(section, text)], "remove": [(section, text)]}: what a rebuild would change. Nothing is written here."""
    wanted = {"Speaking style": style_facts(owner_texts) if owner_texts is not None else None,
              "Frequent words": frequent_words(owner_texts) if owner_texts is not None else None,
              "Relationships": relationship_facts(store) if store is not None else None}
    plan = {"add": [], "remove": []}
    for title in AUTO_SECTIONS:
        new = wanted[title]
        if new is None:
            continue                                                   # no data for this section was given: leave it alone
        refused = {key_of(f["text"]) for f in card.facts(title, ("no",))}
        mine = {key_of(f["text"]) for f in card.facts(title, ("owner",))}      # you wrote a line about it yourself: do not add a rival
        for f in card.facts(title, ("auto",)):
            plan["remove"].append((title, f["text"]))
        for key, text in new:
            if key.casefold() in refused or key.casefold() in mine:
                continue
            plan["add"].append((title, "%s: %s" % (key, text)))
    # lines that would be removed and added again unchanged are not a change
    same = set(plan["add"]) & set(plan["remove"])
    plan["add"] = [x for x in plan["add"] if x not in same]
    plan["remove"] = [x for x in plan["remove"] if x not in same]
    return plan


def apply_plan(card, plan):
    for title, text in plan["remove"]:
        s = card.section(title)
        for it in list(s[1]):
            if it["kind"] == "fact" and it["tag"] == "auto" and it["text"] == text:
                s[1].remove(it)
                break
    for title, text in plan["add"]:
        _put(card.section(title)[1], {"kind": "fact", "tag": "auto", "text": text})


# ---------------------------------------------------------------------------------------------- the card inside the chat prompt
def prompt_text(card, count_tokens, budget_tokens):
    """Whole facts only, in PROMPT_ORDER, never past the budget. [no] lines never go in. -> (text, facts_used, facts_left_out)."""
    lines, used, left = [], 0, 0
    for title in PROMPT_ORDER:
        for it in card.facts(title, ("owner", "auto")):
            line = "%s: %s" % (title, it["text"]) if title != "Name" else "Name: " + it["text"]
            cost = count_tokens(line + "\n")
            if used + cost > budget_tokens:
                left += 1
                continue
            lines.append(line)
            used += cost
    return "\n".join(lines), len(lines), left
