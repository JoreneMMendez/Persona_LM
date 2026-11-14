"""Removing personal information from text BEFORE it leaves your computer (e.g. before uploading to a rented GPU).

Everything here is plain regular expressions and dictionaries, runs locally and uses no network.

What it removes, and how reliable that is:
  * names of the chat participants (full name, first name, last name, any nickname you list)  -> reliable for the names it is GIVEN
  * e-mail addresses, web links, @handles                                                      -> reliable
  * phone numbers (7-15 digits, with spaces, dashes, dots, brackets, +country code)            -> reliable for common formats
  * long digit runs (9+ digits: ID, card, account numbers)                                     -> reliable
  * street addresses                                                                           -> ONLY A HEURISTIC (English/Philippine patterns), it WILL miss some
  * names of people who are not chat participants (a friend you only mention)                  -> NOT detectable without a name list: use `extra_names`

So after anonymizing, `audit()` looks at the result with looser patterns and reports anything suspicious, and you should skim a sample
of the output yourself before uploading. Nothing in software can promise that text is 100% free of personal information.
"""
import collections
import re

# Name parts that are also everyday words. They are only replaced when written with a capital letter exactly like in the name
# ("Will" yes, "will" no), otherwise ordinary sentences would be damaged.
COMMON_WORD_NAMES = {
    "will", "mark", "grace", "joy", "hope", "rose", "faith", "june", "may", "april", "august", "summer", "sky", "art", "bill", "rob", "pat",
    "sue", "ray", "dawn", "gene", "gale", "guy", "jack", "jay", "jean", "jess", "lane", "max", "ned", "page", "paul", "river", "rich",
    "sunny", "sandy", "lily", "ivy", "iris", "ruby", "pearl", "amber", "angel", "heaven", "love", "star", "dear", "bay", "tan", "go", "lim",
    "ong", "sy", "co",
}

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
EMAIL_RE = re.compile(r"[\w.+%-]+@[\w-]+(?:\.[\w-]+)+")
HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_.]{3,}")
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().-]{5,}\d(?![\w])")
DATE_RE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
LOOSE_DIGITS_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")

_STREET = r"(?i:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Highway|Hwy|Extension|Ext|Court|Ct|Place|Pl)"
ADDRESS_RES = [
    # "123 Rizal Street", "45B Mabini St." (a number, 1-3 capitalised words, then a street word)
    re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'.-]*\s+){1,3}" + _STREET + r"\b\.?"),
    # "Brgy. 123", "Purok 5", "Blk 4", "Lot 12", "Phase 2", "Unit 5B", "Bldg 3" (only with a number, so "block ko" is untouched)
    re.compile(r"\b(?:Brgy|Barangay|Purok|Blk|Block|Lot|Subd|Phase|Unit|Floor|Flr|Bldg|Building)\.?\s*#?\s*\d+[A-Za-z]?\b", re.I),
    # "Barangay Commonwealth", "Brgy. San Roque" (followed by capitalised words)
    re.compile(r"\b(?:Brgy|Barangay)\.?\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?"),
]


def _name_variants(name):
    """The strings that should count as this person: the full name and each part of 3+ letters."""
    name = " ".join(name.split())
    parts = [p.strip(".,;:()\"'") for p in name.split()]
    out = {name} if name else set()
    out.update(p for p in parts if len(p) >= 3)
    return out


class Anonymizer:
    """scrub(text) replaces personal information; label(name) gives the stand-in for a sender name.

    me_names:    the names YOU appear under (they become [ME])
    people:      every other participant name in the chats you import (they become Person_01, Person_02, ...)
    extra_names: people who are only MENTIONED (not participants), one per line in a file or a list
    aliases:     {"Mikee": "Michael Santos"}  nicknames, mapped to the same stand-in as the real name
    enabled=False turns everything off (text is returned unchanged): use only for training that never leaves your computer.
    paranoid:    drop a whole message (scrub returns None) if it contains something address-like or a long number, instead of masking it.
    """

    def __init__(self, me_names=(), people=(), extra_names=(), aliases=None, enabled=True, paranoid=False):
        self.enabled = enabled
        self.paranoid = paranoid
        self.counts = collections.Counter()
        self.ambiguous = []
        self.me_names = [n for n in me_names if n]
        self._stand_in = {}              # variant (lower-case) -> stand-in
        self._exact = {}                 # variants that must match case-exactly (common words) -> stand-in
        self._person_no = 0
        self.real_to_standin = {}
        for n in sorted(set(self.me_names)):
            self._register(n, "[ME]")
        for n in sorted(set(people) | set(extra_names or ())):
            if not self._is_me(n):
                self._register(n, self._next_person())
        for alias, real in (aliases or {}).items():
            target = self.real_to_standin.get(real) or ("[ME]" if self._is_me(real) else self._next_person())
            self._register(alias, target, as_alias=True)
        self._build()

    # ------------------------------------------------------------ names
    def _is_me(self, name):
        return name.casefold() in {m.casefold() for m in self.me_names}

    def _next_person(self):
        self._person_no += 1
        return "Person_%02d" % self._person_no

    def _register(self, name, stand_in, as_alias=False):
        if stand_in not in ("[ME]",) and name in self.real_to_standin and not as_alias:
            return
        self.real_to_standin.setdefault(name, stand_in)
        for v in _name_variants(name):
            table = self._exact if v.casefold() in COMMON_WORD_NAMES else self._stand_in
            key = v if table is self._exact else v.casefold()
            if key in table and table[key] != stand_in:
                self.ambiguous.append((v, table[key], stand_in))      # two people share a first name: the first one keeps it
                continue
            table[key] = stand_in

    def _build(self):
        def alt(words):
            return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        self._ci_re = re.compile(r"(?<!\w)(?:%s)(?!\w)" % alt(self._stand_in), re.I | re.U) if self._stand_in else None
        self._cs_re = re.compile(r"(?<!\w)(?:%s)(?!\w)" % alt(self._exact), re.U) if self._exact else None
        # Fast path: a name regex with thousands of alternatives is slow, and most messages contain no name. Any match of a name variant
        # must contain at least one of that variant's words as a whole word, so a message that shares NO word with the known names can be
        # skipped without changing the result. (Set to None to switch the shortcut off, e.g. in tests.)
        self._words = {w.casefold() for v in list(self._stand_in) + list(self._exact) for w in re.findall(r"\w+", v)}

    def _may_contain_name(self, text):
        if self._words is None:
            return True
        return not self._words.isdisjoint(w.casefold() for w in re.findall(r"\w+", text))

    def label(self, name):
        """What to show for a sender name."""
        if not self.enabled:
            return name
        if self._is_me(name):
            return "[ME]"
        return self.real_to_standin.get(name, "Person_??")      # "Person_??": a sender that was not registered (should not happen)

    # ------------------------------------------------------------ text
    def _phone(self, m):
        s = m.group(0)
        digits = sum(c.isdigit() for c in s)
        if DATE_RE.match(s.strip()) or not 7 <= digits <= 15:
            return s
        self.counts["phone"] += 1
        return "[PHONE]"

    def scrub(self, text):
        """Returns the cleaned text, or None when `paranoid` says the whole message should be dropped."""
        if not self.enabled:
            return text
        sensitive = False

        def sub(regex, repl, kind, s):
            def f(m):
                self.counts[kind] += 1
                return repl
            return regex.sub(f, s)

        text = sub(URL_RE, "[URL]", "url", text)
        text = sub(EMAIL_RE, "[EMAIL]", "email", text)
        text = sub(HANDLE_RE, "[HANDLE]", "handle", text)
        text = PHONE_CANDIDATE_RE.sub(self._phone, text)
        before = text
        text = sub(LONG_NUMBER_RE, "[NUMBER]", "number", text)
        sensitive |= text != before
        for rx in ADDRESS_RES:
            before = text
            text = sub(rx, "[ADDRESS]", "address", text)
            sensitive |= text != before
        if self.paranoid and sensitive:
            self.counts["dropped_paranoid"] += 1
            return None

        def name_sub(table, exact):
            def f(m):
                w = m.group(0)
                self.counts["name"] += 1
                return table[w if exact else w.casefold()]
            return f
        if self._may_contain_name(text):
            if self._cs_re:
                text = self._cs_re.sub(name_sub(self._exact, True), text)
            if self._ci_re:
                text = self._ci_re.sub(name_sub(self._stand_in, False), text)
        return text

    # ------------------------------------------------------------ the second look
    def audit(self, text):
        """Look at ALREADY-scrubbed text. Returns (leaks, suspicious):
        leaks      = things that must not be there (a real name, an e-mail, a link, a phone number): the output is NOT safe to upload
        suspicious = looser patterns (7+ digit runs, '@word', 'http') that deserve a human look"""
        leaks, suspicious = [], []
        if self._may_contain_name(text):
            if self._cs_re:
                leaks += [("name", m.group(0)) for m in self._cs_re.finditer(text)]
            if self._ci_re:
                leaks += [("name", m.group(0)) for m in self._ci_re.finditer(text)]
        leaks += [("email", m.group(0)) for m in EMAIL_RE.finditer(text)]
        leaks += [("url", m.group(0)) for m in URL_RE.finditer(text)]
        for m in PHONE_CANDIDATE_RE.finditer(text):
            digits = sum(c.isdigit() for c in m.group(0))
            if 7 <= digits <= 15 and not DATE_RE.match(m.group(0).strip()):
                leaks.append(("phone", m.group(0)))
        suspicious += [("digits", m.group(0)) for m in LOOSE_DIGITS_RE.finditer(text)]
        suspicious += [("at-sign", m.group(0)) for m in re.finditer(r"@\w+", text)]
        suspicious += [("http", m.group(0)) for m in re.finditer(r"https?:|www\.", text, re.I)]
        return leaks, suspicious


def capitalised_word_report(texts, known=(), top=40):
    """Help to find names the program does not know about: the most frequent capitalised words that are NOT the first word of a sentence.
    Look through the list; add real names to your extra-names file and import again."""
    known = {k.casefold() for k in known}
    counts = collections.Counter()
    for t in texts:
        for sentence in re.split(r"[.!?\n]+\s*", t):
            words = re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
            if re.match(r"\s*[A-Z][a-z]{2,}\b", sentence):
                words = words[1:]                          # the sentence's own first word is capitalised anyway
            for w in words:
                if w.casefold() not in known:
                    counts[w] += 1
    return counts.most_common(top)
