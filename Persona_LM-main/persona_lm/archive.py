"""Chat archive: the AI can "see" what was said in chats from long ago.

Why this exists: the model's window is fixed at training time (4096 tokens for new s10m / s30m models; older checkpoints keep the 512 they were trained with). It can never hold 200 old chats at once.
So instead every exchange (your message + the AI's reply) is appended to one file, and for each new message the few OLD exchanges
that are most relevant to it are looked up and put into the prompt (inside the same <|memory|> block that the memory graph uses).

How an old exchange is found:
  * words of your newest message(s) are matched against the words of every archived exchange (rarer words count more);
  * a small bonus for newer exchanges (half-life 180 days), so a tie goes to the more recent chat;
  * exchanges that the model can already see in the current window are skipped (no point in showing them twice).

File format: one JSON object per line (JSON Lines), appended to, never rewritten:
  {"id": 12, "session": "s20261115-091500", "t": 1763198100.0, "user": "...", "ai": "..."}
Several programs can append safely; a half-written last line (power cut) is ignored when reading.

Nothing here needs PyTorch or NumPy.
"""
import json
import math
import os
import time
from collections import Counter, defaultdict

from .memory import words_of

HALF_LIFE_DAYS = 180.0
MAX_TEXT = 1500                  # characters kept per message in the archive (longer ones are cut)
LINE_CHARS = 220                 # characters of each message shown in the prompt line


class ArchiveProblem(Exception):
    """Something the owner can fix. The message is written for the owner."""


def _cut(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def when_text(t, now):
    """'3 days ago', '2 months ago' ... for the prompt line."""
    days = max(0.0, (now - t) / 86400.0)
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 30:
        return "%d days ago" % days
    if days < 365:
        return "%d months ago" % (days // 30)
    return "%d years ago" % (days // 365)


class ChatArchive:
    def __init__(self, path, clock=time.time):
        self.path = path
        self.clock = clock
        self.items = []                      # [{"id", "session", "t", "user", "ai"}]
        self._index = None                   # built lazily: (word sets, document frequency)
        self._load()

    # ------------------------------------------------------------ file
    def _load(self):
        self.items = []
        self._index = None
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        it = json.loads(line)
                    except ValueError:
                        continue             # a half-written line: skip it
                    if isinstance(it, dict) and "user" in it and "ai" in it:
                        it.setdefault("id", len(self.items) + 1)
                        it.setdefault("session", "")
                        it.setdefault("t", 0.0)
                        self.items.append(it)
        except OSError as e:
            raise ArchiveProblem("Cannot read the chat archive %s: %s" % (self.path, e))

    def __len__(self):
        return len(self.items)

    def sessions(self):
        return len({it["session"] for it in self.items})

    def append(self, session, user, ai, t=None):
        """Add one exchange. Returns the stored item."""
        it = {"id": (self.items[-1]["id"] + 1) if self.items else 1, "session": session,
              "t": float(self.clock() if t is None else t), "user": _cut(user, MAX_TEXT), "ai": _cut(ai, MAX_TEXT)}
        folder = os.path.dirname(os.path.abspath(self.path))
        try:
            os.makedirs(folder, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        except OSError as e:
            raise ArchiveProblem("Cannot write the chat archive %s: %s" % (self.path, e))
        self.items.append(it)
        self._index = None
        return it

    # ------------------------------------------------------------ search
    def _build_index(self):
        sets = [set(words_of(it["user"] + " " + it["ai"])) for it in self.items]
        df = Counter(w for s in sets for w in s)
        inverted = defaultdict(list)
        for i, s in enumerate(sets):
            for w in s:
                inverted[w].append(i)
        self._index = (sets, df, inverted)

    def search(self, query, k=3, skip_ids=(), now=None, half_life=HALF_LIFE_DAYS):
        """The archived exchanges that best match `query`: [(item, score, matched_words)], best first.
        An exchange with no word in common with the query is never returned."""
        q = set(words_of(query))
        if not q or not self.items:
            return []
        if self._index is None:
            self._build_index()
        sets, df, inverted = self._index
        now = self.clock() if now is None else now
        n = len(self.items)
        skip = set(skip_ids)
        candidates = set()
        for w in q:
            candidates.update(inverted.get(w, ()))
        scored = []
        for i in candidates:
            it = self.items[i]
            if it["id"] in skip:
                continue
            common = q & sets[i]
            rel = sum(math.log(1 + n / (1 + df[w])) for w in common) / math.sqrt(max(len(sets[i]), 1))
            age_days = max(0.0, (now - it["t"]) / 86400.0)
            recency = 0.5 + 0.5 * math.pow(0.5, age_days / half_life)      # 1.0 for new, down to 0.5 for very old
            scored.append((rel * recency, it, sorted(common)))
        scored.sort(key=lambda t: (-t[0], -t[1]["id"]))
        return [(it, round(sc, 4), common) for sc, it, common in scored[:k]]

    @staticmethod
    def line_for_prompt(it, now, chars=LINE_CHARS):
        return "- earlier chat (%s): you said \"%s\" / I said \"%s\"" % (when_text(it["t"], now), _cut(it["user"], chars), _cut(it["ai"], chars))

    def recall(self, query, k=3, budget_tokens=None, count_tokens=None, skip_ids=(), now=None):
        """Like search, but stops when the lines would use more than budget_tokens (count_tokens(str) -> int).
        A line that is too long is tried again with shorter quotes (220, 120, then 60 characters per message).
        Returns [(item, score, words, line)]."""
        now = self.clock() if now is None else now
        out, used = [], 0
        for it, sc, words in self.search(query, k=k * 3, skip_ids=skip_ids, now=now):
            if len(out) >= k:
                break
            if budget_tokens is None or count_tokens is None:
                out.append((it, sc, words, self.line_for_prompt(it, now)))
                continue
            for chars in (LINE_CHARS, 120, 60):
                line = self.line_for_prompt(it, now, chars)
                cost = count_tokens(line)
                if used + cost <= budget_tokens:
                    used += cost
                    out.append((it, sc, words, line))
                    break
        return out

    def overview(self, now=None):
        if not self.items:
            return "The chat archive is empty (it fills as you chat)."
        now = self.clock() if now is None else now
        first, last = self.items[0], self.items[-1]
        return "Chat archive: %d exchanges in %d chats, from %s to %s (%s)." % (
            len(self.items), self.sessions(), time.strftime("%Y-%m-%d", time.localtime(first["t"])),
            time.strftime("%Y-%m-%d", time.localtime(last["t"])), self.path)
