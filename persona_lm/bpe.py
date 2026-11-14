"""Byte-level BPE tokenizer, trained and applied from scratch.

Vocabulary layout (ids): special tokens first, then the 256 byte tokens, then one token per learned merge.
"""
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict

from .pretok import BYTE2UNI, UNI2BYTE, split_words

# Special tokens come FIRST in the vocabulary, so adding one later would renumber every other token and break all saved data,
# checkpoints and memories. To make that impossible, a block of N_SPECIAL_SLOTS ids is reserved once, now:
#   0-4   the original five (their ids never change)
#   5-9   named for features planned in later milestones (memory, persona, mood, speaker, summary)
#   10-31 spare slots: a future feature takes one of these by being *renamed*, which changes no other id.
# A tokenizer file saved before this change keeps its own 5 specials (Tokenizer.load reads them from the file).
N_SPECIAL_SLOTS = 32
_CORE_SPECIALS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_PLANNED_SPECIALS = ["<|memory|>", "<|persona|>", "<|mood|>", "<|speaker|>", "<|summary|>"]
_N_RESERVED = N_SPECIAL_SLOTS - len(_CORE_SPECIALS) - len(_PLANNED_SPECIALS)
SPECIALS = _CORE_SPECIALS + _PLANNED_SPECIALS + ["<|reserved_%02d|>" % i for i in range(_N_RESERVED)]
EOS, SYSTEM, USER, ASSISTANT, END = range(5)
MEMORY, PERSONA, MOOD, SPEAKER, SUMMARY = range(5, 10)
ROLE_ID = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT,
           "persona": PERSONA}       # Stage 8b: a persona block in chat data/prompt is just another message  <|persona|> text <|end|>  (never trained on by default)


def _special_re(specials):
    return re.compile("(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")")


class Tokenizer:
    def __init__(self, merges, specials=None):
        self.specials = list(specials or SPECIALS)
        self.merges = [tuple(m) for m in merges]
        self.id_to_tok = list(self.specials) + [BYTE2UNI[b] for b in range(256)] + [a + b for a, b in self.merges]
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        if len(self.tok_to_id) != len(self.id_to_tok):
            raise ValueError("duplicate token in vocabulary")
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self._cache = {}
        self._special_re = _special_re(self.specials)
        self._special_set = set(self.specials)
        self.n_special = len(self.specials)

    @property
    def vocab_size(self):
        return len(self.id_to_tok)

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, texts, vocab_size, specials=None, min_freq=2, log=None):
        specials = list(specials or SPECIALS)
        sre = _special_re(specials)
        counts = Counter()
        for t in texts:
            for part in sre.split(t):
                if part in specials or not part:
                    continue
                counts.update(split_words(part))
        words = [[BYTE2UNI[b] for b in w.encode("utf-8")] for w in counts]
        freqs = list(counts.values())
        n_merges = vocab_size - len(specials) - 256
        if n_merges < 0:
            raise ValueError("vocab_size must be at least %d" % (len(specials) + 256))

        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for wi, sym in enumerate(words):
            for p in zip(sym, sym[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        vocab = set(BYTE2UNI.values())
        merges = []
        while len(merges) < n_merges and heap:
            negc, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur != -negc or cur <= 0:
                continue                      # stale heap entry
            if cur < min_freq:
                break
            new = pair[0] + pair[1]
            if new in vocab:                  # same string reachable by another split: skip this pair
                pair_counts[pair] = 0
                continue
            a, b = pair
            changed = set()
            for wi in list(where[pair]):
                sym = words[wi]
                f = freqs[wi]
                for p in zip(sym, sym[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                out = []
                i = 0
                while i < len(sym):
                    if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(sym[i])
                        i += 1
                for p in set(zip(sym, sym[1:])):
                    where[p].discard(wi)
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    changed.add(p)
            vocab.add(new)
            merges.append(pair)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
            if log and len(merges) % 500 == 0:
                log("tokenizer: %d / %d merges" % (len(merges), n_merges))
        return cls(merges, specials)

    # ---------------------------------------------------------------- use
    def _bpe_word(self, word):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        sym = [BYTE2UNI[b] for b in word.encode("utf-8")]
        while len(sym) > 1:
            best = None
            best_rank = None
            for p in zip(sym, sym[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            a, b = best
            out = []
            i = 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            sym = out
        ids = [self.tok_to_id[s] for s in sym]
        if len(self._cache) < 200000:
            self._cache[word] = ids
        return ids

    def encode(self, text, allow_special=True):
        """Text -> ids. With allow_special, the literal special-token strings become their ids."""
        ids = []
        parts = self._special_re.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self._special_set:
                ids.append(self.tok_to_id[part])
                continue
            for w in split_words(part):
                ids.extend(self._bpe_word(w))
        return ids

    def decode(self, ids):
        out = []
        buf = bytearray()
        for i in ids:
            t = self.id_to_tok[i]
            if i < self.n_special:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(t)
            else:
                buf.extend(UNI2BYTE[ch] for ch in t)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    # ---------------------------------------------------------------- files
    def fingerprint(self):
        """SHA-256 of the vocabulary definition. Stored in data folders and checkpoints so a model is never
        used with a tokenizer it was not trained with (that would silently produce nonsense)."""
        blob = json.dumps({"specials": self.specials, "merges": [list(m) for m in self.merges]}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"specials": self.specials, "merges": [list(m) for m in self.merges]}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["merges"], d["specials"])
