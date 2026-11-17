"""T1: the plain-words text bank (persona_lm/menu_text.py) covers every menu item."""
import re
import unittest

from persona_lm import menu_text as mt
from persona_lm.menu import ITEMS, plain_text

KEYS = ("title", "what", "when", "need", "result", "time", "safe")
MAX_WORDS = 18


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def all_texts():
    for label, rec in mt.ITEM_TEXT.items():
        for k in KEYS:
            yield "ITEM_TEXT[%s].%s" % (label, k), rec[k]
    for term, text in mt.GLOSSARY.items():
        yield "GLOSSARY[%s]" % term, text
    for i, (_pat, what, why, todo) in enumerate(mt.FRIENDLY_ERRORS):
        for name, text in (("what", what), ("why", why), ("todo", todo)):
            yield "FRIENDLY_ERRORS[%d].%s" % (i, name), text


class TextBank(unittest.TestCase):
    def test_every_item_has_every_key(self):
        for item in ITEMS:
            rec = plain_text(item)
            for k in KEYS:
                self.assertTrue(str(rec.get(k, "")).strip(), "%s: missing %s" % (item["label"], k))

    def test_no_orphan_records(self):
        self.assertEqual(set(mt.ITEM_TEXT), {i["label"] for i in ITEMS})

    def test_sentences_are_short(self):
        for where, text in all_texts():
            for s in sentences(text):
                self.assertLessEqual(len(s.split()), MAX_WORDS, "%s: %r" % (where, s))

    def test_glossary_terms_used_are_defined(self):
        defined = " | ".join(t.lower() for t in mt.GLOSSARY)
        for where, text in all_texts():
            low = text.lower()
            for word in mt.JARGON:
                if word in low:
                    self.assertIn(word, defined, "%s uses %r but the glossary does not define it" % (where, word))

    def test_glossary_has_the_planned_terms(self):
        low = " | ".join(t.lower() for t in mt.GLOSSARY)
        for need in ("model file", "tokenizer", "prepared text files", "run", "base model", "fine-tune",
                     "test score", "overfitting", "batch", "accum", "learning speed", "gpu", "cpu", "torch", "recipe"):
            self.assertIn(need, low)

    def test_friendly_errors_cover_the_planned_cases(self):
        samples = ["CUDA out of memory", "FileNotFoundError: x", "KeyError: 'merges'", "No module named 'torch'",
                   "OSError: [Errno 28] No space left on device", "URLError: getaddrinfo failed", "KeyboardInterrupt"]
        for text in samples:
            hits = [e for e in mt.FRIENDLY_ERRORS if re.search(e[0], text, re.I)]
            self.assertTrue(hits, "no friendly error for " + text)
        for pat, what, why, todo in mt.FRIENDLY_ERRORS:
            re.compile(pat)
            self.assertTrue(what and why and todo)

    def test_text_is_plain_ascii_safe(self):
        for where, text in all_texts():
            text.encode("cp1252")


if __name__ == "__main__":
    unittest.main()
