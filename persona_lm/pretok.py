"""Word splitter equivalent to the GPT-2 regex, written by hand (Python's `re` has no \\p{L}).

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

It is the same rule llama.cpp uses for tokenizer.ggml.pre = "gpt-2", so a model trained here tokenizes the same way there.
"""
import unicodedata


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable character map (so every byte string is a string of visible characters)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE2UNI = bytes_to_unicode()
UNI2BYTE = {v: k for k, v in BYTE2UNI.items()}

_CONTRACTIONS = ("ll", "ve", "re", "s", "d", "m", "t")


def _is_letter(c):
    return unicodedata.category(c)[0] == "L"


def _is_number(c):
    return unicodedata.category(c)[0] == "N"


def _is_other(c):
    return (not c.isspace()) and not _is_letter(c) and not _is_number(c)


def split_words(text):
    """Split text into pieces; "".join(pieces) == text always."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "'":
            hit = None
            for s in _CONTRACTIONS:
                if text.startswith(s, i + 1):
                    hit = s
                    break
            if hit is not None:
                out.append(text[i:i + 1 + len(hit)])
                i += 1 + len(hit)
                continue
        j = i + 1 if (c == " " and i + 1 < n) else i   # optional single leading space
        for pred in (_is_letter, _is_number, _is_other):
            if j < n and pred(text[j]):
                k = j
                while k < n and pred(text[k]):
                    k += 1
                out.append(text[i:k])
                i = k
                break
        else:
            k = i
            while k < n and text[k].isspace():
                k += 1
            if k < n and k - i > 1:
                k -= 1          # \s+(?!\S): leave the last whitespace for the next piece
            out.append(text[i:k])
            i = k
    return out
