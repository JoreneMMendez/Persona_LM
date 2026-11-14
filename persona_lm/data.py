"""Turning files into token files, and sampling training batches from them.

Pretraining text:  .txt / .md files (or folders of them), .jsonl / .json with strings or {"text": ...}.
Chat data:         .jsonl / .json with {"messages": [{"role","content"}, ...]}  or  {"prompt","reply"}  or  {"instruction","output"}.
Output folder:     train.bin / val.bin (uint16 token ids), train_mask.bin / val_mask.bin (uint8, chat only), meta.json.
"""
import json
import os

import numpy as np

from .bpe import EOS, END, ROLE_ID

TEXT_EXT = {".txt", ".md"}
JSON_EXT = {".json", ".jsonl"}


def _files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in sorted(os.walk(p)):
                for n in sorted(names):
                    yield os.path.join(root, n)
        else:
            yield p


def _json_records(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data


def _text_blocks(path, block_chars=1 << 20):
    """Yield (text, is_last) blocks that end at a line break, so words are never cut in half."""
    with open(path, encoding="utf-8", errors="replace") as f:
        buf = ""
        while True:
            chunk = f.read(block_chars)
            if not chunk:
                break
            buf += chunk
            cut = buf.rfind("\n")
            if cut > 0 and len(buf) > block_chars // 2:
                yield buf[:cut + 1], False
                buf = buf[cut + 1:]
        yield buf, True


def pretrain_docs(paths, tok):
    """Yields (ids ndarray, mask None, end_of_doc)."""
    for path in _files(paths):
        ext = os.path.splitext(path)[1].lower()
        if ext in TEXT_EXT:
            for text, last in _text_blocks(path):
                yield np.array(tok.encode(text, allow_special=False), dtype=np.uint16), None, last
        elif ext in JSON_EXT:
            for rec in _json_records(path):
                text = rec if isinstance(rec, str) else (rec.get("text") or rec.get("content") if isinstance(rec, dict) else None)
                if isinstance(text, str) and text.strip():
                    yield np.array(tok.encode(text, allow_special=False), dtype=np.uint16), None, True


def as_messages(rec, with_train=False):
    """Accepted chat shapes -> list of (role, content) or None.
    with_train=True returns (role, content, train) triples: a message in {"messages": [...]} may carry "train": false/true to override
    the default (loss on assistant turns only). The Messenger importer uses this to put the loss on just the reply it wants the model to learn."""
    if not isinstance(rec, dict):
        return None
    if isinstance(rec.get("messages"), list):
        msgs = [(m.get("role"), m.get("content"), m.get("train")) for m in rec["messages"] if isinstance(m, dict)]
    elif "prompt" in rec and any(k in rec for k in ("reply", "response", "completion")):
        reply = rec.get("reply", rec.get("response", rec.get("completion")))
        msgs = [("user", rec["prompt"], None), ("assistant", reply, None)]
    elif "instruction" in rec and "output" in rec:
        user = rec["instruction"] + ("\n" + rec["input"] if rec.get("input") else "")
        msgs = [("user", user, None), ("assistant", rec["output"], None)]
    else:
        return None
    msgs = [(r, c, t) for r, c, t in msgs if r in ROLE_ID and isinstance(c, str) and c]
    if not any(r == "assistant" and (t is None or t) for r, _, t in msgs):
        return None
    if with_train:
        return [(r, c, (r == "assistant") if t is None else bool(t)) for r, c, t in msgs]
    return [(r, c) for r, c, _ in msgs]


def encode_chat(tok, msgs):
    """<|role|> content <|end|> for every message; the loss mask is 1 only on assistant content and its <|end|>."""
    ids, mask = [], []
    for item in msgs:
        role, content = item[0], item[1]
        train = int(item[2]) if len(item) > 2 else int(role == "assistant")
        body = tok.encode(content, allow_special=False)       # allow_special=False: text in a message can never forge a role token
        ids += [ROLE_ID[role]] + body + [END]
        mask += [0] + [train] * (len(body) + 1)
    ids.append(EOS)
    mask.append(0)
    return np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)


def chat_docs(paths, tok):
    for path in _files(paths):
        if os.path.splitext(path)[1].lower() in JSON_EXT:
            for rec in _json_records(path):
                msgs = as_messages(rec, with_train=True)
                if msgs:
                    ids, mask = encode_chat(tok, msgs)
                    yield ids, mask, False


def prepare(tok, docs, out_dir, with_mask, val_every=50, log=print, val_docs=None):
    """docs: iterable of (ids, mask, end_of_doc). Every val_every-th document goes to validation.
    val_docs: if given, ALL of docs go to training and ALL of val_docs to validation (no automatic split). Use it when the split must be
    made by hand, e.g. by conversation/time, so that overlapping context windows of one conversation never sit on both sides."""
    if tok.vocab_size > 65535:
        raise ValueError("vocabulary too large for uint16 token files")
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for split in ("train", "val"):
        files[split] = (open(os.path.join(out_dir, split + ".bin"), "wb"),
                        open(os.path.join(out_dir, split + "_mask.bin"), "wb") if with_mask else None)
    counts = {"train": 0, "val": 0}
    doc_i = 0
    explicit_val = val_docs is not None
    if explicit_val:
        for split, source in (("train", docs), ("val", val_docs)):
            f, fm = files[split]
            for ids, mask, end in source:
                f.write(ids.tobytes())
                counts[split] += len(ids)
                if fm is not None:
                    fm.write(mask.tobytes())
                if not with_mask and end:
                    f.write(np.array([EOS], dtype=np.uint16).tobytes())
                    counts[split] += 1
        docs = ()                                       # consumed above; skip the automatic loop
    for ids, mask, end in docs:
        split = "val" if (doc_i % val_every == val_every - 1) else "train"
        f, fm = files[split]
        f.write(ids.tobytes())
        counts[split] += len(ids)
        if fm is not None:
            fm.write(mask.tobytes())
        if not with_mask and end:
            f.write(np.array([EOS], dtype=np.uint16).tobytes())
            counts[split] += 1
        if with_mask or end:
            doc_i += 1
        if doc_i and doc_i % 1000 == 0 and (with_mask or end):
            log("prepared %d documents, %d tokens" % (doc_i, counts["train"] + counts["val"]))
    for f, fm in files.values():
        f.close()
        if fm:
            fm.close()
    if not explicit_val and counts["val"] == 0 and counts["train"] > 100:           # too few documents for a document-level split
        tr = np.fromfile(os.path.join(out_dir, "train.bin"), dtype=np.uint16)
        cut = len(tr) - max(len(tr) // 20, 1)
        tr[:cut].tofile(os.path.join(out_dir, "train.bin"))
        tr[cut:].tofile(os.path.join(out_dir, "val.bin"))
        if with_mask:
            mk = np.fromfile(os.path.join(out_dir, "train_mask.bin"), dtype=np.uint8)
            mk[:cut].tofile(os.path.join(out_dir, "train_mask.bin"))
            mk[cut:].tofile(os.path.join(out_dir, "val_mask.bin"))
        counts = {"train": cut, "val": len(tr) - cut}
    meta = {"vocab_size": tok.vocab_size, "kind": "chat" if with_mask else "pretrain", "train_tokens": counts["train"], "val_tokens": counts["val"],
            "tokenizer_sha256": tok.fingerprint()}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("done: %d train tokens, %d validation tokens" % (counts["train"], counts["val"]))
    return meta


class Stream:
    """Random windows from a token file (memory-mapped, so corpora bigger than RAM work)."""

    def __init__(self, data_dir, split="train"):
        self.tokens = np.memmap(os.path.join(data_dir, split + ".bin"), dtype=np.uint16, mode="r")
        mp = os.path.join(data_dir, split + "_mask.bin")
        self.mask = np.memmap(mp, dtype=np.uint8, mode="r") if os.path.exists(mp) else None

    def __len__(self):
        return len(self.tokens)

    def batch(self, rng, B, T):
        n = len(self.tokens)
        if n < T + 2:
            raise ValueError("only %d tokens in the data: need more data or a smaller context" % n)
        starts = rng.integers(0, n - T - 1, size=B)
        x = np.stack([self.tokens[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        m = np.stack([self.mask[s + 1:s + T + 1] for s in starts]) if self.mask is not None else None
        return x, y, m
