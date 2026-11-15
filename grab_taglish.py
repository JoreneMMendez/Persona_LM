"""Download Taglish / Tagalog forum conversations (PinoyExchange, "gabtan99/pex-conversations") as pretraining text.

No extra packages needed (it does NOT use `datasets` or `ijson`; only Python's standard library).

How it works: the dataset is ONE 658 MB json file. The old version streamed it through `datasets`, which has to fetch the whole
file before it gives out the first row, so the screen sat at 0% for ages (and on a slow line, for an hour).
This version reads the file piece by piece, writes threads as they arrive, and STOPS as soon as the size cap is reached,
so a 100 MB request downloads only about 100-150 MB, not 658 MB. If the connection drops it resumes where it stopped.
If a full copy is already on disk (pex_raw/raw.json) it uses that instead and downloads nothing.

Run:     python grab_taglish.py            -> public_text/taglish/pex.jsonl   ({"text": one whole thread per line})
         python grab_taglish.py 100        -> stop after about 100 MB of text
Menu:    the language-data button calls main(max_mb) here; nothing else to do.
Optional: set HF_TOKEN (free Hugging Face token) for faster downloads.

NOTE: the dataset's license is listed as "unknown" and the posts come from real forum users. Use it for your own private
training only; do not re-upload the data, and do not publish a model trained on it without checking the site's terms.

Next:    python clean_corpus.py   (or python -m persona_lm corpus --lang taglish --mode clean)
"""
import codecs
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

OUT_PATH = os.path.join("public_text", "taglish", "pex.jsonl")
MIN_CHARS = 200
URL = "https://huggingface.co/datasets/gabtan99/pex-conversations/resolve/main/raw.json"
LOCAL_COPIES = (os.path.join("pex_raw", "raw.json"), os.path.join("public_text", "taglish", "raw.json"))


def flatten(node, out, depth=0):
    """A thread is {"text": ..., "replies": [same shape, ...]}. Returns the messages in reading order, one per line."""
    if isinstance(node, dict):
        text = (node.get("text") or "").strip()
        if text:
            out.append(text)
        for r in node.get("replies") or []:
            flatten(r, out, depth + 1) if depth < 200 else None
    elif isinstance(node, list):
        for n in node:
            flatten(n, out, depth)
    return out


class WebStream:
    """A file-like object over the download. If the connection breaks it reconnects and continues from the same byte."""

    def __init__(self, url, retries=10):
        self.url, self.retries = url, retries
        self.pos, self.total, self.resp = 0, 0, None

    def _open(self):
        headers = {"User-Agent": "PersonaLM-grab-taglish"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        if self.pos:
            headers["Range"] = "bytes=%d-" % self.pos
        self.resp = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=60)
        if self.pos and getattr(self.resp, "status", 206) != 206:
            raise OSError("the server ignored the resume request")
        if not self.pos:
            self.total = int(self.resp.headers.get("Content-Length") or 0)

    def read(self, n):
        tries = 0
        while True:
            try:
                if self.resp is None:
                    self._open()
                data = self.resp.read(n)
                if not data and self.total and self.pos < self.total:
                    raise ConnectionError("connection closed early")
                self.pos += len(data)
                return data
            except (OSError, http.client.HTTPException) as e:
                tries += 1
                if tries > self.retries:
                    raise OSError("Download kept failing (%s). Check the internet connection and run it again." % e)
                print("  connection problem (%s); retry %d/%d in a few seconds..." % (str(e)[:80], tries, self.retries), flush=True)
                self.close()
                time.sleep(min(30, 2 * tries))

    def close(self):
        if self.resp is not None:
            try:
                self.resp.close()
            except Exception:
                pass
            self.resp = None


def iter_items(raw):
    """Yields the items of the first json list in the file, one at a time, without loading the whole file.
    `raw` only needs .read(n) -> bytes. Works for [item, ...] and for {"conversations": [item, ...]}."""
    dec = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    buf, i, eof = "", 0, False

    def fill():
        nonlocal buf, i, eof
        if eof:
            return False
        b = raw.read(1 << 20)
        if i > (1 << 20):
            buf, i = buf[i:], 0
        if not b:
            eof = True
            buf += utf8.decode(b"", True)
            return False
        buf += utf8.decode(b)
        return True

    while True:                                   # find the "[" that opens the list
        j = buf.find("[", i)
        if j >= 0:
            i = j + 1
            break
        i = len(buf)
        if not fill():
            return
    while True:                                   # decode one element at a time
        while True:
            while i < len(buf) and buf[i] in " \t\r\n,":
                i += 1
            if i < len(buf):
                break
            if not fill():
                return
        if buf[i] == "]":
            return
        while True:
            try:
                obj, end = dec.raw_decode(buf, i)
                break
            except json.JSONDecodeError:
                if not fill():
                    raise ValueError("raw.json ends in the middle of an item (broken download?). Delete it and run again.")
        i = end
        yield obj


def threads_in(item):
    if isinstance(item, dict) and isinstance(item.get("conversations"), list):
        return item["conversations"]
    return [item]


class _Enough(Exception):
    pass


def main(max_mb):
    cap = int(max_mb * 1_000_000) if max_mb else 0
    src = next((p for p in LOCAL_COPIES if os.path.exists(p) and os.path.getsize(p) > 0), None)
    if src:
        print("Reading the copy already on disk (%s); nothing to download." % src, flush=True)
        raw = open(src, "rb")
    else:
        print("Downloading from Hugging Face%s" % ("; it stops once about %d MB of text is saved, so only part of the 658 MB file is fetched." % max_mb
              if max_mb else " (the whole 658 MB file)."), flush=True)
        raw = WebStream(URL)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    total = threads = 0
    last = time.time()
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for item in iter_items(raw):
                for thread in threads_in(item):
                    text = "\n".join(flatten(thread, []))
                    if len(text) < MIN_CHARS:
                        continue
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    total += len(text)
                    threads += 1
                    if threads % 2000 == 0 or time.time() - last > 20:
                        last = time.time()
                        got = " (downloaded %d MB)" % (raw.pos // 1_000_000) if hasattr(raw, "pos") else ""
                        print("  %d threads, %d MB%s" % (threads, total // 1_000_000, got), flush=True)
                    if cap and total >= cap:
                        raise _Enough
    except _Enough:
        pass
    except KeyboardInterrupt:
        print("Stopped; keeping what was written.")
    finally:
        raw.close()
    if threads == 0:
        print("Found no threads in the file. It may be in a different format than expected.")
    print("%d threads, %d MB -> %s" % (threads, total // 1_000_000, OUT_PATH))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
