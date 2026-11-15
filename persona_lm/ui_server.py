"""The local web UI server (stages U0 to U5b). Python standard library only, offline, this computer only.

    python -m persona_lm ui                 # opens http://127.0.0.1:8777/ in your browser
    python -m persona_lm ui --port 9000 --no-open --root C:\\path\\to\\project

What this file guarantees (each point has a test in tests/test_ui_u0.py):
  * It listens on 127.0.0.1 only. A different --host is refused (there is no flag to allow it).
  * A random token is made at every start. The page needs it once (in the link the command prints and opens); the server then
    sets a cookie and redirects to a clean address. Every /api/ request must also send the token in the X-PLM-Token header.
    A web page on another site cannot send that header (we send no CORS headers), so it cannot drive this server.
  * The Host header must be 127.0.0.1:PORT or localhost:PORT (blocks DNS rebinding). If an Origin header is present it must be
    this same address. POST bodies must be JSON and are limited to 64 KB.
  * Static files come from a fixed list (STATIC); there is no "serve any path" route, so .. and encoded tricks reach nothing.
  * Project files are only touched through PathGuard: inside the project folder, not through a link out of it, and never
    data_msgr/private/ or any folder called private (or the raw Messenger exports). Stage U0 only checks that a few known
    names exist; it never lists or reads chat data.
  * There is NO route that runs a command, approves a proposal, changes the identity or writes memory. Later stages add routes
    one at a time; the typed confirmations stay in the terminal unless a stage says otherwise and was approved by the owner.
    U1a adds the chat routes (/api/chat/...; see ui_chat.py): a message is sent with say(), never handle(), so "/approve 1" is only text.
    U2a adds the memory routes (/api/memory/...; see ui_memory.py) for the owner at the keyboard: they touch memory.json only, through
    MemoryStore. Forget needs its id typed back; maintenance applies only the plan that was previewed. Chat text never reaches them.
  * Nothing the owner types is logged by this server (the per-request log line is switched off).
"""
import hmac
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .ui_chat import ChatHub, ChatProblem
from .ui_data import DataHub, DataProblem
from .ui_memory import MemoryHub, MemoryHubProblem
from .ui_mind import MindHub, MindProblem
from .ui_job import JobHub, JobProblem
from .ui_train import TrainHub, TrainProblem

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "ui_static")
DEFAULT_PORT = 8777
MAX_BODY = 65536
DRAIN_LIMIT = 1 << 20                     # most an unwanted request body we read (and discard) before answering
COOKIE = "plm_session"
TOKEN_HEADER = "X-PLM-Token"

# The ONLY files the server will send. URL path -> (file in ui_static, content type).
STATIC = {
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
    "/static/memory.js": ("memory.js", "text/javascript; charset=utf-8"),
    "/static/mind.js": ("mind.js", "text/javascript; charset=utf-8"),
    "/static/data.js": ("data.js", "text/javascript; charset=utf-8"),
    "/static/personas.js": ("personas.js", "text/javascript; charset=utf-8"),
    "/static/training.js": ("training.js", "text/javascript; charset=utf-8"),
    "/static/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
INDEX_FILE = "index.html"

# Sent with every response. No inline script or style, no outside host: the page works with the network unplugged.
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

# Folder names the server never reads, lists or serves, wherever they are in the project.
BLOCKED_NAMES = {"private", "raw", "messenger_export", "facebook_export"}


class UIProblem(Exception):
    pass


class PathGuard:
    """Decides which project paths the UI may touch. Whitelist first (an explicit list of top-level names), then the blocks."""

    # Top-level names a later stage may read (U0 only checks that they exist). Chat data folders are here so the Data page can
    # count files; the private/ folder inside data_msgr is blocked below.
    ALLOWED_TOP = (
        "memory.json", "chat_archive.jsonl", "mood.json", "person_card.md", "persona_settings.json", "project.json", "identity_core", "work", "work2", "work3",
        "data_chat", "data_chat4096", "data_pre", "saved_chats", "data_msgr", "data_msgr_chats", "data_msgr_archive", "exports", "docs", "HANDOFF.md", "README.md",
    )

    def __init__(self, root):
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            raise UIProblem("the project folder %r does not exist" % root)

    def resolve(self, rel):
        """Returns the real absolute path for a path relative to the project, or raises UIProblem. Never touches the disk beyond realpath."""
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise UIProblem("bad path")
        if os.path.isabs(rel) or re.match(r"^[A-Za-z]:", rel) or rel.startswith(("\\", "/")):
            raise UIProblem("absolute paths are not allowed")
        parts = [p for p in re.split(r"[\\/]+", rel) if p not in ("", ".")]
        if not parts or ".." in parts:
            raise UIProblem("paths with .. are not allowed")
        if parts[0] not in self.ALLOWED_TOP:
            raise UIProblem("%r is not on the list of places the UI may use" % parts[0])
        for p in parts:
            low = p.lower()
            if low in BLOCKED_NAMES or low.startswith("private"):
                raise UIProblem("that folder is private and is never read by the UI")
        full = os.path.realpath(os.path.join(self.root, *parts))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise UIProblem("that path leads outside the project folder")
        for p in os.path.relpath(full, self.root).split(os.sep):       # a link inside the project can point at private/
            if p.lower() in BLOCKED_NAMES or p.lower().startswith("private"):
                raise UIProblem("that folder is private and is never read by the UI")
        return full

    def exists(self, rel):
        try:
            return os.path.exists(self.resolve(rel))
        except UIProblem:
            return False


# What is stored where, for the Settings/Privacy page. (path, plain label, what it holds, how sensitive)
STORAGE = [
    ("data_msgr", "Messenger import (anonymised)", "your chats after names, numbers and links were replaced; train.jsonl / val.jsonl", "high"),
    ("data_msgr_chats", "Cleaned chats, one file per chat", "output of clean_chat.py", "high"),
    ("data_msgr_archive", "Old-chat archive", "past exchanges the AI can recall with --archive", "high"),
    ("memory.json", "Memory graph", "what the AI remembers about you and others", "high"),
    ("person_card.md", "Person card", "how you speak, your habits and values", "high"),
    ("mood.json", "Mood", "the AI's current feeling and its history", "medium"),
    ("identity_core", "Identity core", "who the AI is, versions, probe answers, history log, proposals", "medium"),
    ("persona_settings.json", "Settings file", "your export path and the name you are called in chats", "medium"),
    ("work", "Training work folder", "tokenizer, checkpoints, logs (run folders)", "medium"),
    ("work2", "Second work folder (s30m)", "tokenizer copy, checkpoints, logs", "medium"),
    ("work3", "Third work folder (s30m runs)", "checkpoints and logs of the s30m runs", "medium"),
    ("data_chat", "Prepared chat tokens", "your chats as numbers (token files)", "high"),
    ("data_pre", "Prepared public text tokens", "public web text as numbers (not private)", "low"),
    ("saved_chats", "Saved conversations", "chats you saved from the Chat page, as plain text (what you and the AI wrote)", "high"),
    ("exports", "Export packages", "a copy of the AI to use elsewhere; may contain memory and identity", "high"),
]

NEVER_LEAVES = [
    "This page and the server run only on this computer (127.0.0.1). Nothing is sent to the internet by the UI.",
    "The page loads no fonts, scripts, images or analytics from any outside site (the security header forbids it).",
    "The UI does not read the data_msgr/private/ folder or the raw Messenger exports, and it has no route to list or download them.",
    "What you type in the chat is not written to a server log. It is saved only where a feature says so (memory, archive, saved chats).",
    "The AI's words can never run a command, approve a change, edit the identity or write memory. Those steps need you.",
]

SECTIONS = [
    {"id": "chat", "label": "Chat", "stage": "U1", "icon": "chat", "ready": True},
    {"id": "memory", "label": "Memory", "stage": "U2", "icon": "graph", "ready": True},
    {"id": "mood", "label": "Mood and Person", "stage": "U2", "icon": "heart", "ready": True},
    {"id": "prompt", "label": "Personas", "stage": "U3", "icon": "layers", "ready": True},
    {"id": "data", "label": "Data", "stage": "U4", "icon": "database", "ready": True},
    {"id": "training", "label": "Training", "stage": "U5", "icon": "chart", "ready": True},
    {"id": "identity", "label": "Identity", "stage": "U6", "icon": "shield"},
    {"id": "proposals", "label": "Proposals", "stage": "U6", "icon": "inbox"},
    {"id": "models", "label": "Models and compare", "stage": "U7", "icon": "compare"},
    {"id": "export", "label": "Export", "stage": "U7", "icon": "box"},
    {"id": "settings", "label": "Settings and privacy", "stage": "U0", "icon": "lock", "ready": True},
]


def _read_static(name):
    with open(os.path.join(STATIC_DIR, name), "rb") as f:
        return f.read()


class UIState:
    def __init__(self, root, port, token=None):
        self.guard = PathGuard(root)
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self.started = time.time()
        self.chat = ChatHub(self.guard)
        self.memory = MemoryHub(self.guard)
        self.mind = MindHub(self.guard)
        self.data = DataHub(self.guard)
        self.train = TrainHub(self.guard)
        self.job = JobHub(self.guard)

    def allowed_hosts(self):
        return ("127.0.0.1:%d" % self.port, "localhost:%d" % self.port)

    def allowed_origins(self):
        return ("http://127.0.0.1:%d" % self.port, "http://localhost:%d" % self.port)

    def info(self):
        return {"name": "PersonaLM", "version": __version__, "project": os.path.basename(self.guard.root) or self.guard.root,
                "sections": SECTIONS, "stage": "U5b", "uptime_s": int(time.time() - self.started)}

    def privacy(self):
        items = []
        for rel, label, holds, level in STORAGE:
            full = None
            try:
                full = self.guard.resolve(rel)
            except UIProblem:
                pass
            items.append({"path": rel, "label": label, "holds": holds, "sensitivity": level,
                          "present": bool(full and os.path.exists(full)), "kind": ("folder" if full and os.path.isdir(full) else "file")})
        return {"root_name": os.path.basename(self.guard.root) or self.guard.root, "stored": items, "never_leaves": NEVER_LEAVES,
                "private_folder": "data_msgr/private/ is never read, listed or served by this UI"}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "persona_lm_ui"
        _body_done = False
        protocol_version = "HTTP/1.0"
        timeout = 15                                       # a client that stalls cannot hold a thread for ever

        def log_message(self, *args):                      # no per-request lines: they could show paths or message texts
            pass

        # ---- helpers
        def _headers_common(self, ctype, length, extra=None):
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            for k, v in (extra or []):
                self.send_header(k, v)

        def _drain_body(self):
            """Read and throw away any request body the handler did not read. If we answer (403, 405, 413 ...) and close while the
            client's body is still unread, Windows resets the connection and the client gets WinError 10053/10054 instead of our
            answer. Reading it first lets the answer arrive. At most DRAIN_LIMIT bytes are read; a bigger claim is not our problem."""
            if self._body_done or self.command not in ("POST", "PUT", "PATCH", "DELETE"):
                return
            self._body_done = True
            try:
                left = min(int(self.headers.get("Content-Length", "0")), DRAIN_LIMIT)
            except ValueError:
                return
            try:
                while left > 0:
                    chunk = self.rfile.read(min(left, 16384))
                    if not chunk:
                        break
                    left -= len(chunk)
            except OSError:
                pass

        def _send_bytes(self, code, body, ctype, extra=None):
            self._drain_body()
            self.send_response(code)
            self._headers_common(ctype, len(body), extra)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send_bytes(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, code, msg):
            self._json(code, {"ok": False, "error": msg})

        def _host_ok(self):
            return self.headers.get("Host", "") in state.allowed_hosts()

        def _origin_ok(self):
            origin = self.headers.get("Origin")
            return origin is None or origin in state.allowed_origins()

        def _token_ok(self, value):
            return bool(value) and hmac.compare_digest(value.encode("utf-8", "replace"), state.token.encode("utf-8"))

        def _cookie(self):
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    return v
            return ""

        def _gate(self, need_api_token):
            """Host and Origin first, for every request. Returns True when the request may go on (else an answer was sent)."""
            if not self._host_ok():
                self._error(403, "bad Host header")
                return False
            if not self._origin_ok():
                self._error(403, "bad Origin header")
                return False
            if need_api_token and not self._token_ok(self.headers.get(TOKEN_HEADER, "")):
                self._error(401, "missing or wrong token")
                return False
            return True

        # ---- GET / HEAD
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path == "/" or path == "/index.html":
                return self._index(url)
            if path in STATIC:
                if not self._gate(False):
                    return
                name, ctype = STATIC[path]
                return self._send_bytes(200, _read_static(name), ctype)
            if path.startswith("/api/"):
                if not self._gate(True):
                    return
                if path == "/api/health":
                    return self._json(200, {"ok": True})
                if path == "/api/info":
                    return self._json(200, {"ok": True, **state.info()})
                if path == "/api/privacy":
                    return self._json(200, {"ok": True, **state.privacy()})
                if path in ("/api/chat/sources", "/api/chat/state", "/api/chat/poll", "/api/chat/prompt"):
                    return self._chat_get(path, url)
                if path == "/api/personas":
                    try:
                        return self._json(200, {"ok": True, **state.chat.personas(parse_qs(url.query).get("core", [None])[0])})
                    except ChatProblem as e:
                        return self._error(e.code, str(e))
                if path in MEMORY_GET:
                    return self._memory(path, url, None)
                if path in MIND_GET:
                    return self._mind(path, None)
                if path in DATA_GET:
                    return self._data(path, url)
                if path in TRAIN_GET:
                    return self._train(path, url)
                return self._error(404, "no such route")
            if not self._gate(False):
                return
            self._error(404, "not found")

        def _memory(self, path, url, body):
            """All /api/memory/ routes. The owner's page calls these; the chat route cannot (see ui_memory.py)."""
            h = state.memory
            try:
                if path == "/api/memory/list":
                    qs = parse_qs(url.query)
                    one = lambda k, d="": qs.get(k, [d])[0]
                    try:
                        limit, offset = int(one("limit", "100")), int(one("offset", "0"))
                    except ValueError:
                        raise MemoryHubProblem("limit and offset must be whole numbers.")
                    out = h.list(q=one("q"), status=one("status", "all"), sort=one("sort", "newest"), pinned=one("pinned") == "1", limit=limit, offset=offset)
                elif path == "/api/memory/plan":
                    out = h.plan()
                else:
                    out = {"/api/memory/add": h.add, "/api/memory/edit": h.edit, "/api/memory/forget": h.forget, "/api/memory/maintain": h.maintain}[path](body)
                return self._json(200, {"ok": True, **out})
            except MemoryHubProblem as e:
                return self._error(e.code, str(e))

        def _data(self, path, url):
            """Data page: read only, counts and verdicts, never message text (see ui_data.py)."""
            h = state.data
            qs = parse_qs(url.query)
            try:
                if path == "/api/data/overview":
                    out = h.overview()
                elif path == "/api/data/audit":
                    out = h.audit(qs.get("folder", ["data_msgr"])[0])
                elif path == "/api/data/clean-preview":
                    out = h.clean_preview({k: v[0] for k, v in qs.items()})
                elif path == "/api/data/repeats":
                    out = h.repeats({k: v[0] for k, v in qs.items()}, text=qs.get("text", [""])[0] == "1")
                else:
                    out = h.report(qs.get("folder", ["data_chat"])[0], qs.get("size", ["s30m"])[0])
                return self._json(200, {"ok": True, **out})
            except DataProblem as e:
                return self._error(e.code, str(e))

        def _train(self, path, url):
            """Training page, part 1: read only (see ui_train.py). Nothing here starts a program or writes a file."""
            h = state.train
            qs = parse_qs(url.query)
            one = lambda k, d="": qs.get(k, [d])[0]
            try:
                if path == "/api/train/plan":
                    out = h.plan(one("workdir", "work"))
                elif path == "/api/train/checkpoints":
                    out = h.checkpoints()
                elif path == "/api/train/history":
                    out = h.history(one("run"))
                elif path == "/api/train/job":
                    out = state.job.status(one("workdir", "work"))
                else:
                    out = h.log(one("run"), one("file"))
                return self._json(200, {"ok": True, **out})
            except TrainProblem as e:                       # JobProblem is a TrainProblem
                return self._error(e.code, str(e))

        def _mind(self, path, body):
            """Graph, mood (read only) and Person card lines (owner edits). See ui_mind.py."""
            h = state.mind
            try:
                out = {"/api/graph": h.graph, "/api/mood": h.mood, "/api/person": h.person}.get(path) or {
                    "/api/person/add": lambda: h.add(body), "/api/person/edit": lambda: h.edit(body), "/api/person/remove": lambda: h.remove(body)}[path]
                return self._json(200, {"ok": True, **out()})
            except MindProblem as e:
                return self._error(e.code, str(e))

        def _chat_get(self, path, url):
            try:
                if path == "/api/chat/sources":
                    return self._json(200, {"ok": True, **state.chat.sources()})
                if path == "/api/chat/state":
                    return self._json(200, {"ok": True, **state.chat.state()})
                if path == "/api/chat/prompt":
                    return self._json(200, {"ok": True, **state.chat.prompt()})
                return self._json(200, {"ok": True, **state.chat.poll(parse_qs(url.query).get("job", [""])[0])})
            except ChatProblem as e:
                return self._error(e.code, str(e))

        def _index(self, url):
            if not self._gate(False):
                return
            given = parse_qs(url.query).get("t", [""])[0]
            if given:
                if not self._token_ok(given):
                    return self._error(401, "wrong token: use the link the ui command printed")
                cookie = "%s=%s; HttpOnly; SameSite=Strict; Path=/" % (COOKIE, state.token)
                return self._send_bytes(302, b"", "text/plain", [("Location", "/"), ("Set-Cookie", cookie)])
            if not self._token_ok(self._cookie()):
                body = (b"<!doctype html><meta charset=utf-8><title>PersonaLM</title><body style='font:16px system-ui;padding:2rem'>"
                        b"<h1>PersonaLM</h1><p>Open the link that the <code>ui</code> command printed in the terminal "
                        b"(it contains a one-time key for this session).</p>")
                return self._send_bytes(401, body, "text/html; charset=utf-8")
            page = _read_static(INDEX_FILE).replace(b"__TOKEN__", state.token.encode("ascii"))
            return self._send_bytes(200, page, "text/html; charset=utf-8")

        # ---- POST
        def do_POST(self):
            path = urlsplit(self.path).path
            if not self._gate(True):
                return
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._error(415, "send JSON (Content-Type: application/json)")
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error(400, "bad Content-Length")
            if n < 0 or n > MAX_BODY:
                return self._error(413, "body too large (limit %d bytes)" % MAX_BODY)
            raw = self.rfile.read(n) if n else b"{}"
            self._body_done = True
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return self._error(400, "body is not valid JSON")
            if not isinstance(body, dict):
                return self._error(400, "body must be a JSON object")
            if path == "/api/ping":                         # lets the page check that its token still works
                return self._json(200, {"ok": True})
            if path in CHAT_POST:
                try:
                    h = state.chat
                    out = {"/api/chat/load": lambda: h.load(body), "/api/chat/send": lambda: h.send(body.get("message")), "/api/chat/stop": h.stop,
                           "/api/chat/count": lambda: h.count(body.get("text")), "/api/chat/settings": lambda: h.settings(body),
                           "/api/chat/reset": h.reset, "/api/chat/save": h.save}[path]()
                    return self._json(200, {"ok": True, **out})
                except ChatProblem as e:
                    return self._error(e.code, str(e))
            if path == "/api/personas/use":
                try:
                    return self._json(200, {"ok": True, **state.chat.use_persona(body.get("name"))})
                except ChatProblem as e:
                    return self._error(e.code, str(e))
            if path == "/api/data/clean-apply":
                try:
                    return self._json(200, {"ok": True, **state.data.clean_apply(body)})
                except DataProblem as e:
                    return self._error(e.code, str(e))
            if path in TRAIN_POST:                           # U5b: a step NAME only; the arguments come from the project plan (ui_job.py)
                try:
                    return self._json(200, {"ok": True, **(state.job.start(body) if path == "/api/train/start" else state.job.stop(body))})
                except TrainProblem as e:
                    return self._error(e.code, str(e))
            if path in MEMORY_POST:
                return self._memory(path, urlsplit(self.path), body)
            if path in MIND_POST:
                return self._mind(path, body)
            return self._error(404, "no such route")

        def do_PUT(self):
            self._error(405, "method not allowed")

        do_DELETE = do_PATCH = do_PUT

    return Handler


CHAT_GET = ["/api/chat/poll", "/api/chat/prompt", "/api/chat/sources", "/api/chat/state"]
CHAT_POST = ["/api/chat/count", "/api/chat/load", "/api/chat/reset", "/api/chat/save", "/api/chat/send", "/api/chat/settings", "/api/chat/stop"]
MEMORY_GET = ["/api/memory/list", "/api/memory/plan"]
MEMORY_POST = ["/api/memory/add", "/api/memory/edit", "/api/memory/forget", "/api/memory/maintain"]
PERSONA_GET = ["/api/personas"]
PERSONA_POST = ["/api/personas/use"]
MIND_GET = ["/api/graph", "/api/mood", "/api/person"]
DATA_GET = ["/api/data/audit", "/api/data/clean-preview", "/api/data/overview", "/api/data/repeats", "/api/data/report"]
DATA_POST = ["/api/data/clean-apply"]
TRAIN_GET = ["/api/train/checkpoints", "/api/train/history", "/api/train/job", "/api/train/log", "/api/train/plan"]       # U5a read only; U5b adds job (status)
TRAIN_POST = ["/api/train/start", "/api/train/stop"]                                                                    # U5b: owner buttons, see ui_job.py
MIND_POST = ["/api/person/add", "/api/person/edit", "/api/person/remove"]
ROUTES = {"GET": ["/", "/api/health", "/api/info", "/api/privacy"] + CHAT_GET + PERSONA_GET + MEMORY_GET + MIND_GET + DATA_GET + TRAIN_GET + sorted(STATIC), "POST": ["/api/ping"] + CHAT_POST + PERSONA_POST + MEMORY_POST + MIND_POST + DATA_POST + TRAIN_POST}


def make_server(root=".", host="127.0.0.1", port=DEFAULT_PORT, token=None):
    """Returns (ThreadingHTTPServer, UIState), not started. port=0 picks a free port."""
    if host not in ("127.0.0.1", "localhost"):
        raise UIProblem("the UI only listens on this computer (127.0.0.1); %r is refused on purpose" % host)
    state = UIState(root, port, token)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    httpd.daemon_threads = True
    state.port = httpd.server_address[1]
    return httpd, state


def link(state):
    return "http://127.0.0.1:%d/?t=%s" % (state.port, state.token)


def serve(root=".", port=DEFAULT_PORT, open_browser=True, out=print):
    httpd, state = make_server(root, port=port)
    url = link(state)
    out("PersonaLM UI for the project folder: %s" % state.guard.root)
    out("Open this link (it holds a one-time key for this session; do not share it):")
    out("  " + url)
    out("It listens on 127.0.0.1 only. Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out("\nStopped.")
    finally:
        httpd.server_close()
