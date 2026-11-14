"""A small local web server around a loaded AI (Stage 9b). Python standard library only (http.server, json): no extra dependency, no PyTorch.

    python -m persona_lm serve --package my_ai_package            # http://127.0.0.1:8765

Endpoints (JSON in, JSON out):
    GET  /health            {"ok": true}                                   (needs no token)
    GET  /info              name, version, persona, personas, mood label, whether memory is on
    POST /chat              {"message": "hello"}  ->  {"reply": "...", "notes": [...], "mood": "joy" | null, "persona": "..." | null}
    POST /reset             forgets the conversation (memory and mood files stay)
    POST /persona           {"name": "calm mentor"}  (or "none"); OFF unless the server was started with --allow-persona-switch

Rules:
  * It listens on this computer only (127.0.0.1) unless you give another --host AND a --token (then every request except /health must send
    the header  Authorization: Bearer <token> ). There is no encryption (plain HTTP): do not put it on the internet.
  * ONE conversation per server (one AI, one history). Requests are handled one at a time.
  * The text in /chat is a plain message, never a command (same as runtime.chat). There is no endpoint to approve, reject, edit memory or edit the identity.
  * The AI's words can never switch a persona; /persona is a separate request from your program, and it is off by default.
  * Nothing the person says or the AI replies is written to any log by the server (only startup lines and errors are printed).
"""
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .runtime import RuntimeProblem

LOOPBACK = ("127.0.0.1", "localhost", "::1")
MAX_BODY = 65536
MAX_MESSAGE = 4000


class ServerProblem(Exception):
    pass


def make_server(ai, host="127.0.0.1", port=8765, token=None, allow_persona_switch=False):
    """Returns a ThreadingHTTPServer (not started). port=0 picks a free port (server.server_address[1])."""
    if host not in LOOPBACK and not token:
        raise ServerProblem("Listening on %r (not just this computer) needs --token SECRET, because anyone who can reach the port could talk to your AI." % host)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "persona_lm"
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):                      # silence the default per-request line: it must never carry message texts
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if not token:
                return True
            got = self.headers.get("Authorization", "")
            return got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode("utf-8"), token.encode("utf-8"))

        def _body(self):
            """The JSON object of the request, or None after an error answer has been sent."""
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(411, {"error": "Content-Length is required."})
                return None
            if n < 0 or n > MAX_BODY:
                # Read and throw away a moderately oversized body before answering. If we answer first and
                # close with unread data, Windows resets the connection and the client never sees the 413.
                left = n if 0 < n <= 16 * MAX_BODY else 0
                while left > 0:
                    chunk = self.rfile.read(min(left, 65536))
                    if not chunk:
                        break
                    left -= len(chunk)
                self._send(413, {"error": "The request is too big (at most %d bytes)." % MAX_BODY})
                return None
            try:
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(obj, dict):
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "The body must be a JSON object."})
                return None
            return obj

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if not self._authorized():
                return self._send(401, {"error": "A valid token is needed."})
            if self.path == "/info":
                with lock:
                    return self._send(200, {"name": ai.name, "version": ai.version, "persona": ai.persona,
                                            "personas": ai.personas.names() if ai.personas is not None else [],
                                            "mood": ai.mood.label if ai.mood is not None else None, "memory": ai.memory is not None,
                                            "persona_switch_allowed": allow_persona_switch})
            self._send(404, {"error": "Unknown path."})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": "A valid token is needed."})
            if self.path not in ("/chat", "/reset", "/persona"):
                return self._send(404, {"error": "Unknown path."})
            obj = self._body()
            if obj is None:
                return
            with lock:
                if self.path == "/reset":
                    ai.reset()
                    return self._send(200, {"ok": True})
                if self.path == "/persona":
                    if not allow_persona_switch:
                        return self._send(403, {"error": "Switching personas over HTTP is off. Start the server with --allow-persona-switch to allow it."})
                    name = obj.get("name")
                    if not isinstance(name, str) or not name.strip():
                        return self._send(400, {"error": "Send {\"name\": \"persona name\"} (or \"none\")."})
                    try:
                        return self._send(200, {"ok": True, "message": ai.switch_persona(name), "persona": ai.persona})
                    except RuntimeProblem as e:
                        return self._send(422, {"error": str(e)})
                msg = obj.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    return self._send(400, {"error": "Send {\"message\": \"your text\"}."})
                if len(msg) > MAX_MESSAGE:
                    return self._send(413, {"error": "The message is too long (at most %d characters)." % MAX_MESSAGE})
                try:
                    reply = ai.chat(msg)
                except RuntimeProblem as e:
                    return self._send(422, {"error": str(e)})
                self._send(200, {"reply": reply, "notes": list(ai.last_notes), "mood": ai.mood.label if ai.mood is not None else None, "persona": ai.persona})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def serve(ai, host="127.0.0.1", port=8765, token=None, allow_persona_switch=False, out=print):
    srv = make_server(ai, host, port, token, allow_persona_switch)
    out("Serving %s on http://%s:%d  (Ctrl+C stops it). Plain HTTP, one conversation. %s" % (
        ai.name or "the AI", host, srv.server_address[1], "Token required." if token else "This computer only."))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        out("Stopped.")
    finally:
        srv.server_close()
