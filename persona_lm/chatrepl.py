"""The chat screen (Stage 3): a conversation loop with slash commands.

Design rules (they matter for Stage 7, when the AI may PROPOSE changes):
  * Commands are read ONLY from what the person types. The model's replies are never looked at for commands, so nothing the AI
    writes (or anything in its training data) can run /approve, /reject or any other command.
  * /memory /mood /identity /approve /reject exist already but are honest stubs: they say which stage builds them and change nothing.
  * The class does no keyboard reading itself (`handle(line)` takes a string), so tests never wait for a keyboard.
"""
import os
import time

from .bpe import ASSISTANT, END, ROLE_ID

# command -> (stage that builds it, what it will do)
PLANNED = {
    "/memory":   (4, "show what the AI remembers (people, topics, events) and why"),
    "/mood":     (5, "show the AI's current emotional state"),
    "/identity": (6, "show the AI's identity card (name, personality, values) and its version history"),
    "/approve":  (7, "approve a change the AI has PROPOSED to its memory, settings or training"),
    "/reject":   (7, "reject a change the AI has proposed"),
}

HELP = """Commands (type them at the  you>  prompt):
  /help             this list
  /reset            forget the conversation and start again
  /history          show the conversation so far
  /save [file]      write the conversation to a text file on this computer (never overwrites)
  /system <text>    set the system line (how the AI is introduced to itself) and start again
  /temp <0-2>       randomness of the replies (0 = always the most likely word, 0.8 = default)
  /max <n>          longest reply in tokens
  /quit             leave
  //text            send a message that starts with a slash, e.g. //shrug
Not built yet (they only explain what they will do): /memory /mood /identity /approve /reject"""


class ChatSession:
    """One conversation. generate(prompt_ids, seed=..., max_new=..., temperature=..., top_k=...) -> list of new token ids."""

    def __init__(self, tok, ctx, generate, system=None, save_dir=".", settings=None, seed=0, clock=time.time):
        self.tok, self.ctx, self.generate = tok, ctx, generate
        self.system = system
        self.save_dir = save_dir
        self.settings = {"max_new": 120, "temperature": 0.8, "top_k": 40}
        self.settings.update(settings or {})
        self.seed = seed
        self.clock = clock
        self.turn = 0
        self.history = []                       # [(role, text)], the whole conversation (the model only sees the newest part that fits)
        self.last_dropped = 0
        self._told_about_window = False

    # ------------------------------------------------------------ what the model sees
    def _message_ids(self, role, text):
        return [ROLE_ID[role]] + self.tok.encode(text, allow_special=False) + [END]      # allow_special=False: typed text can never forge a role token

    def prompt_budget(self):
        return max(self.ctx - self.settings["max_new"], self.ctx // 2)

    def build_prompt(self):
        """System line first, then as many of the NEWEST whole messages as fit. Returns (ids, dropped_message_count) or (None, reason)."""
        budget = self.prompt_budget() - 1                             # 1 = the final "assistant" marker
        system_ids = self._message_ids("system", self.system) if self.system else []
        if len(system_ids) > budget // 2:
            return None, "The system line is too long (%d tokens; at most %d)." % (len(system_ids), budget // 2)
        room = budget - len(system_ids)
        keep = []
        for role, text in reversed(self.history):
            ids = self._message_ids(role, text)
            if len(ids) > room:
                break
            keep.append(ids)
            room -= len(ids)
        if not keep:
            last = self._message_ids(*self.history[-1])
            return None, "That message is too long for the model's window (%d tokens; at most %d). Send something shorter." % (len(last), budget - len(system_ids))
        dropped = len(self.history) - len(keep)
        ids = list(system_ids)
        for m in reversed(keep):
            ids += m
        return ids + [ASSISTANT], dropped

    # ------------------------------------------------------------ one line typed by the person
    def handle(self, line):
        """Returns (text_to_show or None, quit?)."""
        line = line.strip()
        if not line:
            return None, False
        if line.startswith("//"):
            return self.say(line[1:]), False
        if line.startswith("/"):
            return self.command(line)
        return self.say(line), False

    def say(self, text):
        self.history.append(("user", text))
        ids, info = self.build_prompt()
        if ids is None:
            self.history.pop()
            return info
        self.last_dropped = info
        reply = self.tok.decode(self.generate(ids, seed=self.seed + self.turn, **self.settings)).strip()
        self.turn += 1
        if not reply:
            self.history.pop()
            return "(the model produced an empty reply: just send your message again, or try /temp 1.0)"
        self.history.append(("assistant", reply))
        out = "ai> " + reply
        if info and not self._told_about_window:
            self._told_about_window = True
            out += "\n(note: the AI's window is %d tokens, so it no longer sees the oldest %d message(s) of this conversation. /history still shows everything.)" % (self.ctx, info)
        return out

    def command(self, line):
        name, _, arg = line.partition(" ")
        name, arg = name.lower(), arg.strip()
        if name in ("/quit", "/exit"):
            return None, True
        if name == "/help":
            return HELP, False
        if name == "/reset":
            self.history, self.turn, self._told_about_window = [], 0, False
            return "Conversation cleared.", False
        if name == "/history":
            if not self.history:
                return "(nothing yet)", False
            return "\n".join(("you> " if r == "user" else "ai> " if r == "assistant" else "[system] ") + t for r, t in
                             ([("system", self.system)] if self.system else []) + self.history), False
        if name == "/system":
            if not arg:
                return "Current system line: %s" % (self.system or "(none)"), False
            if len(self._message_ids("system", arg)) > (self.prompt_budget() - 1) // 2:
                return "That system line is too long for the model's window. Nothing changed.", False
            self.system = arg
            self.history, self.turn, self._told_about_window = [], 0, False
            return "System line set; conversation cleared.", False
        if name == "/temp":
            try:
                v = float(arg)
                if not 0 <= v <= 2:
                    raise ValueError
            except ValueError:
                return "Use a number from 0 to 2, e.g. /temp 0.8 (now: %s)." % self.settings["temperature"], False
            self.settings["temperature"] = v
            return "temperature = %s" % v, False
        if name == "/max":
            try:
                v = int(arg)
                if not 1 <= v <= self.ctx - 8:
                    raise ValueError
            except ValueError:
                return "Use a whole number from 1 to %d, e.g. /max 120 (now: %d)." % (self.ctx - 8, self.settings["max_new"]), False
            self.settings["max_new"] = v
            return "longest reply = %d tokens" % v, False
        if name == "/save":
            return self.save(arg), False
        if name in PLANNED:
            stage, what = PLANNED[name]
            msg = "%s is not built yet. It is planned for Stage %d: it will %s. Nothing was changed." % (name, stage, what)
            if name in ("/approve", "/reject"):
                msg += " (When it exists, approvals will be read only from your keyboard, never from anything the AI writes.)"
            return msg, False
        return "Unknown command %s. Type /help. (To send a message that starts with a slash, start it with two slashes.)" % name, False

    def save(self, arg):
        if not self.history:
            return "Nothing to save yet."
        name = arg or time.strftime("chat_%Y%m%d_%H%M%S.txt", time.localtime(self.clock()))
        path = name if os.path.isabs(name) else os.path.join(self.save_dir, name)
        if os.path.exists(path):
            return "%s already exists; nothing was overwritten. Pick another name: /save other_name.txt" % path
        lines = []
        if self.system:
            lines.append("[system] " + self.system)
        lines += [("you: " if r == "user" else "ai: ") + t for r, t in self.history]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            return "Could not write %s (%s)." % (path, e)
        return "Saved %s (it stays on this computer; it contains what you wrote)." % os.path.abspath(path)


def run_repl(session, input_fn=None, out=print):
    """The loop. input_fn is a parameter so tests can type for the person; None = the real keyboard.
    (input is looked up when the loop runs, not when this file loads, so a test guard that replaces it is always seen.)"""
    if input_fn is None:
        input_fn = input
    out("Chat ready. Type a message, /help for commands, /quit to leave.")
    while True:
        try:
            line = input_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            out("")
            return
        text, quit_ = session.handle(line)
        if text:
            out(text)
        if quit_:
            return
