"""The chat screen (Stage 3): a conversation loop with slash commands.

Design rules (they matter for Stage 7, when the AI may PROPOSE changes):
  * Commands are read ONLY from what the person types. The model's replies are never looked at for commands, so nothing the AI
    writes (or anything in its training data) can run /approve, /reject or any other command.
  * /proposals /approve /reject (Stage 7b) work when the chat was started with --core DIR. The AI may FILE a proposal by writing a line like
        [propose memory-add] Maria loves mango #joy *0.8 | why: you said it twice
    in its reply (at most 3 per reply, duplicates skipped, anything invalid ignored). That is all its words can do: they never approve, reject, undo or
    run a command. /approve ID shows the exact change and then needs you to TYPE the number again; it works only when the chat is run in a terminal
    window (not from a pipe or a script). Undo is `python -m persona_lm proposal undo ID`.
  * /identity (Stage 6) shows the identity core when the chat was started with --core DIR. It only READS: protected changes can be made only
    with the `identity` commands, typed in a terminal, never from inside the chat and never from anything the AI writes.
  * /mood (Stage 5) works when the chat was started with --mood FILE. Only what the person TYPES moves the mood (each message is tagged by the
    emotion tagger); the model's own words never do, a refused or empty-reply turn leaves the mood as it was. The mood goes into the prompt as one
    block  <|mood|> label valence arousal <|end|>  after the system line, and nudges the sampling temperature by at most 15 %.
  * --person FILE (Stage 5) puts the owner's Person card into the prompt as  <|speaker|> Name: ... <|end|>  (whole facts only, within a budget).
  * /memory /remember /forget (Stage 4) work only when the chat was started with a memory file (--memory). Memories are added or removed only
    by these typed commands; the AI's own words never reach the memory file. Recalled memories are put into the prompt after the system
    line as one block  <|memory|> - fact - fact <|end|>  and cost part of the window.
  * The class does no keyboard reading itself (`handle(line)` takes a string), so tests never wait for a keyboard.
"""
import os
import sys
import time

from .bpe import ASSISTANT, END, MEMORY, MOOD, ROLE_ID, SPEAKER
from . import emotion as EMO
from . import person as PER
from . import identity as ID
from . import proposals as PRO
from .memory import MemoryConflict, MemoryProblem, parse_remember


HELP = """Commands (type them at the  you>  prompt):
  /help             this list
  /reset            forget the conversation and start again
  /history          show the conversation so far
  /memory           (needs --memory) what the AI remembers; /memory search <words> shows what it would recall for those words
  /remember <text>  (needs --memory) remember something, e.g.  /remember [[person:Maria]] is my sister #love *0.9 !pin
                    ([[Name]] or [[kind:Name]] makes a node; #emotion, *importance 0-1 and !pin go at the END)
  /forget <id>      (needs --memory) delete one memory for good, e.g. /forget m0003
  /mood             (needs --mood) the AI's current feeling; /mood tag <text> shows how a text would be read; /mood reset goes back to the baseline
  /identity         (needs --core) the AI's identity card, its versions and recent history (read only)
  /proposals        (needs --core) changes the AI has proposed and that wait for you
  /approve <id>     (needs --core) look at a proposed change, then type its number to apply it (undo: the `proposal undo` command)
  /reject <id>      (needs --core) throw a proposed change away
  /save [file]      write the conversation to a text file on this computer (never overwrites)
  /system <text>    set the system line (how the AI is introduced to itself) and start again
  /temp <0-2>       randomness of the replies (0 = always the most likely word, 0.8 = default)
  /max <n>          longest reply in tokens
  /quit             leave
  //text            send a message that starts with a slash, e.g. //shrug
The AI may propose changes by writing a line  [propose KIND] ... | why: ...  in its reply; it can never approve them."""


class ChatSession:
    """One conversation. generate(prompt_ids, seed=..., max_new=..., temperature=..., top_k=...) -> list of new token ids."""

    def __init__(self, tok, ctx, generate, system=None, save_dir=".", settings=None, seed=0, clock=time.time, memory=None, memory_tokens=None,
                 mood=None, tagger=None, person=None, person_tokens=None, identity=None, proposals=None):
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
        self.memory = memory                    # a MemoryStore, or None (memory switched off)
        self.memory_tokens = memory_tokens if memory_tokens is not None else max(16, ctx // 5)
        self.last_recalled = []                 # memory ids put into the newest prompt
        self.last_unfit = 0                     # how many matching memories were left out because they did not fit the budget
        self.mood = mood                        # an emotion.Mood, or None (mood switched off)
        self.tagger = tagger or (EMO.Tagger() if mood is not None else None)
        self.last_tag = None                    # how the newest typed message was read
        self.person = person                    # a person.Card, or None
        self.person_tokens = person_tokens if person_tokens is not None else max(16, ctx // 6)
        self.last_person_left = 0                # card facts that did not fit the budget
        self._told_about_card = False
        self.identity = identity                # an identity.Core, or None (no identity core in this chat)
        self.proposals = proposals              # a proposals.Proposals, or None (proposals switched off)
        self._awaiting = None                   # ("approve", "p0001"): the next typed line must be that number

    # ------------------------------------------------------------ what the model sees
    def _message_ids(self, role, text):
        return [ROLE_ID[role]] + self.tok.encode(text, allow_special=False) + [END]      # allow_special=False: typed text can never forge a role token

    def prompt_budget(self):
        return max(self.ctx - self.settings["max_new"], self.ctx // 2)

    def _memory_block(self, room):
        """<|memory|> - fact - fact <|end|> for the memories that fit the newest user message, or [] (also sets self.last_recalled).
        It never takes more than a third of the room left after the system line, so the conversation itself still fits."""
        self.last_recalled, self.last_unfit = [], 0
        if self.memory is None or not self.history:
            return []
        cap = min(self.memory_tokens, room // 3) - 2                       # 2 = the <|memory|> and <|end|> markers
        query = " ".join(t for r, t in self.history[-3:] if r == "user")      # the last user message(s): what the person is talking about now
        found = self.memory.retrieve(query, k=6, budget_tokens=max(cap, 0), count_tokens=lambda line: len(self.tok.encode(line + "\n", allow_special=False)),
                                     now=self.clock()) if cap > 0 else []
        if not found:
            self.last_unfit = len(self.memory.retrieve(query, k=6, now=self.clock()))     # matches that exist but did not fit
            return []
        self.last_recalled = [m["id"] for m, _, _ in found]
        text = "\n".join(self.memory.line_for_prompt(m) for m, _, _ in found)
        return [MEMORY] + self.tok.encode(text, allow_special=False) + [END]

    def _speaker_block(self, room):
        """<|speaker|> Name: ... <|end|> from the Person card; whole facts only; at most a quarter of the room left after the system line."""
        self.last_person_left = 0
        if self.person is None:
            return []
        cap = min(self.person_tokens, room // 4) - 2
        if cap <= 0:
            self.last_person_left = sum(len(self.person.facts(t, ("owner", "auto"))) for t in PER.PROMPT_ORDER)
            return []
        text, used, left = PER.prompt_text(self.person, lambda s: len(self.tok.encode(s, allow_special=False)), cap)
        self.last_person_left = left
        return [SPEAKER] + self.tok.encode(text, allow_special=False) + [END] if used else []

    def _mood_block(self):
        if self.mood is None:
            return []
        return [MOOD] + self.tok.encode(self.mood.prompt_text(), allow_special=False) + [END]

    def build_prompt(self):
        """System line first, then as many of the NEWEST whole messages as fit. Returns (ids, dropped_message_count) or (None, reason)."""
        budget = self.prompt_budget() - 1                             # 1 = the final "assistant" marker
        system_ids = self._message_ids("system", self.system) if self.system else []
        if len(system_ids) > budget // 2:
            return None, "The system line is too long (%d tokens; at most %d)." % (len(system_ids), budget // 2)
        speaker_ids = self._speaker_block(budget - len(system_ids))
        mood_ids = self._mood_block()
        memory_ids = self._memory_block(budget - len(system_ids) - len(speaker_ids) - len(mood_ids))
        room = budget - len(system_ids) - len(speaker_ids) - len(mood_ids) - len(memory_ids)
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
        ids = list(system_ids) + speaker_ids + mood_ids + memory_ids
        for m in reversed(keep):
            ids += m
        return ids + [ASSISTANT], dropped

    # ------------------------------------------------------------ one line typed by the person
    def handle(self, line):
        """Returns (text_to_show or None, quit?)."""
        line = line.strip()
        if self._awaiting:                                          # a typed confirmation is pending: this line answers it, nothing else does
            return self.confirm(line), False
        if not line:
            return None, False
        if line.startswith("//"):
            return self.say(line[1:]), False
        if line.startswith("/"):
            return self.command(line)
        return self.say(line), False

    def say(self, text):
        self.history.append(("user", text))
        snap = None
        if self.mood is not None:                                   # the typed message moves the mood BEFORE the prompt is built, so the AI sees it
            snap = self.mood.snapshot()
            self.last_tag = self.tagger.tag(text)
            self.mood.update(self.last_tag, now=self.clock())
        ids, info = self.build_prompt()
        if ids is None:
            self.history.pop()
            if snap is not None:
                self.mood.restore(snap)                               # a refused message changes nothing
            return info
        self.last_dropped = info
        settings = dict(self.settings)
        if self.mood is not None:
            settings["temperature"] = round(min(2.0, settings["temperature"] * self.mood.temperature_factor()), 4) if settings["temperature"] > 0 else 0
        reply = self.tok.decode(self.generate(ids, seed=self.seed + self.turn, **settings)).strip()
        self.turn += 1
        if not reply:
            self.history.pop()
            if snap is not None:
                self.mood.restore(snap)
            return "(the model produced an empty reply: just send your message again, or try /temp 1.0)"
        self.history.append(("assistant", reply))
        out = "ai> " + reply
        if self.proposals is not None:                              # the AI's words may only FILE proposals (never decide them)
            try:
                for prop, note in PRO.submit_from_reply(self.proposals, reply):
                    out += "\n" + (note or "(the AI proposed %s: %s. Nothing has changed. /approve %s to look at it, /reject %s to drop it)" % (
                        prop["id"], prop["kind"], prop["id"], prop["id"]))
            except OSError as e:
                out += "\n(could not save a proposal: %s)" % e
        if self.mood is not None:
            out += "\n(feeling now: %s; /mood shows more)" % self.mood.label
            try:
                self.mood.save()
            except OSError as e:
                out += "\n(could not save the mood file: %s)" % e
        if self.last_unfit and not self.last_recalled:
            out += "\n(note: %d memor%s matched but did not fit in the memory budget of %d tokens, so the AI did not see %s. " \
                   "A bigger window or --memory-tokens helps.)" % (self.last_unfit, "y" if self.last_unfit == 1 else "ies", self.memory_tokens,
                                                                   "it" if self.last_unfit == 1 else "them")
        if self.last_recalled:
            out += "\n(recalled: %s; /memory shows them)" % ", ".join(self.last_recalled)
            self.memory.touch(self.last_recalled, now=self.clock())
            try:
                self.memory.save()
            except (MemoryConflict, OSError) as e:
                out += "\n(could not save the memory file: %s)" % e
        if self.last_person_left and not self._told_about_card:
            self._told_about_card = True
            out += "\n(note: %d line(s) of your Person card did not fit the card budget of %d tokens, so the AI did not see them. --person-tokens or a bigger window helps; the most useful lines come first.)" % (
                self.last_person_left, self.person_tokens)
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
        if name in ("/memory", "/remember", "/forget"):
            return self.memory_command(name, arg), False
        if name == "/mood":
            return self.mood_command(arg), False
        if name == "/identity":
            if self.identity is None:
                return "No identity core in this chat. Start it with:  python -m persona_lm chat --core identity_core ...   (make one with: identity init)", False
            return self.identity.describe(), False
        if name in ("/proposals", "/approve", "/reject"):
            return self.proposal_command(name, arg), False
        return "Unknown command %s. Type /help. (To send a message that starts with a slash, start it with two slashes.)" % name, False

    def proposal_command(self, name, arg):
        if self.proposals is None:
            return "No proposal store in this chat. Start it with:  python -m persona_lm chat --core identity_core ...   Nothing was changed."
        pr = self.proposals
        try:
            if name == "/proposals":
                rows = pr.listing("pending")
                return "\n".join("%s  %-18s %s" % (p["id"], p["kind"], p["why"][:70]) for p in rows) if rows else "No proposals are waiting."
            if not arg:
                return "Say which one, e.g. %s p0001  (/proposals lists them)." % name
            p = pr.get(arg.split()[0])
            if name == "/reject":
                if ID.REQUIRE_TTY and not (sys.stdin and sys.stdin.isatty()):
                    return "Decisions are accepted only when you type in a terminal window (not from a pipe or a script). Nothing was changed."
                return pr.reject(p["id"])
            if ID.REQUIRE_TTY and not (sys.stdin and sys.stdin.isatty()):
                return "Approvals are accepted only when you type in a terminal window (not from a pipe or a script). Nothing was changed."
            if p["status"] != "pending":
                return "%s is %s, not waiting." % (p["id"], p["status"])
            self._awaiting = ("approve", p["id"])
            return pr.describe(p) + "\nType  %s  to apply this change (anything else cancels; nothing has changed yet):" % p["id"]
        except (PRO.ProposalProblem, ID.IdentityProblem, MemoryProblem, PER.PersonProblem, OSError) as e:
            return "%s Nothing was changed." % e

    def confirm(self, line):
        kind, pid = self._awaiting
        self._awaiting = None
        if line.strip().lower() != pid:
            return "Not confirmed. Nothing was changed."
        lines = []
        try:
            msg = self.proposals.approve(pid, ask=lambda prompt: line, out=lines.append)
        except (PRO.ProposalProblem, ID.IdentityProblem, MemoryProblem, PER.PersonProblem, OSError) as e:
            return "%s Nothing was changed." % e
        if self.memory is not None and self.proposals.memory_path and os.path.abspath(self.memory.path or "") == os.path.abspath(self.proposals.memory_path):
            self.memory.reload()                                    # an approved memory change was made by another store object: read it back
        if self.person is not None and self.proposals.person_path:
            try:
                self.person = PER.load(self.proposals.person_path)
            except PER.PersonProblem:
                pass
        return msg

    def mood_command(self, arg):
        if self.mood is None:
            return "Mood is not switched on for this chat. Start it with:  python -m persona_lm chat ... --mood mood.json"
        sub, _, rest = arg.partition(" ")
        sub = sub.lower()
        if sub == "tag":
            if not rest.strip():
                return "Usage: /mood tag <text>   (shows how that text would be read; changes nothing)"
            return self.tagger.tag(rest).describe()
        if sub == "reset":
            self.mood.reset()
            try:
                self.mood.save()
            except OSError as e:
                return "Back at the baseline, but the mood file could not be saved: %s" % e
            return "Mood reset to the baseline.\n" + self.mood.describe()
        if sub:
            return "Usage: /mood   or   /mood tag <text>   or   /mood reset"
        lines = [self.mood.describe()]
        if self.last_tag is not None:
            lines.append("Your last message was read as: " + self.last_tag.describe())
        lines.append("In the prompt the AI sees:  <|mood|> %s <|end|>.  Real changes in tone need a model trained with mood blocks (not done yet); " % self.mood.prompt_text()
                     + "until then the only effect is a small change of the sampling temperature.")
        return "\n".join(lines)

    def memory_command(self, name, arg):
        if self.memory is None:
            return "Memory is not switched on for this chat. Start it with:  python -m persona_lm chat ... --memory memory.json"
        try:
            if name == "/remember":
                if not arg:
                    return "Usage: /remember <text>   e.g. /remember [[person:Maria]] is my sister #love *0.9 !pin"
                text, opts = parse_remember(arg)
                mem = self.memory.add(text, **opts)
                self.memory.log("add", [mem["id"]])
                self.memory.save()
                return "Remembered as %s. %s" % (mem["id"], self.memory.format_memory(mem, self.clock()))
            if name == "/forget":
                if not arg:
                    return "Usage: /forget <id>   e.g. /forget m0003 (see ids with /memory search <words>)"
                mid = self.memory.forget(arg)
                self.memory.log("forget", [mid])
                self.memory.save(keep_backup=False)
                return "Forgot %s for good." % mid
            # /memory and /memory search <words>
            sub, _, rest = arg.partition(" ")
            if sub.lower() == "search":
                if not rest.strip():
                    return "Usage: /memory search <words>"
                found = self.memory.retrieve(rest, k=8, now=self.clock())
                if not found:
                    return "Nothing in memory matches those words."
                return "\n".join("%s   (score %.2f: %s)" % (self.memory.format_memory(m, self.clock()), sc, why) for m, sc, why in found)
            lines = [self.memory.overview(self.clock())]
            if self.last_recalled:
                lines.append("Used in the last reply:")
                lines += ["  " + self.memory.format_memory(self.memory.memories[i], self.clock()) for i in self.last_recalled if i in self.memory.memories]
            lines.append("(/memory search <words>, /remember <text>, /forget <id>)")
            return "\n".join(lines)
        except MemoryConflict as e:
            return "Not saved: %s" % e
        except MemoryProblem as e:
            return str(e)

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
