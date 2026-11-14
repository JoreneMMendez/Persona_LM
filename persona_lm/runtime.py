"""The runtime library (Stage 9a): load a finished AI and talk to it from your own Python program. No PyTorch, no training code.

    import persona_lm.runtime as rt
    ai = rt.load("identity_core")            # a folder made by `identity init` (uses its current version, tokenizer, memory, mood, person card)
    print(ai.chat("hello"))                  # the reply, as text
    ai.memory                                # the MemoryStore (or None)
    ai.mood                                  # the Mood (or None)
    ai.switch_persona("calm mentor")         # puts a mask on (logged); "none" takes it off

Or without an identity core:  rt.load(ckpt="run/ckpt.npz", tokenizer="tok.json", system="You are Luna.")

What this file imports: numpy, the tokenizer, the model's forward pass (inference only), the KV cache, and the chat/memory/mood/persona/identity
code. It does NOT import PyTorch, the training loop, the optimizer, the data tools or the command-line program (tests/test_stage9a.py checks that).

Safety rules (the same as in the chat screen):
  * chat(text) sends the text to the AI as a plain message. It is NEVER read as a command: a message that starts with a slash is just a message.
    (The chat screen's /approve, /persona and so on exist only in the typed chat; this library has no approve or reject at all.)
  * The AI's own words never change anything but the conversation: they cannot switch a persona, move the identity, edit memory or approve a proposal.
    It may FILE a proposal by writing a [propose ...] line (when the identity core is loaded); filed proposals only wait. Approve them with the
    typed `proposal` / chat commands in a terminal.
  * switch_persona is a method of YOUR program (the owner's side). It writes to the lineage log, and it never touches the identity core.
"""
import os

from . import emotion as EMO
from . import identity as ID
from . import person as PER
from . import personas as PERS
from . import proposals as PRO
from .bpe import Tokenizer
from .chatrepl import ChatSession
from .ckpt import load_ckpt
from .fastgen import sample_ids_cached
from .generate import sample_ids
from .memory import MemoryStore
from .model import Config


class RuntimeProblem(Exception):
    """Something the caller should read: a bad path, a refused message, an empty reply."""


class AI:
    """A loaded AI. Make one with load(). One AI = one conversation (reset() starts a new one)."""

    def __init__(self, session, core=None, personas=None, tok=None, cfg=None, version=None):
        self.session, self.core, self.personas, self.tok, self.cfg, self.version = session, core, personas, tok, cfg, version
        self.last_notes = []             # extra lines the chat screen would print after a reply (what was recalled, mood, window notes)

    # -- talking
    def chat(self, message):
        """Sends one message and returns the AI's reply text. Raises RuntimeProblem if the message was refused (too long) or the reply was empty."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("chat() needs some text")
        s = self.session
        before = len(s.history)
        shown = s.say(message.strip())                       # say(), never handle(): the text is never read as a command
        if len(s.history) != before + 2 or s.history[-1][0] != "assistant":
            raise RuntimeProblem(shown)                      # refused or empty: the session already put everything back as it was
        reply = s.history[-1][1]
        lines = shown.split("\n")
        self.last_notes = [ln for ln in lines[1:] if ln.strip()] if lines and lines[0].startswith("ai> ") else []
        return reply

    def reset(self):
        """Forgets the conversation (not the memory file, not the mood file)."""
        self.session.command("/reset")

    @property
    def history(self):
        return list(self.session.history)

    # -- the parts
    @property
    def memory(self):
        return self.session.memory

    @property
    def mood(self):
        return self.session.mood

    @property
    def person(self):
        return self.session.person

    @property
    def name(self):
        return self.core.sections["Name"] if self.core is not None else None

    @property
    def persona(self):
        return self.personas.active if self.personas is not None else None

    def switch_persona(self, name):
        """Owner-side switch of the mask (a name, or "none"). Returns a sentence. Logged; the identity core is not touched."""
        if self.personas is None:
            raise RuntimeProblem("This AI was loaded without an identity core, so it has no personas. Load it with load(core_folder).")
        try:
            return self.personas.use(name)
        except PERS.PersonaProblem as e:
            raise RuntimeProblem(str(e))

    def pending_proposals(self):
        """Proposals the AI has filed and that wait for the owner: a list of dicts (read only). Approving needs the typed `proposal approve` command."""
        if self.session.proposals is None:
            return []
        return [dict(p) for p in self.session.proposals.listing("pending")]


def _from_core(core, version):
    """Version name + checkpoint path from an identity core. The checkpoint and tokenizer are checked against the checksums in the core."""
    name = version or core.current
    if not name:
        raise RuntimeProblem("The identity core has no current version yet. Register one (identity add-version) or pass version=... or ckpt=...")
    core.verify_version_files(name)
    return name, core.abs(core.version(name)["ckpt"])


def load(path=None, *, ckpt=None, tokenizer=None, version=None, memory=None, mood=None, person=None, system=None,
         max_new=120, temperature=0.8, top_k=40, seed=0, fast=True, memory_tokens=None, person_tokens=None):
    """Loads an AI. `path` is an identity core folder (or leave it out and give ckpt= and tokenizer=). Anything you pass overrides the core.

    memory / mood / person: file paths. With a core they default to the core's own files. fast=False uses the plain generator (slower, same answers).
    Raises RuntimeProblem with a plain sentence when something is wrong."""
    core = personas = name = None
    try:
        if path is not None:
            if os.path.isfile(path):
                raise RuntimeProblem("%s is a file; pass the identity core FOLDER, or use ckpt=... and tokenizer=..." % path)
            core = ID.Core(path)
            personas = PERS.Personas(core)
            if not ckpt:
                if personas.active and not version:
                    version = personas.version_for(personas.active)          # a version trained for the mask that is on (None = the current one)
                name, ckpt = _from_core(core, version)
            else:
                name = None
            tokenizer = tokenizer or core.resolve("tokenizer")
            for key, given in (("memory", memory), ("mood", mood), ("person", person)):
                if not given and core.resolve(key) and (key == "mood" or os.path.exists(core.resolve(key))):
                    if key == "memory":
                        memory = core.resolve(key)
                    elif key == "mood":
                        mood = core.resolve(key)
                    else:
                        person = core.resolve(key)
        if not ckpt or not tokenizer:
            raise RuntimeProblem("load() needs an identity core folder, or both ckpt= and tokenizer=.")
        store = MemoryStore(memory) if memory else None
        mood_obj = tagger = card = None
        if mood:
            mood_obj = EMO.Mood(mood, baseline=core.baseline() if core else (0.1, 0.0))
            if core:
                mood_obj.baseline = list(core.baseline())
            tagger = EMO.Tagger()
        if person:
            card = PER.load(person)
        got = load_ckpt(ckpt)
        if got is None:
            raise RuntimeProblem("Cannot read the checkpoint %s." % ckpt)
        params, cfg, tok = got[0], Config.from_dict(got[2]["cfg"]), Tokenizer.load(tokenizer)
    except (ID.IdentityProblem, PERS.PersonaProblem, PER.PersonProblem, EMO.EmotionProblem, OSError, ValueError, KeyError) as e:
        raise RuntimeProblem(str(e))
    except Exception as e:                                                   # memory/lineage problems have their own classes
        if type(e).__name__ in ("MemoryProblem", "LineageProblem"):
            raise RuntimeProblem(str(e))
        raise
    sampler = sample_ids_cached if fast else sample_ids
    if system is None and core is not None:
        count = lambda text: len(tok.encode(text, allow_special=False))
        system, _left_out = ID.system_line(core.sections, count, max(((cfg.ctx - max_new) // 2) - 4, 8))
    session = ChatSession(tok, cfg.ctx, lambda ids, **kw: sampler(params, cfg, ids, **kw), system=system, save_dir=".",
                          settings={"max_new": max_new, "temperature": temperature, "top_k": top_k}, seed=seed,
                          memory=store, memory_tokens=memory_tokens, mood=mood_obj, tagger=tagger, person=card, person_tokens=person_tokens,
                          identity=core, personas=personas,
                          proposals=PRO.Proposals(core, memory_path=memory, person_path=person) if core is not None else None)
    return AI(session, core=core, personas=personas, tok=tok, cfg=cfg, version=name)
