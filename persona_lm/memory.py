"""Stage 4: the memory graph (long-term memory that lives OUTSIDE the model's weights).

Pure Python, no NumPy, no PyTorch: this file can later move into the no-training runtime unchanged.

The picture
  * A MEMORY is one remembered fact: text, time, source, emotion tag, importance (0..1), uses, and the NODES it is about.
  * A NODE is a person, topic, event or feeling. A node is only a name; what it "knows" is the memories linked to it.
  * A LINK between two nodes is computed, never stored: two nodes are linked once per ACTIVE memory that mentions both.
    (So links can never go stale after a memory is forgotten, merged or archived.)
  * Memories fade: strength = importance * 0.5 ** (idle days / half-life). Using a memory (it was recalled in a chat) resets its idle
    clock. Pinned memories never fade. Fading never deletes: a weak memory becomes "dormant" (hidden from recall, kept in the file).

Rules that matter for later stages
  * Only the OWNER's typed commands add or remove memories. Nothing the model writes can (Stage 7 will add proposals that need approval).
  * Nothing is deleted except by `forget` (the owner's explicit command). Maintenance (merge duplicates, fade, summarise) is a PLAN first;
    it changes nothing until `apply`.
  * The change log (`<file>.log.jsonl`) records what changed and when, never the memory texts.
  * Retrieval is keyword based (words, names, one hop along links). There are no embeddings: honest and simple, but it will not find
    a memory that shares no words with the question.
"""
import copy
import json
import math
import os
import re
import shutil
import time
from calendar import timegm
from collections import Counter, defaultdict

FORMAT_VERSION = 1
KINDS = ("person", "topic", "event", "feeling")
STATUSES = ("active", "dormant", "archived", "merged")          # only active memories are recalled; the others stay in the file
HALF_LIFE_DAYS = 90.0
MAX_TEXT = 2000
DORMANT_BELOW = 0.05

# Very common words carry no meaning for matching (English + the most common Tagalog/Bisaya function words).
STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been am i me my you your he she it we they them his her "
    "its our their this that these those do does did not no so too very can could will would should have has had what who when where why how "
    "ang ng sa na ay ko mo ka ako ikaw siya kami tayo kayo sila mga si ni kay at pa po nga lang din rin daw raw ba yung yong "
    "og ug nga ang sa si ni kay ako ikaw siya kami kita kamo sila mao naa wala dili na pa lang man ra pud".split())

_WIKI = re.compile(r"\[\[([^\[\]|]{1,60}?)\]\]")
_WORD = re.compile(r"\w+", re.UNICODE)
_EMOTION = re.compile(r"^[a-z][a-z0-9-]{0,23}$")


class MemoryProblem(Exception):
    """Something the owner can fix (bad input, unreadable file). The message is written for the owner."""


class MemoryConflict(MemoryProblem):
    """The file was changed by another program after it was loaded; saving now would overwrite that change."""


def now_iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_time(s):
    """'2026-11-14T10:00:00Z' or '2026-11-14' -> seconds since 1970 (UTC)."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return float(timegm(time.strptime(s, fmt)))
        except (ValueError, TypeError):
            continue
    raise MemoryProblem("Cannot read the date %r. Use 2026-11-14 or 2026-11-14T10:30:00Z." % (s,))


def words_of(text, drop_stop=True):
    ws = [w for w in _WORD.findall(text.casefold()) if len(w) >= 2 or w.isdigit()]
    return [w for w in ws if w not in STOPWORDS] if drop_stop else ws


def clean_name(name):
    name = " ".join(str(name).split())
    if not name or len(name) > 60:
        raise MemoryProblem("A node name must be 1 to 60 characters (got %r)." % name[:70])
    if any(c in name for c in "[]|\n"):
        raise MemoryProblem("A node name cannot contain [ ] or |.")
    return name


def node_id(kind, name):
    return "%s:%s" % (kind, clean_name(name).casefold())


def first_sentence(text, limit=160):
    m = re.search(r"[.!?](\s|$)", text)
    s = text[:m.end()].strip() if m else text.strip()
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


class MemoryStore:
    """One memory file. `path=None` keeps everything in RAM (tests, previews)."""

    def __init__(self, path=None, clock=time.time):
        self.path = path
        self.clock = clock
        self.data = {"format_version": FORMAT_VERSION, "next_memory": 1, "nodes": {}, "memories": {}}
        self._sig = None
        if path and os.path.exists(path):
            self._load()

    # ------------------------------------------------------------------ file handling
    def _signature(self):
        try:
            st = os.stat(self.path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            prev = self.path[:-5] + ".prev.json" if self.path.endswith(".json") else self.path + ".prev"
            raise MemoryProblem("Cannot read the memory file %s (%s). Nothing was changed. %s" % (
                self.path, e, ("A backup of the previous save may exist: " + prev) if os.path.exists(prev) else ""))
        if not isinstance(data, dict) or not isinstance(data.get("memories"), dict) or not isinstance(data.get("nodes"), dict):
            raise MemoryProblem("%s is not a memory file (missing 'nodes' or 'memories')." % self.path)
        if data.get("format_version", 0) > FORMAT_VERSION:
            raise MemoryProblem("%s was written by a newer version of this program (format %s, this one reads up to %d)." % (
                self.path, data.get("format_version"), FORMAT_VERSION))
        data.setdefault("next_memory", 1 + max([int(k[1:]) for k in data["memories"] if re.fullmatch(r"m\d+", k)] or [0]))
        data["format_version"] = FORMAT_VERSION
        self.data = data
        self._sig = self._signature()

    def reload(self):
        """Read the file again (used after another part of the program, e.g. an approved proposal, changed it)."""
        if self.path and os.path.exists(self.path):
            self._load()

    def _backup_path(self):
        return self.path[:-5] + ".prev.json" if self.path.endswith(".json") else self.path + ".prev"

    def _log_path(self):
        return self.path[:-5] + ".log.jsonl" if self.path.endswith(".json") else self.path + ".log.jsonl"

    def save(self, keep_backup=True):
        """Atomic write (temp file, then rename). The previous version is kept as <file>.prev.json for one-step undo by hand.
        Refuses if another program changed the file since it was loaded."""
        if not self.path:
            return
        if self._signature() != self._sig:
            raise MemoryConflict("%s was changed by another program after it was loaded. Nothing was saved. "
                                 "Close the other program (or restart this one) and try again." % self.path)
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        if keep_backup and os.path.exists(self.path):
            shutil.copy2(self.path, self._backup_path())
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
        if not keep_backup and os.path.exists(self._backup_path()):
            os.remove(self._backup_path())               # `forget` really forgets: the old text must not survive in the backup
        self._sig = self._signature()

    def log(self, action, ids=(), note=None):
        """Append to the change log. Never writes memory texts."""
        if not self.path:
            return
        row = {"time": now_iso(self.clock()), "action": action, "ids": list(ids)}
        if note:
            row["note"] = note
        with open(self._log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ nodes
    @property
    def nodes(self):
        return self.data["nodes"]

    @property
    def memories(self):
        return self.data["memories"]

    def find_node(self, name):
        """Node whose name matches (any kind), or None."""
        key = clean_name(name).casefold()
        for n in self.nodes.values():
            if n["name"].casefold() == key:
                return n
        return None

    def ensure_node(self, name, kind="topic"):
        if kind not in KINDS:
            raise MemoryProblem("Unknown kind %r. Kinds: %s." % (kind, ", ".join(KINDS)))
        name = clean_name(name)
        nid = node_id(kind, name)
        if nid not in self.nodes:
            self.nodes[nid] = {"id": nid, "kind": kind, "name": name, "created": now_iso(self.clock())}
        return self.nodes[nid]

    def _resolve_links(self, text):
        """[[Name]] / [[kind:Name]] in `text` -> (clean text without the brackets, [node ids]). Existing node names that appear as whole
        words are linked too. New nodes named by [[...]] are created (the caller commits them only if the memory is accepted)."""
        explicit = []

        def take(m):
            raw = m.group(1).strip()
            kind, name = None, raw
            if ":" in raw and raw.split(":", 1)[0].strip().lower() in KINDS:
                kind, name = raw.split(":", 1)[0].strip().lower(), raw.split(":", 1)[1].strip()
            name = clean_name(name)
            explicit.append((kind, name))
            return name
        clean = _WIKI.sub(take, text)
        ids = []
        for kind, name in explicit:
            existing = self.find_node(name) if kind is None else None
            node = existing or self.ensure_node(name, kind or "topic")
            if node["id"] not in ids:
                ids.append(node["id"])
        for n in self.nodes.values():                                    # plain mentions of nodes we already know
            if n["id"] in ids or len(n["name"]) < 2:
                continue
            if re.search(r"(?<!\w)" + re.escape(n["name"]) + r"(?!\w)", clean, re.IGNORECASE):
                ids.append(n["id"])
        return clean, ids

    # ------------------------------------------------------------------ memories
    def add(self, text, emotion="neutral", importance=0.5, source="owner", when=None, pinned=False, nodes=None):
        """Remember something. `nodes`: extra [kind:name or name] to link. Returns the new memory dict."""
        text = " ".join(str(text).split())
        if not text:
            raise MemoryProblem("There is nothing to remember: the text is empty.")
        if len(text) > MAX_TEXT:
            raise MemoryProblem("That memory is %d characters; the limit is %d. Split it into several memories." % (len(text), MAX_TEXT))
        emotion = (emotion or "neutral").strip().lower()
        if not _EMOTION.match(emotion):
            raise MemoryProblem("The emotion tag must be one lowercase word (letters, digits, '-'), up to 24 characters, e.g. joy, sadness, stress.")
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            raise MemoryProblem("Importance must be a number from 0 to 1.")
        if not 0 <= importance <= 1:
            raise MemoryProblem("Importance must be a number from 0 to 1 (0.5 is normal, 0.9 is very important).")
        when_t = self.clock() if when is None else (when if isinstance(when, (int, float)) else parse_time(when))
        backup = copy.deepcopy(self.data["nodes"])
        try:
            clean, ids = self._resolve_links(text)
            for extra in (nodes or []):
                kind, name = None, extra
                if ":" in extra and extra.split(":", 1)[0].strip().lower() in KINDS:
                    kind, name = extra.split(":", 1)[0].strip().lower(), extra.split(":", 1)[1]
                node = (self.find_node(name) if kind is None else None) or self.ensure_node(name, kind or "topic")
                if node["id"] not in ids:
                    ids.append(node["id"])
        except MemoryProblem:
            self.data["nodes"] = backup                                      # a rejected memory leaves no half-made nodes
            raise
        mid = "m%04d" % self.data["next_memory"]
        self.data["next_memory"] += 1
        mem = {"id": mid, "text": clean, "timestamp": now_iso(when_t), "source": source, "emotion": emotion,
               "importance": round(importance, 3), "nodes": ids, "uses": 0, "last_used": now_iso(when_t),
               "status": "active", "pinned": bool(pinned)}
        self.memories[mid] = mem
        return mem

    def get(self, mid):
        mid = mid.strip().lower()
        if re.fullmatch(r"\d+", mid):
            mid = "m%04d" % int(mid)
        if mid not in self.memories:
            raise MemoryProblem("There is no memory %s." % mid)
        return self.memories[mid]

    def forget(self, mid):
        """Delete one memory for good (the owner's explicit command). Nodes left with no memory are removed too.
        A summary that was built from it keeps its own text (it is a separate memory); `forget` that too if needed."""
        mem = self.get(mid)
        del self.memories[mem["id"]]
        for m in self.memories.values():
            if m.get("summary_of") and mem["id"] in m["summary_of"]:
                m["summary_of"] = [x for x in m["summary_of"] if x != mem["id"]]
            if m.get("merged_into") == mem["id"]:
                m["merged_into"] = None
        self.prune_nodes()
        return mem["id"]

    def prune_nodes(self):
        used = {nid for m in self.memories.values() for nid in m["nodes"]}
        for nid in [n for n in self.nodes if n not in used]:
            del self.nodes[nid]

    def set_importance(self, mid, importance, pinned=None):
        mem = self.get(mid)
        if not 0 <= float(importance) <= 1:
            raise MemoryProblem("Importance must be a number from 0 to 1.")
        mem["importance"] = round(float(importance), 3)
        if pinned is not None:
            mem["pinned"] = bool(pinned)
        return mem

    def edit(self, mid, text=None, emotion=None):
        """Change a memory's text and/or emotion tag (the owner's explicit edit). The links to nodes are worked out again from the new
        text; nodes left with no memory are removed. The caller saves."""
        mem = self.get(mid)
        if emotion is not None:
            emotion = emotion.strip().lower()
            if not _EMOTION.match(emotion):
                raise MemoryProblem("The emotion tag must be one lowercase word (letters, digits, '-'), up to 24 characters, e.g. joy, sadness, stress.")
        if text is not None:
            text = " ".join(str(text).split())
            if not text:
                raise MemoryProblem("The text is empty. To remove a memory, forget it instead.")
            if len(text) > MAX_TEXT:
                raise MemoryProblem("That memory is %d characters; the limit is %d. Split it into several memories." % (len(text), MAX_TEXT))
            backup = copy.deepcopy(self.data["nodes"])
            try:
                clean, ids = self._resolve_links(text)
            except MemoryProblem:
                self.data["nodes"] = backup
                raise
            mem["text"], mem["nodes"] = clean, ids
            self.prune_nodes()
        if emotion is not None:
            mem["emotion"] = emotion
        return mem

    # ------------------------------------------------------------------ strength, links
    def strength(self, mem, now=None, half_life=HALF_LIFE_DAYS):
        if mem.get("pinned"):
            return mem["importance"]
        now = self.clock() if now is None else now
        idle_days = max(0.0, (now - max(parse_time(mem["last_used"]), parse_time(mem["timestamp"]))) / 86400.0)
        return mem["importance"] * 0.5 ** (idle_days / half_life)

    def active(self):
        return [m for m in self.memories.values() if m["status"] == "active"]

    def edges(self, now=None, half_life=HALF_LIFE_DAYS, include=("active",)):
        """{(node_a, node_b): {'count': memories shared, 'weight': sum of their strengths}} with node_a < node_b."""
        out = defaultdict(lambda: {"count": 0, "weight": 0.0})
        for m in self.memories.values():
            if m["status"] not in include:
                continue
            s = self.strength(m, now, half_life)
            ns = sorted(set(m["nodes"]))
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    e = out[(ns[i], ns[j])]
                    e["count"] += 1
                    e["weight"] += s
        return dict(out)

    def node_scores(self, now=None, half_life=HALF_LIFE_DAYS):
        """How big a node's circle is: the strength of its active memories plus a quarter of its link weights."""
        score = Counter()
        for m in self.active():
            s = self.strength(m, now, half_life)
            for nid in m["nodes"]:
                score[nid] += s
        for (a, b), e in self.edges(now, half_life).items():
            score[a] += 0.25 * e["weight"]
            score[b] += 0.25 * e["weight"]
        return {nid: round(score.get(nid, 0.0), 4) for nid in self.nodes}

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query, k=5, budget_tokens=None, count_tokens=None, now=None, half_life=HALF_LIFE_DAYS):
        """Memories that help answer `query`, best first: [(memory, score, why)]. Changes nothing (see `touch`).
        Score = relevance x (0.5 + strength). Relevance: query words found in the memory (rarer words count more), +1 for a node the query
        names, +0.3 for a memory about a node linked to a named node. A memory with zero relevance is never returned.
        budget_tokens + count_tokens(str)->int: stop adding memories when the lines would use more than the budget."""
        now = self.clock() if now is None else now
        q_words = set(words_of(query))
        active = self.active()
        if not active or not (q_words or self.nodes):
            return []
        mem_words = {m["id"]: set(words_of(m["text"])) for m in active}
        df = Counter(w for ws in mem_words.values() for w in ws)
        n = len(active)
        q_nodes = {nid for nid, node in self.nodes.items() if re.search(r"(?<!\w)" + re.escape(node["name"]) + r"(?!\w)", query, re.IGNORECASE)}
        neighbours = set()
        if q_nodes:
            for (a, b) in self.edges(now, half_life):
                if a in q_nodes and b not in q_nodes:
                    neighbours.add(b)
                elif b in q_nodes and a not in q_nodes:
                    neighbours.add(a)
        scored = []
        for m in active:
            ws = mem_words[m["id"]]
            common = q_words & ws
            rel = sum(math.log(1 + n / (1 + df[w])) for w in common) / math.sqrt(max(len(ws), 1))
            why = []
            if common:
                why.append("words: " + ", ".join(sorted(common)))
            hit = [nid for nid in m["nodes"] if nid in q_nodes]
            if hit:
                rel += 1.0
                why.append("about " + ", ".join(self.nodes[h]["name"] for h in hit))
            near = [nid for nid in m["nodes"] if nid in neighbours]
            if near and not hit:
                rel += 0.3
                why.append("linked to " + ", ".join(self.nodes[h]["name"] for h in near))
            if rel <= 0:
                continue
            scored.append((rel * (0.5 + self.strength(m, now, half_life)), m, "; ".join(why)))
        scored.sort(key=lambda t: (-t[0], t[1]["id"]))
        out, used = [], 0
        for score, m, why in scored:
            if len(out) >= k:
                break
            if budget_tokens is not None and count_tokens is not None:
                cost = count_tokens(self.line_for_prompt(m))
                if used + cost > budget_tokens:
                    continue
                used += cost
            out.append((m, round(score, 4), why))
        return out

    @staticmethod
    def line_for_prompt(mem):
        return "- " + mem["text"]

    def touch(self, ids, now=None):
        """The memories were used in a reply: count the use and restart their fading clock."""
        now = self.clock() if now is None else now
        for mid in ids:
            if mid in self.memories:
                self.memories[mid]["uses"] += 1
                self.memories[mid]["last_used"] = now_iso(now)

    # ------------------------------------------------------------------ maintenance (a plan first, then apply)
    def plan_maintenance(self, now=None, half_life=HALF_LIFE_DAYS, dormant_below=DORMANT_BELOW, dup_threshold=0.8,
                         cluster_min=6, cluster_min_age_days=30):
        """Returns what maintenance WOULD do. Changes nothing.
          fade:      active, unpinned memories whose strength fell below `dormant_below` -> dormant
          revive:    dormant, unpinned memories used since (strength back above 2x the limit) -> active
          duplicates:[(keep, drop, similarity)] same nodes and nearly the same words -> the weaker one is marked merged
          summaries: nodes with >= cluster_min old active memories -> one EXTRACTIVE summary memory; the originals become archived"""
        now = self.clock() if now is None else now
        fade = [m["id"] for m in self.active() if not m.get("pinned") and self.strength(m, now, half_life) < dormant_below]
        revive = [m["id"] for m in self.memories.values() if m["status"] == "dormant" and not m.get("pinned")
                  and self.strength(m, now, half_life) >= 2 * dormant_below]
        dups, gone = [], set(fade)
        groups = defaultdict(list)
        for m in self.active():
            if m["id"] not in gone and not m.get("summary_of"):
                groups[tuple(sorted(m["nodes"]))].append(m)
        for group in groups.values():
            sets = {m["id"]: set(words_of(m["text"])) for m in group}
            group.sort(key=lambda m: (-m["importance"], m["id"]))
            for i, a in enumerate(group):
                if a["id"] in gone or len(sets[a["id"]]) < 3:
                    continue
                for b in group[i + 1:]:
                    if b["id"] in gone or len(sets[b["id"]]) < 3:
                        continue
                    inter = len(sets[a["id"]] & sets[b["id"]])
                    sim = inter / float(len(sets[a["id"]] | sets[b["id"]]))
                    if sim >= dup_threshold:
                        dups.append((a["id"], b["id"], round(sim, 3)))
                        gone.add(b["id"])
        sums, taken = [], set(gone)
        per_node = defaultdict(list)
        for m in self.active():
            if m["id"] in taken or m.get("pinned") or m.get("summary_of"):
                continue
            if (now - parse_time(m["timestamp"])) / 86400.0 < cluster_min_age_days:
                continue
            for nid in m["nodes"]:
                per_node[nid].append(m)
        for nid, ms in sorted(per_node.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            ms = [m for m in ms if m["id"] not in taken]
            if len(ms) < cluster_min:
                continue
            ms.sort(key=lambda m: (-m["importance"], m["id"]))
            times = sorted(m["timestamp"][:10] for m in ms)
            text = "Summary of %d memories about %s (%s to %s): %s" % (
                len(ms), self.nodes[nid]["name"], times[0], times[-1], " ".join(first_sentence(m["text"]) for m in ms[:3]))
            if len(text) > MAX_TEXT:
                text = text[:MAX_TEXT - 1] + "…"
            sums.append({"node": nid, "ids": [m["id"] for m in ms], "text": text})
            taken.update(m["id"] for m in ms)
        return {"fade": fade, "revive": revive, "duplicates": dups, "summaries": sums}

    def apply_maintenance(self, plan):
        """Carry out a plan from plan_maintenance. Returns a short report. The caller saves."""
        done = {"fade": 0, "revive": 0, "merged": 0, "summaries": 0}
        for mid in plan["fade"]:
            if mid in self.memories and self.memories[mid]["status"] == "active":
                self.memories[mid]["status"] = "dormant"
                done["fade"] += 1
        for mid in plan["revive"]:
            if mid in self.memories and self.memories[mid]["status"] == "dormant":
                self.memories[mid]["status"] = "active"
                done["revive"] += 1
        for keep, drop, _ in plan["duplicates"]:
            if keep in self.memories and drop in self.memories and self.memories[drop]["status"] == "active":
                k, d = self.memories[keep], self.memories[drop]
                k["importance"] = max(k["importance"], d["importance"])
                k["uses"] += d["uses"]
                k["pinned"] = bool(k.get("pinned") or d.get("pinned"))
                k["last_used"] = max(k["last_used"], d["last_used"])
                d["status"], d["merged_into"] = "merged", keep
                done["merged"] += 1
        for s in plan["summaries"]:
            ids = [i for i in s["ids"] if i in self.memories and self.memories[i]["status"] == "active"]
            if len(ids) < 2:
                continue
            emo = Counter(self.memories[i]["emotion"] for i in ids).most_common(1)[0][0]
            mem = self.add(s["text"], emotion=emo, importance=max(self.memories[i]["importance"] for i in ids), source="summary",
                           when=self.clock(), nodes=[self.nodes[s["node"]]["kind"] + ":" + self.nodes[s["node"]]["name"]])
            mem["summary_of"] = ids
            for i in ids:
                self.memories[i]["status"] = "archived"
                self.memories[i]["archived_into"] = mem["id"]
            done["summaries"] += 1
        return done

    # ------------------------------------------------------------------ overview text for the chat screen and CLI
    def overview(self, now=None, top=5):
        now = self.clock() if now is None else now
        counts = Counter(m["status"] for m in self.memories.values())
        scores = self.node_scores(now)
        lines = ["%d memories (%s), %d nodes (%s)." % (
            len(self.memories), ", ".join("%d %s" % (counts[s], s) for s in STATUSES if counts[s]) or "none",
            len(self.nodes), ", ".join("%d %s" % (c, k) for k, c in sorted(Counter(n["kind"] for n in self.nodes.values()).items())) or "none")]
        best = sorted(self.nodes.values(), key=lambda n: (-scores[n["id"]], n["id"]))[:top]
        if best:
            lines.append("Biggest nodes: " + ", ".join("%s (%s)" % (n["name"], n["kind"]) for n in best))
        return "\n".join(lines)

    def format_memory(self, mem, now=None):
        now = self.clock() if now is None else now
        return "%s [%s, %s, importance %.2f, strength %.2f%s%s] %s%s" % (
            mem["id"], mem["status"], mem["emotion"], mem["importance"], self.strength(mem, now), ", pinned" if mem.get("pinned") else "",
            ", used %dx" % mem["uses"] if mem["uses"] else "", mem["text"],
            ("  -> " + ", ".join(self.nodes[n]["name"] for n in mem["nodes"] if n in self.nodes)) if mem["nodes"] else "")


# ---------------------------------------------------------------------- parsing of the /remember line
def parse_remember(line):
    """'Maria loves mangoes #joy *0.8 !pin' -> (text, {'emotion': 'joy', 'importance': 0.8, 'pinned': True}).
    The options are taken only from the END of the line; anything else stays in the text."""
    toks = line.split()
    opts = {}
    while toks:
        t = toks[-1]
        if re.fullmatch(r"#[A-Za-z][A-Za-z0-9-]{0,23}", t) and "emotion" not in opts:
            opts["emotion"] = t[1:].lower()
        elif re.fullmatch(r"\*\d*\.?\d+", t) and "importance" not in opts:
            opts["importance"] = float(t[1:])
        elif t.lower() in ("!pin", "!pinned") and "pinned" not in opts:
            opts["pinned"] = True
        else:
            break
        toks.pop()
    return " ".join(toks), opts
