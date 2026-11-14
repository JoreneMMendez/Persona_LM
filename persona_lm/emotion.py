"""Emotion (Stage 5): a tagger that guesses the feeling in a message, and a mood that moves a little with every message.

Plain words for the owner
  * TAGGER: reads one message and guesses ONE of ten feelings (joy, love, amusement, surprise, calm, neutral, tired, sadness, fear, anger),
    how sure it is, and where the feeling sits on two scales: valence (unpleasant -1 ... pleasant +1) and arousal (sleepy -1 ... excited +1).
    Step 1 is a hand-made word list (English, Tagalog, Bisaya, emoji). Step 2 is a small learner (naive Bayes: it counts which words appear
    in messages of each feeling) trained on YOUR messages, so it picks up your own slang. The data has no feeling labels, so the learner starts
    from the word list's guesses ("weak labels") and from the labels you correct by hand (`emotion review`). Corrections count 5 times more.
    The word lists are my best knowledge, never tested on real chats: expect mistakes, and fix them with `--lexicon` and `emotion review`.
  * MOOD: two numbers (valence, arousal) saved in mood.json. Each message you type pushes them a little toward that message's feeling;
    between messages they drift back to the baseline. The label is simply the nearest feeling on the two scales.
  * Only what you TYPE changes the mood. The AI's own words never do (same rule as memory and commands).

Pure Python (no NumPy, no PyTorch).
"""
import json
import math
import os
import re
import time
from collections import Counter

FORMAT_VERSION = 1

# label -> (valence, arousal)
LABELS = {
    "joy":       (0.8, 0.5),
    "love":      (0.9, 0.2),
    "amusement": (0.6, 0.5),
    "surprise":  (0.2, 0.8),
    "calm":      (0.4, -0.5),
    "neutral":   (0.0, 0.0),
    "tired":     (-0.3, -0.7),
    "sadness":   (-0.7, -0.4),
    "fear":      (-0.45, 0.8),
    "anger":     (-0.75, 0.55),
}
LABEL_ORDER = list(LABELS)                    # fixed order = fixed tie-breaks = the same answer on every machine

# label -> (strong words: weight 1.0, weak words: weight 0.5). Lower case; elongated spellings ("soooo") are folded before the lookup.
# Tagalog and Bisaya entries are the author's best knowledge and have NOT been checked against real chats. Words that mean something else in
# the other language (for example Tagalog "suko" = give up, "sayang" = what a pity) are left out on purpose.
LEXICON = {
    "joy": (
        "happy glad yay yey yehey yeheyy excited awesome amazing congrats congratulations hooray woohoo masaya saya tuwa natuwa nakakatuwa "
        "malipay nalipay kalipay lipay lipayon grateful blessed great 😀 😄 😁 😊 🥳 🎉 😃".split(),
        "nice good thanks thank salamat maayo mabuti ayos goods yes oo 👍 🙂".split()),
    "love": (
        "love loved miss missed missing mahal iloveyou luv gihigugma palangga pangga namiss gimiss sweet ❤ ❤️ 💕 😍 🥰 😘 💖 💗 💓 😚".split(),
        "care caring crush hug hugs yakap".split()),
    "amusement": (
        "lol lmao rofl charot hilarious funny nakakatawa tawa katawa makatawa lingaw makalingaw 😂 🤣 😆 😹".split(),
        "joke jokes kidding jk 😜 😝 😅".split()),
    "surprise": (
        "wow omg omfg gulat nagulat hala nahingangha unexpected shocked shock 😲 😮 😯 🤯".split(),
        "grabe really seryoso seriously huh what 😳".split()),
    "calm": (
        "calm relax relaxed peaceful chill kalma kalmado tahimik pahuway".split(),
        "okay ok oks okie fine sige noted 😌".split()),
    "tired": (
        "tired sleepy exhausted burnout pagod pagal antok inaantok puyat kapoy gikapoy nakakapagod nakapoy kakapoy 😴 🥱 😪 😩".split(),
        "hays haist hayst sleep tulog matulog drained".split()),
    "sadness": (
        "sad cry crying cried lonely heartbroken malungkot lungkot nalulungkot iyak umiyak naiyak nakakalungkot nasaktan masakit luoy nalooy "
        "naguol guol nasakitan depressed down 😢 😭 😞 😔 💔 😿".split(),
        "hurt sorry pain 🥺 ☹ ☹️ 🙁".split()),
    "fear": (
        "scared afraid nervous anxious worry worried panic takot natatakot kinakabahan kaba kabado nakulba kulba kahadlok nahadlok hadlok "
        "pangamba anxiety 😨 😰 😱 😟".split(),
        "uneasy unsure nagalala nagaalala".split()),
    "anger": (
        "angry mad annoyed annoying hate hated pissed furious galit nagalit inis naiinis badtrip bwisit bwiset buwisit nasuko nalagot "
        "naglagot kasuko nakakainis nakakagalit nakakabadtrip irritated 😡 😠 🤬 😤".split(),
        "stupid ugh tsk tss irita".split()),
}
LEX_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_NEGATORS = frozenset("not no never dont cant isnt wasnt arent werent wont didnt doesnt hindi hnd di dili wala walay".split())
_WORD = re.compile(r"[\w\u2764\uFE0F]+|[^\w\s]", re.UNICODE)         # words, or any single other symbol (emoji travel one by one)
_LAUGH = re.compile(r"^(?:ha|he|hi|ho|ja|je)+$")
_REPEAT3 = re.compile(r"(.)\1{2,}")
_EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF))


def _is_symbol_char(ch):
    o = ord(ch)
    return any(a <= o <= b for a, b in _EMOJI_RANGES) or ch in "\u2764\uFE0F\u2639"


def tokens_of(text):
    """Lower-case words and single emoji. Apostrophes are dropped so that don't and dont are one word."""
    text = text.casefold().replace("\u2019", "'").replace("'", "")
    out = []
    for t in _WORD.findall(text):
        if len(t) == 1 and not t.isalnum() and not _is_symbol_char(t):
            continue                                                  # plain punctuation is not a token
        out.append(t)
    return out


def _variants(w):
    """The word as typed, then with stretched letters squeezed to two, then to one: 'soooo' -> 'soo' -> 'so'."""
    yield w
    two = _REPEAT3.sub(lambda m: m.group(1) * 2, w)
    if two != w:
        yield two
    one = _REPEAT3.sub(lambda m: m.group(1), w)
    if one not in (w, two):
        yield one
    if two != w and re.search(r"(.)\1", two):
        yield re.sub(r"(.)\1", r"\1", two)


def _squeezed(w):
    """The most squeezed spelling of a word (every run of repeated letters is cut to one): 'hahaha' stays, 'soooo' -> 'so'."""
    return list(_variants(w))[-1]


def build_lookup(extra=None):
    """word -> [(label, weight)]. `extra` is {label: [words]} from your own file (strong words)."""
    lookup = {}
    for label, (strong, weak) in LEXICON.items():
        for w in strong:
            lookup.setdefault(w, []).append((label, 1.0))
        for w in weak:
            lookup.setdefault(w, []).append((label, 0.5))
    for label, words in (extra or {}).items():
        for w in words:
            lookup.setdefault(str(w).casefold(), []).append((label, 1.0))
    return lookup


def load_lexicon_file(path):
    """JSON like {"anger": ["inis", "bwisit"], "joy": ["lipay"]}: extra strong words. Unknown labels are refused."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise EmotionProblem("Cannot read the word list file %s (%s)." % (path, e))
    if not isinstance(data, dict):
        raise EmotionProblem("%s must look like {\"anger\": [\"word\", ...], \"joy\": [...]}." % path)
    for label, words in data.items():
        if label not in LABELS or not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            raise EmotionProblem("%s: %r is not a known feeling with a list of words. Feelings: %s." % (path, label, ", ".join(LABEL_ORDER)))
    return data


class EmotionProblem(Exception):
    pass


class Tag:
    """The guess for one message."""
    def __init__(self, label, confidence, scores, evidence, negated=0):
        self.label, self.confidence, self.scores, self.evidence, self.negated = label, confidence, scores, evidence, negated

    @property
    def valence(self):
        return LABELS[self.label][0]

    @property
    def arousal(self):
        return LABELS[self.label][1]

    def describe(self):
        why = ", ".join(self.evidence) if self.evidence else "no known feeling words"
        return "%s (sure %.0f%%; valence %+.1f, arousal %+.1f; %s%s)" % (
            self.label, 100 * self.confidence, self.valence, self.arousal, why, "; ignored %d negated word(s)" % self.negated if self.negated else "")


def lexicon_scores(text, lookup):
    """-> (scores {label: weight sum}, evidence [words], negated count). A feeling word right after a 'not' is ignored, not flipped."""
    toks = tokens_of(text)
    scores, evidence, negated = Counter(), [], 0
    for i, t in enumerate(toks):
        hits = None
        for v in _variants(t):
            if v in lookup:
                hits = lookup[v]
                break
        if hits is None and len(t) >= 4 and _LAUGH.match(t):
            hits = [("amusement", 1.0)]
        if hits is None:
            continue
        if any(p in _NEGATORS for p in toks[max(0, i - 2):i]):
            negated += 1
            continue
        for label, w in hits:
            scores[label] += w
        evidence.append(t)
    return scores, evidence, negated


def _blend_from_lexicon(scores):
    """Word-list scores -> a distribution over the ten feelings. Leftover doubt goes to neutral, so one weak hit stays unsure."""
    total, doubt = sum(scores.values()), 0.6                  # `doubt` is a pretend piece of evidence for neutral: one strong word wins, one weak word does not
    dist = {l: scores.get(l, 0.0) / (total + doubt) for l in LABEL_ORDER}
    dist["neutral"] += doubt / (total + doubt)
    return dist


# ---------------------------------------------------------------------------------------------- the learner (naive Bayes)
def features_of(text):
    """Words, plus 3-letter pieces of longer words (so 'nalulungkot' and 'lungkot' share pieces)."""
    feats = []
    for t in tokens_of(text):
        t = _squeezed(t)
        feats.append("w:" + t)
        if len(t) >= 5 and t.isalpha():
            padded = "^" + t + "$"
            feats += ["c:" + padded[i:i + 3] for i in range(len(padded) - 2)]
    return feats


class NaiveBayes:
    def __init__(self, data=None):
        self.data = data or {"format_version": FORMAT_VERSION, "kind": "naive-bayes", "labels": LABEL_ORDER, "class_weight": {}, "feature_counts": {},
                             "totals": {}, "trained_on": {}}

    @classmethod
    def train(cls, rows, min_count=2, alpha=0.5):
        """rows: [(text, label, weight)]. Features seen fewer than `min_count` times in total are dropped (names, typos, one-offs)."""
        class_weight, counts = Counter(), {l: Counter() for l in LABEL_ORDER}
        for text, label, weight in rows:
            if label not in LABELS:
                raise EmotionProblem("unknown feeling %r" % label)
            class_weight[label] += weight
            for f in set(features_of(text)):
                counts[label][f] += weight
        seen = Counter()
        for l in LABEL_ORDER:
            for f, c in counts[l].items():
                seen[f] += c
        keep = {f for f, c in seen.items() if c >= min_count}
        fc = {l: {f: round(c, 3) for f, c in counts[l].items() if f in keep} for l in LABEL_ORDER if class_weight[l] > 0}
        model = cls({"format_version": FORMAT_VERSION, "kind": "naive-bayes", "labels": LABEL_ORDER, "alpha": alpha,
                     "class_weight": {l: round(class_weight[l], 3) for l in LABEL_ORDER if class_weight[l] > 0},
                     "feature_counts": fc, "totals": {l: round(sum(fc[l].values()), 3) for l in fc}, "vocab": len(keep),
                     "trained_on": {"messages": len(rows)}})
        return model

    def probabilities(self, text):
        d = self.data
        classes = list(d["class_weight"])
        if not classes:
            return {l: (1.0 if l == "neutral" else 0.0) for l in LABEL_ORDER}
        total = sum(d["class_weight"].values())
        feats = set(features_of(text))
        logp = {}
        for l in classes:
            lp = math.log(d["class_weight"][l] / total)
            fc, tot = d["feature_counts"][l], d["totals"][l]
            denom = tot + d["alpha"] * (d["vocab"] + 1)
            for f in feats:
                if any(f in d["feature_counts"][o] for o in classes):          # only features the model knows: unknown ones tell nothing
                    lp += math.log((fc.get(f, 0.0) + d["alpha"]) / denom)
            logp[l] = lp
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return {l: (math.exp(logp[l] - m) / z if l in logp else 0.0) for l in LABEL_ORDER}

    def save(self, path):
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(path):
            _copy(path, path[:-5] + ".prev.json" if path.endswith(".json") else path + ".prev")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise EmotionProblem("Cannot read the emotion model %s (%s)." % (path, e))
        if not isinstance(data, dict) or data.get("kind") != "naive-bayes":
            raise EmotionProblem("%s is not an emotion model made by `emotion train`." % path)
        if data.get("format_version", 0) > FORMAT_VERSION:
            raise EmotionProblem("%s was written by a newer version of this program." % path)
        return cls(data)


def _copy(src, dst):
    import shutil
    shutil.copy2(src, dst)


class Tagger:
    """Word list alone, or word list + trained learner (half and half)."""
    def __init__(self, model=None, extra_lexicon=None):
        self.model = model
        self.lookup = build_lookup(extra_lexicon)

    def tag(self, text):
        scores, evidence, negated = lexicon_scores(text, self.lookup)
        if self.model is None:
            dist = _blend_from_lexicon(scores)
            if not scores:
                return Tag("neutral", 0.0, dist, evidence, negated)         # nothing to go on is "no evidence", not "sure it is neutral"
        elif scores:
            lex = _blend_from_lexicon(scores)
            nb = self.model.probabilities(text)
            dist = {l: 0.5 * lex[l] + 0.5 * nb[l] for l in LABEL_ORDER}      # word list and learner agree -> surer; disagree -> less sure
        else:
            dist = self.model.probabilities(text)                           # no feeling word: only the learner can know (your own slang)
        label = max(LABEL_ORDER, key=lambda l: (round(dist[l], 12), -LABEL_ORDER.index(l)))
        return Tag(label, dist[label], dist, evidence, negated)


# ---------------------------------------------------------------------------------------------- samples (weak labels + your corrections)
def owner_messages(paths):
    """The owner's own messages (assistant turns with loss on) from train/val .jsonl made by messenger-import. Each distinct text once."""
    from . import data as D
    seen, out = set(), []
    for path in D._files(paths):
        for rec in D._json_records(path):
            msgs = D.as_messages(rec, with_train=True)
            for role, content, train in (msgs or []):
                if role == "assistant" and train and content not in seen:
                    seen.add(content)
                    out.append(content)
    return out


def make_samples(texts, tagger, neutral_ratio=3.0, limit=None, seed=1):
    """Weak labels for each message. Messages with no feeling word are labelled neutral, but only about `neutral_ratio` times as many of
    them as there are feeling messages are kept, or the learner would just learn 'always neutral'."""
    import random
    rows = []
    for t in texts:
        if len(t) > 1000:
            continue
        tag = tagger.tag(t)
        rows.append({"text": t, "label": tag.label, "confidence": round(tag.confidence, 3), "fixed": "", "evidence": bool(tag.evidence)})
    feeling = [r for r in rows if r["label"] != "neutral"]
    plain = [r for r in rows if r["label"] == "neutral"]
    rng = random.Random(seed)
    rng.shuffle(plain)
    keep = feeling + plain[:int(math.ceil(neutral_ratio * max(len(feeling), 1)))] if neutral_ratio is not None else feeling + plain
    rng.shuffle(keep)
    if limit:
        keep = keep[:limit]
    for i, r in enumerate(keep, 1):
        r["id"] = i
    return keep


def write_samples(path, rows, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise EmotionProblem("%s already exists; nothing was overwritten (it may hold your corrections). Pick another name." % path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_samples(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    r = json.loads(line)
                    if not isinstance(r.get("text"), str) or r.get("label") not in LABELS or r.get("fixed", "") not in list(LABELS) + [""]:
                        raise ValueError("line %d has an unknown feeling or no text" % n)
                    rows.append(r)
    except (OSError, ValueError) as e:
        raise EmotionProblem("Cannot read the samples file %s (%s)." % (path, e))
    return rows


def review(path, input_fn, out=print, count=20, seed=1, include_neutral=False):
    """Show messages one by one; you type a feeling to correct it, Enter to accept the guess, s to skip, q to stop.
    Every answer is saved at once. `input_fn` is a parameter so that tests type for the person."""
    import random
    rows = read_samples(path)
    todo = [r for r in rows if not r.get("fixed") and (include_neutral or r["label"] != "neutral" or r.get("evidence"))]
    random.Random(seed).shuffle(todo)
    todo = todo[:count]
    if not todo:
        out("Nothing left to review (every message with a feeling guess already has your answer, or the file is empty).")
        return 0
    out("Feelings: %s" % ", ".join(LABEL_ORDER))
    out("For each message: type a feeling to correct the guess, press Enter if the guess is right, s = skip, q = stop. Answers are saved at once.")
    done = 0
    for r in todo:
        out("")
        out("[%d] %s" % (r["id"], r["text"]))
        while True:
            try:
                ans = input_fn("guess = %s (%.0f%% sure). Your answer> " % (r["label"], 100 * r.get("confidence", 0))).strip().casefold()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans == "q":
                out("Stopped. %d answered this time." % done)
                return done
            if ans == "s":
                break
            if ans == "":
                r["fixed"] = r["label"]
            elif ans in LABELS:
                r["fixed"] = ans
            else:
                out("  not a feeling. Choose from: %s (or Enter, s, q)" % ", ".join(LABEL_ORDER))
                continue
            done += 1
            write_samples(path, rows, overwrite=True)
            break
    out("Reviewed %d. Now run `emotion train` to learn from them." % done)
    return done


def train_from_samples(rows, owner_weight=5.0, holdout=0.2, min_count=2):
    """-> (model, report). The held-out part (every 5th row) is NOT used for training; the report says how the model does on it."""
    train_rows, test_rows = [], []
    for r in rows:
        (test_rows if r["id"] % int(round(1 / holdout)) == 0 else train_rows).append(r)
    def weighted(rs):
        return [(r["text"], r["fixed"] or r["label"], owner_weight if r["fixed"] else 1.0) for r in rs]
    model = NaiveBayes.train(weighted(train_rows) if len(train_rows) >= 10 else weighted(rows), min_count=min_count)
    full = NaiveBayes.train(weighted(rows), min_count=min_count)
    tagger_words = Tagger(None)
    tagger_both = Tagger(model)
    rep = {"messages": len(rows), "fixed": sum(1 for r in rows if r["fixed"]), "held_out": len(test_rows)}
    fixed_test = [r for r in test_rows if r["fixed"]]
    if test_rows:
        rep["agrees_with_word_list"] = sum(1 for r in test_rows if tagger_both.tag(r["text"]).label == (r["fixed"] or r["label"])) / len(test_rows)
    if fixed_test:
        rep["fixed_held_out"] = len(fixed_test)
        rep["fixed_right_words_only"] = sum(1 for r in fixed_test if tagger_words.tag(r["text"]).label == r["fixed"]) / len(fixed_test)
        rep["fixed_right_with_model"] = sum(1 for r in fixed_test if tagger_both.tag(r["text"]).label == r["fixed"]) / len(fixed_test)
    full.data["trained_on"] = {"messages": len(rows), "owner_corrected": rep["fixed"]}
    return full, rep


# ---------------------------------------------------------------------------------------------- the mood
def nearest_label(valence, arousal):
    return min(LABEL_ORDER, key=lambda l: ((LABELS[l][0] - valence) ** 2 + (LABELS[l][1] - arousal) ** 2, LABEL_ORDER.index(l)))


def _clamp(x):
    return max(-1.0, min(1.0, x))


class Mood:
    """valence/arousal in [-1, 1] + the baseline they drift back to. `path=None` keeps it in RAM."""
    RATE = 0.6                     # how far one fully sure message pulls the mood toward its feeling
    HALF_LIFE_HOURS = 6.0          # time for the distance from the baseline to halve while nobody is talking
    MAX_HISTORY = 30

    def __init__(self, path=None, clock=time.time, baseline=(0.1, 0.0)):
        self.path, self.clock = path, clock
        self.valence, self.arousal = baseline
        self.baseline = list(baseline)
        self.updated = clock()
        self.turns = 0
        self.history = []                          # [{"label", "confidence", "time"}] of the messages that moved the mood
        if path and os.path.exists(path):
            self._load()

    @property
    def label(self):
        return nearest_label(self.valence, self.arousal)

    @property
    def intensity(self):
        """Distance from the baseline, 0 (at rest) to about 1."""
        return min(1.0, math.hypot(self.valence - self.baseline[0], self.arousal - self.baseline[1]) / 1.2)

    def _drift(self, now):
        hours = max(0.0, (now - self.updated) / 3600.0)
        keep = 0.5 ** (hours / self.HALF_LIFE_HOURS)
        self.valence = _clamp(self.baseline[0] + (self.valence - self.baseline[0]) * keep)
        self.arousal = _clamp(self.baseline[1] + (self.arousal - self.baseline[1]) * keep)
        self.updated = now

    def update(self, tag, now=None):
        """One typed message: drift back for the time that passed, then move toward the message's feeling in proportion to how sure the tag is."""
        now = self.clock() if now is None else now
        self._drift(now)
        pull = self.RATE * max(0.0, min(1.0, tag.confidence))
        if pull > 0 and tag.label != "neutral":
            self.valence = _clamp(self.valence + pull * (tag.valence - self.valence))
            self.arousal = _clamp(self.arousal + pull * (tag.arousal - self.arousal))
            self.history = (self.history + [{"label": tag.label, "confidence": round(tag.confidence, 3), "time": round(now, 1)}])[-self.MAX_HISTORY:]
        self.turns += 1

    def reset(self):
        self.valence, self.arousal = self.baseline
        self.updated = self.clock()

    def snapshot(self):
        return (self.valence, self.arousal, self.updated, self.turns, list(self.history))

    def restore(self, snap):
        self.valence, self.arousal, self.updated, self.turns, self.history = snap[0], snap[1], snap[2], snap[3], list(snap[4])

    def prompt_text(self):
        """The text inside the <|mood|> block: label, valence, arousal (one decimal). Future fine-tune data must use exactly this shape."""
        return "%s %+.1f %+.1f" % (self.label, self.valence, self.arousal)

    def temperature_factor(self):
        """Excited moods sample slightly more freely, sleepy ones slightly less (+-15 %). This is the ONLY effect on tone until the model is trained on mood blocks."""
        return 1.0 + 0.15 * self.arousal

    def describe(self):
        return "Feeling: %s (valence %+.2f, arousal %+.2f, %.0f%% away from the baseline %+.2f/%+.2f). Messages seen: %d. Temperature factor: x%.2f." % (
            self.label, self.valence, self.arousal, 100 * self.intensity, self.baseline[0], self.baseline[1], self.turns, self.temperature_factor())

    # -- file
    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.valence, self.arousal = _clamp(float(d["valence"])), _clamp(float(d["arousal"]))
            self.baseline = [_clamp(float(d["baseline"][0])), _clamp(float(d["baseline"][1]))]
            self.updated, self.turns, self.history = float(d["updated"]), int(d.get("turns", 0)), list(d.get("history", []))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            raise EmotionProblem("Cannot read the mood file %s (%s). Nothing was changed; delete the file to start from the baseline." % (self.path, e))

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"format_version": FORMAT_VERSION, "valence": round(self.valence, 4), "arousal": round(self.arousal, 4),
                       "baseline": self.baseline, "updated": self.updated, "turns": self.turns, "history": self.history}, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
