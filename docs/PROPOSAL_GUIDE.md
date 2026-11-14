# Proposals: how the AI asks to change itself (Stage 7a)

**The idea in one line:** the AI may *suggest* a change. Nothing happens until *you* type the proposal's number at the keyboard.

This half (7a) covers four kinds of change:

| Kind | What it does |
|---|---|
| `memory-add` | remember something new |
| `memory-forget` | forget one memory |
| `memory-importance` | change how important one memory is (and pin it) |
| `card-add` | add a line to your Person card |

The AI's name, personality, temperament and values (the identity core) can **not** be changed by a proposal. Those keep the Stage 6 rules (typed phrase, waiting time, second typed phrase). Config changes and fine-tune proposals, and `/approve` inside the chat, come in Stage 7b.

Always start in the project folder:

```
cd C:\Users\user\persona_lm_project
```

## 1. File a proposal (changes nothing)

Replace the file names with yours. If you ran `identity set-file memory ...` and `identity set-file person ...` you can leave out `--memory` and `--person`.

```
python -m persona_lm proposal --dir identity_core --memory memory.json --person person_card.md memory-add "Maria's birthday is in May" --why "you mentioned it twice" --importance 0.7
```

You will see the exact change, why, and how to undo it, and the number (for example `p0001`).

## 2. Look at what is waiting

```
python -m persona_lm proposal --dir identity_core list
```

```
python -m persona_lm proposal --dir identity_core show p0001
```

## 3. Approve (type the number) or reject

```
python -m persona_lm proposal --dir identity_core approve p0001
```

It shows the change again and asks you to type the number (`p0001`). Anything else cancels and nothing changes. This only works when you type in a terminal window yourself: a script or a pipe is refused on purpose.

```
python -m persona_lm proposal --dir identity_core reject p0001 --note "not true"
```

## 4. Undo an approved change

```
python -m persona_lm proposal --dir identity_core undo p0001
```

Undo is also typed. It refuses (and touches nothing) if you changed that memory or card line by hand since, so it never overwrites your own edits.

**Forgotten memories:** to make `memory-forget` undoable, the memory's text is kept inside `proposals.json` (in the identity folder). If you really want it gone, erase that copy (then it can no longer be undone):

```
python -m persona_lm proposal --dir identity_core purge-undo p0001
```

## What is written down

Every approve, reject and undo is added to the lineage history (`identity_core/lineage.jsonl`) with the proposal number, the kind and a fingerprint, **never the memory text**.

## Honest limits (Stage 7a)

- The chat does not create proposals yet (7b). For now you file them with the commands above, which is also how the tests simulate the AI.
- Tested on synthetic data and tiny practice models only; never on your real memories or on Windows.
- `proposals.json` is a plain file: someone who can edit your files could edit it, but an edited proposal is refused at approval (its fingerprint no longer matches). The history chain is tamper-evident, not tamper-proof.
- Approving changes memory/Person-card files that another program might be editing at the same time: if the memory file changed meanwhile, the approval stops with an error and nothing is changed.
