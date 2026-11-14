# Proposals: how the AI asks to change itself (Stages 7a + 7b)

**The idea in one line:** the AI may *suggest* a change. Nothing happens until *you* type the proposal's number at the keyboard.

There are six kinds of change:

| Kind | What it does |
|---|---|
| `memory-add` | remember something new |
| `memory-forget` | forget one memory |
| `memory-importance` | change how important one memory is (and pin it) |
| `card-add` | add a line to your Person card |
| `config` | change the identity setting `warn_drift` (the safeguards `threshold` and `cooldown_hours` can NOT be proposed; only you change them with `identity settings`) |
| `finetune` | a training run. Approving only **records the exact command**; the program never starts it, you run it yourself |

The AI's name, personality, temperament and values (the identity core) can **not** be changed by a proposal. Those keep the Stage 6 rules (typed phrase, waiting time, second typed phrase). LoRA adapters do not exist in this project yet (Stage 8b), so only `finetune` can be proposed.

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

## In the chat

Start the chat with your identity folder, and proposals switch on:

```
python -m persona_lm chat --core identity_core --memory memory.json --person person_card.md
```

- The AI can **file** a proposal by writing a line like `[propose memory-add] Maria loves mango #joy *0.8 | why: you said it twice` in its reply (kinds: `memory-add`, `memory-forget ID`, `memory-importance ID VALUE`, `card-add Section: text`, `config warn_drift 0.2`, `finetune run=... data=... size=... steps=... lr=... [init=...]`). At most 3 per reply; duplicates and invalid ones are ignored and reported. This is **all** its words can do.
- `/proposals` lists the waiting ones. `/reject p0001` throws one away.
- `/approve p0001` shows the exact change and asks you to **type `p0001` again**. Anything else cancels and the message is not sent to the AI. It works only when you type in a terminal window (a pipe or script is refused).
- After an approval the chat re-reads the memory file and your card, so you do not need to restart it.
- Undo is not inside the chat: use `proposal undo p0001` in a terminal.

Config and training proposals from the command line:

```
python -m persona_lm proposal --dir identity_core config warn_drift 0.25 --why "the warnings are too noisy"
```

```
python -m persona_lm proposal --dir identity_core finetune --run runs/ft2 --data-dir data/chat --size s10m --steps 200 --lr 0.0001 --init-from runs/pre/ckpt.npz --why "new chat data"
```

A `finetune` proposal is refused if the run folder already has a checkpoint (a proposal never reuses or overwrites a run).

## What is written down

Every approve, reject and undo is added to the lineage history (`identity_core/lineage.jsonl`) with the proposal number, the kind and a fingerprint, **never the memory text**.

## Honest limits (Stages 7a + 7b)

- **The AI's proposal lines are untested with a real trained model.** The practice models speak gibberish, so the tests use a fake model that writes the lines. A real model will only write them if its fine-tune data contains such lines (design debt for Stage 3 part 2).
- Only `warn_drift` can be proposed as a setting; chat knobs like temperature are not saved anywhere, so there is nothing to propose.
- The `finetune` proposal checks the numbers and the file names, not that the data folder is right or that the run will work.
- Tested on synthetic data and tiny practice models only; never on your real memories or on Windows.
- `proposals.json` is a plain file: someone who can edit your files could edit it, but an edited proposal is refused at approval (its fingerprint no longer matches). The history chain is tamper-evident, not tamper-proof.
- Approving changes memory/Person-card files that another program might be editing at the same time: if the memory file changed meanwhile, the approval stops with an error and nothing is changed.
