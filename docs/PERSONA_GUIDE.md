# Personas (Stage 8a): masks the AI can wear

A **persona** is a short instruction such as "You speak slowly and kindly and ask good questions." The AI reads it before it answers.
It is a **mask**: it changes how the AI is *asked* to speak, never *who it is*.

## What a persona is, and what it is not (read this once)

- A persona is **prompt text**. On its own nothing is trained into the model's weights. Stage 8b adds an **optional** trained side (see the end of this guide).
- Your tiny practice models have never seen a persona block, so they will mostly ignore it. A real model fine-tuned with persona blocks will follow it better
  (the real fine-tune data must contain `<|persona|>` blocks, like the memory and mood blocks).
- The persona **never edits the identity core**. The AI's name, personality, temperament and values stay exactly as they are, and the identity line is still
  the first thing in every prompt. The persona block always ends with: "You are still <name> underneath; this is only a mask."
- Only **you** can add, change, remove or switch a persona, by typing. Nothing the AI writes can do it, and neither can anything in training data.
- Every add / edit / remove / switch is one line in the lineage log (`identity lineage`). The log keeps names and a fingerprint of the text, **not the text itself**.

Personas live in `identity_core/personas.json` (a plain file). They need an identity core (`identity init`).

## Commands (replace `identity_core` if your folder has another name; each takes under a second)

Make a mask:

```
python -m persona_lm persona add "calm mentor" You speak slowly and kindly and ask good questions.
```

See your masks:

```
python -m persona_lm persona list
```

See exactly what the AI would be shown:

```
python -m persona_lm persona show "calm mentor"
```

Put it on (it stays on across chats until you change it):

```
python -m persona_lm persona use "calm mentor"
```

Take it off (the AI speaks as itself again):

```
python -m persona_lm persona use none
```

Change or delete it:

```
python -m persona_lm persona edit "calm mentor" Be brief and warm.
```

```
python -m persona_lm persona remove "calm mentor"
```

If your identity folder is not called `identity_core`, add `--dir YOUR_FOLDER` right after the word `persona`.

## In the chat

Start the chat with `--core identity_core` (as before). Then:

- `/persona` lists the masks (a `*` marks the one that is on)
- `/persona calm mentor` puts one on; `/persona none` takes it off
- the conversation so far is kept when you switch (`/reset` starts fresh)

## Limits

- Name: 1-30 letters, digits, spaces, dots, dashes, underscores. `none`, `default` and `off` mean "no mask" and cannot be names. At most 20 personas.
- Text: at most 300 characters, and it may not contain `<|` or `|>` (they would forge special tokens).
- If the text is too long for the model's small window, the chat **refuses the message and says so**; it never silently drops the mask.
  Shorten it with `persona edit`, or `/persona none`.
- The persona file is plain text on your computer. Anyone who can edit your files can edit it; a file that is broken or contains forbidden text is refused when
  the chat starts, before the model loads.

## Optional: train a model to wear the mask (Stage 8b)

Prompt text alone is weak on a small model. To make a mask stick you can fine-tune on **your own example replies**. Be clear about what this is:

- It makes a **full fine-tuned copy of the weights** for that mask. It is **not** a small adapter (LoRA is not built).
- It is optional and untested on real data. The mask still works as prompt text without it.
- Nothing is trained automatically. The commands below only prepare files and record what you did; you run the training yourself.

1. Write examples (what someone says, and how this mask would answer). A few dozen is a start; at most 200 per mask:

```
python -m persona_lm persona example "calm mentor" "I failed my test" "That hurts. What will you try next?"
```

2. Make the training file. It never overwrites. Each line holds the identity line, the persona block and the message with no learning on them, and your reply with learning on it. The persona block is token-for-token what the chat will show the model:

```
python -m persona_lm persona make-data "calm mentor" --out data\mentor.jsonl
```

3. Train it with your usual steps: `prepare --kind chat` on that file, then `train` with `--init-from` your current checkpoint into a **new** run folder. Then register and score the new version like any other:

```
python -m persona_lm identity add-version mentor1 --ckpt runs\mentor1\ckpt.npz
```

4. Tell the persona which version was trained for it. While that mask is on, `chat --core` uses that version (unless you name one with `--version`). Starting a chat with the mask on says so:

```
python -m persona_lm persona attach "calm mentor" mentor1
```

   If you edit the mask text afterwards, the list shows **STALE** (the version learned the old text). `persona detach NAME` removes the link. A version that the regression test **rejected** is refused.

## Does the mask hide who the AI is? `persona check`

```
python -m persona_lm persona check "calm mentor"
```

It asks the identity probes (name, owner, values, memories) with the mask **on** and compares with the answers recorded **without** a mask for your current version (`identity record` first). It judges only **whether the facts about who the AI is are still stated** (against your threshold). Overall wording and tone will differ; that is what a mask is for, so they are shown but not judged. It changes no identity file, promotes nothing and adds one history line with numbers only.

- If it says the model "gave no answer to any probe", the mask text does not fit in the model's window next to the identity line. The tiny practice models have a 64-token window, so **no mask fits there**; this is normal for them. Real `s10m` models have 512.
- A mask is refused (not silently shortened) when it takes more than a third of the room left after the identity line.
- This compares words, not minds. Read a few answers yourself.

Not built: a persona proposed by the AI (only you create personas), and adapters (LoRA).
