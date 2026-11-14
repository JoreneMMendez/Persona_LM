# Stage 3 guide: the guided pipeline

Everything runs on your own computer. Replace `WORK` with a folder name you choose (for example `C:\Users\user\persona_work`). Each command below should finish in a few seconds, except the training steps (hours). If anything takes more than 2 minutes and is not a training step, press Ctrl+C and send me the screen.

## What the pipeline does
1. **tokenizer**: learns the vocabulary. *This one cannot be undone*: every trained model is tied to it. It only runs after you confirm.
2. **pretrain-data / pretrain**: teaches the model language from public text (not your chats).
3. **chat-data / finetune**: teaches it your way of chatting, from your anonymized Messenger examples.
4. **chat**: talk to it.

## Step A: record your decisions (changes nothing else)
Replace the two file paths with the real `train.jsonl` and `val.jsonl` made by `messenger-import`, and `en,tl` with your languages.

```powershell
python -m persona_lm pipeline init --dir WORK --languages en,tl --vocab 16000 --size s10m --ctx 512 --pretrain-text PUBLIC_TEXT_FOLDER --chat-train PATH\train.jsonl --chat-val PATH\val.jsonl --batch 4 --accum 8
```

Leave out `--pretrain-text` if you have no public text yet. The plan will say that fine-tuning on your chats alone teaches very little.

## Step B: read the plan

```powershell
python -m persona_lm pipeline plan --dir WORK
```

It lists every step with the exact command, marks the NEXT one, and prints WARNINGs (for example a language whose alphabet is missing from your text).

## Step C: say OK (only when you are sure)
This locks the languages, vocabulary size and window length. Do it only after you understood and accepted them.

```powershell
python -m persona_lm pipeline confirm --dir WORK --yes
```

## Step D: run one step at a time
Run this again and again; each run does the next undone step. Training steps can be stopped with Ctrl+C and continued by running the same command.

```powershell
python -m persona_lm pipeline run --dir WORK
```

Before a training step, look at what the data can support:

```powershell
python -m persona_lm data-report --data-dir WORK\data_pre --size s10m --tokenizer WORK\tok.json
```

After or during training, see where everything stands (also warns if the validation loss started to rise):

```powershell
python -m persona_lm status WORK
```

## Step E: chat
`pipeline plan` prints the exact chat command at the bottom. Inside the chat, type `/help`. `/save name.txt` writes the conversation to a file on your computer. `/memory`, `/mood`, `/identity`, `/approve`, `/reject` exist but only explain which later stage builds them.

## What is not done yet
Early stopping (the run does not stop by itself when it starts to memorise; the plan uses about 3 passes over your chats, and `status` warns). The real tokenizer is not made until you confirm in Step C.
