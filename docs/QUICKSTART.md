# Quickstart: the first 10 commands, Windows troubleshooting, and how much data you need

Written for Windows + PowerShell, project folder `C:\Users\user\persona_lm_project`. Everything here is a normal command from README.md in short form.
Never print, upload or commit your chat data (`data_msgr*`, `data_chat*`, `data_clean*`, `*.jsonl`, `*.bin`, `private/`), and never replace `work\tok.json`.

## 1. The first 10 commands

Open PowerShell in the project folder.

```powershell
cd C:\Users\user\persona_lm_project
```

| # | What | Command |
|---|---|---|
| 1 | Does this computer work? (tiny training runs, console, menu, GPU) | `python -m persona_lm doctor --windows` |
| 2 | Do the files match each other? (tokenizer, token folders, checkpoints) | `python -m persona_lm doctor` |
| 3 | Is the GPU code right on this GPU? | `python -m persona_lm check-gpu` |
| 4 | How fast is it? | `python -m persona_lm benchmark --size s30m --vocab 8000` |
| 5 | What do I have, and what should I do next? | `python -m persona_lm status` |
| 6 | Is my chat data big enough? | `python -m persona_lm data-report --data-dir data_chat4096 --size s30m` |
| 7 | How many steps (about 3 passes) and the exact train command? | `python -m persona_lm plan-steps --data-dir data_chat4096 --size s30m --run work3\run_chat4 --init-from work3\run_pre\ckpt.npz` |
| 8 | Train (Ctrl+C is safe; the same command continues) | the command that step 7 printed |
| 9 | Does the new model answer the message, or give one stock reply? | `python -m persona_lm chat-probe --ckpt work3\run_chat4\ckpt.best.npz` |
| 10 | Talk to it | `python -m persona_lm chat --ckpt work3\run_chat2\ckpt.npz --tokenizer work\tok.json --temperature 0.5 --top-k 20 --max-new 40` |

Or open the menu with `python -m persona_lm`, or the web page with:

```powershell
python -m persona_lm ui --root C:\Users\user\persona_lm_project
```

### After a change (the standard check)

```powershell
python -m unittest discover -s tests
git status
git ls-files          # look for chat data, .jsonl, .npz, .bin, work*, private/ : none of them may be listed
```

## 2. How much data do I need?

One rule of thumb: a model learns language well when it reads about **20 tokens per parameter**. Chat tuning on your own messages is different:
it only teaches your style on top of a base that already knows the language.

| Model | Parameters (vocabulary 8000) | Tokens for 20 per parameter | Window |
|---|---|---|---|
| `tiny` | 0.6 M | 12 M | 64 |
| `s10m` | 13.7 M | 274 M | 4096 |
| `s30m` | about 29 to 34 M (depends on the real vocabulary) | about 590 to 670 M | 4096 |
| `s125m` (the "about 100M" model) | about 91 M | about 1.8 B | 4096 |

* **Pretraining (the base).** Check `data_pre_new` with `data-report`: it prints "trained tokens per parameter". Far below 10 means the base is weak; more public text helps more than longer training.
* **Chat tuning (your style).** Your own words are the limit (about 1,440 messages). Stay near **3 passes** over the data (`plan-steps` does the arithmetic). Your 3000-step run was about 12 passes; it got worse after step 800 and answered everything with a stock reply.
* **Before a bigger model (`s125m`).** Do not start it until the pretraining text is near 1.8 B tokens AND you have more of your own text, because a bigger model memorises your small chat set even faster. It is also about 10 times slower per token than `s10m` (README, "Honest limits"), and whether it fits a 6 GB GPU at window 4096 is **unverified**: run `benchmark --size s125m --vocab 8000 --try-batches 1,2` first.
* **Only more of your own text raises the ceiling.** Older Messenger exports, other apps; add them with `clean_data.py` (see HANDOFF.md, "How to add more chats").

## 3. Troubleshooting (Windows)

**`python` is not found.** Install Python 3.10 or newer from python.org and tick "Add Python to PATH", then open a NEW PowerShell window.

**The chat gives the same reply to everything ("Kuyog ko", "BWHAHAHA", "Asa ka?").**
That is overfitting. Run `python -m persona_lm chat-probe --ckpt <run>\ckpt.npz`: it prints how many different replies came back. Then: chat with `ckpt.best.npz` instead of `ckpt.npz`, retrain with the command from `plan-steps` (about 3 passes, `--early-stop 3`), and never use `work3\run_chat` (the 3000-step run). Replies are hidden by default because a model can repeat private text; add `--show` to read them on your screen.

**The chat only continues text like a news article.** You are using a base run (`run_pre`). Use a chat-tuned run (`run_chat2`, `run_chat3`, ...).

**`KeyError: 'merges'` or "not a tokenizer".** A file such as `work\tok.seal.json` was given as the tokenizer. The real one is `work\tok.json`. Never create a new one: every model depends on it.

**Out of memory on the GPU.** Use `--batch 1 --accum 4`. If it still fails, check that nothing else uses the GPU (games, browser video). The result of `--batch x --accum` stays the same as long as the product is the same.

**PyTorch is installed but training is slow / `doctor --windows` says "cannot use a GPU".** You have the CPU-only build, or the NVIDIA driver is missing. See `docs/GPU_GUIDE.md`, steps 3 to 5 (`nvidia-smi`, the CUDA build of PyTorch, `check-gpu`).

**The training panel looks broken, or there are strange characters.** Train with `--plain`, or set `PERSONA_PLAIN=1`. Windows Terminal draws the panel best. If the old console cannot show the box characters, the panel and menu switch to plain ASCII by themselves. `python -m persona_lm doctor --windows` draws the panel into a console-like test stream and tells you if something fails.

**"Stop and save" in the web UI hangs.** Use Force stop. Training saves a checkpoint every `--ckpt-every` steps anyway, so a forced stop loses at most that much.

**The checkpoint and the tokenizer do not match ("DIFFERENT tokenizer").** `python -m persona_lm doctor` finds it. Use the tokenizer the model was trained with; do not train on those token files, run `prepare` again with `work\tok.json`.

**Training stopped by itself.** That is `--early-stop 3`: validation loss did not improve three checks in a row. The best weights are in `ckpt.best.npz`. `ckpt.npz` is still the latest, so the same command can continue if you want to.

**Odd file errors while training (file in use, access denied).** Keep the project outside OneDrive and out of a very long path (Windows stops at 260 characters). `doctor --windows` warns about both. Antivirus programs can also lock `ckpt.npz` for a moment: exclude the project folder.

**`Expand-Archive` complains.** Use `Expand-Archive -Force "$HOME\Downloads\STAGE_N_files.zip" .` from inside the project folder. It overwrites only the files that are in the zip.

**`git push` is rejected.** Run `git pull --rebase origin main` first, then `git push origin main`. Before every push check `git ls-files` (no chat data, `.jsonl`, `.npz`, `.bin`, `work*`, `private/`).

## 4. Where to read more

README.md (all commands), HANDOFF.md (status, decisions, stage log), `docs/GPU_GUIDE.md` (GPU), `docs/MESSENGER_GUIDE.md` (your chats), `docs/EXPORT_GUIDE.md` (phone / GGUF).
