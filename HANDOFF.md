# HANDOFF: persona_lm (a language model from scratch, pure Python + NumPy, optional PyTorch)

**DO NOT DELETE THIS FILE.** It is the project's memory between sessions and between AIs. `tests/test_handoff.py` fails if it disappears or loses a section (keep the headings "START HERE", "5. Stage log", "7. Milestone plan", "12. Next action").
Anything marked (unverified) was not checked on the owner's computer.

## CURRENT FOCUS (read this first)
The owner wants to work on the NEW CHANGES: the staged cleanup and improvements in the prompt at the end of this file ("Prompt for an AI to clean and improve the project"). Start with Stage 1 unless the owner says another stage. If the owner types just "stages" or "new changes", that means: continue the staged plan from the first stage that is not marked DONE below.
Stage status: Stage 1 not done | Stage 2 not done | Stage 3 partly done (menu ignores non-tokenizer files, 2026-11-16) | Stage 4 not done | Stage 5 not done | Stage 6 not done | Stage 7 not done. (Update this line at the end of every stage.)

## START HERE
- **CURRENT GOAL (owner, 2026-11-16): work on the cleanup/improvement STAGES 1 to 7 in the last section ("Prompt for an AI to clean and improve the project").** When a new session starts, do not ask what to do: continue with the next stage that is not done yet (see "Stage progress" below), do only that one stage, and finish it with the changed-files zip plus the PowerShell commands (copy files, run tests, `git add`, `commit`, `pull --rebase`, `push origin main`).
- Stage progress (update after every stage): Stage 1 not done, Stage 2 not done, Stage 3 not done (the menu tokenizer fix is already in), Stage 4 not done, Stage 5 not done, Stage 6 not done, Stage 7 not done.
- Owner: Jorene. Windows, PowerShell, project folder `C:\Users\user\persona_lm_project`. GPU: RTX 3050 6 GB laptop. PyTorch + CUDA, bf16.
- What it is: tokenizer, transformer with hand-written backprop, AdamW, resumable training, chat tuning, sampling, GGUF export, local web UI. Personal chat AI trained on the owner's Messenger chats (Cebuano/Bisaya, Tagalog, English).
- Run tests: `python -m unittest discover -s tests -v`. Commands are in README.md.
- HARD RULES for any AI working here:
  1. Never print, upload or commit private chat data (`data_msgr*`, `data_chat*`, `data_clean*`, `*.jsonl`, `*.bin`, `private/`).
  2. Never overwrite or regenerate `work\tok.json` (the frozen tokenizer; every model depends on it). Never change the tokenizer or checkpoint format or any token id.
  3. Never delete files. Move them to `_to_delete\` and tell the owner.
  4. Never touch the owner's data or model folders (`work*`, `data_*`, `public_*`, `kaggle_upload`, `*.npz`) unless the owner asked for exactly that.
  5. Do not skip or delete tests to make them pass.
- Delivery rule (owner's standing request): the owner may run out of credit at any time. Work in small stages, and at the end of EVERY stage give only the changed files (as a zip) plus the PowerShell commands that copy them in, run the tests, then `git add <changed files>`, `git commit`, `git pull --rebase origin main` and `git push origin main` (see the prompt at the end of this file).
- Adding more chats: always use `clean_data.py` (see "How to add more chats" at the end). Never feed raw chat files to `prepare`.
- Before the AI's tokens run out, update this file (owner's standing request).

## 1. Layout
- `persona_lm/` package: model.py, bpe.py, train.py, torch_backend.py, generate.py, fastgen.py, chatrepl.py, memory.py, emotion.py, identity.py, personas.py, proposals.py, runtime.py, export_pkg.py, gguf.py, menu.py, pretty.py, corpus*.py, ui_*.py (web UI) and `cli.py` (the command line).
- Root scripts: `clean_data.py` (clean each chat file, the one to use), `clean_chat.py` (cleaning rules, also used by the UI), `setup_s30m.py`, `prep_mood_data.py`, `cleanup_files.py`, and the public-text scripts `grab_data.py` / `grab_english.py` / `grab_taglish.py` / `ceb_domains.py` / `clean_ceb.py` / `clean_corpus.py` / `make_mix.py` / `make_tok_sample.py` / `check_tok.py` / `quick_check.py` (reused by `persona_lm/corpus.py`, do not delete). `train_s30m_kaggle.ipynb` = s30m pretraining on Kaggle.
- Folders on the owner's computer (ignored by git): `work\` (s10m, `tok.json`, data_pre, run_pre, run_chat), `work2\` (only a copy of data_pre), `work3\` (s30m: run_pre, run_chat, run_chat2, run_chat3), `data_chat\`, `data_chat4096\`, `data_clean\`, `data_pre_new\`, `kaggle_upload\data_pre`.

## 2. Models
- s10m (13.7M, old, `work\`). s30m = 33.5M parameters: 8 layers, d_model 512, 8 heads, d_ff 1376, context 4096. s30m uses the same tokenizer as s10m (`work\tok.json`).
- Pretraining and chat tuning must use the same `--size`. `ckpt.npz` holds weights, optimizer, RNG and step; the previous save is `ckpt.prev.npz`.
- `work3\run_pre` is only the pretrained BASE: it continues text like a news article and cannot chat. Chat with a chat-tuned run (`run_chat2`).

## 3. Data
- Chat data: `messenger-import` (window mode, up to 8 context turns, loss only on the owner's replies) -> `clean_data.py` -> `prepare --kind chat`. Window mode repeats each message as context in up to 9 examples, so the model memorizes quickly.
- The owner has only about 1,440 messages of their own. This, not training time, is the main limit of chat quality.

## 4. Training facts that matter
- Overfitting: the 3000-step chat run (`work3\run_chat`, about 12 passes) reached val 0.937 at step 800, then val rose to 1.351 while train loss fell to 0.27. It ignores the input and answers with stock replies. Do not use it. Stay near 3 passes.
- Best model: `work3\run_chat2` (800 steps, lr 1e-4 -> 1e-5, warmup 20, batch 1, accum 4): val 1.239 (100), 1.057 (200), 0.986 (300), 0.939 (400), 0.926 (500), 0.921 (600), 0.916 (700), **0.910 (800)**. Replies fit the chat style but do not follow questions. Checked 2026-11-16: it loads and chats.
- Data size: `data_chat4096` = 4,116,902 train / 433,857 val tokens (about 16,384 tokens per step, so 800 steps = about 3 passes).
- `work3\run_chat3` was trained from the cleaned mix (train 574,600 tokens, about 105 steps for 3 passes). Result not yet checked.
- The trainer keeps only the latest checkpoint and the one before it, so the best step is lost unless the run is stopped there.
- 6 GB GPU: the default micro-batch (8) at ctx 4096 runs out of memory. Use `--batch 1 --accum 4`.
- Pretraining target is about 20 tokens per parameter (about 670M tokens for s30m). Check the real number for `data_pre_new` on the Web UI Data page. Fine-tuning only teaches style; the base limits how well it can answer.

## 5. Stage log
- Stage 1: GPU backend (PyTorch), `check-gpu`, `benchmark`.
- Stage 2: Messenger import (anonymized, loss only on the owner's replies). Tested on synthetic data.
- Stage 3: guided pipeline (`pipeline init/plan/confirm/run`), chat screen with slash commands, `data-report`.
- Stage 4: long-term memory (`memory.json`, recall by words and names).
- Stage 5: emotion tagger, mood, Person card.
- Stage 6: identity core (protected; changed only by typed `identity` commands).
- Stage 7a/7b: proposals; the AI may only suggest changes, the owner approves by typing the number.
- Stage 8a/8b: personas (masks), optional fine-tuned masks.
- Stage 9a/9b: runtime API with KV cache; export package and local server; GGUF export (not tested inside llama.cpp).
- Web UI (`python -m persona_lm ui --root C:\Users\user\persona_lm_project`), 2026-11-15: U0 base, U1 Chat (settings, save, mood badge), U2 Memory (list, graph, mood, Person card), U3 Prompt panel + Personas, U4 Data (import wizard, privacy audit, cleaner), U5 Training (plan, checkpoints, start/stop, live loss chart), U8 Language data (download + clean public text). Plan: `docs/UI_PROMPT.md`.
- Also 2026-11-15: old-chat archive (`--archive`), `clean_data.py`, s30m setup, s30m chat fine-tunes, `--cpu-share` hybrid training, nicer training panel (`pretty.py`), terminal menu item 8/9 (language data, token files).
- 2026-11-16: fixed `KeyError: 'merges'` in the menu: it picked `work\tok.seal.json` (not a tokenizer) because it took the newest `tok*.json`. `menu.py` now lists only real tokenizers (with `merges` and `specials`) and puts `tok.json` first; `bpe.py` `Tokenizer.load` gives a clear error for a wrong file.
- Tests: about 870 (some skipped without PyTorch).

## 6. Known problems
- Chat quality is limited by the small model and little personal data (sections 3 and 4).
- Tests that need PyTorch (`tests/test_hybrid.py`, 4 checks) were skipped when written: the CPU+GPU hybrid training code is UNTESTED on real hardware until the owner runs `python -m unittest tests.test_hybrid -v` and a real `--cpu-share auto` run.
- Not yet checked on a real Windows console: the training panel (`pretty.py`), the boxed menu forms and terminal progress screen (`corpus_ui.py`), and "Stop and save" in the web UI (use Force stop if it hangs). If a panel looks broken, use `--plain`.
- Web UI pages were checked in headless Chromium only. Open each page once on the owner's PC (light, dark, narrow width).
- GGUF output untested in llama.cpp. Tagalog/Bisaya emotion word lists unchecked. The model was not fine-tuned on memory/mood blocks, so memory use is untested.
- The chat REPL without `--system` / `--mood` may differ from the training prompt format (unverified).
- Real chat copies (`*.bak`) were once found in the repo: check `git ls-files` before every push.

## 7. Milestone plan
1. DONE: s30m chat tuning at 800 steps (`work3\run_chat2`, val 0.910).
2. Get more of the owner's own text (older Messenger exports, other apps). This is the only change that raises the ceiling.
3. Check tokens per parameter of `data_pre_new`; continue pretraining if it is far under 20 (Kaggle notebook).
4. Improve chat sampling (lower temperature, top-p, repetition penalty).
5. Export to GGUF for the phone and test in llama.cpp.
6. Fine-tune with memory/mood blocks so the model learns to use them.
7. A larger model (about 100M parameters) if the data grows enough.

## 8. Commands
- Chat: `python -m persona_lm chat --ckpt work3\run_chat2\ckpt.npz --tokenizer work\tok.json --temperature 0.5 --top-k 20 --max-new 40`
- Retrain (about 3 passes only): `python -m persona_lm train --run work3\run_chat2 --data-dir data_chat4096 --size s30m --init-from work3\run_pre\ckpt.npz --steps 800 --lr 0.0001 --min-lr 0.00001 --warmup 20 --eval-every 100 --batch 1 --accum 4`
- Stop training safely with Ctrl+C; the same command resumes.
- CPU + GPU together: add `--cpu-share auto` (needs `--accum 2` or more; expect 10 to 25 % at best; off by default).
- Training panel off: `--plain` or `PERSONA_PLAIN=1`.

## 9. Privacy
- Chat data is anonymized by default and stays local. `.gitignore` excludes chats, checkpoints, memory, mood, person card and work folders. The web UI never reads the raw Messenger export (the wizard only shows the commands).
- Web UI write areas (all owner-only, token + Host/Origin checked, through PathGuard): `memory.json`, `person_card.md`, `mood.json` (read only), `chat_archive.jsonl` (opt-in), `saved_chats/`, `data_msgr_chats` (cleaner; an old copy is moved to `_to_delete\`, never deleted), training jobs (command always built from the pipeline plan, never from the page), public-text downloads (after Start + confirm; Taglish needs a ticked acknowledgement). Chat text goes through `say()` and can never run a command, approve a proposal or write memory (tested).

## 10. Cleanup status
- `cleanup_files.py` (dry run by default; `--apply` moves safe items into `_to_delete\`) has not been run on the owner's computer. It moves duplicate `ckpt.prev.npz`, duplicate `data_pre` copies and junk only. Superseded runs (`work\run_chat`, `work\run_chat2`, `work3\run_chat` = the overfit run) wait for the owner's decision. Keep `work3\run_pre`, `work3\run_chat2`, `work3\run_chat3` and `work\tok.json`.

## 11. Open questions for the owner
- Is the s10m model in `work\` still needed? Which data folder is the real chat set: `data_chat` or `data_chat4096`?
- Allow the web UI to run `messenger-scan` on an export folder outside the project (read only)? Not done without an explicit yes.
- What is `work\tok.seal.json`? It is not read by the project (see 2026-11-16 fix). Move it to `_to_delete\` if it is not needed.

## 12. Next action
Do the next unfinished stage of the cleanup/improvement plan (START HERE, "CURRENT GOAL"), one stage per message. Then:
Run `python cleanup_files.py` (dry run, then `--apply`). Chat with `work3\run_chat2` (command in section 8). Then gather more of the owner's own messages (milestone 2) and check `data_pre_new` tokens per parameter (milestone 3). Update sections 4 and 5 after each run.

## How to add more chats
Do not merge anything; each file stays its own file.
1. Put the new `.jsonl` file(s) (chat/session/messages format, or `.json` lists) in the project folder.
2. Clean ALL files again in one run, the owner's real chats FIRST, then synthetic/other data:
   `python clean_data.py --input <real chats>.jsonl <new chats>.jsonl <synthetic>.jsonl --max-examples 5000`
   Each input becomes `data_clean\train\<name>_cleanData.jsonl` and `data_clean\val\<name>_cleanData.jsonl` (last 5 % of each chat's sessions). The script replaces the old `*_cleanData` files first, so always list every file you want used. Repeated-reply limits (`--max-dup 3`, `--max-dup-long 5`) are counted over all files together. `--max-examples N` caps each file so a huge synthetic file cannot drown out the real chats (lower it, e.g. 1500, if replies sound generic).
3. Make the tokens (tokenizer stays `work\tok.json`, never regenerate it):
   `python -m persona_lm prepare --kind chat --tokenizer work\tok.json --data data_clean\train --val-data data_clean\val --out data_chat`
4. Fine-tune from the pretrained base into a NEW run folder, about 3 passes (steps = 3 x train tokens / 16,384 at `--batch 1 --accum 4`, ctx 4096):
   `python -m persona_lm train --run work3\run_chatN --data-dir data_chat --size s30m --init-from work3\run_pre\ckpt.npz --steps <steps> --lr 0.0001 --min-lr 0.00001 --warmup 8 --eval-every 15 --batch 1 --accum 4`
   Stop (Ctrl+C) if validation loss starts rising.
5. Chat text is private: never print or upload `data_clean`, `data_chat` or the input files. Leave `clean_data.py --show-top` off.

## Prompt for an AI to clean and improve the project
Give this to an AI coding tool with the project folder open. It works in STAGES so that credit never runs out before the files reach the owner. Paste the whole prompt once, then say "do stage 1", later "do stage 2", and so on.

```
You are working on Persona_LM, my personal chat AI project (pure NumPy/PyTorch, Cebuano/Tagalog/English). Read HANDOFF.md and README.md first.

## HOW YOU MUST WORK (credit protection)
- I may run out of credit at any moment. Work in STAGES (listed below). Do ONE stage per message, and only the stage I ask for.
- At the END OF EVERY STAGE (never later), before anything else, you must give me:
  1. the changed files only, as one zip (or each file attached),
  2. a 3-line summary (what changed, test result, what you did NOT touch),
  3. the PowerShell commands below, filled in with the real file names.
- If you think credit is running low, stop the current task, finish what already works, and deliver (1)-(3) NOW. A small finished stage is better than a big unfinished one.
- Keep replies short. Do not repeat file contents, do not re-read files you already read, and read only the parts of a file you need.
- Update HANDOFF.md (sections 4, 5 and 12) at the end of each stage and include it in the zip.

## COMMANDS YOU MUST ALWAYS END WITH (PowerShell, run in C:\Users\user\persona_lm_project)
```powershell
# 1. unzip the changed files over the project (or copy them in)
Expand-Archive -Force "$HOME\Downloads\STAGE_N_files.zip" .
# 2. run the tests
python -m unittest discover -s tests
# 3. commit and push to main (add ONLY the changed files, never "git add .")
git status
git checkout main
git add <list every changed file here>
git commit -m "<short message for this stage>"
git pull --rebase origin main
git push origin main
```
Before the push, tell me to run `git ls-files` and check that no chat data, `.jsonl`, `.npz`, `.bin`, `work*` or `private/` files are tracked.

## HARD RULES (never break these)
1. NEVER delete, overwrite, move, rename or edit my data or models: work/, work2/, work3/, data_msgr*, data_chat*, data_clean*, data_pre*, public_text/, public_clean/, public_mix/, private/, kaggle_upload/, any *.jsonl, *.bin, *.npz, *.csv, and work\tok.json (the frozen tokenizer; models depend on it).
2. NEVER print, open for display, upload or commit my private chat contents.
3. Do not delete any file. If something looks useless, move it to _to_delete\ and list it for me. Use cleanup_files.py in dry-run mode first.
4. Do not change the tokenizer format, checkpoint format, or any id/vocab layout. Old checkpoints and token files must still load.
5. One topic per commit. Run the full test suite before and after each stage. If a test fails, fix it or undo that change. Never skip or delete tests to make them pass.
6. If you are not sure something is unused, grep for it (code, tests, docs, notebook, UI) and leave it alone if you can't prove it.
7. Do not push or merge yourself. I run the commands.

## STAGES
### Stage 1: Audit (no changes to files)
Run the tests (baseline). List unused functions/imports/modules, duplicate code, obsolete root scripts and stale docs, each with a risk level. Wait for my OK. Deliver: the list only (no zip needed).

### Stage 2: Safe cleanup (after my OK on the list)
Remove unused imports and dead code inside persona_lm/. Move obsolete root scripts to _to_delete\ (never delete). Check .gitignore covers all data/model folders and private files. Deliver: zip + commands.

### Stage 3: Menu and chat fixes
- Menu offers only real tokenizers (keep the current menu.py), and warns when a checkpoint's tokenizer fingerprint does not match.
- Chat defaults to the newest chat-tuned run (run_chat*), not run_pre, with a warning that a base run cannot chat.
- Friendly errors instead of raw KeyError/tracebacks for wrong file or missing folder.
Deliver: zip + commands.

### Stage 4: Sampling
Add chat flags --top-p and --rep-penalty (default off so behaviour does not change), a stop on repeated lines, and tests. Deliver: zip + commands.

### Stage 5: Safety tools
A "doctor" command (checks tokenizer/checkpoint/data fingerprints match, says what is wrong in plain words), automatic backup of ckpt + tokenizer before any training run, best-checkpoint saving, early stop when validation loss rises. Deliver: zip + commands.

### Stage 6: Data tools
De-duplication report, real vs synthetic ratio, train/val leak check. Read-only: results go to a new folder, originals are never changed. Deliver: zip + commands.

### Stage 7: Proposals (no code until I choose)
Suggest more improvements, each with what it does, effort (S/M/L), risk to my data, and how to test it (for example a larger ~100M model with a data-size guide, Windows checks, a better README). Wait for my choice.
```
