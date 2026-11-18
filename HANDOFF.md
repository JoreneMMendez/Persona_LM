# HANDOFF: persona_lm (a language model from scratch, pure Python + NumPy, optional PyTorch)

**DO NOT DELETE THIS FILE.** It is the project's memory between sessions and between AIs. `tests/test_handoff.py` fails if it disappears or loses a section (keep the headings "START HERE", "5. Stage log", "7. Milestone plan", "12. Next action").
Anything marked (unverified) was not checked on the owner's computer.

## CURRENT FOCUS (read this first)
The owner's NEW request (2026-11-17): rework the terminal menu so people with no technical background can use it. The plan is written as a prompt with 7 stages (T1 to T7) in the next section. **The prompt comes first and NOTHING is built until the owner types `build T<n>`.** The older cleanup stages 1 to 7 are all DONE (history is kept at the end of this file).
If the owner types just "stages" or "status": print the stage status line below and build nothing. Only the exact words `build T<n>` start work, one stage at a time, in order.
Terminal UI stage status: T1 DONE | T2 DONE | T3 DONE | T4 DONE | T5 not started | T6 not started | T7 not started. (Update this line at the end of every stage.)
Old cleanup stages (history): Stage 1 to 7 DONE (2026-11-16; owner chose P1, P2, P6, P8 in Stage 7).

## TERMINAL UI REWORK: PROMPT FIRST, BUILD ONLY ON COMMAND (owner request, 2026-11-17)
**Status: T1 to T4 are built. T5 to T7 are NOT built.** Stage status: T1 DONE | T2 DONE | T3 DONE | T4 DONE | T5 not started | T6 not started | T7 not started. (Update this line at the end of every stage.)
What the owner wants: the terminal menu (`python -m persona_lm menu`) reworked so a person with NO technical background can use it. Every section says what it does, in plain words. Built in 7 small stages, and only when the owner types `build T1`, `build T2`, and so on.
How to use: (1) open a new AI session with the project zip + this file, (2) paste the prompt below ONCE, (3) the AI answers only "Ready", (4) type `build T1`; when it is delivered and tested, type `build T2`; and so on. Nothing is built without the words `build T<n>`.

````text
You are continuing Persona_LM (zip + HANDOFF.md attached). The job: rework the terminal menu so a non-technical person can operate it. The plan below has 7 stages, T1 to T7.

## BUILD GATE (rule 0, above everything else)
- Until I type exactly "build T<n>" (n = 1 to 7), do NOT read project files, edit, create, run anything, or plan in detail. Answer with ONE line: "Ready. Say 'build T1'." (use the first stage not marked DONE).
- "build T<n>" = build only that stage, then stop. If an earlier stage is not DONE, answer in one line ("T3 needs T2 first. Say 'build T2'.") and do nothing else.
- "stages" or "status" = print the stage status line from HANDOFF.md, build nothing. Words like "continue" or "go on" are NOT a build command.
- A question about a stage ("what will T4 do?") = answer in 5 lines or less from the stage text below, without reading files.

## READER AND PLAIN-WORDS STANDARD (applies to all text you write)
- The reader is a smart adult who has never used a terminal. No jargon without a plain explanation: say "model file" (not checkpoint), "prepared text files" (not token files), "test score, lower is better" (not val loss). Keep the technical name visible but grey and second, e.g. "Teach the AI (train)", so docs still match.
- Every screen answers four questions: What is this? What happens if I pick it? What do I need first? What do I get, and where is it saved?
- Sentences of 18 words or less. Lines wrap to the terminal width (design for 60 columns). Say what is safe: "This does not change your chats or your models."
- Every error says what went wrong, why, and what to do next. No tracebacks (PERSONA_DEBUG=1 may still show one).

## HARD RULES (all rules in HANDOFF.md "HARD RULES" still apply, plus these)
1. Do not change what any menu item runs: command names, flags, `build_argv`, checkpoint format, tokenizer format, `work\tok.json`. This is a TEXT and SCREEN rework only.
2. Keep the keys `label`, `cmd`, `fields`, `settings`, `preset` of every ITEMS entry, and keep the ITEMS list ORDER (tests use ITEMS[0] = train, ITEMS[1] = chat, ITEMS[QUIT] = quit). Show a different order only through a separate display order, and map the number keys to it.
3. Windows console must still work: ASCII fallback (Skin unicode_ok), NO_COLOR, `--plain`, and the numbered fallback menu (no boxes) must show the same plain text.
4. Never print or read the owner's chats. The menu may look only at file NAMES and sizes (scan()), as it does now.
5. All existing tests must pass. If a test pins a string you intentionally renamed, change only that expected string and list the test in your summary. Never skip or delete tests.
6. Do not push or merge. The owner runs the commands.

## TOKEN-EFFICIENCY BUILD RULES (follow all of them)
A. Reading
 1. Never read HANDOFF.md fully. Read only CURRENT FOCUS, START HERE, and the stage you are building (grep -n for the heading, then sed -n for those lines).
 2. Never open a big file whole: menu.py (about 1070 lines), cli.py (about 1360), pretty.py (about 300). Get an outline with `grep -n "^def \|^class " FILE`, then `sed -n A,Bp FILE` for ONLY the functions named in the stage's "Read" line. Read each piece once and remember it.
 3. Never read tests whole (grep the name of the test, or just run it). Never open data, model or work folders, ui_static, or docs that the stage does not list.
 4. At most 8 read commands before your first edit. If you need more, you are reading the wrong place: use grep instead.
B. Writing
 5. All plain-language text lives in ONE data file, `persona_lm/menu_text.py` (plain dicts and strings, no logic), created in T1. Later stages add strings there with small edits, so text changes never touch logic code.
 6. Edit with small targeted str_replace. Never rewrite or re-print a whole file. Do not reformat old code or rename functions. Create a new file in ONE call, no drafts, no placeholders.
 7. One short comment line per new function at most.
C. Testing
 8. Order: tests for the new text first (tests/test_menu_text.py, created in T1, grown each stage), then code. Run ONLY the affected files while working: tests/test_menu.py, tests/test_menu_finetune.py, tests/test_menu_text.py (+ tests/test_pretty.py if pretty.py changed, tests/test_corpus_ui.py if corpus_ui.py changed). ONE full-suite run at the very end of the stage.
 9. Run quietly: `python -m unittest FILE 2>&1 | tail -15`. Never use -v on the full suite. On a failure read only the failing assertion, fix it, rerun that one file. After 3 failed repairs of the same thing, undo that edit and report it.
 10. No visual checking loops. Instead, render each new screen ONCE into a StringIO at widths 60 and 80 inside a test and assert: no line wider than the width, every text key exists, no UnicodeEncodeError on a cp1252 stream (see StrictStream in persona_lm/selfcheck.py), no jargon word that is missing from the glossary.
D. Replies
 11. One status line at the start ("Building T<n>: <3 words>"), no narration while working. Final reply: 15 lines or less plus the delivery block. Never paste code, file contents or test logs.
 12. Do not ask questions unless blocked. Use the defaults in the stage, and state your assumptions in ONE line.
 13. No extra features, no polish pass, no re-checking finished work. Ideas go in ONE "Later ideas" line.
E. Credit protection
 14. A stage should touch 4 files or fewer and change about 300 lines or fewer (mostly text). If it is bigger, build the first working part, tick only that part, list what is left, and stop.
 15. If credit seems low: stop, finish what already works, deliver now. Never leave half-written files or failing tests.
 16. At the end of the stage update HANDOFF.md: the stage status line (CURRENT FOCUS and this section), at most 12 lines in the Stage log (section 5), and section 12. Put HANDOFF.md in the zip.

## THE 7 STAGES
T1 Plain-words text bank (no screen changes yet)
 Read: menu.py ITEMS (about lines 386-423), HELP and SETTINGS (about 219-300). Files: NEW persona_lm/menu_text.py, NEW tests/test_menu_text.py, a small hook in menu.py.
 Build: for every ITEMS entry a record with: title (plain name), what (1-2 sentences), when (when to use it), need (what must exist first), result (what you get and where it is saved), time (rough time), safe (what it does NOT change). A GLOSSARY (model file, tokenizer, prepared text files, run, base model, fine-tune, test score/overfitting, batch/accum, learning speed, GPU/CPU/torch, recipe, ...), one plain sentence each. FRIENDLY_ERRORS: pattern -> 3-part message (what happened / why / what to do) for out of memory, missing file or folder, wrong tokenizer, torch not installed, disk full, no internet, stopped by Ctrl+C.
 Done when: tests prove every ITEMS entry has every key, sentences are 18 words or less, every glossary term used in the text is defined, and old tests still pass.

T2 Main menu screen
 Read: Menu.header/render/select, _fallback, scan() found keys. Files: menu.py, menu_text.py, tests/test_menu.py (+ test_menu_finetune.py only if a pinned string changed).
 Build: group the list under headings: START HERE, TALK TO YOUR AI, TEACH YOUR AI, PREPARE DATA, CHECKS AND TOOLS, then Quit. Show the plain title with the technical name in grey. The Details panel shows What it does / When to use / You need / You get / Time / Safe. The Project panel becomes "Your project" with ready-checks in words ("Your AI model: ready / none yet", "Prepared chat data: ...", "Graphics card: found / not found, training will be slow") and ONE "Next suggestion" line made from scan(). The numbered fallback shows the group headings and one plain line per item.
 Done when: both screens render at width 60 and 80 with no overflow, number keys still pick the item shown, old tests pass.

T3 Question forms
 Read: Form.lines/key/problem, collect, _field and the _d_* defaults. Files: menu.py, menu_text.py, tests.
 Build: every question gets a plain label, a one-line help under the highlighted row, an example, and an explained default ("Enter = newest run: run_chat3"). "?" shows the longer help. Problems are plain ("That folder already exists. Pick a new name like run_chat4, or it will continue the old run."). Footer in words: "* = you must answer, Esc = go back". The typed-line fallback (collect) prints the same help.
 Done when: every field of every item has label, help and example in menu_text.py (test), forms render at width 60.

T4 Confirmation and settings screens
 Read: Confirm, confirm_box, confirm_train, confirm_corpus, _train_lines, _corpus_lines, edit_settings, SETTINGS. Files: menu.py, menu_text.py, tests.
 Build: a "You are about to ..." box in plain words: what will happen, what will be created (folder names), what will NOT be changed, time and disk estimate ONLY where the code already has the numbers (else "unknown until it starts"). The exact command goes last under "Technical command (for experts)". Default answer is No for long actions. Rewrite SETTINGS what/up/down/tip in plain words (18 words or less each) and add three named presets with one plain line each: "Safe (recommended)", "Gentle chat tuning", "Stop it memorizing". Warnings in words ("more than about 4 passes over the data makes it memorize").
 Done when: no setting text uses a term missing from the glossary (test), confirm screens fit width 60.

T5 Guided setup (new item "Start here: step by step")
 Read: scan() and run_menu loop. Files: menu.py, menu_text.py, tests.
 Build: ONE new ITEMS entry placed just before Quit in the list (never at index 0 or 1) but shown FIRST by the display order. It shows a checklist with DONE / NEXT / LATER per step: 1 check this computer, 2 get language data, 3 make prepared text files, 4 teach the AI, 5 try it (chat), 6 chat for real. The state comes only from scan(). Pick a step to open the matching existing menu item. A step that is not a menu item (adding the owner's Messenger chats, clean_data.py) shows the exact command in a copy-ready box with one plain sentence; the menu never reads the Messenger export.
 Done when: tests cover the state of the checklist for an empty folder, a folder with data only, and a folder with a trained run.

T6 Live screens, results, errors, help
 Read: TrainDisplay labels in pretty.py, the except blocks in run_menu and in cli.main (message text only), the Press-Enter step. Files: pretty.py, menu.py, cli.py (error text only), menu_text.py, tests.
 Build: training panel labels in plain words ("Test score, lower is better", "Learning speed", "Time left", the overfitting warning in a sentence); `--plain` output unchanged in structure. After every command a "What happened / What to do next" box (finished, failed, or stopped with Ctrl+C: "Safe to stop. Choose the same item again to continue."). FRIENDLY_ERRORS used in the menu and in cli.main (PERSONA_DEBUG=1 still shows the traceback). New key "?" or "h" in the menu = a Help page: the glossary, the keys, "if you are stuck", and where your files are.
 Done when: tests for each friendly error, test_pretty.py passes (change only intentionally renamed strings).

T7 Check, Windows safety, docs
 Read: selfcheck.py (the menu/panel check only), README.md menu part, docs/QUICKSTART.md headings. Files: tests/test_menu_text.py, selfcheck.py (only if a menu check already exists there), README.md, docs/QUICKSTART.md.
 Build: a sweep test that renders EVERY screen at widths 60, 80, 100, with NO_COLOR, with a cp1252 StrictStream and in the numbered fallback (no overflow, no UnicodeEncodeError, reading level: average sentence 18 words or less). A short "Use the menu (no typing commands)" section in README.md and QUICKSTART.md. An 8-line "try it on your PC" checklist for the owner. Mark the real Windows console look as UNVERIFIED until the owner confirms.
 Done when: the full suite passes and the owner has the checklist.

## DELIVERY (end of EVERY stage, nothing else first)
1. One zip with only the changed files: STAGE_T<n>_files.zip (include HANDOFF.md).
2. A 3-line summary: what changed, test result, what you did NOT touch.
3. The PowerShell commands as a NUMBERED list of SINGLE lines, in C:\Users\user\persona_lm_project. Under each one a plain sentence: what it does and what I should see. Order: unzip over the project, run the tests, `git status`, `git checkout main`, `git add <each changed file>` (never `git add .`), `git commit -m "..."`, `git pull --rebase origin main`, `git push origin main`. Before the push tell me to run `git ls-files` and check that no chats, .jsonl, .npz, .bin, work* or private/ files are tracked.
4. One line: "Try it: python -m persona_lm menu" and what I should look for.

Now answer only: "Ready. Say 'build T1'."
````

## START HERE
- **CURRENT GOAL (owner, 2026-11-17): the terminal UI rework, stages T1 to T7 (section "TERMINAL UI REWORK" above). The prompt is written; build ONLY when the owner types `build T<n>`.** Do not start a stage on your own, and do not ask what to build: answer "Ready. Say 'build T2'." (or the next stage that is not DONE). When a stage is built, do only that stage and finish it with the changed-files zip plus numbered single-line PowerShell commands.
- Old cleanup plan (stages 1 to 7, 2026-11-16): all DONE (P1, P2, P6, P8 built; P3 tokens-per-parameter report, P4 window-mode repeat reducer, P5 memory/mood fine-tune data, P7 GGUF test kit were offered and NOT chosen).
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

- 2026-11-17: owner asked for a terminal UI rework for non-technical users. Only the PROMPT was written (section "TERMINAL UI REWORK", 7 stages T1 to T7, build gate, token-efficiency build rules, delivery format). No code changed. T1 to T7: not started.
- T1 (2026-11-17) DONE: plain-words text bank. NEW `persona_lm/menu_text.py` (data only): ITEM_TEXT (keyed by the ITEMS `label`; keys title, what, when, need, result, time, safe) for all 12 items, GLOSSARY (24 terms, one sentence each), FRIENDLY_ERRORS (7 entries: pattern, what happened, why, what to do: out of memory, missing file, wrong tokenizer, torch missing, disk full, no internet, Ctrl+C), JARGON (words that must be in the glossary). `menu.py`: one import and `plain_text(item)`; ITEMS untouched, no screen changed. NEW `tests/test_menu_text.py` (7 tests: every item has every key, no orphan records, sentences of 18 words or less, jargon is defined, planned glossary terms, errors match real error texts, cp1252-safe). Later stages add strings to menu_text.py.
  Not done in T1 (by plan): screens, grouping, forms, confirm boxes, help page. Times in `time` are rough words, not measured (unverified).
- T2 (2026-11-18) DONE: main menu screen. `menu_text.py` gained GROUPS (START HERE / TALK TO YOUR AI / TEACH YOUR AI / PREPARE DATA / CHECKS AND TOOLS, then Quit), TECH_NAMES, DETAIL_FIELDS, SCREEN, NEXT, and a `short` line per item (some titles/texts shortened so Details fit 80 columns). `menu.py`: LAYOUT/DISPLAY (display order; ITEMS order and keys untouched), `number_key(i)`, `project_checks(found)`, `next_suggestion(found)` (from scan() only), `Menu.render/select` rewritten (list left, Details with What/When/Need/Get/Time/Safe right, \"Your project\" box + Next suggestion below), `_fallback` shows the same groups and words, wrapped to the width. Number keys and Up/Down follow the order shown; the menu starts on \"See what you have\". The screen is now MENU_H = 27 lines tall (was 12), needs a console of at least 29 lines (else the numbered list). Only 1-9 work as single keys (items 10 and 11 by arrows; the numbered list accepts 10, 11). \"Graphics card\" line reports PyTorch installed, not the card itself (scan() cannot tell). Tests changed (renamed rows, new display order): test_menu.py (Screen x3, Loop x2, 4 number keys), test_menu_finetune.py (MenuScrolls x2). Tests grew in test_menu_text.py (8 new). Full suite 1036 OK, 6 skipped. UNVERIFIED: real Windows console look.
- T3 (2026-11-18) DONE: question forms. `menu_text.py` gained FIELD_TEXT (every field of every item: label, help, more, example, default note), FORM (footer, Enter notes, info lines) and PROBLEMS (plain error texts). `menu.py`: `field_text`, `enter_note`; `Form.lines/problem/key` and `collect` use them: plain label, one-line help, `Example:`, `Enter = <what the default is>` (e.g. next free name: run_chat4), footer `* = you must answer, Esc = go back`; `?` shows the longer help (in a text answer `?` is a normal letter once you have typed or changed something; the typed-line `collect` treats an answer of just `?` as help). The About box is 8 lines (was 7) and the form has one more line (footer). The checks `_check_new_run`, `_check_base_ckpt`, `_check_langs`, `_check_recipe_id` now return plain messages (the existing-run one suggests the next free name). Commands, flags, ITEMS and `build_argv` untouched. Pinned strings changed in tests: test_menu.py ('choose at least one of' -> 'separated by commas', 2 places), test_menu_finetune.py ('no such file' -> 'not found', 'CONTINUE' -> 'continue the old run'). New tests: tests/test_menu_text.py (Forms, 5). UNVERIFIED: real Windows console look.
- T4 (2026-11-18) DONE: confirmation and settings screens. `menu_text.py` gained SETTING_TEXT (plain name/short/what/up/down/affects/tip for the 9 training settings), PRESET_TEXT (3 named presets), SETTINGS_UI, CONFIRM. `menu.py`: `SETTINGS` keeps only key/kind/lo/hi and takes its words from SETTING_TEXT; `PRESET_VALUES`/`QUICK`: 1 Safe (recommended) = batch 1, accum 4, eval_every 100 (assumes a small graphics card, as in section 4), 2 Gentle chat tuning = lr 1e-4, min_lr 1e-5, warmup 50, steps 3000, 3 Stop it memorizing = weight_decay 0.2, lr 3e-4, min_lr 3e-5, eval_every 100. `edit_settings` (typed-line screen, wraps to the width) shows plain names, \"passes over your data\" and a warning above about 4 passes when the size of train.bin is known (`_train_tokens`: file SIZE only), else a general memorizing sentence. `_train_lines`/`_corpus_lines` give \"You are about to ...\" / What will happen / What will be created / What will NOT be changed / time and disk (numbers only where the code has them: the download size is the sum of the languages' default_mb, else \"unknown until it starts\") and the exact command last under \"Technical command (for experts)\". DEFAULT IS NOW NO: Enter cancels in `Confirm` (boxed) and in `confirm_train`/`confirm_corpus` (typed: `[y/N]`); Taglish still needs the whole word `yes`. Commands, flags, ITEMS and build_argv untouched. Pinned tests changed (default No, preset order, flow answers): test_menu.py (Confirm, Forms x3, LanguageData x3, Loop, Settings preset test). New tests: tests/test_menu_text.py (Confirmations 4, SettingsScreens 4). UNVERIFIED: real Windows console look; the 'Safe' numbers are a judgement call.

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
- Same reply to everything? `python -m persona_lm chat-probe --ckpt work3\run_chat2\ckpt.npz work3\run_chat3\ckpt.npz` (numbers only; `--show` prints replies on screen).
- Steps for about 3 passes + ready train command: `python -m persona_lm plan-steps --data-dir data_chat4096 --size s30m --run work3\run_chat4 --init-from work3\run_pre\ckpt.npz`.
- Self-check of the Windows PC: `python -m persona_lm doctor --windows`.
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
- `work\tok.seal.json` is written and read by `python -m persona_lm tok-check` (it records the frozen tokenizer's fingerprint). It is not a tokenizer; keep it (the menu already ignores it).

## 12. Next action
FIRST: the terminal UI rework. The prompt is in the section "TERMINAL UI REWORK" near the top. T1 to T4 are DONE. T1 to T4 are DONE. Wait for the owner to type `build T5` (then T6, T7, in order). The items below are the older to-do list and stay valid.
All 7 cleanup stages are DONE. Stage 7 delivered P1, P2, P6 and P8 (see the Stage 7 log). Next, on the owner's computer: run `python -m persona_lm doctor --windows` (the torch/hybrid/panel/menu parts have not run on real Windows hardware yet), `chat-probe` on `run_chat2` and `run_chat3`, and `plan-steps` before the next chat run. Open proposals if the owner wants them: P3, P4, P5, P7. Then:
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
(HISTORY: all 7 cleanup stages below are DONE. The active prompt is "TERMINAL UI REWORK" near the top of this file.)
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


## Stage 1/2 log (2026-11-16)
- Stage 1 audit: baseline 915 tests, 1 failure (tests/test_menu.py fixture writes `{}` as tok.json; menu now needs merges+specials; fix in Stage 3).
- Stage 2: removed unused imports (buildtokens sys, pipeline shlex, proposals time, ui_job re), removed unused `generate.chat_loop` and its `import sys`, widened .gitignore. Owner moves `run_scraper.py`, `python_reddit_bisaya_scraper.py`, `bisaya_chats.csv`, `bisaya_data.csv` to `_to_delete\`. Not touched: re-export imports in cli/identity/ui_server, doc renames.
- Stage 3 (2026-11-16): menu chat default = newest run_chat* (base run only as fallback); `chat` warns for a base run and for a tokenizer fingerprint mismatch (checkpoint `tokenizer_sha256`); friendly errors for missing ckpt/tokenizer and for KeyError/FileNotFoundError/JSON errors in `main` (`PERSONA_DEBUG=1` shows the traceback). Fixed the test_menu fixture. New tests: tests/test_stage3_chatfixes.py. 922 tests pass.
- Stage 4 (2026-11-16): `chat`/`generate` flags `--top-p` (not set = the built-in 0.95 as before), `--rep-penalty` (1.0 = off) and `--stop-repeat` (off); `generate.is_looping`, `pick_token(rep_penalty, recent)`; plain and cached sampling give the same tokens. Tests: tests/test_stage4_sampling.py.
- Stage 5 (2026-11-16): safety tools. New `persona_lm/safety.py` (shared by NumPy and PyTorch loops) and `persona_lm/doctor.py`. (1) Automatic backup: before a run starts/resumes, `ckpt.npz` (+ the tokenizer when `--tokenizer PATH` is given) is COPIED to `<run>\backups\<time>\` with `backup.json`; identical backups are not repeated; low disk space or a copy error only prints a warning; `--no-backup` turns it off. Backups are never pruned by the code (delete old ones yourself when disk is short). (2) Best checkpoint: every new lowest validation loss is saved to `<run>\ckpt.best.npz` (weights only, no optimizer, loads for chat/export like any checkpoint; `--save-best 0` turns it off). (3) Early stop: `--early-stop N` stops after N validation checks in a row without a new best (default 0 = off); the normal `ckpt.npz` is still written, so resuming works. Training maths, data order, checkpoint format and tokenizer are unchanged. (4) `python -m persona_lm doctor [--root DIR] [--tokenizer PATH] [--deep]`: read-only; checks tokenizer loads, token folders (fingerprint, vocab size, file sizes vs meta.json, chat mask file), checkpoints (fingerprint, vocab size, readable, ckpt.prev fallback, overfitting = latest val loss >5 % above best), `--deep` also looks for NaN/inf. Exit code 1 when there is a PROBLEM. Tests: tests/test_stage5_safety.py (16). 944 tests pass (6 skipped; PyTorch is not installed in the sandbox, so the torch-side hooks are identical code but UNTESTED on real hardware (unverified): run one short `train --backend torch` once).
- Stage 6 (2026-11-16): data tools, READ-ONLY. `python -m persona_lm data-audit --train data_clean\\train --val data_clean\\val --out data_audit --real merged_chats` (new module `persona_lm/datatools.py`). Writes `report.md`, `report.json`, `exact_duplicates.tsv` into a NEW folder (refuses an existing non-empty folder or one inside the data folders; never overwrites). It reports (1) exact duplicate examples, how often replies repeat and how many exceed clean_data's limits (3 short / 5 long); (2) real vs synthetic share of the TRAINED replies (`--real` = parts of file names); (3) train/val leak check: exact examples in both, (chat, session) in both (skipped for files without real session numbers), share of long val replies/messages that also occur in training. Only counts, file names, line numbers and hashes are written; no chat text is printed or saved. Exit code 1 only for a leak (PROBLEM). Tests: tests/test_stage6_datatools.py (7).
- Stage 7 (2026-11-16): owner chose P1, P2, P6, P8. (P1) `python -m persona_lm chat-probe --ckpt RUN_OR_CKPT [more ...]` (`persona_lm/probe.py`): sends 20 built-in plain messages (English, Tagalog, Bisaya; none from the owner's chats; `--prompts-file` for your own) in the chat prompt format and counts different replies, the share of the most common reply, replies that echo the message and empty replies. Verdict OK / warning / PROBLEM (STOCK REPLY when one reply is 30 % or more of the answers, or under 50 % of replies differ). Numbers only unless `--show`; several checkpoints are compared in a table; exit code 1 on a PROBLEM. Reads checkpoints only. (P2) `python -m persona_lm plan-steps --data-dir D --size S [--passes 3] [--batch 1 --accum 4] [--run R --init-from B] [--from-run OLD_RUN] [--tokens-per-sec N]` (`persona_lm/planner.py`): steps = passes x train tokens / (batch x accum x ctx) (4,116,902 tokens -> 754 steps for 3 passes; the 800-step run was 3.2 passes), about 8 validation checks per run, warm-up, lr 1e-4 -> 1e-5 for chat data. Warns about more than 4 passes, validation data under 2 windows (early stop cannot work), under 8 steps, under 50k trained tokens. Prints a `train` command with `--early-stop 3 --save-best 1` that the real parser accepts (tested). `--from-run` says at which step and pass an earlier run's validation loss was best. Read-only. (P6) `python -m persona_lm doctor --windows [--no-tests]` (`persona_lm/selfcheck.py`; the normal `doctor` is unchanged): Python/NumPy/PyTorch/GPU/bf16, disk, long path, OneDrive, console encoding and colour codes; a tiny NumPy training run + chat sample; a tiny PyTorch run with tokens/s; `--cpu-share 1` hybrid on a CUDA GPU; `tests.test_hybrid` in a subprocess; the training panel and a menu form drawn into a `StrictStream` that raises UnicodeEncodeError like a cp1252 console (the Skin falls back to ASCII by itself; forcing unicode crashes, tested). All on made-up text in a temp folder; never touches the project. Each check is guarded: a crash becomes a PROBLEM line. (P8) `docs/QUICKSTART.md` (first 10 commands, data-size guide incl. the s125m ~91M model, Windows troubleshooting), linked from README.md; tests check every command and flag in it against the real parser. Tests: tests/test_stage7_chosen.py (26). 977 tests pass (6 skipped; PyTorch is not installed in the sandbox). UNVERIFIED on real hardware: the PyTorch smoke run, the hybrid run, the `msvcrt`/colour-code checks and the Windows console behaviour in `doctor --windows`. The `chat-probe` thresholds (30 % / 15 % / 50 % / 75 %) are rules of thumb, not measured on the owner's models.
- Dataset recipes (2026-11-16): `hfdata` terminal command + menu item + `colab_hfdata.ipynb`: stream a public Hugging Face dataset (first recipe: pykeio/oshichats-v2), filter, clean, tokenize with the frozen tok.json. See "Newest state: dataset recipes" at the end.

## Newest state: dataset recipes (hfdata), 2026-11-16
- Owner's request: a Colab pipeline for `pykeio/oshichats-v2` (English, `languages[0] > 0.9`, adjustable `score` threshold, `donation == False`, up to 5M cleaned messages) that fits THIS project, plus "a new interface on the terminal" to repeat it for future datasets. Built `persona_lm/hfdata.py` (engine, wizard, commands), `hfdata` registered in `cli.py`, a menu item "Dataset recipes (create / run)" in `menu.py`, `colab_hfdata.ipynb`, `dataset_recipes/oshichats.json`, `docs/DATASET_GUIDE.md`, `tests/test_hfdata.py` (32 tests, no network, a fake opener). Full suite: 947 OK, 6 skipped.
- Decisions: NO new tokenizer and NO new model (the original prompt asked for an 8k BPE and a 30M GPT): `work\tok.json` stays frozen and `s30m` is trained by the normal `train` command (`--init-from` the pretrained checkpoint, lower lr). The web UI is untouched (its "no dataset name from the page" rule still holds); the new interface is terminal only.
- Recipe = JSON in `dataset_recipes/` (fixed filter operators, no eval, no trust_remote_code, repo id must look like owner/name). `hfdata run` writes shards + `state.json` in `public_text/hf_<id>/` (resume point), then `public_clean/<id>/<id>.jsonl` ({"text"}; messages of one `stream.id` grouped ~30 per document because `prepare` adds an end-of-document token per document) and `chat_clean.txt` (one message per line). `author` is never written. Every file goes through `.part`. A changed recipe needs `--restart` (old work is moved to `_to_delete\`).
- Tag handling: `<|liver|fans|maybe-collaborator:...|>` -> `[streamer]` / `[fans]` (recipe `clean.tags`), `<|:emote:|>` dropped (`clean.emotes`), repeated characters/words collapsed to 3, short messages dropped, links/handles scrubbed with `corpus.scrub_public`.
- NOT verified by the AI that wrote this (no internet / no GPU there): the real `datasets` streaming of the 30 GB file (only a fake opener was tested), the exact field names of oshichats-v2 (`message`, `score`, `languages`, `donation`, `stream.id` were taken from the dataset card; `hfdata peek oshichats` shows what is really there), the Colab notebook cells, `train --init-from` with these settings on a T4, and the one-line live status on the owner's PowerShell (`--plain` or PERSONA_PLAIN=1 turns it off). The score threshold 2.0 in the example recipe is a PLACEHOLDER: pick it from `hfdata peek`.
- Open: how much oshichats text to mix with Tagalog/Cebuano for the continued pretraining (`hfdata tokens --also ...`), and whether to use the oshichats licence (CC-BY-4.0 on the card) for anything beyond private training.
- Menu fine-tune item (2026-11-16): `menu.py` has a new item 3 "Fine-tune a model (chat tuning)" (same `train --init-from` command; asks base ckpt, NEW run folder run_chat<n>, chat token folder, size; starts the settings screen on lr 1e-4, min_lr 1e-5, warmup 50, steps 3000 for a NEW run only). The menu list now scrolls when there are more items than rows (it had silently cut off the last items). `tests/test_menu_finetune.py` (12 tests); two brittle position-based tests in `tests/test_menu.py` now look items up by command. Full suite: 1021 OK, 6 skipped.
