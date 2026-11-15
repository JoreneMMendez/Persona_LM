# HANDOFF: persona_lm (a language model from scratch, pure Python + NumPy, optional PyTorch)

**DO NOT DELETE THIS FILE.** It is the project's memory between sessions and between AIs. `tests/test_handoff.py` fails if it disappears or loses a section.
Rebuilt on 2026-11-15 because the previous copy had been overwritten by notes from an unrelated project (CSC 201 / BOFO documentation). The facts below come from README.md, the code, and the 2026-11-15 training log. Anything marked (unverified) was not checked. If an older copy of this file exists in git history or a backup, merge its stage details into section 5.

## START HERE
- Owner: Jorene. Windows, PowerShell, project folder `C:\Users\user\persona_lm_project`. GPU: RTX 3050 6 GB laptop. PyTorch 2.14 + CUDA, bf16.
- What it is: tokenizer, transformer with hand-written backprop, AdamW, resumable training, chat tuning, sampling, GGUF export. Headless, command line. Personal chat AI trained on the owner's Messenger chats (Cebuano/Bisaya, Tagalog, English).
- Run tests: `python -m unittest discover -s tests -v`. Read README.md for the commands.
- Rules for any AI working here: never print or upload private chat data (`data_msgr*`, `data_chat*`, `*.jsonl`, `*.bin`, `private/`); never overwrite `work\tok.json` (frozen tokenizer); never delete files, move them to `_to_delete\` and tell the owner.
- Delivery rule for changes (owner's standing request): when you change the project, give only the changed files (as a zip) to use less credit, and end every reply with the PowerShell command that applies them.
- ADDING MORE CHATS: always use `clean_data.py` (see "How to add more chats" at the end of this file). Never feed raw chat files to `prepare` and never merge files by hand.
- Before the AI's tokens run out, update this file (owner's standing request).

## 1. Layout
- `persona_lm/` package (model.py, bpe.py, train.py, torch_backend.py, chatrepl.py, generate.py, fastgen.py, memory.py, emotion.py, identity.py, personas.py, proposals.py, runtime.py, export_pkg.py, gguf.py, ...). `cli.py` is the command line.
- Root scripts: `clean_chat.py` (clean chat data, merge, de-duplicate stock replies), `setup_s30m.py` (make work2 for s30m), `grab_data.py` / `grab_english.py` / `ceb_domains.py` / `clean_ceb.py` / `clean_corpus.py` / `quick_check.py` / `make_mix.py` / `make_tok_sample.py` / `check_tok.py` (one-off public-text preparation), `prep_mood_data.py`, `train_s30m_kaggle.ipynb` (s30m pretraining on Kaggle).
- Folders on the owner's computer (not in the zip, ignored by git): `work\` (s10m, tok.json, data_pre, run_pre, run_chat), `work2\` (only a copy of data_pre), `work3\` (s30m: run_pre, run_chat, run_chat2), `data_chat\`, `data_chat4096\` (chat tokens used by the s30m run), `kaggle_upload\data_pre`.

## 2. Models
- s10m (13.7M, old, work\). s30m = 33.5M parameters: 8 layers, d_model 512, 8 heads, d_ff 1376, context 4096. s30m must use the same tokenizer as s10m (`work\tok.json`).
- Pretraining and chat tuning must use the same `--size`. Checkpoints (`ckpt.npz`) hold weights, optimizer, RNG and step; the previous one is `ckpt.prev.npz`.

## 3. Data
- Chat data comes from `messenger-import` (window mode, up to 8 context turns, loss only on the owner's replies), then `clean_chat.py`, then `prepare --kind chat`.
- Window mode repeats every message as context in up to 9 examples. This makes the model memorize quickly.

## 4. Training facts that matter
- s30m chat fine-tune, 2026-11-15: 3000 steps, about 16,400 tokens per step (probably `--batch 1 --accum 4` at ctx 4096), lr 1e-4 to 1e-5, from `work3\run_pre\ckpt.npz`.
- Result: validation loss bottomed at step 800 (0.937) and then rose to 1.351 at step 3000 while train loss fell to 0.27. This is overfitting. The 3000-step checkpoint (`work3\run_chat`) chats badly: it ignores the input and answers with stock replies ("Kuyog ko", "BWHAHAHA", "Asa ka?").
- Data size: data_chat4096 = 4,116,902 train tokens, 433,857 val tokens (from data_msgr_chats, tokenizer sha f187b974...). 3000 steps x 16,384 tokens = about 12 passes; 800 steps = about 3 passes. The model overfits after about 3 passes (corrected 2026-11-15: an earlier note said 0.8 passes, which was wrong).
- Retrain 2026-11-15 (`work3\run_chat2`, 800 steps, lr 1e-4 -> 1e-5, warmup 20, batch 1 accum 4): val 1.239 (100), 1.057 (200), 0.986 (300), 0.939 (400), 0.926 (500), 0.921 (600), 0.916 (700), **0.910 (800)**. This is the best model so far; use it. Replies now fit the chat style (group-chat topics) but it still does not answer questions properly. The Messenger export has only about 1,440 messages by the owner, which is the real limit.
- The 150-message-context re-import (`messenger-import --context 150`) was never completed (cancelled at the chat-selection prompt). Not recommended: longer context repeats the same messages more and overfits faster.
- The trainer keeps only the latest checkpoint and the one before it, so the best step is lost unless the run is stopped there.
- 6 GB GPU: the default micro-batch (8) at ctx 4096 runs out of memory. Use `--batch 1 --accum 4`.

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
- Stage U0: local web UI (`python -m persona_lm ui`).
- Stage U1a (2026-11-15): web UI Chat core. See "Newest state: web UI U1a" below.
- Stage U1b (2026-11-15): web UI chat controls. See "Newest state: web UI U1b" below.
- Stage U2a (2026-11-15): web UI Memory list. See "Newest state: web UI U2a" below.
- Stage U2b (2026-11-15): web UI graph, mood, Person card. See "Newest state: web UI U2b" below.
- 2026-11-15: old-chat archive (`--archive`), `clean_chat.py` merge mode, s30m setup, s30m chat fine-tune run (overfit, see section 4).
- Stage U3a (2026-11-15): web UI Prompt panel. See "Newest state: web UI U3a" below.
- Stage U3b (2026-11-15): web UI personas page and mask switcher. See "Newest state: web UI U3b" below.
- Stage U4a (2026-11-15): web UI Data page, part 1. See "Newest state: web UI U4a" below.
- Stage U4b (2026-11-15): web UI Data page, part 2 (the cleaner). See "Newest state: web UI U4b" below.
- Stage U5a (2026-11-15): web UI Training page, part 1 (plan, checkpoints, run numbers and log; read only). See "Newest state: web UI U5a" below.
- Stage U5b (2026-11-15): web UI Training page, part 2 (start/stop/resume, live loss chart, rising-val warning). See "Newest state: web UI U5b" below.
- Stage U8 (2026-11-15): web UI "Language data" page: download + clean + check English / Tagalog / Cebuano / Taglish public text from the menu. See "Newest state: web UI U8" below.
- Tests: 722 after U4b (705 after U4a, 688 after U3b; `python -m unittest discover -s tests`, 2 skipped without PyTorch); `test_handoff.py` needs this file.

## 6. Known problems
- Chat quality is poor at the overfit checkpoint (section 4).
- GGUF output untested in llama.cpp. Tagalog/Bisaya emotion word lists unchecked. Memory/archive use by the model is untested (the model was not fine-tuned on memory blocks).
- The chat REPL started without `--system` / `--mood` may differ from the training prompt format (unverified; check whether the training data has a system or mood line).
- `.gitignore` was missing `work3/` and `data_chat4096/` (fixed 2026-11-15).

## 7. Milestone plan
1. DONE 2026-11-15: s30m chat tuning at 800 steps, val 0.910 (`work3\run_chat2`).
2. Get more of the owner's own text (older Messenger exports, other apps); this is the only change that raises the ceiling. Also try `clean_chat.py --max-dup` lower, or session mode.
3. Test chat with full sentences, temperature 0.3 to 0.5.
4. Export to GGUF for the phone and test in llama.cpp.
5. Fine-tune with memory/mood blocks so the model learns to use them.

## 8. Commands
- Retrain (already done once; for new data keep about 3 passes): `python -m persona_lm train --run work3\run_chat2 --data-dir data_chat4096 --size s30m --init-from work3\run_pre\ckpt.npz --steps 800 --lr 0.0001 --min-lr 0.00001 --warmup 20 --eval-every 100 --batch 1 --accum 4`
- Chat: `python -m persona_lm chat --ckpt work3\run_chat2\ckpt.npz --tokenizer work\tok.json`
- Stop training safely with Ctrl+C; the same command resumes.

## 9. Privacy
- Chat data is anonymized by default and stays local. `.gitignore` excludes chats, checkpoints, memory, mood, person card and work folders. Real chat copies (`*.bak`) were once found in the repo: check `git ls-files` before every push.

## 10. Cleanup status
- `cleanup_files.py` (dry run by default; --apply moves safe items into `_to_delete\`) was written on 2026-11-15; not yet run on the owner's computer. It moves duplicate `ckpt.prev.npz`, duplicate `data_pre` copies and junk only. Superseded runs (work\run_chat, work\run_chat2, work3\run_chat = the overfit 3000-step run) wait for the owner's decision.

## 11. Open questions for the owner
- Is the s10m model in `work\` still needed? Which data folder is the real chat set: `data_chat` or `data_chat4096`?

## 12. Next action
Run `python cleanup_files.py` (dry run, then --apply) now that training is finished. Chat with `work3\run_chat2` using full sentences and `/temp 0.5`. Then gather more of the owner's own messages (milestone 2) or export to GGUF (milestone 4). Update sections 4 and 5 after each run.

## Newest state: web UI U1a (chat core), 2026-11-15
Web UI plan and stage prompt: `docs/UI_PROMPT.md` (U0 and U1a done; next is U1b, chat controls). One stage per session.
- Built: choose a checkpoint + tokenizer (or an identity core folder) from files inside the project, load it, send messages, bubbles, typing dots, live reply while it is written, Stop, Enter/Shift+Enter, token counter (used / window, draft count, warning when the oldest messages fall out of the window). Chat is now the start page.
- Files added: `persona_lm/ui_chat.py` (ChatHub), `persona_lm/ui_static/chat.js`, `tests/test_ui_u1a.py`. Changed: `ui_server.py` (routes, `work3` added to PathGuard.ALLOWED_TOP), `fastgen.py` (optional `should_stop` / `on_token`, default off, same answers), `app.js`, `app.css`, `tests/test_ui_u0.py` (route table and stage name).
- Endpoints: GET `/api/chat/sources|state|poll?job=`; POST `/api/chat/load|send|stop|count`. All need token + Host/Origin checks. A message goes through `AI.chat()` -> `say()`, never `handle()`: "/approve 1" is only text. No route approves, edits identity or writes memory.
- Defaults: temperature 0.5 (HANDOFF section 7 advice), max_new 120, top_k 40. U1b adds the settings drawer. Loading again starts a new conversation. Reply is kept (marked "stopped") if you press Stop mid-way.
- Limits: not checked in a real browser (none available in the build session): run the browser check once on your PC (light, dark, narrow width, console). Memory, mood badge, archive recall are not wired into the chat page yet (U1b/U2/U3); checkpoint mode chats without them. Identity core mode loads the core's own memory/mood/Person card as the runtime does (the model's memory use is untested, see section 6).
- Run: `python -m persona_lm ui --root C:\Users\user\persona_lm_project`, open the printed link, Chat page, pick `work3/run_chat2/ckpt.npz`, tokenizer `work/tok.json`.

## Newest state: web UI U1b (chat controls), 2026-11-15
Next stage: U2a (memory list). `docs/UI_PROMPT.md` has the plan.
- Built: Settings drawer (temperature, top-k, longest reply; same limits as the typed /temp and /max; not while a reply is being written), Save (new file `saved_chats/chat_DATE_TIME[_n].txt`, name made by the server, never overwrites), Clear (two-step button; keeps memory.json and mood.json), live mood badge (label; hover shows valence/arousal).
- Mood is OFF by default in checkpoint mode: tick "Use a mood" on the load screen (then it reads/writes `mood.json`, and adds a mood line to the prompt; the model was not fine-tuned on it, so replies may change). In identity core mode the core's own mood is used. Only the typed message moves the mood (tested: the AI's words cannot).
- Endpoints added: POST `/api/chat/settings|reset|save`. `/api/chat/state` and finished polls now carry `mood` and `max_new_limit`. `load` accepts `mood: true` (exactly true) and, for cores, `version`.
- Files: changed `ui_chat.py`, `ui_server.py` (`saved_chats` in PathGuard and the Privacy page), `chat.js`, `app.css`, `.gitignore` (`saved_chats/`), `tests/test_ui_u0.py`, `tests/test_ui_u1a.py`; added `tests/test_ui_u1b.py`.
- Limits: still not checked in a real browser; saved chats contain what you wrote (private, git-ignored); Save/Clear/Settings are disabled while a reply is written.

## Newest state: web UI U2a (memory list), 2026-11-15
(U2a done; U2b followed, see below.)
- Built: Memory page (sidebar "Memory"): search (words, names, ids, feeling), filters (status, pinned only), sort (newest/oldest/strongest/most important), Show more, add, edit (text, feeling, importance, pin), Pin/Unpin, Forget (two clicks), and "Tidy up" = maintenance preview, then Apply.
- Files: added `persona_lm/ui_memory.py` (MemoryHub), `ui_static/memory.js`, `tests/test_ui_u2a.py`. Changed `ui_server.py` (routes, stage name), `app.js`, `app.css`, `tests/test_ui_u0.py` (route table: the memory routes are now allowed and listed).
- Endpoints: GET `/api/memory/list|plan`; POST `/api/memory/add|edit|forget|maintain`. All need token + Host/Origin. They touch only `memory.json` (PathGuard) through MemoryStore: same validation, atomic save and conflict check as the typed commands; log lines never hold the text. Forget needs `confirm` = the id and saves without backup (like the CLI). Tidy-up applies only the previewed plan (`plan_id`); if memory changed since, nothing is applied. Chat text cannot reach these routes (chat still uses say()).
- SAFEGUARD NOTE: before U2a no route wrote memory (a test said so). U2a adds owner-only memory writes as the stage list asked; the test now lists exactly these routes. Identity, proposals and approvals are still untouched.
- Limits: memory.json in the project root only (an identity core's own memory file is not shown yet). Editing text re-links `[[Name]]` and plain mentions of known nodes; links made by hand with the CLI `--node` are kept. If the chat page has a model loaded with memory, its in-RAM copy is not refreshed: reload the model to see UI changes there. Browser check done once (headless Chromium: search, add, edit error, tidy, forget, light/dark/phone width, no console errors except the deliberate 400).
- Run: `python -m persona_lm ui --root C:\Users\user\persona_lm_project`, open Memory.

## Newest state: web UI U2b (graph, mood, Person card), 2026-11-15
Next stage: U3a (prompt panel). `docs/UI_PROMPT.md` has the plan.
- Built: Memory page now has List | Graph tabs. Graph: force layout (same picture every time), circle size = strength, lines = shared memories, drag to pan, wheel or +/- to zoom, click a node (or use the text node list) to see its active memories and neighbours. Only the 150 biggest nodes are drawn. Mood and Person page: mood now (label, valence/arousal, drift applied in RAM only), mood map, history chart (valence and arousal of the labels that moved the mood), last messages; Person card editor: add, edit, change type (owner/auto/no) and remove ONE line at a time.
- Files: added `persona_lm/ui_mind.py` (MindHub), `ui_static/mind.js`, `tests/test_ui_u2b.py`. Changed `ui_server.py`, `app.js`, `memory.js` (tabs), `app.css`, `tests/test_ui_u0.py` (route table).
- Endpoints: GET `/api/graph`, `/api/mood`, `/api/person` (read only; mood.json is never written); POST `/api/person/add|edit|remove`. Token + Host/Origin on all. A line is changed only if the `old` text the page saw still matches (else 409). Editing a line's text makes it an owner line so a rebuild keeps it; the old file is kept as person_card.prev.md. Only person_card.md, mood.json, memory.json are touched (PathGuard).
- SAFEGUARD NOTE: second owner-only write area after U2a (memory). Mood cannot be changed from the UI; `person build` stays in the terminal.
- Limits: the mood history stores only labels, so the chart shows each label's typical valence/arousal, not the exact values at that time. Dormant/archived memories are not drawn. Identity core's own memory/mood/card files are not shown yet. Browser check done once (headless Chromium: graph click/zoom, mood page, card add/edit, phone width, no console errors).

## Newest state: web UI U3a (prompt panel), 2026-11-15
Next stage: U3b (persona switcher + personas page). `docs/UI_PROMPT.md` has the plan.
- Built: Chat page has a "Prompt" button. It opens a side panel (beside the chat on wide screens, a drawer on narrow ones) that shows the EXACT text the model read for the newest message: token meter (used / prompt budget / window), blocks in order (system line, persona, Person card, mood line, memory + old-chat recall, then the kept messages and the open "AI writes here" marker), the memories and old chats that were recalled, how many oldest messages were left out, and a warning when memory matches existed but did not fit the memory budget. Refreshes by itself after each reply while open.
- New in the loader (checkpoint mode only, both OFF by default): "Use memories (memory.json)" and "Recall old chats (chat_archive.jsonl)". Before this, checkpoint chat had no memory or archive at all. Identity-core mode still uses the core's own memory/mood/Person card (no archive).
- Files: changed `ui_chat.py` (prompt snapshot taken inside the generate hook, so it is the real prompt, not a rebuild; `prompt()`), `ui_server.py` (GET `/api/chat/prompt`, `chat_archive.jsonl` added to PathGuard.ALLOWED_TOP, stage name), `runtime.py` (`load(archive=...)`, optional, default None), `chat.js`, `app.css`, `.gitignore` (`chat_archive.jsonl`), `tests/test_ui_u0.py`; added `tests/test_ui_u3a.py` (11 tests).
- Endpoint: GET `/api/chat/prompt` (token + Host/Origin like every /api route; read only; 409 if no model; no POST route). Only the literal `true` turns memory/archive on; no path is ever taken from the page.
- SAFEGUARD NOTE: with "Recall old chats" ticked, every finished exchange is appended to chat_archive.jsonl (same as the terminal `--archive`). Opt-in, off by default, one fixed file name inside the project. Using memories also updates the "used" count in memory.json (same as the terminal). Model text still cannot run a command or write memory (a test sends "/forget all" and checks it is only a message).
- Limits: not checked in a real browser (none in the build session): run the browser check once on your PC (light, dark, narrow width, console). The panel shows what was captured for the last message; after Clear or reloading the model it says nothing was sent yet. Long sections are cut at 1,500 characters in the panel only (the model still got all of it). The panel does not show why a memory scored high, only which were chosen.
- Run: `python -m persona_lm ui --root C:\Users\user\persona_lm_project`, Chat > Change model > tick "Use memories", load `work3/run_chat2/ckpt.npz`, send a message, press Prompt.

## Newest state: web UI U3b (personas page + mask switcher), 2026-11-15
Next stage: U4a (Data page part 1: Messenger scan/import wizard, privacy audit, data-report verdict). `docs/UI_PROMPT.md` has the plan.
- Built: sidebar "Personas" page (the old "Prompt and personas" entry; the prompt panel itself lives in Chat since U3a). It lists the masks of the identity core loaded in Chat, with text, example count, trained version and a STALE warning, and the exact sentence the model reads for the active mask. Buttons: "Wear this mask" / "Take off". A selector lists other core folders read only. Chat header gets a "No mask / NAME" selector when the loaded core has masks. Adding, editing and removing masks stay in the terminal (the page shows the commands).
- Files: added `ui_static/personas.js`, `tests/test_ui_u3b.py` (12 tests). Changed `ui_chat.py` (`personas()`, `use_persona()`, `_cores()` split out, `persona`/`personas` in chat state), `ui_server.py` (routes, section ready, stage name), `app.js`, `chat.js`, `app.css`, `tests/test_ui_u0.py` (route table + PERSONA set).
- Endpoints: GET `/api/personas[?core=identity_core/NAME]` (read only), POST `/api/personas/use` `{name}` ("none" takes the mask off). Token + Host/Origin on both. The switch calls `AI.switch_persona()` (same as typed `/persona`): it is logged in the core's lineage log and saved in the core's persona file; weights and identity text are not touched. Refused while a reply is being written, with no identity core loaded, or for an unknown name. The conversation is kept.
- SAFEGUARD NOTE: fourth owner-side write (after memory, Person card, and opt-in archive). Only the page's own button reaches it; chat text goes through say() so "/persona joker" typed in Chat is only a message (tested). Identity, proposals, approvals untouched.
- Limits: not checked in a real browser (none in the build session): run the browser check once. Only the core loaded in Chat can be switched; other cores are read only. A mask whose text is too long for the model's window makes the next send fail with a clear message (nothing is kept) until you take it off. A trained version attached to a mask is NOT swapped in by the UI (the typed command does not either; only prompt text changes).
- Run: `python -m persona_lm ui --root C:\Users\user\persona_lm_project`, Chat > Identity core tab > load, then open Personas.

## Newest state: web UI U4a (Data page, part 1), 2026-11-15
Next stage: U4b (cleaner + duplicate report). `docs/UI_PROMPT.md` has the plan.
- Built: sidebar "Data" page. (1) Import wizard: 5 steps (scan, import, audit, prepare, report) marked done / NEXT from what exists on disk, each with the exact PowerShell command and a Copy button. (2) Privacy audit: reads `data_msgr/stats.json` only and shows anonymised yes/no, leaks, suspicious, masked counts, dropped kinds, a verdict (ok / warn / bad). (3) Data-report verdict for one token folder + model size: tokens, loss tokens, tokens per parameter, tokenizer fingerprint check against `work/tok.json`, validation warning, and steps for about 3 passes (batch 1, accum 4).
- DESIGN DECISION (safeguard kept): the Messenger scan and import are NOT run from the UI. The raw export is outside the project and the hard rules say the UI never reads it. The wizard shows the commands and checks what they produced. Running the scan from the page would need a new owner decision (see below).
- Files: added `persona_lm/ui_data.py` (DataHub), `ui_static/data.js`, `tests/test_ui_u4a.py` (17 tests). Changed `ui_server.py` (routes, STATIC, stage name, `data_chat4096` added to PathGuard.ALLOWED_TOP so its token counts can be shown), `app.js`, `app.css`, `tests/test_ui_u0.py` (stage name).
- Endpoints (GET only, token + Host/Origin): `/api/data/overview`, `/api/data/audit`, `/api/data/report?folder=&size=`. Folder names come from fixed lists (IMPORT_DIRS, TOKEN_DIRS); the page cannot send a path. No POST route was added.
- Privacy: no message text, name or number is sent. `private/`, `train.jsonl`, `val.jsonl` are never opened. `persona_settings.json` is read for one yes/no (export folder remembered); its path and names are not sent. A test plants a secret in private/, train.jsonl and audit.txt and checks it never appears.
- BUG FIXED: `/static/personas.js` (U3b) was missing from `STATIC` in ui_server.py, so the browser got a 404, the module graph failed and NO page rendered (stuck on "Connecting..."). Added, plus a test that every .js/.css file and every `./x.js` import is served.
- Browser check: done once with headless Chromium on the Data page (light, dark, 380 px wide, no console errors, no sideways scroll). The other pages were not re-checked after the personas.js fix: open each once on your PC.
- Limits: the audit view shows only what stats.json holds. Reviewing `private/audit.txt` and `capitalised_words.txt` is done by opening those files yourself. Data-report on a big pretraining folder (`data_pre`) scans the whole file and can take a few seconds.
- Open question for the owner: allow the page to run `messenger-scan` on an export folder that is OUTSIDE the project (read only)? That weakens the "UI never reads the raw export" rule, so it is not done without an explicit yes.
- Run: `python -m persona_lm ui --root C:\Users\user\persona_lm_project`, open Data.

## Newest state: web UI U4b (Data page, part 2: the cleaner), 2026-11-15
Next stage: U5a (Training page, part 1). `docs/UI_PROMPT.md` has the plan.
- Built: "Clean the chats" card on the Data page. Settings (max short / longer identical reply, keep other people's lines) -> Preview (in memory, writes nothing): examples before -> after for train and val, what was removed (call notices, other people's group lines, repeated short/long replies, no-target examples), chats -> files, and a repeated-replies histogram before / after (buckets 2-3, 4-10, 11-50, over 50 times; top-10 share). Button "Show the most repeated replies" shows reply TEXT (opt-in). Button "Write data_msgr_chats" (two clicks) writes the previewed plan and shows the `prepare` command.
- Refactor: `clean_chat.py` now has `load_dir`, `run_clean`, `write_output`; `main()` calls them (same output, its 13 tests pass). The UI loads the project's own `clean_chat.py` and calls those functions, so the page and `python clean_chat.py --dir data_msgr` write identical files (a test compares them). Only `--dir data_msgr` mode is in the UI; `--input FILE ...` stays in the terminal.
- Files: added `tests/test_ui_u4b.py` (17 tests). Changed `clean_chat.py`, `persona_lm/ui_data.py`, `ui_server.py` (3 routes, stage name), `ui_static/data.js`, `tests/test_ui_u0.py` (route table, DATA whitelist), `tests/test_ui_u4a.py` (the one allowed data POST).
- Endpoints: GET `/api/data/clean-preview`, `/api/data/repeats[&text=1]`; POST `/api/data/clean-apply {plan_id}`. Token + Host/Origin on all. Params are whole numbers 0 to 100 (else 400).
- SAFEGUARD NOTE (fifth owner-side write): clean-apply writes ONLY `data_msgr_chats` (through PathGuard, so a link out of the project is refused with 403), only for a plan that was previewed on `train.jsonl` / `val.jsonl` that have not changed since (size + mtime, else 409), once per plan. It never deletes: an existing `data_msgr_chats` is MOVED to `_to_delete\data_msgr_chats_DATE_TIME` (the terminal command deletes old chat files in it; the page does not). Chat text cannot reach it (tested).
- PRIVACY NOTE: preview and the default repeats report send counts only. The opt-in text view is the only place chat text reaches the page: refused (403) if stats.json says the import was not anonymised, cut to 30 characters, at most 15 rows, text from the anonymised files only. Say if you want this removed.
- Limits: the cleaner reads the whole `train.jsonl` into the server's memory (refused over 400 MB; use the terminal then); preview and apply each take a few seconds on a big import. Browser check done once (headless Chromium: preview, repeats text, write, light, dark, 380 px wide, no console errors; the earlier pages were not re-checked).
- Run: `python -m persona_lm ui --root C:\\Users\\user\\persona_lm_project`, Data > Clean the chats > Preview.

## Newest state: web UI U5a (Training page, part 1), 2026-11-15
Next stage: U5b (start/stop prepare + train as a background process, live loss chart, rising-val warning, resume). `docs/UI_PROMPT.md` has the plan.
- Built: sidebar "Training" page, READ ONLY. (1) Pipeline plan for work / work2 / work3: the same steps as `pipeline plan` marked done / NEXT / skipped, the exact command with a Copy button, "decisions confirmed" badge (confirming stays in the terminal), no-project case shows the `pipeline init` command. (2) Checkpoints: every run folder with a ckpt, step / planned steps, model shape, best and last val loss with a "rising" warning (last > best x 1.03), tokenizer fingerprint vs `work/tok.json` (matches / DIFFERENT), file size and date, "only .prev" flag. (3) Details for one run: last step, train loss, best val, a validation table, and the log (numbers from metrics.jsonl, or the tail of a `*.log` file in the run folder).
- Files: added `persona_lm/ui_train.py` (TrainHub), `ui_static/training.js`, `tests/test_ui_u5a.py` (22 tests). Changed `ui_server.py` (4 GET routes, STATIC, stage name, section ready), `ui_static/app.js`, `app.css`, `tests/test_ui_u0.py` (stage name).
- Endpoints (GET only, token + Host/Origin on all): `/api/train/plan?workdir=`, `/api/train/checkpoints`, `/api/train/history?run=work3/run_chat2`, `/api/train/log?run=&file=`. No POST route, no process is started, nothing is written (a test checks the module source).
- Safeguards: only work, work2, work3 and one run level below them (PathGuard); a run is `WORKFOLDER/RUNFOLDER` only; log files must be named `*.log` and sit directly in the run folder (last 64 KB, 200 lines); only the checkpoint's small meta block is read, never the weights, no pickle; the plan does not open the chat files named in project.json (their paths appear only inside the commands).
- Limits: no loss chart yet (U5b). The trainer prints to the terminal, so a `.log` file exists only if you redirect output (PowerShell: `... | Tee-Object work3\\run_chat2\\train.log`); otherwise the log view is rebuilt from metrics.jsonl. The best step is not kept by the trainer, so a run past its best step can only be retrained. Browser check done once (headless Chromium: light, dark, 380 px wide, no console errors; a stray "null" text and a phone-width sideways scroll were found and fixed).
- Run: `python -m persona_lm ui --root C:\\Users\\user\\persona_lm_project`, open Training.

## Newest state: web UI U5b (Training page, part 2), 2026-11-15
Next stage: U6a (Identity page). `docs/UI_PROMPT.md` has the plan.
- Built: Training page card "Run a step": Start / "Resume from step N" for the NEXT pipeline step (pretrain-data, pretrain, chat-data, finetune), shown with its exact command and a confirm button. While it runs: progress bar, live loss chart (train line, val dots, ring on best step), log tail, "Stop and save" and "Force stop". Warning when val loss is more than 3 percent above its best (also in the run Details view and the chart there). Runs in the background and keeps going if the page or UI server closes; the page picks it up again.
- SAFEGUARD NOTE (owner asked for this stage): first route that starts a program. The page sends only {workdir, step, confirm:true}; the command always comes from the project's pipeline plan (same as `pipeline run`). Not startable here: the tokenizer step (freezes tok.json), a step that is not NEXT, decisions not confirmed, tok.json fingerprint mismatch, any step using a file outside the project or in data_msgr/private (shown as a terminal command). One job at a time. No shell, one Popen, list arguments.
- Endpoints: GET `/api/train/job?workdir=` (job state + which steps can start); POST `/api/train/start`, `/api/train/stop` (`{}` = save a checkpoint then end, `{"force":true}` = terminate). `/api/train/history` now also returns `rising`, `evals_since_best`, `last_val`.
- Files: added `persona_lm/ui_job.py` (JobHub), `persona_lm/ui_jobrun.py` (wrapper: heartbeat + stop-file watcher, runs the normal CLI), `tests/test_ui_u5b.py` (29 tests), `tests/fake_jobrun.py` (stand-in trainer for tests). Changed `ui_server.py`, `ui_train.py` (history only; still read only), `ui_static/training.js`, `app.css`, `tests/test_ui_u0.py`, `tests/test_ui_u5a.py` (route table, stage name, SVG namespace allowed).
- Files the job writes: `<work>/ui_job.json`, `<run>/ui_train.log` (training) or `<work>/ui_prepare.log`, `<work>/ui_job.stop` while a stop is asked. The log is also listed in the run Details log picker.
- Limits: stop uses SIGINT raised inside the wrapper (same path as Ctrl+C); tested on Linux only, NOT on Windows: if "Stop and save" does not end the run within a minute, use Force stop (loses progress since the last checkpoint, default every 100 steps). A job with no heartbeat for 120 s shows as "no sign of life". The best step is still not kept by the trainer. Browser check done once (headless Chromium: light, dark, 380 px, no console errors, no sideways scroll; start, chart, warning, stop all worked with the stand-in trainer, not with real training).
- Tests: 773 (2 skipped without PyTorch).
- Run: `python -m persona_lm ui --root C:\\Users\\user\\persona_lm_project`, open Training, press Start on the NEXT step.


## Newest state: web UI U8 (Language data page: download + clean), 2026-11-15
- Built: sidebar section **Language data** (`#/corpus`). Tick English / Tagalog / Cebuano / Taglish, choose "download, clean, check" or "only clean what is downloaded", optionally "also rebuild the training mix", press Start (confirm box shows the exact command). Background job with step N of M, log tail, Stop (finish the current document) and Force stop. Keeps running if the page or UI is closed. A box at the bottom says what each work folder's `pretrain_text` reads (warns if it points at the RAW `public_text/`) and gives the `prepare` command for new token files.
- Same thing from the terminal: `python -m persona_lm corpus [--lang en,tl,ceb,taglish] [--mode all|clean] [--mb en=420 tl=300 taglish=100] [--mix]`.
- Files: `persona_lm/corpus.py` (the pipeline; REUSES grab_english.stream_into, grab_data.grab, ceb_domains.main -> clean_ceb.main, grab_taglish.main, clean_corpus.clean_tl, make_mix.build), `persona_lm/ui_corpus.py` (CorpusHub), `persona_lm/ui_static/corpus.js`, `cli.py` (corpus command), `ui_server.py` (routes, section, PathGuard + STORAGE entries), `tests/test_ui_corpus.py` + `tests/fake_corpusrun.py` (39 tests, no network; the REAL cleaning scripts run on stub downloads).
- Endpoints: GET `/api/corpus/overview`, GET `/api/corpus/job`, POST `/api/corpus/start` {languages, mode, mb, mix, confirm, ack_taglish}, POST `/api/corpus/stop` {force?}. The page sends language ids, a mode and numbers only; the server builds the command.
- SAFEGUARD CHANGES (owner asked for the download): (1) the UI's background job can now reach the internet, only after Start + confirm, and only to fetch public text; the Privacy page wording was changed to say so. (2) PathGuard ALLOWED_TOP gained `public_text`, `public_clean`, `public_mix` (public text only; private/ stays blocked). Side effect: a project whose `pretrain_text` points into public_mix/ or public_clean/ can now start its "Prepare the public text" step from the Training page. UI stage name is now "U8".
- Safety: every output goes to `<name>.part` first and is moved into place only when its step finished, so a stop, a crash or a network drop never replaces a good file with a half-written one. English that ends below 90 % of the asked size is treated as a cut-off download and not used. Taglish (PinoyExchange, licence unknown, real users) needs a ticked acknowledgement; e-mails, links, @handles and phone numbers are replaced while cleaning (new `clean_taglish`, same patterns as the Messenger anonymiser).
- Needs `pip install datasets` for the download (the page shows this when it is missing). Disk check before a download (about 2.5 x the requested size).
- Limits: the real downloads were NOT run in the build session (no network); the cleaning rules and the job runner were tested on stub downloads and through the real `ui_jobrun`. The page was not looked at in a browser (none available): do one pass (light, dark, narrow width). `make_mix` budgets come from make_mix.DEFAULT_MB (Taglish 60 MB). New text is only used after tokenizing again (`prepare` into a NEW folder such as `data_pre_new`, same frozen tokenizer); the Training page cannot redo a finished step. Taglish was not part of the original three-language mix: if you use it, add `public_mix\taglish.jsonl` to the tokenizing command (the page's command already does). Tests: 868 after U8.

## Newest state: clean_data.py (one *_cleanData file per input), 2026-11-15
- Built: root script `clean_data.py`. `python clean_data.py --input merged_chats_clean_mood.jsonl synthetic_chats_100k.jsonl` cleans each .json/.jsonl (or every file in a folder) SEPARATELY and writes `data_clean\train\<name>_cleanData.jsonl` plus `data_clean\val\<name>_cleanData.jsonl` (last 5 % of each chat's sessions) and `data_clean\report.json` (counts only). Nothing is merged. Files already named *cleanData* are skipped.
- Cleaning = the clean_chat.py rules (imported from it) plus: bad lines/records/empty messages dropped, exact duplicate examples dropped. Repeated-reply limits (--max-dup 3, --max-dup-long 5) are counted over ALL files together, so list the owner's real chats FIRST.
- All files are used for training: `prepare --kind chat --tokenizer work\tok.json --data data_clean\train --val-data data_clean\val --out data_chat` reads whole folders.
- Measured on the owner's two files: merged_chats_clean_mood 1016 -> 973 train + 43 val; synthetic_chats_100k 100000 -> 34757 train + 4854 val (57,047 short and 3,342 long repeated replies dropped: the synthetic file is extremely repetitive). The synthetic file is about 35x bigger than the real one, so it dominates both training and validation; `--max-examples N` caps each file.
- Chat text is not printed unless `--show-top` is given. Tests: tests/test_clean_data.py.

## How to add more chats (owner's standing request, 2026-11-15)
When the owner has new chat files (.jsonl in the chat/session/messages format, or .json lists), do this. Do not merge anything; each file stays its own file.
1. Put the new .jsonl file(s) in the project folder `C:\Users\user\persona_lm_project`.
2. Clean ALL the files again in one run, the owner's real chats FIRST, then synthetic/other data:
   `python clean_data.py --input <real chats>.jsonl <new chats>.jsonl <synthetic>.jsonl --max-examples 5000`
   Each input becomes `data_clean\train\<name>_cleanData.jsonl` (and `data_clean\val\<name>_cleanData.jsonl`). The script deletes the old *_cleanData files first, so always list every file you want used, not just the new one. Already-cleaned files (name contains "cleanData") are skipped if you point it at a folder.
   `--max-examples N` caps each file so a huge synthetic file cannot drown out the owner's real chats (it is a cap per file; the real chat file is usually below it). Lower N (e.g. 1500) if replies sound generic.
3. Make the tokens (uses every file in both folders, tokenizer stays `work\tok.json`, never regenerate it):
   `python -m persona_lm prepare --kind chat --tokenizer work\tok.json --data data_clean\train --val-data data_clean\val --out data_chat`
4. Fine-tune from the pretrained base into a NEW run folder, about 3 passes only (steps = 3 x train tokens / 16,384 at `--batch 1 --accum 4`, ctx 4096):
   `python -m persona_lm train --run work3\run_chatN --data-dir data_chat --size s30m --init-from work3\run_pre\ckpt.npz --steps <steps> --lr 0.0001 --min-lr 0.00001 --warmup 8 --eval-every 15 --batch 1 --accum 4`
   Stop (Ctrl+C) if validation loss starts rising; the trainer keeps only the latest checkpoint and the one before it.
5. Chat text is private: never print or upload data_clean, data_chat or the input files. `clean_data.py --show-top` prints reply text; leave it off.
- State on 2026-11-15: `merged_chats_clean_mood.jsonl` (1,016 examples) + `synthetic_chats_100k.jsonl` (100,000 examples, very repetitive) cleaned with `--max-examples 5000` -> train 973 + 3,638, val 43 + 257 -> `prepare` gave 574,600 train tokens and 37,190 val tokens (about 35 steps per pass, so about 105 steps for 3 passes). First fine-tune from this data: `work3\run_chat3` (result not yet checked).
- Why fine-tuning alone cannot teach it to talk: pre-training teaches the language, fine-tuning only style. The base `work3\run_pre` limits how well it can answer; check its tokens per parameter on the Data page (about 20 per parameter is the usual target, 4 passes over the same text is about the useful limit).
- Old checkpoints listed in the chat dropdown can be moved to `_to_delete\runs` (keep work3\run_pre, work3\run_chat2, work3\run_chat3; never move work\tok.json).

## Newest state: CPU + GPU together for training (--cpu-share), 2026-11-15
- Owner's request: "update the project so it can use both CPU and GPU for training". Built in `persona_lm/torch_backend.py` (`choose_cpu_share`, `make_cpu_params`, `sync_cpu_params`, `hybrid_accumulate_step`, `probe_cpu_share`; `train_torch(cpu_share=, cpu_threads=)`) and `persona_lm/cli.py` (`train --cpu-share 0|auto|N --cpu-threads N`). Default is OFF (`0`): without the flag training is exactly as before.
- How it works: each optimizer step has `--accum` micro-batches. The GPU does most; a worker thread does the last N on the CPU in fp32 with its own copy of the weights (refreshed from the GPU each step). CPU gradients are added to the GPU gradients before clipping and AdamW, so the update is the same as GPU-only (up to rounding). Checkpoint, data order and optimizer are unchanged, so a run can be continued with or without the flag.
- `auto` times one micro-batch on each device (own RNG, the data order is not touched) and uses the CPU only if the predicted step is at least 5 % faster; otherwise it logs that the CPU is too slow and the GPU works alone. Needs `--accum 2` or more (the GPU always keeps at least one micro-batch). Ignored on a CPU-only run and with the NumPy backend.
- Expect a modest gain at best (the laptop CPU is much slower than the RTX 3050; about 10 to 25 % when it works), extra RAM (about 3 GB or more at ctx 4096 for the CPU micro-batch), more heat and fan noise, and possible thermal throttling on a laptop. Evaluation always runs on the GPU. Example: `python -m persona_lm train ... --batch 1 --accum 4 --cpu-share auto`.
- Tests: `tests/test_hybrid.py`. The arithmetic and command-line tests ran (full suite 789 OK, 6 skipped). The 4 PyTorch checks (split gradients equal plain gradients, loss scale, clamping, error from the CPU thread) were SKIPPED because the AI that wrote this had no PyTorch and no GPU: the hybrid code itself is UNTESTED until the owner runs `python -m unittest tests.test_hybrid -v` and a real `--cpu-share auto` run on the laptop.

## Newest state: nicer terminal for training, 2026-11-15
- Owner's request: "make the terminal have better UI". Built `persona_lm/pretty.py` (`TrainDisplay`, `make_log`) and wired into `cmd_train` in `persona_lm/cli.py` (new flag `train --plain`).
- On a real terminal, `train` shows a live panel: progress bar, loss, validation with an arrow and the best step so far, lr, tokens/s, elapsed, ETA, a loss sparkline and a red warning when validation is more than 3 % above its best (same rule as the web UI). Other messages (backend, resumed at step, hybrid) print above the panel. When training ends there is a summary with the best validation step and a note that the trainer only keeps the latest two saves.
- IT DOES NOT CHANGE THE LOG LINES. Training still produces the same `step N/M loss ... tok/s ETA ... h` lines; the panel is only a different way to show them and only when stdout is a terminal (at least 80 columns). Files, pipes, the web UI's training runs and tests get plain `print`, so nothing that parses the logs breaks. `--plain` or `PERSONA_PLAIN=1` switches it off, `NO_COLOR=1` removes colours, a console that cannot show block characters gets an ASCII bar.
- Windows: ANSI is switched on with SetConsoleMode (Windows 10 and later); if that fails it falls back to plain lines. NOT yet seen on the owner's real PowerShell: if the panel looks broken (stacked or garbled lines), use `--plain` and report what the terminal shows.
- Tests: `tests/test_pretty.py` (12 tests, fake terminal). Only `train` has the panel; `prepare`, `clean_data.py` and the chat REPL are unchanged.

