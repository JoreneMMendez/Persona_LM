# HANDOFF: persona_lm (a language model from scratch, pure Python + NumPy, optional PyTorch)

**DO NOT DELETE THIS FILE.** It is the project's memory between sessions and between AIs. `tests/test_handoff.py` fails if it disappears or loses a section.
Rebuilt on 2026-11-15 because the previous copy had been overwritten by notes from an unrelated project (CSC 201 / BOFO documentation). The facts below come from README.md, the code, and the 2026-11-15 training log. Anything marked (unverified) was not checked. If an older copy of this file exists in git history or a backup, merge its stage details into section 5.

## START HERE
- Owner: Jorene. Windows, PowerShell, project folder `C:\Users\user\persona_lm_project`. GPU: RTX 3050 6 GB laptop. PyTorch 2.14 + CUDA, bf16.
- What it is: tokenizer, transformer with hand-written backprop, AdamW, resumable training, chat tuning, sampling, GGUF export. Headless, command line. Personal chat AI trained on the owner's Messenger chats (Cebuano/Bisaya, Tagalog, English).
- Run tests: `python -m unittest discover -s tests -v`. Read README.md for the commands.
- Rules for any AI working here: never print or upload private chat data (`data_msgr*`, `data_chat*`, `*.jsonl`, `*.bin`, `private/`); never overwrite `work\tok.json` (frozen tokenizer); never delete files, move them to `_to_delete\` and tell the owner.
- Delivery rule for changes (owner's standing request): when you change the project, give only the changed files (as a zip) to use less credit, and end every reply with the PowerShell command that applies them.
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
- Tests: 665 after U2b (`python -m unittest discover -s tests`, 2 skipped without PyTorch); `test_handoff.py` needs this file.

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
