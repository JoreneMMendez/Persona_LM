# PersonaLM: handoff document

> **DO NOT DELETE OR REPLACE THIS FILE.** It lives at the root of the project (`persona_lm_project/HANDOFF.md`) and travels inside `persona_lm.zip`.
> **Update it at the end of every stage** (append to the stage log in section 5, refresh sections 4, 6, 7, 12). A test (`tests/test_handoff.py`) fails if it is missing or loses its START HERE / stage-log / next-action sections.
> Last updated: 2026-11-14, end of **Stage 5** (emotion + Person card); before that Stage 4 (memory graph), Stage 3 part 1, Stage 2 (Messenger importer) and Stage 1 (GPU). Give this file to the next AI **together with `persona_lm.zip`**.

## START HERE (read this first, about 2 minutes)

**State (2026-11-14).** Stage 0 audit: done. Stage 1 GPU backend: done and measured (see section 5). Stage 2 Messenger importer + anonymizer: done (synthetic data only). Stage 3 is HALF done ("part 1": pipeline, chat screen, reports). **Stage 4 memory graph: done** (owner ran 187 tests OK on their PC). **Stage 5 emotion + Person card: done** (`emotion.py`, `person.py`, chat `/mood`, `--mood --emotion-model --lexicon --person`, CLI groups `emotion`, `mood`, `person`, `docs/EMOTION_GUIDE.md`, `tests/test_stage5.py`; synthetic data only; NOT yet run by the owner). Old text follows: **Stage 4 (`memory.py`, `memory_export.py`, chat `/memory /remember /forget`, `memory` CLI, `docs/MEMORY_GUIDE.md`; tests on synthetic data; graph page checked in headless Chromium).
**OWNER'S DECISION (2026-11-14): build Stages 4-9 FIRST on tiny practice models; do the real tokenizer + real pretraining/fine-tuning (Stage 3 part 2) LAST, after their Messenger export exists.** Stage 6 (identity core + lineage + regression tests) is next. The owner said "next stage" to advance; do one stage per go.
Tests: **262 (260 pass + 2 PyTorch tests that run on the owner's machine)** in the author's sandbox; the owner's run of the 135 tests of the previous zip passed (OK) on Python 3.14.

**Answered by the owner (2026-11-14):** languages = English, Tagalog, Bisaya (Cebuano), Taglish, Bisaya+Tagalog, any mix of the three (vocab 16,000 recommended, not yet frozen; Bisaya public text is scarce, Cebuano Wikipedia is partly machine-made: say so when choosing the corpus); OK on `ctx=512` and the 32 reserved special tokens (after a plain-words explanation); "its own person speaking in the owner's style" = YES; unit tests 135 OK on their PC; `benchmark` done.
**Still waiting on the owner (later, not blocking Stages 5-9):** their Messenger export (they will export it after all stages are built) and the output of `messenger-scan`; the pretraining corpus choice. Do NOT freeze or train the real tokenizer until the Messenger data exists and they say OK (`pipeline confirm --yes`).

**What you can do now:** Stage 5, then 6-9, each tested on tiny/synthetic data. **Design debt to remember:** the real fine-tune data must contain `<|memory|>` blocks (and later mood/persona blocks) or the trained model will ignore them; `messenger-import` does not produce them yet.

**How to work with this owner (learned the hard way):**
- Not an expert. Windows 11 Home, PowerShell. Laptop: Acer Aspire 7, RTX 3050 6GB, 15.7 GB RAM, Python 3.14.6, torch 2.14.1+cu130, numpy 2.5.2. The project now lives at `C:\Users\user\persona_lm_project` (moved from the Desktop on 2026-11-14; always `cd C:\Users\user\persona_lm_project` before running commands). It is being put on the owner's GitHub as a PRIVATE repo named `persona_lm` (a `.gitignore` excludes `private/`, `*.jsonl`, checkpoints, models, `data/`, `runs/`, `exports/`; never commit real Messenger data). The owner copies commands straight from the chat (markdown link syntax in a pasted command broke one: write plain URLs/paths, never `[text](url)` inside a command).
- Give each command in its own code block. Never put two commands next to each other without a line break (a pasted pair got glued into one and failed). Never write a placeholder path without saying "replace this" (their export path command ran with my literal placeholder).
- **OWNER'S STANDING RULE (do not delete this line; keep it when handing over to another AI): every command that touches the downloaded zip must use the `*` wildcard in the file name, because their browser renames downloads (`persona_lm (6).zip`). Never write a fixed name like `persona_lm.zip`. Use the newest match, e.g. `Get-ChildItem "$env:USERPROFILE\Downloads\persona_lm*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Expand-Archive -DestinationPath C:\Users\user -Force`.** (Interpretation of the owner's words "the command must have this *": confirm with them if unsure.)
- Say how long a command should take, and tell them to press Ctrl+C and paste the screen if it takes more than 2 minutes. A test that waited for the keyboard cost them 3 hours.
- Always deliver the **complete project zip, not a patch**, plus the updated `HANDOFF.md` ("merge all files, don't give me just the update"). Before delivering: update this file, run the suite inside a pseudo-terminal (section 11), unzip the zip into a clean folder and run the tests from there, then present both files.
- Say plainly what was NOT tested. Own mistakes briefly, fix them, verify, move on.
- Their network once failed DNS lookups (a DNS server of 8.8.8.8 was configured); PyTorch downloads are about 2 GB and slow, so avoid asking for large downloads.

---

## 1. What this project is

The owner is building a **personal AI from scratch**, trained on their own Facebook Messenger messages, running locally.
The starting point is `persona_lm`: pure Python + NumPy, with a byte-level BPE tokenizer, a Llama-style transformer with
**hand-written backprop**, AdamW, resumable training, chat tuning, sampling and a GGUF exporter. It is CPU-only and has no
memory, emotion, identity or self-modification features.

The owner does **not** care how long it takes. They care about **correctness and learning how it works**.
They are not an expert: explain things in plain language, avoid jargon, and say what is and is not tested.

## 2. Working rules from the owner (follow these)

- Python 3.10+. **Explain every new dependency before adding it.**
- Keep code readable (the owner wants to learn from it).
- Keep **runtime code free of training-only dependencies** (no PyTorch in `persona_lm.runtime`).
- **Never claim something works unless it was tested.** Say "untested" when it is.
- If a goal is unrealistic at this scale, say so plainly, but still build it rather than refusing.
- Build **one milestone at a time**, in the order in section 6. After each: how to run it, what works, honest known limitations.
- Tests for every milestone. The existing 26 tests (gradient, resume, etc.) must keep passing.
- Ask at most 3 clarifying questions before starting if something blocks.
- Plan approval: the owner said "build it now" (2026-11-12), which was taken as approval of the plan in section 7.

## 3. Requirements (condensed from the owner's spec, in priority order)

### Goal 1: GPU training
- GPU backend using the local NVIDIA GPU (CUDA) with automatic CPU fallback. Recommend PyTorch or CuPy and explain why.
- Same code must run on rented GPUs (Colab, Kaggle, RunPod, Lambda, Vast.ai). Give a step-by-step guide for one beginner-friendly option.
- Checkpoints resume across machines (local <-> cloud). Mixed precision (fp16/bf16) and gradient accumulation for small VRAM.
- A `benchmark` command printing tokens/sec.

### Goal 2: Terminal-only
- Everything through a CLI: prepare data, train, chat, inspect memory, approve/reject changes, export, serve.
- Chat is a REPL with slash commands: `/memory /mood /identity /approve /reject /save /quit`.

### Goal 3: Train on Facebook Messenger export
- Importer for the JSON export (`messages[]`, `sender_name`, `timestamp_ms`, `content`). Fix the **mojibake** (UTF-8 bytes stored as latin-1 text).
- Handle group chats and multiple conversation folders. Skip photos, stickers, calls, system messages. Merge consecutive messages from the same sender.
- Owner chooses which sender is "me" and which chats to include.
- Chat format where the AI learns to respond the way the owner would, with the previous N messages as context. **Loss counts only on the owner's messages.**
- Everything stays local. Anonymize names and strip phone numbers, emails, addresses before any data goes to a cloud GPU.
- Be honest about data size and propose a pretrain-then-finetune plan.

### Goal 4: Memory graph (Obsidian-style)
- Nodes (people, topics, events, feelings) with weighted links. Bigger circles = more important/connected.
- Each memory: text, timestamp, source, emotion tag, importance score, links.
- Retrieval into context while chatting. Maintenance: merge duplicates, decay unused, summarize clusters.
- Export Markdown with `[[wiki-links]]` for Obsidian, plus a local HTML force-directed graph view.

### Goal 5: Emotion and knowing the owner
- Emotional state (valence/arousal + label) updated each turn, influencing tone. Train an emotion tagger from the owner's messages.
- A "Person card" (name, relationships, habits, speaking style, topics, values): built from data, updated over time, plain editable file.

### Goal 5b: Identity continuity across every upgrade
The AI must stay the **same individual** through every retrain, fine-tune, size or architecture change.
- **Identity lives outside the weights**: an "identity core" of files every version loads: (a) Person card + the AI's own self-description
  (name, personality, speaking style, values, relation to owner), (b) full memory graph, (c) emotional baseline, (d) lineage log
  (every version, date, change, who approved it).
- **Weights are replaceable.** Upgrades reproduce old behavior via: continued training (same architecture); distillation
  (old model's outputs + chat data as teacher) when size/architecture changes; and the **same frozen tokenizer, never changed**.
- **Identity regression tests**: fixed probe prompts + saved answers; each candidate version is scored (similarity to old answers,
  memory recall, tone/emotion match). Below the owner's threshold = auto-rejected, cannot be promoted.
- **Protected identity core**: changes to name/values/core personality/temperament need a diff, the owner's **typed** confirmation,
  a cooldown, and can **never** happen in the same step as a model upgrade.
- **Growth, not drift**: drift score between versions, warn if too fast. Keep every version for rollback; allow chatting with an older version.
- **Personas are masks**: switching persona never edits the identity core; the AI always knows it is the same one underneath.
- **Be honest**: personality/memory continuity is achievable; an identical "mind" across different weights is not. Report how close each upgrade is.

### Goal 6: Self-modification, always with the owner's confirmation
- The AI may *propose*, never apply. Allowed: (a) memory + Person card edits, (b) new fine-tuning runs / LoRA adapters, (c) config changes.
- Each proposal shows what/why/diff/how to undo. Every approved change is versioned in the lineage log and rollbackable.
- **Neither the AI's output nor anything in the training data can approve a change.** Only the owner's input at the terminal (design: read approval from the interactive TTY only).

### Goal 7: Personas
- Switchable personas ("me", "a calm mentor", "a character"): system prompt + optional trained adapter. Be clear what is trained into weights vs. just prompting.

### Goal 8: Export / embed in the owner's own app
- One `export` command makes a portable package: (a) weights (GGUF via the existing exporter; safetensors if PyTorch is used),
  (b) frozen tokenizer, (c) identity core, (d) memory graph (JSON + Markdown), (e) personas + LoRA adapters,
  (f) manifest (version, size, lineage entry, checksums, license notes).
- Clean inference library `persona_lm.runtime`: `ai = load("package/")`, `reply = ai.chat(message)`, `ai.memory`, `ai.mood`,
  `ai.switch_persona(name)`. No training dependencies.
- Must run on CPU/phone: verify without GPU and without PyTorch; GGUF must load in llama.cpp (state exactly what was and was not tested).
- Memory/mood/Person card keep updating inside the app, with changes still needing confirmation and still logged. Packages can be re-imported for further growth.
- Integration guide with example code: (1) Python app, (2) mobile/desktop via llama.cpp, (3) local HTTP server `persona_lm serve`.
- **Privacy warning before export** (packages contain personal data) and a "weights only, no memories" option.

## 4. Project layout now (after Stage 2)

Flat package `persona_lm/`. The planned split into `core/`, `runtime/`, `train/` subpackages was **deliberately deferred** to Milestone 9 (when the
no-torch runtime must be separated), to avoid a risky refactor now. `backend.py` and everything except `torch_backend.py` never import PyTorch.

| File | Role | Stage 1 change |
|---|---|---|
| `model.py` | Config, presets, NumPy forward + hand-written backprop | `loss_and_grads(..., denom=)`; `s10m`/`s30m` default `ctx` 256 -> **512** |
| `bpe.py`, `pretok.py` | Byte-level BPE, GPT-2 splitter | **32 reserved special-token ids** (`N_SPECIAL_SLOTS`), `Tokenizer.fingerprint()` (SHA-256) |
| `data.py` | Token files, chat encoding with loss mask, sampler | `meta.json` now stores `tokenizer_sha256` |
| `ckpt.py` | Atomic `.npz` checkpoints | `FORMAT_VERSION = 2` (`format_version` in meta); old files still load |
| `train.py` | NumPy training loop | Refactored: shared `start_or_resume()`, `step_denominators()`, tokenizer check, new hp `loss_norm` |
| `backend.py` (new) | `choose_backend()`: auto/torch/numpy, no torch import | new |
| `torch_backend.py` (new) | PyTorch model (same maths), `TorchAdamW`, fp16 `LossScaler`, bf16/fp16 autocast, `train_torch`, `benchmark_torch`, `selfcheck` | new; **maths verified on the owner's RTX 3050 (check-gpu 9/9)**, speed/low-precision training still to measure |
| `cli.py` | commands | `train --backend --device --precision --deterministic`, `benchmark --try-batches`, new `check-gpu` |
| `optim.py`, `generate.py`, `gguf.py` | unchanged | |
| `messenger.py` (new, Stage 2) | Messenger JSON export reader (mojibake fix, filters, sessions, merged turns), chat/window/session example builders, time split, size verdict, `import_export()` | new |
| `anonymize.py` (new, Stage 2) | `Anonymizer`: names/phones/e-mails/links/handles/addresses/long numbers; `audit()` second look; `capitalised_word_report()` | new |
| `data.py` | | Stage 2: per-message `train` flag (`as_messages(rec, with_train=True)`, `encode_chat` accepts 3-tuples), `prepare(..., val_docs=)` explicit validation set |
| `cli.py` | | Stage 2: `messenger-scan`, `messenger-import`, `prepare --val-data`, console-safe `main()`, tokenizer training dedups chat messages |
| `docs/MESSENGER_GUIDE.md` (new, Stage 2) | How to export from Facebook, run the importer, what anonymization does/does not do, data-size honesty, pretrain-then-finetune commands | new |
| `tests/test_messenger.py` (new, Stage 2) | 58 tests on a synthetic export; a module guard makes any accidental keyboard read fail instead of hang | new |
| `pipeline.py` (new, Stage 3) | `project.json` (languages, vocab, ctx, size, paths), `init/confirm/plan/run`: one step at a time; tokenizer step needs confirmation and is never overwritten; fingerprint check on later steps; locked fields; script check (warns if a wanted script is missing from the tokenizer text) | new |
| `chatrepl.py` (new, Stage 3) | `ChatSession` + `run_repl`: slash commands (`/help /reset /history /save /system /temp /max /quit`, stubs `/memory /mood /identity /approve /reject`), whole-message context trimming that keeps the system line, `//` escape; commands come only from the keyboard, never from model text | new; `generate.chat_loop` is now unused (kept) |
| `report.py` (new, Stage 3) | `data-report` (tokens, examples, loss tokens, tokens/parameter verdict, passes over the data, tokenizer match) and `status` (scans a work folder, shows runs, warns when validation loss rises) | new |
| `docs/STAGE3_GUIDE.md` (new, Stage 3) | Owner-facing walkthrough of the guided pipeline | new |
| `tests/test_stage3.py` (new, Stage 3) | 33 tests incl. a whole tokenizer -> pretrain -> fine-tune run on tiny data and the real `chat` command with scripted typing; keyboard guard | new |
| `docs/GPU_GUIDE.md` (new) | Beginner guide: laptop (Windows) + Google Colab + moving runs + privacy | new |
| `tests/test_persona_lm.py` | original 26 tests | unchanged, still pass |
| `tests/test_stage1.py` (new) | 15 NumPy-side tests | new |
| `tests/test_torch_backend.py` (new) | runs `selfcheck` on CPU/GPU; **skipped without PyTorch** | new |
| `tests/test_handoff.py` (new) | 1 test, fails if this file is missing | new |
| `memory.py` (new, Stage 4) | `MemoryStore`: memories + nodes in `memory.json` (atomic save, one backup `.prev.json`, refuses to save over a file changed by another program, change log without texts), `[[wiki-link]]` parsing, strength/fading, computed links, keyword recall (`retrieve`), maintenance plan/apply (fade, merge duplicates, extractive summaries), `parse_remember` | new; pure Python, no NumPy/PyTorch |
| `memory_export.py` (new, Stage 4) | `export_vault` (Obsidian Markdown, never overwrites), `export_graph` (one offline HTML force-directed graph, 600-node cap, dark/light) | new |
| `chatrepl.py` | | Stage 4: `memory=`/`memory_tokens=`, `<\|memory\|>` block after the system line (max a third of the room), `/memory /remember /forget`, "did not fit" note, recalled memories counted as used and saved |
| `cli.py` | | Stage 4: `memory` command group (add/list/show/nodes/search/set/forget/maintain/export-vault/export-graph), `chat --memory --memory-tokens` (memory file checked before the model loads) |
| `emotion.py` (new, Stage 5) | 10 feelings with valence/arousal; word-list tagger (English/Tagalog/Bisaya/emoji, negation ignored, stretched words folded); naive-Bayes learner trained on weak labels + owner corrections (weight 5); samples file (`label`, `review`, `train`); `Mood` (valence/arousal, 6 h half-life drift, rate 0.6, `mood.json`) | new; pure Python |
| `person.py` (new, Stage 5) | Person card = plain Markdown (`[auto]` built, `[no]` banned, untagged = owner's); `plan_build` / `apply_plan` (plan first, owner lines never touched, backup `.prev.md`); style counts, frequent words, relationships from memory people; `prompt_text` | new; pure Python |
| `chatrepl.py` | | Stage 5: `mood=/tagger=/person=`, blocks in order system -> `<\|speaker\|>` (card) -> `<\|mood\|>` `label +v +a` -> `<\|memory\|>`; typed message moves mood before the prompt is built (restored if refused/empty reply); temperature x(1+0.15*arousal); `/mood`, `/mood tag`, `/mood reset` |
| `cli.py` | | Stage 5: `emotion tag/label/review/train/report/tag-memories`, `mood show/reset`, `person init/show/add/build`, `chat --mood --emotion-model --lexicon --person --person-tokens` (files checked before the model loads) |
| `docs/EMOTION_GUIDE.md`, `tests/test_stage5.py` (new, Stage 5) | owner guide; 75 tests | new |
| `docs/MEMORY_GUIDE.md`, `tests/test_stage4.py` (new, Stage 4) | owner guide; 52 tests (store, file safety, recall, maintenance, exports incl. JS checks when Node exists, chat, real `chat` command, CLI) | new |

Run tests: `python -m unittest discover -s tests -v` -> **262 tests: 260 pass, 2 skipped** (the 2 skipped are the PyTorch ones; with PyTorch installed they run too) in the author's sandbox, also inside a pseudo-terminal.

## 5. Stage log

### Stage 0: audit (done)
Read all code, ran the baseline (26/26 pass), verified two suspicions by experiment (masked accumulation bug; sampler order is invariant to micro-batch split).

### Stage 1: GPU backend (code complete; core maths VERIFIED on the owner's RTX 3050; speed/memory/low-precision training steps still to measure)
**What was built:** PyTorch backend with CUDA/CPU auto-selection and NumPy fallback; mixed precision (bf16 on Ampere+, fp16 with dynamic loss scaling on older GPUs,
fp32 otherwise); gradient accumulation; checkpoints interchangeable between NumPy and PyTorch and between machines; `benchmark` (tokens/s, peak memory,
largest micro-batch that fits); `check-gpu` self-test against the NumPy reference; beginner guide for the laptop and for Colab.
Also fixed audit findings 1, 2 (groundwork), 6 and applied the context/special-token defaults (see section 6).

**Tested in the author's sandbox (NumPy only):**
- masked gradient accumulation equals one big batch (float64, rel. error < 1e-10) for micro-batch splits 4, 2, 1; legacy "micro" mode still differs (proving the test can fail);
- 4x2 vs 2x4 micro-batch/accum splits train to the same weights (rel. diff < 1e-4);
- a pre-fix checkpoint resumes in legacy loss mode; new checkpoints carry `format_version`;
- data from a different tokenizer is refused on resume and on `--init-from`; `prepare` records the fingerprint;
- reserved-special-token layout, old 5-special tokenizer files still load;
- backend fallback when PyTorch is missing; `torch_backend` imports without PyTorch; static name check of `torch_backend.py` finds no undefined names;
- all 26 original tests (gradient finite differences, bit-exact resume, GGUF read-back ...).

**VERIFIED ON REAL HARDWARE (owner ran `python -m persona_lm check-gpu`, 2026-11-13):** NVIDIA GeForce RTX 3050 6GB Laptop GPU (cc 8.6, bf16 fast), Windows 11, Python 3.14.6, PyTorch 2.14.1+cu130, NumPy 2.5.2. **9 checks passed, 0 failed:**
forward logits match NumPy (max abs diff 3.58e-07); loss 4.895444 vs 4.895443; all 20 gradient tensors match (worst rel. error 6.18e-07); masked gradient accumulation == one big batch (3.87e-07); AdamW after 3 steps (1.12e-08);
stop+resume == uninterrupted run on the GPU (max abs weight diff exactly 0.00e+00, tiny model, one machine); NumPy-started run continues on torch (1.06e-05); bf16 and fp16 forward losses close to fp32 (4.9125 / 4.9126 vs 4.9126).
This also shows the project code runs on Python 3.14. (The owner's unit-test run output had not been seen to completion when this was written; the full suite was only seen to start.)

**MEASURED on the owner's RTX 3050 (2026-11-14, `benchmark --size s10m --vocab 16000`, bf16):** micro-batch 1/2/4/8/16 = 15,053 / 22,930 / 35,714 / 39,489 / 40,830 tokens/s, peak GPU memory 0.27 / 0.42 / 0.73 / 1.35 / 2.58 GB (raw forward+backward; real training steps are somewhat slower). 1e9 tokens ~ 6.8 h at best. `s30m` not benchmarked yet.

**STILL NOT TESTED after that check (speed/memory above are now measured; the rest remains):** the self-check's bf16/fp16 test compares only the FORWARD loss on a tiny random model. Not yet exercised on real hardware: a real bf16 TRAINING step at s10m/s30m size (the `benchmark` command does run one), the fp16 dynamic-loss-scaling overflow/skip path,
`--deterministic`, speed (tokens/s), the largest micro-batch that fits in 6 GB, cross-machine resume (local <-> Colab), and `benchmark_torch`/`train_torch` on real data. The original (pre-check) list of what was untested was: forward/gradient parity with NumPy,
AdamW parity, fp16/bf16 behaviour and loss scaling, `train_torch` resume, NumPy->torch continuation, `benchmark_torch`, `--deterministic`. They are covered by
`selfcheck` / `python -m persona_lm check-gpu` / `tests/test_torch_backend.py`, **which the owner must run once on the Acer Aspire 7** (and ideally on Colab) and report the output.
If any check FAILs the next AI should fix `torch_backend.py` first, before starting Stage 3 (training), because later stages depend on it.

**Known limitations / honest notes**
- "Exact resume across machines" is only achievable for data order, optimizer state and schedule. GPU arithmetic differs from CPU and between GPU models at about 1e-6 relative.
  The CPU PyTorch path is intended to be bit-exact on resume (checked by `selfcheck`, untested). `--deterministic` is best effort (`warn_only`) and not guaranteed bit-exact on a GPU.
- fp16 overflow handling skips the update and halves the loss scale (the step still counts toward the learning-rate schedule). Rare, but a design choice.
- Checkpoints store float32 master weights, so low-precision training does not lose precision at rest; checkpoint files are not compressed and have no checksum file yet (only `format_version` and the tokenizer hash). Add checksums with the lineage log (Stage 6).
- The `--loss-norm micro` legacy mode exists only so old runs resume unchanged; new runs use `batch`.
- A GGUF file is still untested inside llama.cpp (unchanged from before).
- Generation (chat) still uses the NumPy CPU forward without a KV cache.
- The PyTorch benchmark measures raw forward+backward; real training steps are somewhat slower.

### Stage 2: Messenger importer + anonymizer (done; tested on SYNTHETIC data only)
**What was built:** `messenger-scan` (read-only table of conversations, busiest senders), `messenger-import` (interactive or flags: which sender is "me" - several names allowed -, which chats,
mode `window` = one example per reply of the owner with the previous `--context` turns, loss only on that reply; or `session`), mojibake repair, filters (photos, stickers, voice, calls, unsent,
shares, link-only, English system notices), merge of consecutive messages, sessions split at 6 h silence, group chats (others labelled `Person_NN: `), validation = most recent 5 % of
SESSIONS (no leakage), anonymization on by default with a self-audit that makes the command exit with code 2 and `safe_to_upload=false` if a known name/e-mail/link/phone survives,
`private/` folder (real names; never upload), `stats.json` with an honest data-size verdict, `prepare --val-data`, per-message `train` flag in the core encoder.
Also: the guide's whole pipeline (tokenizer on public text + owner's messages -> pretrain -> chat fine-tune with `--init-from`) was run end to end on tiny demo data with the NumPy backend.

**Tested (synthetic export imitating Facebook: newest-first split files, mojibake in text/names/titles, photo/sticker/call/unsent/system/share/link-only messages, group chat, archived thread,
planted phone/e-mail/address/link):** mojibake round trip; folder discovery; chronological merge of message_N.json files; each filter; sender ranking; selection parsing and interactive prompt;
sessions; merging; window/session example construction (context trimming oldest-first, over-long reply skipped not truncated, group labels, system line); anonymizer on names (full/first/last/nickname/any case),
phone formats, dates/prices not touched, e-mail/URL/handle, addresses, common-word names, paranoid mode, disabled mode, audit; the name-regex speed shortcut gives identical results to the slow path (3000 random cases);
whole import: no real name/phone/e-mail/link in train/val/stats, real names only in private/, emoji and accents survive, loss exactly on the target reply, validation sessions disjoint from training sessions,
deterministic output, group filter, wrong `--me` error; command line scan -> import -> prepare, and decoding the masked tokens shows only the owner's words carry loss; a fake leaky anonymizer makes the command exit 2;
the two new modules contain no network imports.

**NOT tested / known limitations**
- **Never run on a real Messenger export.** Real exports will contain surprises (message types, languages, participants, huge sizes). Ask the owner to run `messenger-scan` first and report anything odd.
- System notices are recognised in **English** only; other languages' notices would be kept as if typed by a person.
- Anonymization is not perfect and says so: names that are only *mentioned* need `--extra-names` (a frequency report helps); addresses are pattern-based (English/Philippine styles); identifying context is not detected.
  The owner must read a sample of `train.jsonl` before uploading anything.
- The owner's own name becomes `[ME]` in other people's messages (how friends address the owner); others become `Person_NN`. See the design note in section 6.
- Training windows are random token slices: a window may start mid-example and lose its beginning (inherited from `Stream`).
- Everything is loaded into RAM at once; speed measured: scrubbing about 19 microseconds per message regardless of the number of names (after the shortcut).
- Token counts in the verdict are estimated as characters / 3.5 (rough; emoji and Filipino text use more tokens per character).
- Unclosed-file `ResourceWarning`s remain in the original 26 tests.
- **Incident (2026-11-13/14): the test suite HUNG for 3 hours on the owner's PC.** Cause: `test_import_needs_me_when_not_interactive` assumed `sys.stdin` is not a terminal. In an interactive PowerShell it is, so `messenger-import` correctly
  asked "Who are you?" and waited for typing (the prompt was hidden because the test captures stdout). It passed in the author's sandbox only because stdin was a pipe there. **Fixed:** tests now fake the terminal state explicitly (`FakeTerminal`),
  `setUpModule` makes any unexpected `input()` raise instead of wait, the typed-answers path now has its own test, and `choose_me` looks up `input` at call time. **Rule for every future stage: tests must not depend on the real stdin, and the full
  suite must be run inside a pseudo-terminal before delivery** (snippet in section 11).

### Stage 3, part 1: guided pipeline, chat screen, reports (done; tiny synthetic data only; the real tokenizer is NOT frozen)
**What was built:**
- `pipeline init|confirm|plan|run --dir WORK` (`persona_lm/pipeline.py`). `init` records the decisions in `WORK/project.json` (languages, vocab, ctx, size, public text, chat train/val files from `messenger-import`, fine-tune settings); changing languages/vocab/ctx/size withdraws the confirmation; once `tok.json` exists vocab+languages are locked, once a checkpoint exists size+ctx are locked. `confirm --yes` is the owner's explicit OK. `plan` prints every step with its exact command (quoted for Windows paths) and done/NEXT markers, plus script warnings (e.g. Japanese wanted but no kana in the tokenizer text; English+Tagalog are both Latin so letters cannot prove both are present). `run` executes ONE step through the normal CLI (default: next undone). Steps: tokenizer -> pretrain-data -> pretrain -> chat-data -> finetune. Tokenizer step refuses without confirmation and refuses to overwrite `tok.json`; later steps refuse if `tok.json`'s fingerprint differs from the one stored in `project.json`. Without public pretraining text the two pretrain steps are marked skipped (with a warning that chats alone teach very little) and fine-tuning starts from scratch. Fine-tune steps are worked out from the data: `finetune_epochs` (default 3) x tokens / (batch x accum x ctx), at least 50, with lr 1e-4 -> 1e-5, warmup 20, `--init-from` the pretrained checkpoint.
- Chat screen (`chatrepl.py`, used by `chat`): see section 4. Prompt = system line + as many NEWEST whole messages as fit (the old loop cut tokens from the front, even through the system line); the person is told once when old messages stop being seen; too-long message refused; `/save` never overwrites; typed text cannot forge role tokens (`allow_special=False`).
- `data-report --data-dir D [--size --ctx --batch --accum --steps --tokenizer]`; `status [DIR]`.
**Tested (tiny synthetic data, NumPy, no keyboard):** 33 new tests: loss-token and example counts equal a direct encode; tokenizer match/mismatch; verdict levels; `private/` never opened; validation-rise warning; command stubs change nothing and never reach the model; a model reply of `/quit` or `/approve` does not run a command; context trimming/system line/one-time notice; refusal of over-long messages; `/save`; scripted `run_repl`; end-to-end tokenizer -> pretrain -> chat data -> fine-tune through `pipeline run` (fine-tune uses `--init-from`, tokenizer hash identical in both checkpoints); the real `chat` command with scripted typing; frozen-tokenizer protection (overwrite, swap, locked fields); no-pretrain path; the full suite also passes inside a pseudo-terminal.
**NOT tested / limitations**
- Never run on real data, real-size models (`s10m`/`s30m`), the GPU path (`--backend torch`), or Windows. The tests use `--backend numpy` and the `tiny` size. On the owner's PC `pipeline run` will use torch automatically if no `--train-args "--backend numpy"` is given (same `train` command as before).
- **Early stopping is NOT implemented.** Fine-tune length is chosen up front (about 3 passes); `status` warns when validation loss rose, but only the newest checkpoint plus one previous are kept, so the best one cannot be recovered. Idea for part 2: keep `ckpt.best.npz` when val loss improves (touches both backends: do it carefully).
- The pretraining corpus is NOT chosen (no network in the author's sandbox; the owner's languages are unknown). Verdicts in `data-report` are rules of thumb (20 tokens/parameter is a generic web-scale rule, not a measurement for this project).
- Chat generation is still NumPy-only on the CPU, no KV cache: slow at s30m / 512 tokens. Not measured on the owner's PC.
- The `[ME]` / `Person_NN` / "its own person speaking in the owner's style" question (section 6 design note) is still open; the chat screen just feeds typed text as the `user` role with an optional `--system` line.
- Script check only covers a few scripts and cannot separate languages sharing one alphabet.

### Stage 4: memory graph (done; synthetic data only)
**What was built:** see the layout table and `docs/MEMORY_GUIDE.md`. Memory = text, time, source, emotion tag (free word), importance 0-1, nodes (person/topic/event/feeling), uses, last_used, status (active/dormant/archived/merged), pinned. Links are COMPUTED from active memories (count + summed strength), never stored. Strength = importance x 0.5^(idle days/90); using a memory resets the clock; pinned never fades. Recall: query words (rarer = heavier, stopword lists for English/Tagalog/Bisaya) + named nodes (+1) + one hop along links (+0.3), times (0.5 + strength); zero relevance is never returned; token budget skips what does not fit (never cuts a memory). Maintenance is a PLAN until `--apply`; duplicates are only compared inside the same node set (>=80% same words, >=3 words); summaries are EXTRACTIVE (first sentences of the top-3 memories of a node with >=6 unpinned memories older than 30 days), originals become `archived`, nothing is deleted except by the owner's `forget` (which also deletes the backup copy). Only typed commands change memory; model text never does.
**Tested (author's sandbox, Python 3.12, NumPy 2.5):** 52 new tests (187 total, also inside a pseudo-terminal, also with `-W error::ResourceWarning` for the new file); mutation checks: breaking each of forget-backup removal, conflict detection, zero-relevance filter, pinned fading, the one-third budget cap, forged special tokens, cross-node duplicate merging makes at least one test fail. The graph page was opened in headless Chromium (no page errors, light and dark, 600 nodes ~39 fps in software rendering); `node --check` and a layout stress test run inside the unit suite when Node is installed. End-to-end `chat --memory` was run with a real tiny checkpoint.
**NOT tested / limitations:** never run with the owner's real memories, real model, Windows, or their browser/Obsidian (vault checked only structurally: every wiki-link has a note, file names are Windows-safe). Recall is by words, not meaning (no embeddings): "what fruit" will not find "mango". The model has never been trained with `<|memory|>` blocks, so it cannot yet use them (see Design debt). If a save fails with a conflict the in-RAM store has the change but the file does not; restart the chat. Names in `[[ ]]` are limited to 60 characters; nodes have no relationship types (write the relationship as a memory). The memory file is plain text: the chat does not encrypt it.

### Stage 5: emotion + Person card (done; synthetic data only; owner has not run it yet)
**What was built:** see the layout table and `docs/EMOTION_GUIDE.md`. Tagger = word list (+ optional `--lexicon` JSON) blended 50/50 with a naive-Bayes learner when a model exists; with no known word only the learner decides. Weak labels come from the word list; `emotion review` (typed answers, saved at once) gives real labels that count 5x. Mood changes ONLY from typed messages; model text never. Mood block format fixed: `<|mood|>label +0.1 -0.3<|end|>` (future fine-tune data must use exactly this). Person card is built from data only for Speaking style, Frequent words, Relationships; Name/Habits/Values are the owner's (chat files carry no times, nothing proves values). Card goes in the prompt as a `<|speaker|>` block (whole lines, budget ctx/6, at most a quarter of the room).
**Tested (author's sandbox, Python 3.12):** 75 new tests (262 total; also inside a pseudo-terminal; new file also with -W error::ResourceWarning): tagging in 3 languages + emoji, negation, stretching, confidence bounds, learner on synthetic data, pruning, model file safety, samples/review flow incl. EOF, mood maths (bounds, 6 h half-life, backwards clock), chat order/budget/restore-on-refusal, model text cannot change mood, card parse/render round trip, owner lines never touched, `[no]` suppression, idempotent rebuild, CLI workflows incl. the real `chat` command with a tiny checkpoint. Also: `.gitignore` had a UTF-8 BOM (could break its first rule `private/`): rewritten without it.
**NOT tested / limitations:** never run on real messages or on Windows. The Tagalog/Bisaya word lists are the author's best knowledge, unchecked by a speaker: expect errors; correct with `emotion review`. Weak labels teach the learner the word list's own mistakes; the `train` number is agreement, not accuracy. One feeling per message; sarcasm, mixed feelings and meaning are not understood. anger vs fear sit close on the two scales. The model was never trained with mood/speaker blocks, so the only real tone effect now is temperature (+-15 %). Language-mix and frequent-words are counts from tiny word lists; frequency is not meaning. No habits from times. Mood file has no conflict check (atomic write only).

## 6. Decisions

**Made by the owner:**
- Identity: **"its own person"** with its own name and personality, not a copy of the owner.
  *Unconfirmed assumption:* it should speak in the owner's style (slang, humor, phrasing, learned from Messenger data) but be a **separate individual** whose name, opinions and
  self-description live in the identity core. Note the tension with Goal 3 (loss only on the owner's messages teaches the model to speak *as* the owner); resolve in Stage 2/6.
- Hardware (reported by the owner from PowerShell, 2026-11-13): **Acer Aspire 7**, **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (6144 MiB VRAM, Ampere, so **bf16 is fast**; driver 616.64) plus an Intel UHD iGPU,
  **15.7 GB RAM**, Intel Core i5-12450H (8 cores / 12 threads), **Windows 11 Home**, **Python 3.14.6**.
  Consequences: `s10m` and `s30m` should train locally (optimizer+weights+grads need ~16 bytes per parameter before activations: ~0.3-0.55 GB); `s125m` needs ~2+ GB before activations and is realistic only with a tiny micro-batch or on a rented GPU
  (run `benchmark --try-batches` to find out; no speed numbers have been measured on this GPU yet). The laptop should be plugged in and in performance mode.
  **Python 3.14 caveats:** PyTorch officially ships Python 3.14 binaries incl. CUDA for Windows (PyTorch 2.11 support matrix lists Python >=3.10,<=3.14; a PyTorch maintainer confirmed in issue #169929, May 2026, that cp314 CUDA wheels exist, but not for CUDA 12.1/12.4).
  PyTorch 2.11+ made CUDA 13.0 the stable default (cu128 was removed in 2.12; cu126 remains for older drivers). Install with
  `python -m pip install numpy` and `python -m pip install torch --index-url https://download.pytorch.org/whl/cu130` (the owner's driver is far newer than CUDA 13.0 needs, as far as the author knows). **None of this was run by the author**, and
  **this project's own code has not been run on Python 3.14** (author's sandbox used an older Python). If the PyTorch install fails ("No matching distribution found"), install Python 3.12 or 3.13 next to it (`py -3.12`) and use that.
- Languages: the owner wants the model to **"learn any language I train it on"** (not yet named). The byte-level tokenizer can represent every script and emoji (round trip verified for English, Tagalog, Spanish, Japanese, Arabic, Korean, emoji), but a language missing from the
  tokenizer's training text is split into many tokens (demo with a tiny vocabulary: Tagalog 0.91 tokens/char before vs 0.23 after adding Tagalog to the tokenizer text; Japanese 3.0, Korean 2.6, Arabic 1.9 without them), wasting the 512-token window.
  So the tokenizer must be trained on text from ALL desired languages before it is frozen. Recommended vocabulary **16,000** (`s10m` = 16.8M params, `s30m` = 33.5M; 8,000 is enough only for Latin-script languages; 32,000 gives 22.9M / 41.7M).
  A small model shares its capacity between languages: 2-3 languages are reasonable, more will each be thinner. Each language needs pretraining text in it; the owner's chats alone are not enough.
- "Build it now" (2026-11-12) = go-ahead.

**Applied by assumption (owner said they did not understand them, then said "build it now"):**
- `ctx = 512` for `s10m` and `s30m` (was 256). Reason: persona prompt + memories + previous messages must share the window; fixed at training time.
- A **reserved block of 32 special tokens** (5 original, 5 named for later features: memory/persona/mood/speaker/summary, 22 spare), so specials never need renumbering.
Nothing irreversible has happened: no real tokenizer or model has been trained yet. **Explain these again in plain words and let the owner veto before the real tokenizer is trained (Stage 3).**

**Design note (Stage 2) to resolve in Stages 5-6:** the training data teaches "answer the way the OWNER answers a friend" (assistant = the owner, user = friends; the owner's name is `[ME]`).
At chat time the other side is the OWNER talking to an AI that is "its own person". That is a distribution shift. The planned fix is the identity core + a system/persona prompt
(the AI's own name and self-description, "you speak in <owner>'s style and you are talking to <owner>"), possibly with a distillation/rewrite step; the `<|persona|>`, `<|memory|>`, `<|mood|>`, `<|speaker|>` reserved tokens exist for this.
Decide how `[ME]` and `Person_NN` are shown/replaced at chat time (e.g. `[ME]` -> the owner's chosen display name inside the prompt).

**Still open (updated 2026-11-14: languages, ctx/specials OK and persona assumption are now ANSWERED, see START HERE):** the owner's Messenger export size and number of chats (the PowerShell command was run with a placeholder path: `C:\path\to\unzipped\export`, so no numbers yet); the **list of languages** (and OK for vocab 16,000; "recommended" = yes);
confirmation of the "own person in the owner's speaking style" assumption; result of `check-gpu` and of the unit tests on the owner's Python 3.14; OK for `ctx=512` + 32 reserved tokens.

## 7. Milestone plan and status

| # | Milestone | Status |
|---|---|---|
| 0 | Audit | done |
| 1 | GPU backend | **done: check-gpu 9/9 PASS on RTX 3050; real speed/memory (benchmark) and a real training run still to be measured** |
| 2 | Messenger importer + anonymizer | **done (synthetic data only; owner should run `messenger-scan` on the real export)** |
| 3 | Train/chat CLI, pretrain-then-finetune flow (train the real tokenizer here; **owner chose to do part 2 LAST, after Stages 4-9 and the Messenger export**) | **part 1 done** (pipeline, chat screen, status, data-report; tiny synthetic data). **Part 2 waiting on the owner**: languages, ctx/specials OK, pretraining corpus, then freeze the real tokenizer and run real training |
| 4 | Memory graph (store, retrieval, decay/merge/summarize, Obsidian export, HTML graph) | **done** (synthetic data; real fine-tune must include memory blocks) |
| 5 | Emotion + Person card | **done** (synthetic data only; owner should try `emotion label/review/train`, `person build`, `chat --mood --person`) |
| 6 | Identity core + lineage + regression tests (probes, scoring, drift, protected tier, distillation, checksums) | |
| 7 | Self-modification (proposals, diffs, undo, TTY-only approval) | |
| 8 | Personas | |
| 9 | Export package + runtime library (no-torch runtime, KV cache, `serve`, package manifest, privacy gate; consider the `core/runtime/train` split here) | |

Target layout (from the spec): `core/` (no training deps), `runtime/`, `train/`, `data/`, `memory/`, `mind/`, `identity/`, `proposals/`, `personas/`, `export/`, `cli/`.

## 8. Audit findings and status

| # | Finding | Status |
|---|---|---|
| 1 | Gradient accumulation wrong for masked data (rel. gradient error 1.16 on an uneven mask) | **fixed** (Stage 1, `loss_norm="batch"`; old runs keep `"micro"`) |
| 2 | Special tokens occupy the first ids, adding one later renumbers everything | **fixed** by the reserved block; fingerprint check added; the real tokenizer is still to be trained in Stage 3 |
| 3 | Context small (256) | **changed** to 512 for s10m/s30m |
| 4 | Validation split is per document (leak with overlapping windows) | **fixed** for Messenger data (split by session, most recent held out; `prepare --val-data`) |
| 5 | No KV cache in generation | open, Stage 9 |
| 6 | Checkpoints lacked version/tokenizer hash | **fixed** (version + tokenizer hash); checksum file still open (Stage 6) |
| 7 | `ResourceWarning`s from unclosed files in old tests/CLI | minor, open |

Verified good: sampler draws the identical data order for any (batch, accum) split with the same product; checkpoint format is device-independent.

## 9. Environment limits of the previous AI's sandbox

1 CPU core, 4 GB RAM, **no GPU, no network, no PyTorch, no llama.cpp**. NumPy-side and file-format code is testable there; PyTorch/CUDA and llama.cpp loading are not.
If you have a better environment, run `python -m persona_lm check-gpu` first and report real results.

## 10. Honest expectations to keep telling the owner

- A personal Messenger history alone (often ~1-10 MB of the owner's own text) is **not enough** to train a good model from scratch: pretrain on public dialogue-style text, then fine-tune on the owner's messages.
- A small model holds a personality weakly; memory retrieval and the Person card carry more of "knowing the owner" than the weights.
- Identity continuity is measurable (probe scores, drift score), not guaranteed identical across different weights.
- The emotion tagger starts from weak heuristic labels (the data has none); the owner can correct samples.
- Anything uploaded to a rented GPU leaves the owner's computer: only public or anonymized data.

## 11. How to maintain this file

At the end of every stage: add a "Stage N" entry to section 5 (what was built, what was tested, what was NOT tested, limitations), update the status tables in 4, 6, 7, 8,
bump the "Last updated" line, run the tests, and rebuild `persona_lm.zip` with this file inside. Never delete it, and never remove the owner's standing rules in section START HERE (the `*` wildcard rule for zip commands included). If you are a different AI taking over, read sections 1-3 and 5-6 first.

Pseudo-terminal check before every delivery (the sandbox's stdin is a pipe, the owner's is a terminal; behaviours that differ between them must be tested):

```python
import pty, subprocess, sys
master, slave = pty.openpty()
p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
print(p.returncode, p.stdout.decode()[-200:])      # a TimeoutExpired here means something is waiting for the keyboard
```

## 12. Next action

1. The owner extracts the newest zip (wildcard rule), runs `cmd /c "python -m unittest discover -s tests < NUL"` (expect `Ran 262 tests ... OK`), then tries `python -m persona_lm person init --name "Name"`, `emotion tag "pagod na ako"`, `chat ... --mood mood.json --person person_card.md` (see docs/EMOTION_GUIDE.md); ask what looked wrong.
2. **Stage 6: identity core + lineage + regression tests** when the owner says "next" (one stage per go). Reuse: Person card + mood baseline (`Mood.baseline`) belong to the identity core; protected changes need typed confirmation; checksums for checkpoints are still open.
3. Stages 7-9 as in section 7. **Stage 3 part 2 (real tokenizer, pretraining, fine-tune) is deliberately last**: needs the Messenger export, a pretraining corpus (English/Tagalog/Bisaya) and the owner's OK; fine-tune examples must include `<|speaker|>`, `<|mood|>` and `<|memory|>` blocks in the exact formats above.
4. Still open: `s30m` benchmark, cross-machine resume, GGUF in llama.cpp, fp16 loss-scaling path.
