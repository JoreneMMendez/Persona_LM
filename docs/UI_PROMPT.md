# Web UI prompt (U0 to U3b done, then 10 small stages: U4a ... U7a, U7b)

Paste everything in the block below into a NEW AI session together with the newest project zip and `HANDOFF.md`.
Change the last line (`Start with stage: U2a`) to the stage you want: U1a, U1b, U2a, U2b ... U7a, U7b. ONE stage per session.
Status of each stage: HANDOFF.md, section 7 (rows U0 and U1a-U7b). U0 to U3b are built (2026-11-15); the rest follow the list below.
Why 14 small stages: the first plan had 7 big ones (U1-U7) and a session ran out of credit before it finished. Each old stage is now cut in two (a and b), so each one finishes inside one session.

```text
You are continuing the PersonaLM project (zip + HANDOFF.md attached). Do ONLY the stage I name at the bottom.

CREDIT RULES (read first; they exist because an earlier session ran out of credit mid-stage)
- Do NOT read HANDOFF.md fully (it is huge). Read only: the "START HERE" block (first ~30 lines), the Web UI rows in section 7, and the files this stage touches. Use grep to find things; never print whole files you do not need.
- Reuse what exists: persona_lm/ui_server.py, ui_static/*, runtime.py, chatrepl.py, memory.py, identity.py, proposals.py, personas.py, pipeline.py, report.py. Do not duplicate their logic.
- Edit with small targeted changes (str_replace). Do not rewrite or re-print a whole file to change a few lines. Do not reformat old code.
- Order inside a stage: backend endpoint + its tests FIRST (run only that test file), then the page, then ONE full-suite run at the end.
- Browser check: only if headless Chromium is already available; ONE pass (light, dark, narrow width, console errors). If it is not available, say so and skip it. No screenshots loops, no polishing passes.
- HANDOFF.md update: keep it SHORT (about 20 lines): one "Newest state" entry, the stage row in the section 7 table, files added, endpoints, commands, limitations. Append; keep the old text and the sections tests/test_handoff.py checks.
- If the stage cannot be finished inside the budget: finish a smaller WORKING part, tick only that part in HANDOFF.md, list what is left, and stop. Never leave half-written files or failing tests.

GOAL
A local web UI (HTML/CSS/JS) for the whole system, so I can use everything from a good, easy interface instead of typing commands. Every feature must really work, not be a mock-up.

HARD RULES (do not weaken)
- Pure Python + NumPy stays. The UI server uses only the standard library, binds to 127.0.0.1 only, never 0.0.0.0.
- Fully offline: no CDN, no web fonts, no external requests. Plain HTML + CSS + vanilla JS (static files, no inline script/style because of the CSP). No frameworks, no build step.
- Privacy: never read, list or serve data_msgr\private\, raw Messenger exports, or any file outside the project folder. Every path goes through PathGuard. Keep the per-session token, and the Host and Origin checks, on every /api route.
- Safety design stays: model text can never run a command, approve a proposal, change identity or write memory (the chat route uses say(), never handle()). Protected identity changes (CHANGE IDENTITY / APPLY IDENTITY, cooldown) and /approve keep their typed-confirmation rules. If a safe web version is not possible, show the item read-only with the exact terminal command to run. Ask me before changing any safeguard.
- Do not retrain or change the frozen tokenizer (work\tok.json).
- Windows 11, PowerShell, Python 3.14, RTX 3050 laptop. Give me exact commands, ONE at a time. Zip commands use the * wildcard rule from HANDOFF.md.
- Design (already in app.css, reuse it): calm and clean, dark and light themes, sidebar, rounded cards, system font stack, clear empty/loading/error states, toasts, focus rings, aria labels, works down to phone width.

STAGES (each one small; one per session; a = first half, b = second half of the old stage)
U0  DONE. Skeleton: `ui` command, safe server, design system, sidebar shell, Settings/Privacy page.
U1a DONE. Chat core. Choose a checkpoint + tokenizer (or an identity core folder) from files inside the project, load it, send a message, bubbles, typing indicator, Enter to send / Shift+Enter for a new line, stop button, token counter (used / the model's window). Endpoints under /api/chat/. Replies are plain text; use say(), never handle().
U1b DONE. Chat controls. Settings drawer (temperature, top-k, max new tokens), save and reset the conversation, live mood badge.
U2a DONE. Memory list. Searchable list, add / edit / pin / forget, the maintenance plan with a preview before apply.
U2b DONE. Memory graph + Mood + Person card. Graph (pan/zoom, click a node), mood history chart, view and edit the Person card.
U3a DONE. Prompt panel. A side panel showing exactly which memories and old-chat recall (--archive) went into the last prompt.
U3b DONE. Persona switcher (owner-side, through the runtime) and the personas page.
U4a Data page, part 1. Messenger scan/import wizard, privacy audit view, data-report verdict.
U4b Data page, part 2. The cleaner (clean_chat.py, one numbered file per chat, --input mode) with a before/after report, and the duplicate and repeated-reply report.
U5a Training page, part 1. Pipeline plan with a done/NEXT checklist, checkpoints list, run logs read-only.
U5b Training page, part 2. Start/stop of prepare and train as a background process, live loss chart (train and val), a warning when val loss rises, resume.
U6a Identity page. Read-only core view, versions, lineage verify, probe scores and drift, promote/rollback only where the safeguards allow it.
U6b Proposals page. List with a diff; approve/reject only through the existing confirmation rules.
U7a Models compare + Export. Side-by-side replies of two checkpoints on the same prompts; export-package and verify-package with the privacy warning and an explicit confirm.
U7b Final polish. Keyboard shortcuts, accessibility pass, phone width, empty/error states.

FOR EACH STAGE
1. Say in 3 lines what you will build, then build it.
2. Tests: unit tests for each new endpoint, including refusals (wrong token, bad Host, path outside the whitelist, model text trying to trigger a command). Run the stage's test file, then the full suite once.
3. Update HANDOFF.md (short, as above).
4. End with the ONE command I should run next, and what I should see.

Start with stage: U4a
```
