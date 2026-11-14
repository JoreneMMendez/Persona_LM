# Web UI prompt (stages U0-U7)

Paste everything in the block below into a NEW AI session together with the newest project zip and `HANDOFF.md`.
Change the last line (`Start with stage: U0`) to the stage you want: U0, U1, ... U7. One stage per session.
Status of each stage: HANDOFF.md, section 7 (rows U0-U7). U0 is built (2026-11-15); U1-U7 are not.

```text
You are continuing the PersonaLM project (zip + HANDOFF.md attached). Read HANDOFF.md fully first, then do ONLY the stage I name below.

GOAL
Build a local web UI (HTML/CSS/JS) for the whole system, so I can do everything from a good-looking, easy interface instead of typing commands. Every feature must actually work, not be a mock-up.

HARD RULES (from HANDOFF.md, do not weaken)
- Pure Python + NumPy stays. The UI server uses only the standard library (http.server / socketserver), binds to 127.0.0.1 only, never 0.0.0.0.
- Fully offline: no CDN, no web fonts, no external requests. Plain HTML + inline CSS + vanilla JS (ES modules in a few static files are fine). No frameworks, no build step.
- Privacy: never read, list or serve data_msgr\private\, raw Messenger exports, or any file outside the project folder. Whitelist the paths the server may touch. Add a random per-session token (CSRF protection) and check the Host and Origin headers.
- Safety design stays: model text can never run a command, approve a proposal, change identity or write memory. Protected identity changes (CHANGE IDENTITY / APPLY IDENTITY, cooldown) and /approve keep their typed-confirmation rules. If a safe web version is not possible, show the item read-only with the exact terminal command to run. Ask me before changing any safeguard.
- Reuse runtime.py / chatrepl.py / memory.py / identity.py / proposals.py / personas.py / pipeline.py / report.py; do not duplicate their logic. Do not retrain or change the frozen tokenizer (work\tok.json).
- Windows 11, PowerShell, Python 3.14, RTX 3050 laptop. Give me exact commands, ONE at a time. Zip commands use the * wildcard rule from HANDOFF.md.

UI REQUIREMENTS
- Modern, clean, calm design: dark and light themes (follow the system, with a toggle), a sidebar with the sections below, rounded cards, good spacing and typography (system font stack), smooth but light animations, clear empty/loading/error states, toasts, keyboard shortcuts, responsive down to a phone width, accessible (labels, focus rings, contrast, aria).
- Chat page: message bubbles, typing indicator, Enter to send, Shift+Enter for a new line, stop button, a settings drawer (temperature, top-k, max new tokens), a live mood badge, a side panel showing what memory and old-chat recall (--archive) were put into the prompt, persona switcher, save/reset, and a token counter showing the 512-token window.
- Memory: graph view (pan/zoom, click a node), a searchable list, add/edit/pin/forget, the maintenance plan with a preview before apply.
- Mood and Person card: view and edit, with a mood history chart.
- Identity: read-only core view, versions, lineage verify, probe scores and drift, promote/rollback only where the safeguards allow it.
- Proposals: list with a diff; approve/reject only through the existing confirmation rules.
- Data: Messenger scan/import wizard, the cleaner (clean_chat.py: one numbered file per chat, --input mode) with a before/after report, a duplicate and repeated-reply report, the privacy audit view, and the data-report verdict.
- Training: pipeline plan with a done/NEXT checklist, start/stop of prepare and train as a background process, a live loss chart (train and val) read from the run logs, a warning when val loss rises, checkpoints list, resume.
- Models: choose run/checkpoint/tokenizer, side-by-side reply comparison of two checkpoints on the same prompts.
- Export: export-package and verify-package with the privacy warning and an explicit confirm.
- Settings/Privacy page: what is stored where, what never leaves the computer.
- One command starts it: python -m persona_lm ui   (opens the browser; --port, --no-open).

STAGES (one per go; keep each small, test it, then stop and wait for me)
U0  Skeleton: `ui` command, safe local server (token, whitelist, Host/Origin check), design system (CSS tokens, components, themes), sidebar shell, Settings/Privacy page.
U1  Chat page on top of runtime.py (replies, window counter, settings drawer, save/reset).
U2  Memory + Mood + Person card pages (incl. graph).
U3  Prompt panel (memory + old-chat recall) and persona switcher.
U4  Data page: scan/import wizard, clean_chat, reports, audit.
U5  Training page: pipeline checklist, background run, live loss chart, checkpoints.
U6  Identity + Proposals pages (safeguards intact).
U7  Models compare + Export + final polish (keyboard shortcuts, accessibility, mobile).

FOR EACH STAGE
1. Say in 3 lines what you will build, then build it.
2. Tests: unit tests for every endpoint (including refusals: wrong token, bad Host, path outside the whitelist, model text trying to trigger a command). Run the full suite; also open the pages in headless Chromium if available (no console errors, light and dark, narrow width) and say what you could and could not verify.
3. Update HANDOFF.md: a "Newest state" entry for the stage, the "Web UI (stages U0-U7)" status table in section 7 (done / NOT run by the owner / NOT tested), the files added, the endpoints, the safeguards kept, the exact commands to run, and the limitations. Keep the old text. Keep the sections tests/test_handoff.py checks.
4. End with the ONE command I should run next, and what I should see.

Start with stage: U0
```
