# Taking your AI with you: export package, runtime and local server (Stage 9b)

Three pieces, from simple to advanced. None of them needs PyTorch.

## 1. Make the package

Run from the project folder. Choose a NEW folder name (nothing is ever overwritten).

```powershell
python -m persona_lm export-package --dir identity_core --out my_ai_package
```

The first time it only prints a **privacy warning** and writes nothing. Read it, then add `--yes`:

```powershell
python -m persona_lm export-package --dir identity_core --out my_ai_package --yes
```

If the core has no current version yet, add `--version v1` (the name you registered).

What is inside: the weights (`model/ckpt.npz`), the tokenizer, who the AI is (`identity/identity.json`), the memory graph (`memory/memory.json` and the same as Markdown notes in `memory/vault/`), your Person card, the mood, the masks (name + text only), and `package.json` listing every file with its SHA-256 checksum.

Never inside: the lineage log, proposals, the example sentences you wrote for masks, the training state, Messenger data, other versions.

Options: `--weights-only` (just model + tokenizer: nothing about you except what the weights learned), `--no-memory`, `--no-person`, `--no-mood`, `--no-personas`, `--format gguf` or `both` (a `.gguf` file for llama.cpp; `--gguf-type f16|q8_0|f32`).

**Privacy, plainly:** the weights were trained on your messages, and a small model can repeat pieces of its training text. Even `--weights-only` is not "anonymous". Share the package only with people and apps you would trust with your messages.

Check a package at any time (it should take a second):

```powershell
python -m persona_lm verify-package my_ai_package
```

Files the AI rewrites while it is used (memory, mood) are reported as a normal "note", not as damage.

## 2. Use it from Python

```python
import persona_lm.runtime as rt
ai = rt.load_package("my_ai_package")      # verifies the checksums first
print(ai.chat("hello"))
```

Same `ai` object as in `docs/RUNTIME_GUIDE.md`. Differences from loading an identity core: there is no lineage log, so `ai.switch_persona(...)` works but is **not logged and not saved**, and the AI cannot file proposals. Memory and mood files inside the package are updated as it is used.

## 3. Local web server for your own app

```powershell
python -m persona_lm serve --package my_ai_package
```

It listens on `http://127.0.0.1:8765` (this computer only). Press Ctrl+C to stop.

| Request | What it does |
|---|---|
| `GET /health` | `{"ok": true}` |
| `GET /info` | name, version, mask, mood, whether memory is on |
| `POST /chat` with `{"message": "hello"}` | `{"reply": "...", "notes": [...], "mood": "...", "persona": ...}` |
| `POST /reset` | forget the conversation |
| `POST /persona` with `{"name": "calm mentor"}` | switch the mask; **off** unless you start with `--allow-persona-switch` |

Example from PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/chat -ContentType "application/json" -Body '{"message":"hello"}'
```

Example from Python (standard library only):

```python
import json, urllib.request
req = urllib.request.Request("http://127.0.0.1:8765/chat", data=json.dumps({"message": "hello"}).encode(), headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["reply"])
```

Rules and limits:

- One conversation per server. Requests are handled one at a time. Messages up to 4000 characters.
- Text sent to `/chat` is a plain message, never a command. There is no endpoint to approve changes, edit memory or edit the identity.
- Another `--host` than 127.0.0.1 needs `--token SECRET`; then send `Authorization: Bearer SECRET`. The server uses plain HTTP without encryption: do not put it on the internet.
- The server does not log what anyone says.
- `--core identity_core --version v1` serves straight from an identity core instead of a package.

## Honest limits

Tested only with a tiny random model, fake text, and loopback connections in the author's sandbox. Never run on your PC, with a real model, or from another device. GGUF files were written and read back by the project's own reader; they were **not** opened in llama.cpp. The Android/app side is not built; this is the Python side only.
