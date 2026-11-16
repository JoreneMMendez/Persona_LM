# Using your AI from your own Python program (Stage 9a)

This needs only Python and NumPy. PyTorch is **not** needed (and is not used).

## The short version

```python
import persona_lm.runtime as rt

ai = rt.load("identity_core")          # the folder made by `identity init`
print(ai.chat("hello"))                # the AI's reply, as text
```

Run it from the project folder (the one that contains `persona_lm`), for example `cd C:\Users\user\persona_lm_project` first.

`load("identity_core")` uses the core's current version, its frozen tokenizer, and its memory, mood and Person card files (if the core knows them).
If the core has no current version yet, say which one: `rt.load("identity_core", version="v1")`.
Without an identity core: `rt.load(ckpt="run/ckpt.npz", tokenizer="tok.json", system="You are Luna.")`.

## What you get

| | |
|---|---|
| `ai.chat(text)` | one message in, the reply text out. Raises `RuntimeProblem` if the message was refused (too long for the window) or the reply was empty |
| `ai.last_notes` | the extra lines the chat screen prints (what was recalled, the feeling now, window notes) |
| `ai.history`, `ai.reset()` | the conversation so far; start again (memory and mood files stay) |
| `ai.memory` | the memory store (or `None`) |
| `ai.mood` | the mood (or `None`) |
| `ai.person` | the Person card (or `None`) |
| `ai.switch_persona("calm mentor")` | put a mask on; `"none"` takes it off. Logged in the lineage history. Never touches who the AI is |
| `ai.pending_proposals()` | changes the AI proposed and that wait for you (read only) |

Options of `load`: `max_new`, `temperature`, `top_k`, `seed`, `fast=False` (the plain, slower generator; same answers), `memory=`, `mood=`, `person=`, `system=`.

## Safety rules (same as the chat screen)

- `chat("/approve p0001")` is just a message. Nothing typed into `chat()` is ever read as a command.
- The AI's own words cannot switch a persona, edit memory or approve anything. It may only **file** a proposal (a `[propose ...]` line); filed proposals just wait. There is no approve or reject in this library: approve in a terminal with the `proposal` command.
- `switch_persona` is a method your program calls, not something the model can do.

## Faster generation

The runtime uses a KV cache (`persona_lm/fastgen.py`): the model does not re-read the whole conversation for every new word. On a random test model it was about 14 times faster in the author's sandbox; your laptop will differ. When the conversation fills the whole window the speed-up is gone (the plain generator has to start over there too).

## Honest limits

Tested only with a tiny random model and fake replies, never with a real trained model or on your PC. The export package and a local server (`serve`) are Stage 9b.
