# The memory graph (Stage 4): a plain-words guide

Run every command from the project folder: `cd C:\Users\user\persona_lm_project`. Each command below takes under a second unless it says otherwise.

## What it is

The AI's **long-term memory is a file** (`memory.json`), not part of the model. You can read it, back it up, and it survives every retrain.

- A **memory** is one fact: the text, when it happened, a feeling tag, how important it is (0 to 1), and which things it is about.
- A **node** is a person, topic, event or feeling (Maria, mango, Cebu trip, homesick). A node is only a name; it "knows" the memories linked to it.
- A **link** joins two nodes that appear in the same memory. The more memories they share, the thicker the line. Links are worked out from the memories every time, so they can never be out of date.
- **Fading:** a memory you never use slowly gets weaker (it halves every 90 days). Using it in a chat makes it strong again. **Pinned** memories never fade. Fading never deletes: a very weak memory becomes *dormant* (not recalled, still in the file).

How it reaches the AI: before each reply, the chat looks for memories that match what you just said and puts a few of them on the AI's "whiteboard" (the 512-piece window), right after its introduction line. It only takes up to about a fifth of the window, so the conversation still fits.

## Add memories

Put `[[double brackets]]` around names. `[[person:Maria]]`, `[[topic:mango]]`, `[[event:Cebu trip]]`, `[[feeling:homesick]]` say what kind of thing it is. Plain `[[Maria]]` reuses an existing node. After a name exists, writing `Maria` in later memories links it automatically.

```powershell
python -m persona_lm memory add "[[person:Maria]] is my sister and loves mango" "#love" "*0.9" "!pin"
```

At the end of the line you may add `#emotion` (one word), `*0.8` (importance from 0 to 1; 0.5 is normal) and `!pin` (never fades).

To say when it happened, add `--when 2026-05-01`.

## Look at memories

```powershell
python -m persona_lm memory nodes
```

```powershell
python -m persona_lm memory list --node Maria
```

```powershell
python -m persona_lm memory search what does Maria like to eat
```

`search` shows exactly what the AI would recall for those words, with the reason. It changes nothing.

## Remove or change

```powershell
python -m persona_lm memory forget m0003
```

This only shows the memory. Add `--yes` to delete it for good (the backup copy of the file is deleted too, so the text really is gone).

```powershell
python -m persona_lm memory set m0003 --importance 0.9 --pin
```

## Tidy up (maintenance)

```powershell
python -m persona_lm memory maintain
```

This only prints a **plan**: which memories would fade, which look like duplicates (same people/things, 80% same words), and which big old groups (6 or more memories about one node, older than 30 days) could be summarised. Nothing changes until you add `--apply`:

```powershell
python -m persona_lm memory maintain --apply
```

Nothing is thrown away: merged and summarised memories stay in `memory.json` marked `merged` or `archived`, and the previous version of the file is kept as `memory.prev.json`. The summaries are **not written by the AI**: they copy the first sentences of the most important memories. A real written summary needs the trained model (a later stage).

## Use it in a chat

```powershell
python -m persona_lm chat --ckpt runs\chat\ckpt.npz --tokenizer tok.json --memory memory.json
```

Inside the chat: `/memory` (overview and what was used in the last reply), `/memory search <words>`, `/remember <text>`, `/forget <id>`. After a reply that used memories you will see `(recalled: m0001 ...)`.
Only **you**, typing, can add or remove memories. Whatever the AI writes is never stored, even if it writes `/remember`.

## See it: Obsidian and the graph picture

```powershell
python -m persona_lm memory export-vault vault
```

Makes a **new** folder `vault` (it refuses to write into a folder that already has files). Open it as a vault in Obsidian and use its graph view. `People`, `Topics`, `Events`, `Feelings` have one note per node; `Memories` has one note per memory. The vault is a copy: editing it does not change `memory.json`.

```powershell
python -m persona_lm memory export-graph memory_graph.html
```

Makes one web page. Double-click it. Bigger circle = more or stronger memories, thicker line = more shared memories. Drag a circle, drag the background, scroll to zoom, hover to read. It works without internet. It shows at most the 600 biggest nodes.

## Privacy

`memory.json`, the vault and the graph page contain your memories as plain text. They are listed in `.gitignore` so they are not pushed to GitHub by accident. Do not upload them anywhere. The log file `memory.log.jsonl` records what changed and when, but never the memory texts.

## What it cannot do (honest limits)

- **Recall works by words and names, not by meaning.** "What fruit does she like?" will not find "Maria loves mango" because they share no word. Embeddings (meaning search) may come later.
- **The model has never been trained to use the memory block.** The plumbing works and is tested, but the real fine-tune (Stage 3 part 2) has to include examples with memory blocks, or the model will ignore them. This is planned, not done.
- Do not edit `memory.json` with another command while a chat is open: the chat refuses to save over a changed file and tells you.
- Links come only from memories that mention both nodes. There is no "Maria is the sister of Jun" relationship type yet; write it as a memory.
- The graph picture was checked in a real browser (Chromium) by the author, but not in your browser.
