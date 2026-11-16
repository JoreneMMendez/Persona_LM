# Identity core (Stage 6)

The AI's identity lives in files, not in the weights, so it stays the same one after every retrain. Everything is local. Tested on made-up data and tiny practice models only.

Run these in PowerShell from your project folder. Each command is typed by you; protected changes ask you to type a phrase.

1. Make it: `python -m persona_lm identity --dir identity_core init --name "AI name" --owner "Your name" --tokenizer tok.json --memory memory.json --person person_card.md --mood mood.json`
2. Look: `identity --dir identity_core show`
3. Ordinary change (Speaking style, Relation to owner): `identity --dir identity_core set "Speaking style" short and funny`
4. PROTECTED change (Name, Personality, Temperament, Values): `identity --dir identity_core set Values honesty first`. You type `CHANGE IDENTITY`; it is only STAGED. After the cooldown (24 h default) run `identity apply` and type `APPLY IDENTITY`. `identity pending` shows it, `identity cancel` drops it. Never within the cooldown of a model upgrade, in either direction.
5. Versions: `identity add-version v1 --ckpt run/ckpt.npz` (stores the file's SHA-256; the tokenizer is frozen by the first one). `identity make-probes` (once), `identity record v1`, `identity promote v1`.
6. New model: `identity add-version v2 --ckpt run2/ckpt.npz`, then `identity score v2`. Below the threshold (default 0.6, `identity settings --threshold 0.7`) = REJECTED, cannot be promoted. Passed: `identity promote v2`. Go back: `identity rollback v1`.
7. Old model teaches a new one: `identity distill --teacher v1 --out distill.jsonl --data data_msgr/train.jsonl --limit 500`.
8. Check everything: `identity verify` (log chain, self.md, every checkpoint checksum). `identity lineage` shows the history.
9. Chat with it: `chat --core identity_core` (or `--version v1` for an older one). `/identity` inside the chat shows the card.

Hand edits of self.md change nothing until `identity adopt`. The score measures how SIMILAR answers are, not whether it is "the same mind". Keep your checkpoint files: the core stores their checksums, not copies.
