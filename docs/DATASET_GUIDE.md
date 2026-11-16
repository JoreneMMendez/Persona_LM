# Dataset recipes (stream a public Hugging Face dataset, filter, clean, train)

Terminal only. Code: `persona_lm/hfdata.py`. Tests: `tests/test_hfdata.py`. Colab: `colab_hfdata.ipynb`.
Everything here is PUBLIC text. Nothing of your chats is read, written or uploaded.

## The idea
A **recipe** is a small JSON file in `dataset_recipes/`. It says which dataset to stream, which rows to keep (filters), how to clean the text and how many messages you want.
You make one with a wizard, look at the first rows before committing, and run it. For the next dataset you answer the wizard again: no code to edit.

    python -m persona_lm hfdata new                 # wizard: dataset name -> first 20,000 rows -> filters -> cleaning -> saves the recipe
    python -m persona_lm hfdata peek oshichats      # fields, sample messages, score quantiles, what each filter drops
    python -m persona_lm hfdata run oshichats       # stream + filter + clean (Ctrl+C is safe) and write the output
    python -m persona_lm hfdata tokens oshichats    # output -> data_pre_oshichats\ with your frozen work\tok.json
    python -m persona_lm hfdata colab-kit oshichats # a zip for Google Drive (package + recipe + tokenizer + notebook)
    python -m persona_lm hfdata list                # all recipes and how far they got
    python -m persona_lm hfdata show oshichats      # one recipe in plain words

The same things are in the launcher menu: **Dataset recipes (create / run)**.
When there is only one recipe you can leave its id out.

## The example recipe: oshichats
`dataset_recipes/oshichats.json` streams `pykeio/oshichats-v2` (English VTuber live chat, licence CC-BY-4.0 on the dataset card; check it yourself).
It keeps rows with `languages[0] > 0.9` (English), `score > 2.0` and `donation == false`.
**2.0 is only a placeholder.** Run `hfdata peek oshichats` first: it prints the score quantiles of the rows that pass the other filters, so you can pick a threshold from real numbers, then edit the value in the JSON (or use `FILTER_OVERRIDES` in the notebook).
Messages contain tags such as `<|liver:text=BASE64|>` (the streamer's name), `<|fans:text=...|>` and `<|:emote:|>`. They are replaced by `[streamer]` / `[fans]` or dropped (see the cleaning options).
`author` (the username) is never written to disk.

## A recipe file
    {
      "id": "oshichats", "label": "...", "hf_repo": "pykeio/oshichats-v2", "config": null, "revision": null, "split": "train",
      "text_field": "message",
      "filters": [ {"field": "languages[0]", "op": ">", "value": 0.9}, {"field": "score", "op": ">", "value": 2.0}, {"field": "donation", "op": "==", "value": false} ],
      "clean": {"min_chars": 4, "collapse_repeats": 3, "dedup": true, "emotes": "drop", "scrub": true,
                "tags": {"liver": "[streamer]", "fans": "[fans]", "maybe-collaborator": "[streamer]", "*": ""}},
      "group": {"by": "stream.id", "lines": 30},
      "max_kept": 5000000, "max_scanned": 0, "shuffle_buffer": 0, "shard_size": 100000,
      "license": "CC-BY-4.0", "attribution": "...", "private_only": true
    }
* **filters**: `field` is a name or path (`score`, `languages[0]`, `stream.id`); `op` is one of `== != > >= < <= in contains` (a fixed list; nothing is evaluated as code). A missing field or a value of the wrong type fails the filter.
* **clean**: `min_chars` drops short messages (placeholders such as `[streamer]` do not count); `collapse_repeats` N turns "sooooooo", "hahahahaha" and "pog pog pog pog pog" into at most N copies (0 = off); `dedup` removes exact duplicates; `emotes` is `drop`, `word` or `keep`; `scrub` replaces links, e-mails, @handles and phone numbers (same function as the Language data page).
* **group**: put up to `lines` consecutive kept messages with the same `by` value (e.g. one stream) into ONE document, so the model sees chat flow. `null` = one message per document. `prepare` adds an end-of-document token after every document, so 5 million 1-line documents would teach the model very short contexts.
* **max_kept / max_scanned**: stop after this many kept messages / scanned rows (0 = no row limit).
* **shuffle_buffer**: the oshichats file is one long file; read in order it favours the first streams. A buffer N mixes rows only inside a window of N rows, so it helps a little, not completely.

## Files
| What | Where |
|---|---|
| shards (resume point) + state | `public_text/hf_<id>/shard_000001.jsonl ..., state.json` |
| documents for `prepare` | `public_clean/<id>/<id>.jsonl` (`{"text": ...}`) |
| one message per line (to read) | `public_clean/<id>/chat_clean.txt` |
| token files | `data_pre_<id>\` (a finished folder is kept; `--new` makes `_2`) |

## Stopping, resuming, errors
* Every file is written as `.part` and moved into place only when finished.
* Ctrl+C (or a lost connection) finishes the current shard, saves the position and stops. Run the same command to continue. Streaming cannot jump ahead, so a resume reads the dataset again up to the saved row (network only, no processing) and rebuilds the duplicate list from the shards. On a 30 GB file a late resume can take a while.
* Keeping 5,000,000 messages can need scanning most of the 30 GB if the filters keep few rows. Set `max_scanned` if you want a hard limit.
* Changing the source, filters or cleaning of a recipe that already has shards is refused. Add `--restart`: the old work is MOVED to `_to_delete\`, never deleted.
* Memory: the duplicate list holds one 8-byte hash per kept message (about 300 MB for 5 million).

## Mixing it into training (important)
`hfdata tokens ID` makes tokens ONLY from the new text. If you continue pretraining your s30m on live-chat text alone, it can forget Tagalog and Cebuano. Add your existing public text:

    python -m persona_lm hfdata tokens oshichats --also public_mix\tl.jsonl public_mix\ceb.jsonl

Then look at `python -m persona_lm data-report --data-dir data_pre_oshichats --size s30m --tokenizer work\tok.json`. The usual target is about 20 tokens per parameter.
Continue the pretrained model with `train --init-from work3\run_pre\ckpt.npz` and a LOWER learning rate than the first pretrain (the notebook uses 2e-4 down to 2e-5).

## Google Colab (free T4 GPU)
1. `python -m persona_lm hfdata colab-kit oshichats` makes `colab_kit_oshichats.zip`. Unzip it into Google Drive's `MyDrive` (you get `MyDrive/persona/`).
2. Optional: copy `work3\run_pre\ckpt.npz` to `MyDrive/persona/base_ckpt.npz` (public text only) to continue from your pretrained model. Without it the notebook starts from random weights.
3. In Colab: Runtime > Change runtime type > T4 GPU, open `persona/colab_hfdata.ipynb`, run the cells top to bottom. Cell "peek" shows the score quantiles: set `FILTER_OVERRIDES = {"score": ...}` if needed (then `RESTART_SCAN = True` once if a scan had already started).
4. After a disconnect, run all the cells again: the scan, the tokens and the training each continue by themselves (they live in Drive).
The notebook calls the same code as the terminal, so both make the same files. The tokenizer in the kit is `work\tok.json` (only read).

## Licence and privacy
`private_only` is true by default: the cleaned text is for training your own model on your own machine and Drive. Do not re-publish it. Check the licence on the dataset card; the recipe stores the licence and a credit line you type.
