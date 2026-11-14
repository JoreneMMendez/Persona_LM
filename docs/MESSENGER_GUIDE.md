# Training on your Facebook Messenger export

Status: the importer and anonymizer were tested only on a **synthetic** export built to imitate Facebook's quirks (see `tests/test_messenger.py`).
They have **not** been run on a real export. Real exports will contain things the author has not seen; that is why the commands print
reports and why you should read a sample of the output yourself before anything leaves your computer.

Everything below runs on your own computer. The importer and anonymizer contain no network code (a test checks this).

## 1. Get your data from Facebook

Facebook changes its menus from time to time, so look for these words rather than exact clicks: Accounts Center / Settings > **Your information and permissions** >
**Download your information** > choose **messages** only > date range **All time** > format **JSON** (not HTML) > media quality does not matter (low is fine, photos are ignored).
Facebook prepares the file and notifies you; download the zip and unzip it. You should find folders like `.../messages/inbox/<name>_<id>/message_1.json`.
Messages in end-to-end encrypted chats may not be in the export at all.

## 2. Look at what you have (read-only)

    python -m persona_lm messenger-scan "C:\path\to\unzipped\export" --me "Your Name As Facebook Shows It"

A table lists every conversation with a number, how many typed messages it has, how many are yours, and the period. Without `--me` it shows the busiest senders;
one of them is you. If you appear under two names (you renamed your account), repeat the option: `--me "Old Name" --me "New Name"`.

## 3. Import (anonymized by default)

    python -m persona_lm messenger-import "C:\path\to\export" --out data_msgr --me "Your Name" --chats 1,2,3,7-12

Leave out `--me` and `--chats` and it asks you interactively. What it does:

* repairs the broken text encoding ("Ã©" becomes "é", emoji come back);
* keeps only messages you can type: photos, stickers, voice notes, calls, "X named the group", reactions, unsent messages, shared links and messages that are only a link are skipped
  (system notices are recognised in **English** only);
* merges consecutive messages from the same sender into one turn;
* starts a new session after a silence of `--session-gap-hours` (default 6), so context never reaches across days;
* **window mode (default):** one example per reply of yours, with the previous `--context` turns (default 8) as context. Other people are role `user`, you are role `assistant`;
  the loss is placed **only on your reply**. `--mode session` instead uses whole stretches of conversation with loss on all your turns (cheaper, less exact);
* in group chats other people's turns start with `Person_02: ` so the model knows who is talking;
* holds out the **most recent 5 %** of sessions as validation data (whole sessions, so training and validation never share a conversation).

Useful options: `--min-my-messages 20` (skip chats where you hardly wrote), `--no-groups`, `--max-participants 12`, `--max-chars 1400` (one example is about 400 tokens; the model sees 512),
`--system "..."`, `--alias "Mikee=Michael Santos"`, `--extra-names names.txt`, `--paranoid`.

### Output

    data_msgr/train.jsonl, val.jsonl   the chat examples (the only files that may leave your computer, and only if stats.json says "safe_to_upload": true)
    data_msgr/stats.json               counts only, no names
    data_msgr/private/                 REAL names and the map between real and masked names. NEVER upload this folder.

## 4. Anonymization: what it does and does not do

Masked: names of the chat participants (full name, first name, last name, nicknames you list) become `Person_01`, `Person_02`... and **your own names become `[ME]`**;
e-mail addresses, links, @handles, phone numbers (Philippine and international formats), long numbers (IDs, accounts) and street/barangay-style addresses are replaced by `[EMAIL]`, `[URL]`,
`[HANDLE]`, `[PHONE]`, `[NUMBER]`, `[ADDRESS]`. Dates, prices and ordinary words ("block ko siya") are left alone.

After masking, an **audit** looks at the output again and the command **exits with an error (code 2) and marks the data "not safe"** if it still finds a real name, e-mail, link or phone number.
It also lists "suspicious" things (7+ digit runs, @ signs) in `private/audit.txt` for you to look at.

What it **cannot** do, so you must check:

1. **Names of people who are not in the chats** (a friend you only mention, a teacher). The program cannot know them. Read `private/capitalised_words.txt`
   (the most frequent capitalised words), put the real names in a text file (one per line) and re-run with `--extra-names names.txt`.
2. **Addresses** are found by pattern only (English / Philippine styles: "123 Rizal Street", "Brgy. 45", "Blk 4 Lot 12"). A description like "behind the red gate near the church" is not an address pattern.
   The city or barangay name on its own ("Quezon City") is **not** masked. Use `--paranoid` to drop whole messages that contain address-like text or long numbers.
3. **Context that identifies you** ("I'm the only nurse in my batch at X hospital", workplaces, school names, rare events) is not detected.
4. **Name guessing** matches name parts of 3+ letters. A name that is also a common word ("Will", "Grace") is only masked when capitalised; a name part that equals a common word in other
   places gets masked everywhere (for example a surname "Cruz" in any sentence).

So: **read a few hundred lines of `train.jsonl` yourself** before uploading. If you can, keep the real training of the *final* model on your own computer.
`--no-anonymize` exists for data that never leaves your computer: the output then contains real names and is marked not safe to upload.

## 5. Is my data enough? (the honest answer)

The import prints a verdict with your real numbers. The general picture: a personal chat history is typically **a few hundred thousand to a couple of million tokens of your own text**.
A small model like `s10m` (about 14M parameters) would want on the order of **270 million tokens** to train well from scratch (the common rule of thumb is ~20 tokens per parameter).
So your messages alone give a model that mostly memorises phrases. What your data *can* do is teach **style**: how long your replies are, slang, emoji, how you open and close a chat.

Recommended plan (**pretrain, then fine-tune**), using only commands that exist today (this exact sequence was run end to end on tiny demo data with the NumPy backend; it was not run on real data or a GPU):

1. Collect public text **in the languages you chat in** (see the note below). Put it in `pretrain_text/` as `.txt` files. Check each source's license.
2. Train the **tokenizer once** and keep it forever. Train it on the public text *plus* your anonymized messages, so it knows your slang and emoji:

       python -m persona_lm tokenizer --data pretrain_text data_msgr/train.jsonl --vocab 8000 --out tok.json

   (This is the moment the tokenizer gets frozen; changing it later would break old models and memories. The reserved special tokens are already in place.)
3. Pretrain: `prepare --kind pretrain --tokenizer tok.json --data pretrain_text --out data_pre`, then `train --run runs/pre --data-dir data_pre --size s10m ...` on a GPU.
4. Fine-tune on your messages:

       python -m persona_lm prepare --kind chat --tokenizer tok.json --data data_msgr/train.jsonl --val-data data_msgr/val.jsonl --out data_chat
       python -m persona_lm train --run runs/chat --data-dir data_chat --size s10m --init-from runs/pre/ckpt.npz --lr 1e-4 --min-lr 1e-5 --steps 2000

   Watch the validation loss: when it starts rising while training loss keeps falling, stop; the model has begun memorising.
5. The memory graph and Person card (later stages) carry the facts about you. Weights carry style.

**Language note:** if your chats are in Filipino/Taglish (or another mix), a pretraining corpus in English only helps less, and public Filipino text is much smaller. Tell the author which languages
you use so the plan can be adjusted. The window is only 512 tokens and emoji or non-English text use more tokens per character, so `--max-chars` may need to be lower for you.

## 6. Known limitations

* Never run on a real export yet. Expect surprises (new message types, other languages' system notices, odd participants).
* System notices in languages other than English are not recognised and would be kept as if you had typed them.
* Training windows are random slices of the token file, so a slice may start in the middle of an example and lose its beginning.
* All chats are loaded into memory at once; a multi-gigabyte export needs enough RAM.
* Replies to a specific message, reactions, edits and attachments' captions are not modelled.
* Others' turns and your own turns are separated only by the roles; there is no notion of *who* you are talking to in a 1:1 chat beyond the content itself (a later stage can add it through persona/identity prompts).
