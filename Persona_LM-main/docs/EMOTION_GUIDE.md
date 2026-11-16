# Emotion, mood and your Person card (Stage 5)

Everything stays on your computer. Tested on made-up data only: the Tagalog and Bisaya word lists are my best knowledge and may be wrong. You are the judge.

## 1. How a message is read
`python -m persona_lm emotion tag "pagod na ako sooo"` prints one of ten feelings (joy, love, amusement, surprise, calm, neutral, tired, sadness, fear, anger), how sure it is, and two numbers: valence (unpleasant -1 to pleasant +1) and arousal (sleepy -1 to excited +1). A word after "not/hindi/dili" is ignored, not flipped.
Add your own words with a JSON file: `{"anger": ["inis"], "joy": ["lipay"]}` and `--lexicon file.json`.

## 2. Teaching it your way of talking (takes minutes, optional)
1. `emotion label --data data_msgr/train.jsonl --out emotion_samples.jsonl` (new file, never overwrites; contains your messages)
2. `emotion review --samples emotion_samples.jsonl` : for each message type a feeling to correct, Enter if right, s skip, q stop. Do at least 50-100.
3. `emotion train --samples emotion_samples.jsonl --out emotion_model.json`
The number it prints is agreement with the guesses, NOT accuracy. Only your corrections are real answers.
`emotion tag-memories --memory memory.json` gives memories still tagged neutral a feeling (a plan; `--apply` does it).

## 3. Mood
`chat ... --mood mood.json`. Each message YOU type moves two numbers a little toward its feeling; they drift back to a baseline (half in 6 hours). `/mood` shows it, `/mood tag text` shows how a text would be read, `/mood reset` resets. The AI's own words never change it. The AI sees `<|mood|> tired -0.3 -0.7 <|end|>`; because the model has not been trained with mood blocks yet, the only real effect is a small change of randomness (+-15 %). `python -m persona_lm mood show` / `mood reset` work outside the chat.

## 4. Person card
`person init --name "Your name"` makes `person_card.md`. Open it in Notepad.
- Lines with no tag are yours and are never changed.
- `person build --data train.jsonl --memory memory.json` shows a PLAN; `--apply` writes it (old file kept as person_card.prev.md). It fills Speaking style, Frequent words and Relationships only. Habits and Values are yours: `person add Values I care about my family first`.
- `[auto]` = built by the program. Delete the word `[auto]` to keep a line for good; change it to `[no]` to never get a line with that beginning again.
Use it in chat with `--person person_card.md` (the AI sees it as a speaker block).
