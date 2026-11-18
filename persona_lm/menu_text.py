"""Plain-words text for the terminal menu. Data only: no logic, no screens.

Rules for every sentence here: 18 words or less, no jargon without an entry in GLOSSARY.
ITEM_TEXT is keyed by the `label` of the matching entry in menu.ITEMS.
"""

# Words that must be explained in GLOSSARY if they appear in any text of this file.
JARGON = ("checkpoint", "token", "tokenizer", "val loss", "overfit", "gpu", "cuda", "torch", "batch",
          "accum", "fine-tune", "recipe", "base model", "learning rate", "hugging face", "benchmark")

ITEM_TEXT = {
    "Train or resume a run": dict(
        title="Teach your AI",
        short="Your AI reads prepared text and improves.",
        what="Your AI rereads prepared text and slowly gets better. You can also continue a stopped run.",
        when="Use this after you have prepared text files. Choose it again to continue a run that stopped.",
        need="Prepared text files and a run folder name. A graphics card is much faster.",
        result="A run folder with your model file in it.",
        time="Minutes to many hours. It depends on your computer and your text.",
        safe="Stopping with Ctrl+C is safe. Your chats and your other runs are not changed.",
    ),
    "Chat with a model": dict(
        title="Talk to your AI",
        short="Chat with your AI in this window.",
        what="Type a message and your AI answers in this window. Nothing leaves your computer.",
        when="Use this after a run has finished teaching your AI to chat.",
        need="A model file that was taught to chat, and the tokenizer file that goes with it.",
        result="A conversation on screen. Nothing is saved unless you ask the chat to save it.",
        time="Starts in a few seconds. Each answer takes a moment.",
        safe="This does not change your models or your prepared text files.",
    ),
    "Fine-tune a model (chat tuning)": dict(
        title="Teach your AI to chat",
        short="Teach a base model to answer like a chat.",
        what="Start from a base model that only continues text. Gently teach it to answer like a chat.",
        when="Use this once you have a base model and prepared chat files.",
        need="A base model file, prepared chat files, and a NEW run folder name.",
        result="A new run folder with a chat model. The base model is only copied, never changed.",
        time="Usually less than a few hours. The screen shows the estimate when it starts.",
        safe="Your base model and your chats are not changed. Stopping with Ctrl+C is safe.",
    ),
    "Generate text": dict(
        title="Finish a sentence",
        short="Your AI continues a text, once.",
        what="You type the start of a text. Your AI writes how it might continue, once.",
        when="Use this to try a base model quickly. It does not hold a conversation.",
        need="A model file, its tokenizer file, and some words to start from.",
        result="The continued text on screen. Nothing is saved.",
        time="A few seconds.",
        safe="This does not change any file.",
    ),
    "Project status": dict(
        title="See what you have",
        short="Shows what exists and what to do next.",
        what="Shows which models and prepared text files exist in your project folder. It also suggests a next step.",
        when="Use this when you do not know what to do next.",
        need="Nothing.",
        result="A short list on screen. Nothing is saved.",
        time="A second or two.",
        safe="This does not change your chats or your models. It only looks at file names and sizes.",
    ),
    "Open the Web UI": dict(
        title="Open the web page",
        short="Click in your browser instead of typing.",
        what="Opens a page in your browser where you can click instead of typing. It runs only on your computer.",
        when="Use this if you prefer buttons, charts and forms to this menu.",
        need="A web browser. The internet is not needed.",
        result="A page in your browser. This window stays busy until you stop it with Ctrl+C.",
        time="A few seconds to open.",
        safe="Only you can open the page. Your chats and models are not sent anywhere.",
    ),
    "Speed benchmark": dict(
        title="Test computer speed",
        short="Shows how fast teaching will be.",
        what="Measures how fast your computer reads text. Then it estimates how long teaching your AI will take.",
        when="Use this before a long run, to plan your time.",
        need="Nothing. Pick the model size you plan to use.",
        result="A speed number and a time estimate on screen. Nothing is saved.",
        time="About a minute.",
        safe="This does not change your chats or your models.",
    ),
    "Check the GPU": dict(
        title="Check the graphics card",
        short="Checks if the graphics card can help.",
        what="Checks whether your graphics card can be used to teach your AI faster. It prints a short report.",
        when="Use this once, or when teaching feels too slow.",
        need="PyTorch installed. Without a graphics card it still runs, but says so.",
        result="A report on screen. Nothing is saved.",
        time="Under a minute.",
        safe="This does not change your chats or your models.",
    ),
    "Dataset recipes (create / run)": dict(
        title="Get a public dataset",
        short="Use a saved recipe to get public text.",
        what="Reads a public dataset with a saved recipe, keeps what you want, and cleans it.",
        when="Use this for more public text than the basic language data gives.",
        need="Internet, and a recipe name. Choose 'new' to make a recipe by answering questions.",
        result="Cleaned text and prepared text files in new folders.",
        time="Minutes to hours, depending on the dataset size and your internet.",
        safe="Your chats and models are not changed. Old folders are not overwritten.",
    ),
    "Language data (download)": dict(
        title="Download public text",
        short="Get English, Tagalog and Cebuano text.",
        what="Downloads public English, Tagalog, Cebuano and Taglish text. Then it cleans and checks it.",
        when="Use this first, to get text for your base model.",
        need="An internet connection and free disk space. Taglish needs your OK on the screen.",
        result="Folders named public_text and public_clean in your project folder.",
        time="Can take hours on a slow connection.",
        safe="Your chats and your models are not changed. You can stop with Ctrl+C.",
    ),
    "Language data -> token files": dict(
        title="Make prepared text files",
        short="Turn cleaned text into prepared text files.",
        what="Turns the cleaned language data into prepared text files. One click.",
        when="Use this after downloading and cleaning the language data.",
        need="Cleaned language data and the tokenizer file work\\tok.json.",
        result="A NEW folder, usually named data_pre_new.",
        time="Minutes to an hour, depending on how much text you have.",
        safe="The tokenizer file is only read. No internet is used. Old folders are not changed.",
    ),
    "Quit": dict(
        title="Quit",
        short="Leave the menu.",
        what="Leaves the menu and goes back to the command window.",
        when="Use this when you are done.",
        need="Nothing.",
        result="The menu closes.",
        time="Instant.",
        safe="This does not change any file.",
    ),
}

# One plain sentence per term. Keep the technical name in the key.
GLOSSARY = {
    "model file": "The saved brain of your AI. It is a file named ckpt.npz (a checkpoint) inside a run folder.",
    "checkpoint": "Another name for a model file: a saved copy of your AI at one moment.",
    "tokenizer": "A file that cuts text into small pieces so your AI can read it. Never replace it.",
    "token": "One small piece of text, like a short word or part of a word.",
    "prepared text files": "Your text, already cut into pieces. Your AI can read these files quickly. Also called token files.",
    "run": "One round of teaching. It has its own folder with the model file and notes.",
    "base model": "An AI that learned from lots of public text. It continues text but does not chat yet.",
    "fine-tune": "Gently teach a base model a new skill, such as answering like a chat.",
    "test score": "A number that shows how well your AI does on text it never studied. Lower is better. Also called val loss.",
    "val loss": "The test score. Lower is better. It is checked on text that your AI does not study.",
    "overfitting": "Your AI memorizes its lessons and stops understanding new messages. Stop teaching when the test score rises.",
    "batch": "How many pieces of text your computer reads at one time. More needs more graphics memory.",
    "accum": "How many reads are added up before your AI learns once. It acts like a bigger batch.",
    "learning speed": "How big each lesson step is. Too big is unstable. Too small is slow. Also called learning rate.",
    "learning rate": "The learning speed: how big each step of learning is.",
    "GPU": "The graphics card. It teaches your AI much faster than the main processor (the CPU).",
    "CPU": "The main processor of your computer. It works without a graphics card, but slowly.",
    "CUDA": "The software that lets an NVIDIA graphics card do the teaching work.",
    "torch": "PyTorch, a free tool that lets your computer use the graphics card.",
    "recipe": "A saved list of steps for getting and cleaning one public dataset.",
    "Hugging Face": "A public website that shares free datasets.",
    "benchmark": "A short speed test. It shows how fast your computer can teach your AI.",
    "language data": "Public text in English, Tagalog, Cebuano and Taglish. It is used to teach the base model.",
    "Ctrl+C": "Press the Ctrl and C keys together to stop. It is safe: the last save is kept.",
}

# Each entry: (regular expression, what happened, why, what to do). Matched against the error text, any case.
FRIENDLY_ERRORS = [
    ("out of memory|cuda.*memory|cublas_status_alloc",
     "Your graphics card ran out of memory.",
     "The batch was too big for the card.",
     "Choose the same item again. Set batch to 1 and accum to 4, or use half of the old batch."),
    ("filenotfounderror|no such file|cannot find the (file|path)|not found|does not exist",
     "A file or folder was not found.",
     "It may have been renamed, moved, or never made.",
     "Pick 'See what you have' to list what exists. Then check the spelling of the name."),
    ("merges|specials|not a tokenizer|tokenizer.*(mismatch|differ|wrong)|fingerprint",
     "The tokenizer file does not match your model or your text files.",
     "A model can only read text that was cut with its own tokenizer.",
     "Use work\\tok.json. Do not make a new tokenizer. Run the doctor check if you are unsure."),
    ("no module named .?torch|import torch|torch is not installed|pytorch.*(missing|not installed)",
     "PyTorch is not installed.",
     "The graphics card cannot be used without it. The slower processor path still works.",
     "Install PyTorch, or choose the same item again with the CPU option. Then run 'Check the GPU'."),
    ("no space left|disk full|not enough (disk )?space|errno 28",
     "Your disk is full.",
     "There was no room left to save the file.",
     "Free some space, then choose the same item again. Old backup folders inside a run are safe to move."),
    ("urlerror|connectionerror|getaddrinfo|name or service|timed out|no internet|network is unreachable|max retries",
     "The internet could not be reached.",
     "Your connection may be off, slow, or blocked.",
     "Check your connection and choose the same item again. Work that finished is kept."),
    ("keyboardinterrupt|stopped by ctrl\\+c|interrupted",
     "You stopped it with Ctrl+C.",
     "Ctrl+C asks the program to stop.",
     "This is safe. Choose the same item again to continue where it stopped."),
]

# ---- T2: main menu screen -------------------------------------------------------------------
# Headings of the main list, in display order. Items are named by their ITEMS `label`. Quit is always last.
GROUPS = (
    ("START HERE", ("Project status", "Open the Web UI")),
    ("TALK TO YOUR AI", ("Chat with a model", "Generate text")),
    ("TEACH YOUR AI", ("Train or resume a run", "Fine-tune a model (chat tuning)")),
    ("PREPARE DATA", ("Language data (download)", "Language data -> token files", "Dataset recipes (create / run)")),
    ("CHECKS AND TOOLS", ("Speed benchmark", "Check the GPU")),
)

# The technical name, shown in grey after the plain title.
TECH_NAMES = {
    "Train or resume a run": "train", "Chat with a model": "chat", "Fine-tune a model (chat tuning)": "fine-tune",
    "Generate text": "generate", "Project status": "status", "Open the Web UI": "web UI", "Speed benchmark": "benchmark",
    "Check the GPU": "check-gpu", "Dataset recipes (create / run)": "hfdata", "Language data (download)": "corpus",
    "Language data -> token files": "build-tokens",
}

# Labels of the Details panel: (label, ITEM_TEXT key).
DETAIL_FIELDS = (("What it does:", "what"), ("When to use:", "when"), ("You need:", "need"),
                 ("You get:", "result"), ("Time:", "time"), ("Safe:", "safe"))

SCREEN = dict(
    menu_box="Menu", details_box="Details", project_box="Your project",
    hint="Up/Down move   Enter choose   1-9 pick   q quit",
    fallback_title="PersonaLM menu", fallback_prompt="Type a number, or q to quit: ",
    model="Your AI model:", model_none="none yet. Teach one with the menu.",
    data="Prepared text files:", data_none="none yet",
    torch="Graphics card tools (PyTorch):", torch_yes="installed", torch_no="not found, so teaching will be slow",
    next="Next suggestion:",
)

# Next suggestion, made from scan(). {teach} {chat} {finetune} {download} {prepare} are the plain titles of the items.
NEXT = dict(
    empty="You have no prepared text yet. Pick '{download}', then '{prepare}'.",
    data_only="Your text is ready. Pick '{teach}' to start your first run.",
    unfinished="Your newest run, {name}, is {pct}% taught. Pick '{teach}' to continue it.",
    other="Your newest model is {name}. Pick '{chat}' to try it, or '{teach}' to teach it more.",
    base_only="Your newest model, {name}, is a base model and cannot chat yet. Pick '{finetune}'.",
    chat_ready="Your AI {name} can chat. Pick '{chat}' to talk to it.",
)
