# GPU training guide (beginner friendly)

Status: **written but not yet verified on a real GPU.** The author's environment had no GPU and no PyTorch. Step 5 below
(`check-gpu`) is how you find out whether everything works on *your* hardware. If any line says FAIL, copy the whole output and send it back.

## A. Your own laptop (Acer Aspire 7, Windows)

1. **Find out what GPU you have.** Press Ctrl+Shift+Esc (Task Manager) > Performance. Look for "GPU 1" or "NVIDIA ...". Many Aspire 7 models have a
   GTX 1650, RTX 2050 or RTX 3050 with about 4-6 GB of video memory; some models have no NVIDIA GPU at all.
2. **Install Python 3.10 or newer** from python.org (tick "Add Python to PATH").
3. **Install the NVIDIA driver** (GeForce Experience or nvidia.com/drivers), then check `nvidia-smi` works in a terminal.
4. **Install PyTorch.** Go to https://pytorch.org/get-started/locally/ , choose Stable / Windows / Pip / Python / your CUDA version, and run the one command it shows
   (the right command depends on the CUDA version and changes over time, which is why it is not written here). Also run `pip install numpy`.
5. **Check it:** in the project folder run

       python -m persona_lm check-gpu

   You should see your GPU name and a list of `[PASS]` lines. This compares the GPU code with the original NumPy code, number by number.
6. **Measure speed and find the largest micro-batch that fits:**

       python -m persona_lm benchmark --size s10m --vocab 8000 --try-batches 1,2,4,8,16

7. **Train.** Gradient accumulation lets a small GPU act like a big one: the *effective* batch is `--batch` x `--accum`. If you ran out of memory,
   halve `--batch` and double `--accum`; the training result stays the same.

       python -m persona_lm train --run runs/pre --data-dir data_pre --size s10m --batch 4 --accum 8 --steps 50000

   Ctrl+C is safe. Run the same command again to continue.

Tips for a laptop: plug in the charger, choose the "performance" power mode, keep the vents clear. If `--precision auto` picks fp16 (older GPUs such as the GTX 16-series)
and you see "fp16 overflow" messages often, try `--precision fp32`.

## B. Google Colab (the easiest rented GPU to start with)

Colab runs a notebook in your browser on Google's computers. A free GPU is usually available but not guaranteed, sessions end after a while, and limits change,
so check Colab's own pages for the current rules.

1. Open https://colab.research.google.com and sign in with a Google account. Click **New notebook**.
2. Menu **Runtime > Change runtime type**, choose a **GPU**, Save. (Menu names can shift slightly.)
3. Click the folder icon on the left and upload `persona_lm.zip`. Then in a code cell:

       !unzip -q persona_lm.zip
       %cd persona_lm_project

   PyTorch is already installed in Colab. NumPy too.
4. Keep your work when the session ends by mounting Google Drive:

       from google.colab import drive
       drive.mount('/content/drive')

5. Check and measure:

       !python -m persona_lm check-gpu
       !python -m persona_lm benchmark --size s10m --vocab 8000 --try-batches 1,2,4,8,16,32

6. Train, with the run folder on Drive (so a disconnect loses almost nothing):

       !python -m persona_lm train --run /content/drive/MyDrive/personalm/runs/pre --data-dir data_pre --size s10m --batch 16 --accum 2 --steps 50000

   If the session ends, open a new one, mount Drive again and run the **same command**: it continues from the last checkpoint.
   The free Colab GPU (typically a T4) is not an Ampere card, so `--precision auto` will choose fp16, not bf16.

## C. Moving a run between your laptop and the cloud

A run folder contains `ckpt.npz` (weights, optimizer state, random-number state, step) and `metrics.jsonl`. To move a run:

1. Copy the whole run folder (e.g. `runs/pre`) and the data folder it uses (`data_pre`) to the other machine.
2. Run the same `train` command there. It prints `resumed at step N`.
3. You may change `--batch` and `--accum` for the new GPU's memory as long as `batch x accum` stays the same: the data order is identical.

What "exact" means here: the data the model sees, the optimizer state and the schedule are identical on every machine. The arithmetic is not
bit-identical between different GPUs (or CPU vs GPU), so the weights agree to roughly 6 decimal places, not exactly. On the same machine the CPU path is exact.

## D. Privacy

Anything you upload to Colab or any rented GPU goes to someone else's computer. **Only upload public text and anonymized data.**
The Messenger importer and anonymizer (Milestone 2) will be built for exactly this. Until then, do not upload your messages.
