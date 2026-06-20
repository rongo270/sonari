# Training / fine-tuning Whisper — the realistic guide

## The honest summary

- **You cannot out-train Whisper on general audio.** OpenAI trained it on ~5
  million hours of audio using thousands of GPUs. No individual can beat that.
- **You also cannot train it on your laptop.** Your machine has an Intel
  integrated GPU (no CUDA). Training needs an NVIDIA GPU. On your CPU, even a
  *small* fine-tune would take many days for poor results.
- **What you CAN do, and it's genuinely valuable:** *fine-tune* Whisper so it
  gets noticeably better on a **specific** target — your voice, your accent, a
  weak language, or a special vocabulary (names, jargon, slang). We do this with
  **LoRA**, which trains ~1% of the model and fits on a **free** cloud GPU.

So the plan is: **run inference locally** (the `transcribe.py` / `lyrics.py`
tools we built), and **train in the cloud** when you have a specific target.

---

## Where to get a free / cheap GPU

| Option | GPU | Cost | Good for |
|--------|-----|------|----------|
| **Google Colab** | T4 (16 GB) | Free tier | Learning, small/medium models, LoRA |
| **Kaggle Notebooks** | T4 x2 / P100 | Free, ~30 h/week | Same, a bit more reliable |
| **vast.ai / RunPod** | RTX 3090/4090, A100 | ~$0.2–1.5/hr | Serious large-v3 fine-tunes |
| **Lambda / Paperspace** | A100/H100 | Paid | Big jobs |

Start with **Colab free** to learn the pipeline, then rent a 3090/4090 by the
hour if you want to fine-tune `large-v3` properly.

---

## Step-by-step (Google Colab)

1. Go to https://colab.research.google.com → New notebook.
2. Runtime → Change runtime type → **T4 GPU**.
3. In the first cell, install deps:

   ```python
   !pip install -q transformers datasets accelerate peft evaluate jiwer librosa soundfile tensorboard bitsandbytes
   ```

4. Upload `finetune_whisper.py` (the file in this folder) using the left
   sidebar's file panel, or clone it from your own GitHub.

5. **Get your data onto the machine.** Two options:

   - **Your own voice/domain (recommended).** Make a folder `data/` with `.wav`
     files and a `metadata.csv`:

     ```
     file,text
     clip001.wav,"hello this is a test"
     clip002.wav,"the meeting starts at noon"
     ```
     Aim for **at least 1–3 hours** of clean, correctly-transcribed audio for a
     real improvement. Clips of 5–20 seconds each work best. Zip it, upload,
     unzip in Colab.

   - **A public language dataset (Common Voice).** Just pass `--dataset
     common_voice --lang he` (you'll need a free Hugging Face account and to
     accept the dataset terms once).

6. Train. **Start small to confirm everything works**, then scale up:

   ```python
   # quick sanity run (small model, fast):
   !python finetune_whisper.py --base-model openai/whisper-small \
       --dataset local --data-dir ./data --epochs 1

   # the real run (best model, needs more VRAM -> use --load-8bit on a T4):
   !python finetune_whisper.py --base-model openai/whisper-large-v3 \
       --dataset local --data-dir ./data --epochs 3 --batch-size 8 --load-8bit
   ```

7. Watch the **WER** (Word Error Rate) printed each epoch — lower is better.
   Compare it to the base model's WER on the same test set to prove you improved.

8. Download the resulting `whisper-finetuned/` folder (the LoRA adapter).

---

## Using your fine-tuned model back in `transcribe.py`

`transcribe.py` uses faster-whisper (CTranslate2), so a Hugging Face model must
be (a) merged and (b) converted. Do this once, in Colab or locally:

```python
# 1) merge the LoRA adapter into the base model
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
merged = PeftModel.from_pretrained(base, "whisper-finetuned").merge_and_unload()
merged.save_pretrained("whisper-merged")
WhisperProcessor.from_pretrained("whisper-finetuned").save_pretrained("whisper-merged")
```

```bash
# 2) convert to CTranslate2 (the format faster-whisper loads)
pip install ctranslate2 transformers
ct2-transformers-converter --model whisper-merged \
    --output_dir whisper-large-v3-mine --copy_files tokenizer.json preprocessor_config.json \
    --quantization int8
```

Copy the `whisper-large-v3-mine/` folder to your laptop, then:

```bash
python transcribe.py input/test.wav --model "C:/whisper-transcriber/whisper-large-v3-mine"
```

`--model` accepts a local folder path, so your custom model just works.

---

## Tips for an actually-better model

- **Data quality beats quantity.** Wrong transcripts poison training. Clean,
  exact text matters more than volume.
- **Match your real use case.** Train on the same kind of audio you'll feed it
  (same mic, same accent, same noise level).
- **Don't overfit.** If train WER drops but test WER rises, you've trained too
  long — reduce epochs or add more data.
- **Start with `whisper-small`** to validate the whole loop in minutes before
  spending GPU hours on `large-v3`.
