# finetune_distilGPT

LoRA fine-tuning of DistilGPT2 on WikiText-2, in a single Google Colab notebook.

## What it does

The notebook (`finetune_distilGPT.ipynb`) walks through:

1. Loading `distilbert/distilgpt2` and its tokenizer via Hugging Face `transformers`.
2. Loading the `mikasenghaas/wikitext-2` dataset (train/validation/test splits) via `datasets`.
3. Tokenizing the text (max length 512, padded, with `labels` set for causal LM training).
4. Wrapping the base model with a LoRA adapter (`peft`, `r=8`, `lora_alpha=16`, `lora_dropout=0.1`).
5. Fine-tuning with Hugging Face `Trainer` (3 epochs, batch size 8, fp16, `Trainer`/`TrainingArguments`, `Accelerator`).
6. Saving the fine-tuned model to `./finetuned_model`.
7. Running interactive text generation: prompts for input text and generates a continuation with the fine-tuned model (temperature 0.7, top-k 50, top-p 0.9, repetition penalty 1.2).

## Tech stack

- Python (Google Colab / Jupyter notebook)
- PyTorch
- Hugging Face `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`, `Trainer`, `TrainingArguments`)
- Hugging Face `datasets`
- Hugging Face `peft` (LoRA)
- `accelerate`

No `requirements.txt` or `pyproject.toml` is included — dependencies are installed inline in the notebook via `!pip install datasets` / `!pip install accelerate` (transformers/torch/peft are assumed to be preinstalled, as they are on Colab).

## Setup / usage

This project is a single notebook designed to run on Google Colab (it includes an "Open in Colab" badge):

1. Open `finetune_distilGPT.ipynb` in Google Colab (or Jupyter with a GPU runtime).
2. Run the cells top to bottom. This will:
   - Install `datasets` and `accelerate`.
   - Download `distilgpt2` and the WikiText-2 dataset.
   - Tokenize the data and train a LoRA-adapted DistilGPT2 for 3 epochs.
   - Save the resulting model to `./finetuned_model`.
3. In the final cell, enter a text prompt when asked (`Enter text:`) to generate a continuation with the fine-tuned model.

A GPU runtime is effectively required (`fp16=True` in `TrainingArguments`, and the generation cell hardcodes `device="cuda"`).

## Status

**Work in progress / rough proof-of-concept.** The notebook has been run end-to-end at least once (training completed and text was generated from a saved checkpoint in the committed output), but it has several issues that would need cleanup before this is a polished project:

- No `requirements.txt`/environment file — dependencies are only pinned implicitly by what happened to be installed in the Colab session that produced the checked-in output.
- The generation cell hardcodes `device="cuda"`, so it will fail on a CPU-only or non-CUDA environment.
- No train/eval metrics, loss curves, or evaluation results are surfaced beyond raw Trainer output.
- Sample generation output in the notebook is low quality (the model trails off into blank lines), suggesting undertraining or a tokenization/generation issue that hasn't been addressed.
- No modularized scripts — everything lives in one notebook, no separate training/inference scripts or tests.
- No license file.
