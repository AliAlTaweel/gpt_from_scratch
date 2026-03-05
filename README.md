# 🎭 GPT Shakespeare

A miniature GPT-style language model trained on the **Tiny Shakespeare** dataset, built from scratch using PyTorch. Features character-level tokenization, multi-head self-attention, and instruction fine-tuning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Hyperparameters](#hyperparameters)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training Pipeline](#training-pipeline)
- [Instruction Fine-Tuning](#instruction-fine-tuning)
- [Generation](#generation)
- [Requirements](#requirements)

---

## Overview

This project implements a **character-level GPT language model** inspired by Andrej Karpathy's *nanoGPT*. The model is pre-trained on Shakespeare's plays to learn language patterns, then fine-tuned on instruction–response pairs to answer questions about the text.

---

## Architecture

The model is a decoder-only Transformer with the following components:

| Component | Description |
|---|---|
| `Head` | Single causal self-attention head with key, query, and value projections |
| `MultiHeadAttention` | Parallel attention heads concatenated and projected back to `n_embd` |
| `FeedForward` | Two-layer MLP with ReLU activation and 4× expansion factor |
| `Block` | Transformer block with pre-LayerNorm, residual connections, attention + FFN |
| `GPTLanguageModel` | Full model: token + positional embeddings → N blocks → LayerNorm → LM head |

```
Input Tokens
     │
     ▼
Token Embedding + Positional Embedding
     │
     ▼
┌─────────────────────┐
│  Transformer Block  │ × n_layer
│  ┌───────────────┐  │
│  │  LayerNorm    │  │
│  │  Multi-Head   │  │
│  │  Attention    │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │  LayerNorm    │  │
│  │  FeedForward  │  │
│  └───────────────┘  │
└─────────────────────┘
     │
     ▼
LayerNorm → LM Head → Logits
```

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | `512` | Number of sequences per training step |
| `block_size` | `32` | Maximum context length (tokens) |
| `max_iters` | `5000` | Pre-training steps |
| `learning_rate` | `3e-4` | AdamW learning rate |
| `n_embd` | `512` | Embedding dimension |
| `n_head` | `8` | Number of attention heads |
| `n_layer` | `12` | Number of Transformer blocks |
| `dropout` | `0.2` | Dropout probability |
| `eval_interval` | `500` | Steps between loss evaluations |
| `eval_iters` | `200` | Batches averaged per evaluation |

---

## Project Structure

```
.
├── data/
│   └── input.txt          # Tiny Shakespeare dataset (auto-downloaded)
├── gpt_shakespeare.py     # Main training script
├── gpt_shakespeare.pth    # Saved model weights (after training)
└── README.md
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install torch requests
```

### 2. Run Training

```bash
python gpt_shakespeare.py
```

The dataset will be **automatically downloaded** from Karpathy's char-rnn repo if `./data/input.txt` is not found.

---

## Training Pipeline

Training runs in two sequential phases:

### Phase 1 — Pre-training
The model learns character-level language patterns from the full Shakespeare corpus over **5,000 steps**. Loss is reported every 500 steps on both train and validation splits (90/10 split).

```
Starting Pre-training...
Step 0:    train loss 4.5012, val loss 4.5031
Step 500:  train loss 2.1847, val loss 2.2103
...
Step 4500: train loss 1.6234, val loss 1.8901
```

### Phase 2 — Instruction Fine-Tuning
The pre-trained model is further trained for **1,000 steps** on a small set of Shakespeare Q&A pairs. See the [next section](#instruction-fine-tuning) for details.

---

## Instruction Fine-Tuning

A small set of instruction–response pairs is used to adapt the model for question answering about the play:

| Instruction | Response |
|---|---|
| *"Who are the citizens referring to?"* | They are discussing Caius Marcius. |
| *"What is the main conflict?"* | Food shortage and revolt. |
| *"Summarize Menenius' argument."* | The state is like a body, parts work together. |

Both instruction and response are character-encoded and padded/truncated to `block_size`. To extend fine-tuning, add more entries to the `instructions_data` list in the script.

---

## Generation

After training, the model generates text from a prompt:

```python
prompt = "Summarize this scene:"
context = encode_text(prompt).unsqueeze(0).to(device)
generated = model.generate(context, max_new_tokens=100)
print(decode(generated[0].tolist()))
```

The `generate` method samples autoregressively using **multinomial sampling** over the softmax distribution, conditioned on the last `block_size` tokens.

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- `requests`
- CUDA-capable GPU *(optional but recommended — falls back to CPU automatically)*

---

## Credits

- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn) by Andrej Karpathy
- Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)