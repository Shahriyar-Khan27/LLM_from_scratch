# Fine-Tuned LLM Evaluation

An end-to-end implementation of a Large Language Model (LLM) built from scratch in PyTorch — covering tokenization, architecture, pretraining, fine-tuning (classification & instruction), and LLM-as-judge evaluation.

---

## Overview

This notebook walks through the complete lifecycle of building and fine-tuning a GPT-style language model:

- Implementing a GPT model from scratch (tokenizer → attention → transformer blocks → generation)
- Training with cross-entropy loss and evaluating with perplexity
- Loading pretrained OpenAI GPT-2 weights
- Fine-tuning for **spam classification** and **instruction following**
- Evaluating fine-tuned responses using an **LLM-as-judge** (Ollama / Llama 3)

---

## Notebook Structure

### 1. Tokenization
- Regex-based tokenizers (`SimpleTokenizerV1`, `SimpleTokenizerV2`)
- Special tokens: `<|endoftext|>`, `<|unk|>`
- Byte Pair Encoding (BPE) via `tiktoken`
- Vocabulary size comparison across GPT-2 / GPT-3 / GPT-4

### 2. Embeddings
- Sliding window input-target pairs
- `GPTDatasetV1` with PyTorch `DataLoader`
- Token embeddings + positional embeddings

### 3. Attention Mechanisms
- Simplified self-attention with softmax
- Trainable Q/K/V weights (`SelfAttention_v1`, `SelfAttention_v2`)
- Causal (masked) attention with dropout
- Multi-head attention (`MultiHeadAttentionWrapper`, `MultiHeadAttention`)

### 4. GPT Architecture (from scratch)
| Component | Details |
|---|---|
| `DummyGPTModel` | Skeleton architecture |
| `LayerNorm` | Pre-norm layer normalization |
| `GELU` | Activation function |
| `FeedForward` | 2-layer FFN per transformer block |
| `TransformerBlock` | Attention + FFN + residual connections |
| `GPTModel` | Full 124M-parameter model |

### 5. Text Generation
- `generate_text_simple` (greedy decoding)
- Temperature scaling
- Top-k sampling
- Combined temperature + top-k strategy

### 6. Training & Evaluation
- `train_model_simple` training loop (AdamW optimizer)
- Cross-entropy loss and perplexity
- Train / validation loss plotting

### 7. Model Persistence
- Save/load model weights: `torch.save` / `torch.load`
- Checkpoint: weights + optimizer state

### 8. Loading Pretrained Weights
- OpenAI GPT-2 weights loaded via `gpt_download3`
- TensorFlow checkpoint parsing

### 9. Classification Fine-Tuning (Spam Detection)
- Dataset: UCI SMS Spam Collection
- `SpamDataset` with train/val/test splits
- Frozen backbone + 2-class output head
- `train_classifier_simple` training loop
- `classify_review` inference function

### 10. Instruction Fine-Tuning (SFT)
- Dataset formatted in **Alpaca format**
- `InstructionDataset` with train/val/test splits
- Padding and target masking (`ignore_index=-100`)
- Model: GPT-2 Medium (355M parameters)
- Output saved to `gpt2-medium355M-sft.pth`

### 11. LLM-as-Judge Evaluation
- Local inference via **Ollama** (Llama 3)
- `query_model` for single response generation
- `generate_model_scores` for batch scoring (0–100 scale)
- Comparison of base vs. fine-tuned model responses

---

## Model Configurations

| Model | Parameters | Embedding Dim | Heads | Layers |
|---|---|---|---|---|
| GPT-2 Small | 124M | 768 | 12 | 12 |
| GPT-2 Medium | 355M | 1024 | 16 | 24 |

---

## Dependencies

```bash
pip install torch tiktoken numpy pandas matplotlib tqdm psutil tensorflow
```

Also requires:
- [`ollama`](https://ollama.com) with Llama 3 for LLM-as-judge evaluation
- `gpt_download3.py` for loading OpenAI GPT-2 weights

---

## Key Classes Implemented

| Class | Purpose |
|---|---|
| `SimpleTokenizerV1/V2` | Character/word-level tokenizers |
| `GPTDatasetV1` | Sliding window dataset |
| `SpamDataset` | Classification fine-tuning dataset |
| `InstructionDataset` | Instruction fine-tuning dataset |
| `SelfAttention_v1/v2` | Self-attention with trainable weights |
| `CausalAttention` | Masked attention for autoregressive generation |
| `MultiHeadAttention` | Parallel attention heads with weight splits |
| `LayerNorm` | Layer normalization |
| `GELU` | Gaussian Error Linear Unit activation |
| `FeedForward` | Position-wise feed-forward network |
| `TransformerBlock` | Single transformer layer |
| `GPTModel` | Full GPT architecture |

---

## Outputs

| File | Description |
|---|---|
| `model.pth` | Saved GPT model weights |
| `model_and_optimizer.pth` | Weights + optimizer checkpoint |
| `review_classifier.pth` | Fine-tuned spam classifier |
| `gpt2-medium355M-sft.pth` | Instruction fine-tuned GPT-2 Medium |
| `instruction-data-with-response.json` | Test set with model responses |
| `loss-plot.pdf` | Training/validation loss curves |
| `accuracy-plot.pdf` | Classification accuracy curves |
| `temperature-plot.pdf` | Temperature scaling visualization |

---

## References

- *Build a Large Language Model From Scratch* — Sebastian Raschka
- [OpenAI GPT-2](https://github.com/openai/gpt-2)
- [tiktoken](https://github.com/openai/tiktoken)
- [Ollama](https://ollama.com)
