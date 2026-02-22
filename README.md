# LLMs from scratch — Learning Notebook

A personal study notebook documenting my journey of learning how LLMs work, from building a GPT model from scratch to fine-tuning it and evaluating its outputs.

---

## What I Learned

This notebook follows my hands-on exploration of:

- How tokenization works under the hood
- Building attention mechanisms and transformer blocks step by step
- Training a GPT model from scratch using PyTorch
- Fine-tuning a pretrained model for classification and instruction following
- Evaluating a fine-tuned model using another LLM as a judge

---

## My Learning Path

### 1. Tokenization
First, I understood how raw text gets converted into tokens before being fed into a model.
- Built simple tokenizers from scratch using regex (`SimpleTokenizerV1`, `SimpleTokenizerV2`)
- Learned about special tokens like `<|endoftext|>` and `<|unk|>`
- Explored Byte Pair Encoding (BPE) using the `tiktoken` library
- Compared vocabulary sizes across GPT-2, GPT-3, and GPT-4

### 2. Embeddings
- Created sliding window input-target pairs to understand how LLMs are trained
- Built a `GPTDatasetV1` class and used PyTorch `DataLoader`
- Learned the difference between token embeddings and positional embeddings

### 3. Attention Mechanisms
This was the most important part — understanding how attention actually works.
- Started with a simplified self-attention to grasp the concept
- Added trainable Q/K/V weight matrices (`SelfAttention_v1`, `SelfAttention_v2`)
- Understood why we divide by √d to keep gradients stable
- Implemented causal masking so the model can't "see the future"
- Added dropout for regularization
- Finally implemented full multi-head attention (`MultiHeadAttention`)

### 4. Building the GPT Architecture
Built the entire GPT model piece by piece:
| Component | What I learned |
|---|---|
| `DummyGPTModel` | How to scaffold the overall structure |
| `LayerNorm` | Why pre-norm stabilizes training |
| `GELU` | A smoother alternative to ReLU |
| `FeedForward` | The 2-layer FFN inside each transformer block |
| `TransformerBlock` | Combining attention + FFN + residual connections |
| `GPTModel` | The complete 124M-parameter model |

### 5. Text Generation
- Implemented greedy decoding with `generate_text_simple`
- Experimented with **temperature scaling** to control randomness
- Applied **top-k sampling** to filter unlikely tokens
- Combined both strategies for better generation quality

### 6. Training & Evaluation
- Wrote a training loop (`train_model_simple`) using AdamW optimizer
- Calculated **cross-entropy loss** and **perplexity** to measure model quality
- Plotted training vs. validation loss to monitor overfitting

### 7. Saving & Loading Models
- Learned how to save/load model weights with `torch.save` / `torch.load`
- Saved both model weights and optimizer state for resumable training

### 8. Loading OpenAI's Pretrained GPT-2 Weights
- Loaded OpenAI's GPT-2 weights using `gpt_download3`
- Parsed TensorFlow checkpoints and mapped weights into my PyTorch model

### 9. Fine-Tuning for Spam Classification
My first fine-tuning experiment — adapting GPT-2 for a binary classification task.
- Used the UCI SMS Spam Collection dataset
- Built `SpamDataset` with proper train/val/test splits
- Froze the backbone and added a 2-class output head
- Trained with `train_classifier_simple` and tested with `classify_review`

### 10. Instruction Fine-Tuning (SFT)
Fine-tuned GPT-2 Medium (355M) to follow instructions like a chat model.
- Formatted data in **Alpaca format** (instruction + input + response)
- Built `InstructionDataset` with proper padding and target masking (`ignore_index=-100`)
- Trained on GPT-2 Medium and saved the result as `gpt2-medium355M-sft.pth`

### 11. Evaluating with an LLM-as-Judge
Used Llama 3 (via Ollama) locally to score my fine-tuned model's responses.
- `query_model` for generating responses
- `generate_model_scores` to score each response on a 0–100 scale
- Compared base GPT-2 vs. fine-tuned model side-by-side

---

## Model Configs I Worked With

| Model | Parameters | Embedding Dim | Heads | Layers |
|---|---|---|---|---|
| GPT-2 Small | 124M | 768 | 12 | 12 |
| GPT-2 Medium | 355M | 1024 | 16 | 24 |

---

## Setup

```bash
pip install torch tiktoken numpy pandas matplotlib tqdm psutil tensorflow
```

Also needs:
- [Ollama](https://ollama.com) with Llama 3 pulled locally (for evaluation)
- `gpt_download3.py` to fetch OpenAI GPT-2 weights

---

## Files Generated

| File | What it is |
|---|---|
| `model.pth` | My trained GPT weights |
| `model_and_optimizer.pth` | Full checkpoint with optimizer state |
| `review_classifier.pth` | Fine-tuned spam classifier |
| `gpt2-medium355M-sft.pth` | Instruction fine-tuned GPT-2 Medium |
| `instruction-data-with-response.json` | Model responses on test set |
| `loss-plot.pdf` | Training/validation loss curves |
| `accuracy-plot.pdf` | Classifier accuracy over epochs |
| `temperature-plot.pdf` | Effect of temperature on token probabilities |

---

## Reference

Followed along with: *Build a Large Language Model From Scratch* by **Sebastian Raschka**

---

## Credits

Learned this under the guidance of **Dr. Raj Dandekar**, Co-Founder of [Vizuara](https://vizuara.ai). His teaching made these complex concepts approachable and easy to understand.
