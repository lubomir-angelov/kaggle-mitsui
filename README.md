# Kaggle Mitsui Challenge

This repository contains code and resources for the Kaggle Mitsui competition (2025).

The ideas are inspired by:
- [Time Series Library](https://github.com/thuml/)
- [Aurora Weather Forecasting Model](https://github.com/microsoft/aurora) 

## Project Structure

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/kaggle-mitsui.git
   cd kaggle-mitsui
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Download the data:**
    - Place the competition data in the data/ directory.
4. **Run a sample notebook:**
    - Open any notebook in the root folder to get started.
    - Run the jupyter server:
    ```bash
        jupyter notebook
    ```
    - Choose your desired notebook in the browser


## Plan for entries
- ~~Use a lgtbm as a first attempt~~ 
    - added lightgm with sharpe score approximately 0.432
    - tried out cuda to speed up the model
    - lightGBM does not support CUDA on windows see [here](https://github.com/microsoft/LightGBM/issues/3837)
    - installing with 
    ```bash
        pip uninstall lightgbm
        pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON
    ```
        on the notebook cluster also led to the same errors:
    ```bash
        LightGBMError: GPU Tree Learner was not enabled in this build.
        Please recompile with CMake option -DUSE_GPU=1
        add Codeadd Markdown
    ```
    This is now too much effort to resolve, moving on the TimeXer
- Implement TimeXer or one of the other transformers found in the [Time Series Library](https://github.com/thuml/Time-Series-Library?tab=readme-ov-file)
    - train locally and produce weight/checkpoints
    - implement "Aurora Tricks"
    - upload weights/checkpoints in Kaggle and load the model from those, then perform predictions with the MitsuiInference server
- Perform additional steps on the TimXer

### "Aurora tricks"
 TimeXer is a plain-Transformer backbone with patch-wise self-attention plus variate-wise cross-attention, so the same three Aurora-style tricks—

1. **Patch-3-D tokenisation**
2. **Masked-token self-supervised pre-training (MTP + permutation test)**
3. **LoRA / adapter fine-tuning**

—drop straight into the TimeXer pipeline with only minor wiring changes.

---
## TimeXer with Aurora architecture add-ons

## 1 Why the fit is natural

| Aurora / FED trick                                 | TimeXer building block                                                                                                                     | Compatibility notes                                                                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Patchify a **(time × asset × feature)** cube       | TimeXer already slices the *time* axis into patches and keeps a “variate dimension” for exogenous series ([arXiv][1])                      | Extend the patch window from *(τ days × F factors)* to *(τ × A × F)* by reshaping `x → (batch, patch, token)` before the `PatchEmbedding`.                    |
| Masked-token reconstruction + permutation contrast | Cross-attention decoder can predict the missing patch exactly like Aurora’s ViT-UNet; TimeXer’s endogenous-global token acts as the bridge | Add `RandomMask` layer in front of the encoder; use MSE / Huber loss on the reconstructed patch + binary permutation loss between shuffled (exo, endo) pairs. |
| Adapter / LoRA fine-tuning                         | Q/K/V and FFN matrices are standard Transformer shapes                                                                                     | Inject LoRA (rank r=4–8) into `Wq, Wk, Wv, Wo` and optionally the two FFN linear layers; freeze the backbone.                                                 |

---

## 2 Concrete recipe for the Mitsui commodity panel

| Phase                         | Settings                                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Cube build**                | 256-day look-back, **A=16** instruments, **F≈35** engineered factors → patches of *(τ=16, α=4, φ=8)* ⇒ 128 tokens. |
| **Self-supervised pre-train** | Mask 45 % of patches, MSE + permutation loss; AdamW 3 e-4, cosine decay, 30 epochs on one A100 ⇒ ≈ 2 h.            |
| **Freeze + LoRA**             | `r=4`, α=8, dropout 0.05; add `Linear(d_model, 424)` head for the 424 one-step targets.                            |
| **Fine-tune inside Kaggle**   | 10 epochs on CPU (≈ 4 min); early-stop on 20 % hold-out; save only LoRA weights (≃ 5 MB).                          |
| **Serve**                     | Load frozen backbone in \~5 s CPU; inference per daily batch ≈ 40 ms (INT8 backbone).                              |

---

## 3 Minimal adapter patch (PyTorch + PEFT)

```python
from timexer import TimeXer          # repo   :contentReference[oaicite:1]{index=1}
from peft   import LoraConfig, get_peft_model
import torch.nn as nn, torch

backbone = TimeXer(seq_len=256, pred_len=1, d_model=384,
                   patch_len=16, stride=16, n_heads=8, e_layers=4)

backbone.load_state_dict(torch.load("timexer_foundation.pt"), strict=True)
backbone.eval()

peft_cfg = LoraConfig(r=4, lora_alpha=8,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "linear1", "linear2"],
                      lora_dropout=0.05)

model = get_peft_model(backbone, peft_cfg)
model.head = nn.Linear(384, 424, bias=False)   # Kaggle targets
```

Train only `model.head` + LoRA params; everything else stays frozen.

---

## 4 Benefits over vanilla TimeXer

* **Cross-asset regime flips**: 3-D patches let the variate-wise cross-attention see metals + FX in one receptive field.
* **Robust to missing quotes**: masked reconstruction pre-training teaches the network to in-fill trading halts.
* **Few-shot adaptability**: spin up a new LoRA head for a fresh contract with minutes of CPU time.

---

### Bottom line

TimeXer is already **patch-based** and **Transformer-centric**; adding Aurora’s mask-and-reconstruct pre-training and LoRA adapters is mostly bookkeeping.  The result is a lighter-weight foundation model that keeps TimeXer’s strong exogenous-handling ability **and** gains the regime-shift resilience and rapid fine-tune workflow you were after.

[1]: https://arxiv.org/abs/2402.19072?utm_source=chatgpt.com "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
