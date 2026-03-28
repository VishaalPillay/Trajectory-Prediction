# Agent Context — `training/` Module

## Purpose
This folder contains the **training pipeline** — the loss function and the main training loop. It orchestrates data loading, model instantiation, loss computation with WTA warmup, and validation. This spans **Day 2 (baseline) through Day 3 (full model)**.

## Role in the Overall Architecture
```
data/ ──→ models/ ──→ [training/] ──→ evaluation/ ──→ inference.py
                        │
                        ├── loss.py     → WTA + warmup loss
                        └── train.py    → Full training orchestration
```

The training module imports the `Dataset` from `data/`, the model components from `models/`, and the metrics from `evaluation/`. It produces trained checkpoint files (`.pt`) that `inference.py` loads for prediction.

---

## Files

### `loss.py` — Winner-Takes-All (WTA) Loss with Warmup
- **What it does**: Implements `combined_loss(pred, gt, alpha)` — the custom loss function that enables multi-modal prediction without mode collapse.
- **How WTA works**:
  1. Compute per-mode L2 error between each of the 3 predictions and ground truth
  2. Identify the **best (closest) head** for each sample
  3. **Only penalize the best head** — other heads receive zero gradient for that sample
  4. This forces heads to **diversify**: each head is only rewarded when it is the best
- **Why not standard MSE**: MSE punishes every wrong prediction equally. The model's optimal strategy is to predict the same safe average path 3 times → all heads converge → **mode collapse**. This kills multi-modality and fails the judging criterion.
- **⚠️ BUG v1 GUARD — Dead Head Problem**: Pure WTA from epoch 0 causes catastrophic failure:
  - If Head 1 is marginally better by random initialization, it wins every batch
  - Head 1 collects all gradients and improves. Heads 2 and 3 receive **zero gradients forever**
  - Result: 1 trained head + 2 frozen heads = a single-mode model
  - **Fix**: Decaying MSE warmup term:
    | Phase | Alpha | Effect |
    |-------|-------|--------|
    | Epochs 1–N_warmup (e.g. 5) | `alpha = 1.0` | All 3 heads forced to learn basic motion |
    | Epochs N_warmup–N_total | `alpha` decays 1.0 → 0.0 linearly | Cooperative → competitive transition |
    | Final epochs | `alpha = 0.0` | Full specialization, heads compete for modes |
- **Function signature**:
  ```python
  def combined_loss(pred, gt, alpha=0.0):
      # pred: (B, K=3, T, 2)     gt: (B, T, 2)     alpha: warmup weight
      # Returns: scalar loss = L_wta + alpha * L_mse
  ```

### `train.py` — Full Training Loop
- **What it does**: End-to-end training orchestration script. Run with:
  ```bash
  python training/train.py --dataroot /path/to/nuscenes
  ```
- **Responsibilities**:
  1. **Data Loading**: Create `DataLoader` from `data.dataset.TrajectoryDataset` with batch size, shuffling, and augmentation
  2. **Model Init**: Instantiate `Encoder`, `MultiHeadDecoder`, `SocialPooling` from `models/`
  3. **Optimizer**: Adam with learning rate ~1e-3 (tune as needed)
  4. **Training Loop**:
     - Compute alpha warmup schedule per epoch
     - Forward pass: encode → social pool → decode → 3 trajectories
     - Compute `combined_loss(pred, gt, alpha)`
     - Backward pass + optimizer step
  5. **Validation**: Compute `minADE@3` and `minFDE@3` using `evaluation.metrics` every epoch
  6. **Checkpointing**: Save best model by `minADE@3` to `checkpoints/best.pt`
  7. **Logging**: Print epoch-level loss, alpha value, and validation metrics
- **Augmentation**: Applied in `Dataset.__getitem__()`, NOT in the training loop itself
- **Day 2 checkpoint gates** (must all pass before moving to Day 3):
  1. ✅ Model outputs 3 **visually distinct** trajectories for 10+ validation examples
  2. ✅ `minADE@3` computes correctly
  3. ✅ **No dead heads** — all 3 decoder losses are decreasing

---

## Training Schedule
| Hyperparameter | Recommended Value |
|----------------|-------------------|
| `batch_size` | 64 |
| `learning_rate` | 1e-3 |
| `optimizer` | Adam |
| `N_warmup` | 5 epochs |
| `N_total` | 50 epochs |
| `hidden_dim` | 64 or 128 |
| `checkpoint_metric` | `minADE@3` (lower is better) |

## Alpha Warmup Schedule Code
```python
N_warmup = 5
N_total  = 50
for epoch in range(N_total):
    if epoch < N_warmup:
        alpha = 1.0
    else:
        alpha = 1.0 - (epoch - N_warmup) / (N_total - N_warmup)
        alpha = max(alpha, 0.0)
    loss = combined_loss(pred, gt, alpha=alpha)
```

## Dependencies
- `torch` — training loop, optimizer, DataLoader
- `data.dataset` — TrajectoryDataset
- `models.encoder`, `models.decoder`, `models.social_pooling` — model components
- `evaluation.metrics` — minADE@3, minFDE@3 for validation
- `tqdm` — progress bars
- `matplotlib` — optional trajectory visualization

## Common Mistakes to Avoid
1. **Never use pure WTA from epoch 0** — always start with full MSE warmup (`alpha=1.0`)
2. **Never use pure MSE without transitioning to WTA** — heads will converge to identical paths
3. **Always check for dead heads visually** — plot 3 output trajectories, they MUST be different
4. **Validate with minADE@3, not single-mode ADE** — single-mode makes multi-modal results look artificially worse
5. **Save checkpoints by validation metric, not training loss** — training loss can decrease while overfitting
6. **Apply augmentation only during training** — never augment validation or test data