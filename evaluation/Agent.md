# Agent Context — `evaluation/` Module

## Purpose
This folder contains the **evaluation metrics** used to measure prediction quality. It implements `minADE@3` and `minFDE@3` — the official best-of-K metrics that the hackathon judges use for ranking. Getting these metrics wrong means misrepresenting model quality to the leaderboard.

## Role in the Overall Architecture
```
data/ ──→ models/ ──→ training/ ──→ [evaluation/] ──→ inference.py
                         │              ▲
                         │              │
                         └──────────────┘
                     training calls metrics
                     every epoch for validation
```

The `evaluation/` module is used by `training/train.py` during validation to compute metrics every epoch, and by `inference.py` to report final test scores. It is a **Day 2 critical deliverable** — implement and verify metrics before adding any model complexity.

---

## Files

### `metrics.py` — minADE@3 and minFDE@3
- **What it does**: Implements the two primary evaluation metrics for multi-modal trajectory prediction.

#### `min_ade(pred_trajs, gt_traj)` — Minimum Average Displacement Error at K=3
- **Definition**: For K=3 predicted trajectories, compute the average Euclidean distance between each predicted trajectory and the ground truth across all timesteps. Return the **minimum** across the K modes.
- **Input**:
  - `pred_trajs`: `(K=3, T=6, 2)` — 3 predicted trajectories
  - `gt_traj`: `(T=6, 2)` — ground truth trajectory
- **Output**: Scalar float — the best-of-3 average displacement error
- **Formula**:
  ```
  For each mode k:  ADE_k = mean over t of ||pred[k,t] - gt[t]||_2
  minADE@3 = min over k of ADE_k
  ```

#### `min_fde(pred_trajs, gt_traj)` — Minimum Final Displacement Error at K=3
- **Definition**: For K=3 predicted trajectories, compute the Euclidean distance between each predicted **final point** and the ground truth final point. Return the **minimum** across the K modes.
- **Input**:
  - `pred_trajs`: `(K=3, T=6, 2)` — 3 predicted trajectories
  - `gt_traj`: `(T=6, 2)` — ground truth trajectory
- **Output**: Scalar float — the best-of-3 final displacement error
- **Formula**:
  ```
  For each mode k:  FDE_k = ||pred[k, -1] - gt[-1]||_2
  minFDE@3 = min over k of FDE_k
  ```

---

## Why "min" (Best-of-K) Matters
The `min` prefix is **critical** and is what separates correct evaluation from incorrect evaluation:

| Metric | What It Rewards | Result |
|--------|----------------|--------|
| Single-mode ADE | Only one prediction | Multi-modal models look artificially worse |
| Mean ADE@3 | Average of all 3 modes | Penalizes exploration, rewards safe average |
| **minADE@3** | **Best of 3 modes** | **Rewards diverse predictions — correct metric** |

Many competing teams submit with single-mode ADE, which makes their results look far worse than they actually are. We MUST use `minADE@3` and `minFDE@3` to correctly represent our multi-modal model's quality.

## Implementation Reference
```python
def min_ade(pred_trajs, gt_traj):
    # pred_trajs: (K, T, 2)     gt_traj: (T, 2)
    ade_per_mode = np.mean(
        np.linalg.norm(pred_trajs - gt_traj[None], axis=-1),  # (K, T)
        axis=-1  # (K,)
    )
    return float(np.min(ade_per_mode))

def min_fde(pred_trajs, gt_traj):
    fde_per_mode = np.linalg.norm(
        pred_trajs[:, -1, :] - gt_traj[-1, :],  # (K, 2)
        axis=-1
    )
    return float(np.min(fde_per_mode))
```

## Dependencies
- `numpy` — for vectorized distance computation (`np.linalg.norm`, broadcasting)

## Common Mistakes to Avoid
1. **Never evaluate with single-mode ADE** — always use best-of-K (`min`) across the 3 predicted modes
2. **Never average across modes** — that penalizes diverse predictions and defeats the purpose of multi-modality
3. **Always verify metric implementation against known values** before using it to evaluate model changes
4. **Compute metrics on the validation set every epoch** — not just at the end of training
5. **The "final point" in FDE is index `[-1]`** — the last timestep of the prediction window, not the last observation
