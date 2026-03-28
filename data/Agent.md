# Agent Context — `data/` Module

## Purpose
This folder is the **data pipeline** — responsible for extracting, preprocessing, and serving trajectory data from the nuScenes dataset. It is the foundation of the entire project. If the data is wrong, every downstream component fails. This is a **Day 1 critical-path deliverable**.

## Role in the Overall Architecture
```
data/ ──→ models/ ──→ training/ ──→ evaluation/ ──→ inference.py
 ^                       │
 │  .npy arrays flow     │  DataLoader feeds batches
 └───────────────────────┘
```

The `data/` module produces clean NumPy arrays that the `training/` module consumes through a PyTorch `DataLoader`. The `inference.py` script also uses `dataset.py` to load test data at prediction time.

---

## Files

### `extract_nuscenes.py` — Raw nuScenes → NumPy Pipeline
- **What it does**: Connects to the nuScenes API, extracts (x, y) trajectories for pedestrians and cyclists, and saves them as `.npy` files.
- **Key API**: Uses `nuscenes.prediction.PredictHelper` — **NOT raw token parsing**. Tokens from `get_prediction_challenge_split()` are strings, not coordinates.
- **Outputs**: `train_past.npy`, `train_future.npy`, `val_past.npy`, `val_future.npy`
- **Critical shapes**:
  - `past.shape == (N, 4, 2)` — N agents × 4 observed timesteps × (x, y)
  - `future.shape == (N, 6, 2)` — N agents × 6 predicted timesteps × (x, y)
- **Temporal resolution**: 2 Hz → 2 seconds observation = 4 timesteps, 3 seconds prediction = 6 timesteps
- **⚠️ BUG v4 GUARD**: `get_prediction_challenge_split()` returns `'{instance_token}_{sample_token}'` strings. You MUST split on `_` and pass both tokens to `PredictHelper.get_past_for_agent()` / `get_future_for_agent()` with `in_agent_frame=False`.
- **Filtering**: Any token returning fewer than 4 past or 6 future timesteps (edge of scene) must be discarded.
- **Validation**: After extraction, print and visually verify that shapes are `(4, 2)` and `(6, 2)` for sampled agents.

### `preprocess.py` — Normalisation & Feature Engineering
- **What it does**: Implements two critical preprocessing functions applied to every trajectory.
- **`normalize_trajectory(obs_seq)`**:
  - Translates trajectory so the **last observed position becomes (0, 0)**.
  - Rotates trajectory so the **last observed heading aligns with the positive x-axis**.
  - Heading is computed over the **full 2-second window** (first → last point), NOT single-frame delta.
  - **⚠️ BUG v2 GUARD**: For stationary agents (`np.hypot(dx, dy) < 0.1`), skip rotation entirely — `arctan2(0, 0)` produces undefined/random angles.
- **`add_velocity(seq)`**:
  - Computes frame-to-frame displacements `(dx, dy)` via `np.diff`.
  - Pads the first timestep with a copy of the second velocity.
  - Transforms `(T, 2)` → `(T, 4)` with features `(x, y, dx, dy)`.
- **Why this matters**: Without agent-centric normalization, the model must learn the same walking behavior independently at every map coordinate — an enormous waste of capacity. This single step can improve ADE by 20-30%.

### `dataset.py` — PyTorch Dataset Class
- **What it does**: Wraps the `.npy` arrays in a `torch.utils.data.Dataset` for use by the training loop.
- **Must load**: The normalized, velocity-augmented past trajectories and raw future trajectories.
- **Data augmentation** (training only, never at inference):
  1. **Random Rotation**: Rotate full scene by random angle ∈ [0, 2π]
  2. **Horizontal Flip**: Mirror all x-coordinates
  3. **Speed Perturbation**: Scale time axis by random factor ∈ [0.9, 1.1]
- **Returns per sample**: `(past_tensor, future_tensor, neighbor_positions)` — neighbors are needed for social pooling.

---

## Key Constants & Conventions
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `obs_secs` | 2.0 | 2 seconds of observed history |
| `pred_secs` | 3.0 | 3 seconds of predicted future |
| `temporal_hz` | 2 Hz | nuScenes prediction challenge standard |
| `obs_timesteps` | 4 | 2s × 2Hz |
| `pred_timesteps` | 6 | 3s × 2Hz |
| `input_dim` | 4 | (x, y, dx, dy) after velocity features |
| `stationary_threshold` | 0.1 m | Below this total displacement, skip heading rotation |

## Dependencies
- `nuscenes-devkit` (for `NuScenes`, `PredictHelper`, `get_prediction_challenge_split`)
- `numpy`, `torch`, `pandas`

## Common Mistakes to Avoid
1. **Never parse coordinates from raw token strings** — always use `PredictHelper`
2. **Never apply data augmentation at inference/evaluation time**
3. **Always verify shapes**: past `(4, 2)`, future `(6, 2)` — discard any that don't match
4. **Always normalize before adding velocity** — velocity features are in the normalized frame
5. **Use official splits** from `get_prediction_challenge_split()` — do not roll your own train/val split
