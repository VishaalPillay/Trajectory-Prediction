<div align="center">

# 🚗 Intent & Trajectory Prediction
### Hackathon 2026 — Problem Statement 1

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![nuScenes](https://img.shields.io/badge/Dataset-nuScenes-00BFFF?style=for-the-badge)](https://www.nuscenes.org/prediction)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Predicting the future (x, y) positions of pedestrians and cyclists in L4 autonomous driving environments — 2 seconds in, 3 seconds out, 3 distinct possible futures.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Example Outputs & Results](#-example-outputs--results)
- [Project Structure](#-project-structure)
- [Known Issues Fixed](#-known-issues-fixed)

---

## 🔭 Project Overview

In an L4 urban driving environment, **reacting** to where a pedestrian is — is not enough. The vehicle must **predict** where they will be.

This project builds a multi-modal trajectory prediction model that:

| Capability | Detail |
|---|---|
| **Input** | 2 seconds of past `(x, y)` history per agent (4 timesteps @ 2 Hz) |
| **Output** | 3 distinct predicted future paths over 3 seconds (6 timesteps) |
| **Agents** | Pedestrians and cyclists in urban L4 environments |
| **Social Awareness** | Distance-weighted Social Pooling over a 6 m radius |
| **Multi-Modality** | 3-Head GRU decoder with Winner-Takes-All loss + warmup |
| **Evaluation** | minADE@3 and minFDE@3 (best-of-3-modes metric) |

### Why This Approach Wins

Most competing teams fall into predictable failure modes — building overly complex Transformers that don't ship in time, using MSE loss that collapses all outputs into one average path, or using the wrong evaluation metric. This solution is engineered to avoid every known failure mode while fully satisfying both judging criteria: **Social Context** and **Multi-Modality**.

---

## 🧠 Model Architecture

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    INPUT PREPROCESSING                       │
 │   Raw (x,y) → Agent-Centric Normalization → (x,y,dx,dy)     │
 │   [Stationary guard: skip rotation if displacement < 0.1 m] │
 └──────────────────────┬──────────────────────────────────────┘
                        │  (T=4, 4)
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                    GRU ENCODER                               │
 │   Processes 4-timestep (x, y, dx, dy) sequence              │
 │   Output: hidden state h_i  of shape (hidden_dim,)          │
 └──────────────────────┬───────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
          ▼                            ▼
 ┌─────────────────┐        ┌───────────────────────┐
 │   AGENT STATE   │        │   SOCIAL POOLING      │
 │   h_i           │        │   6 m radius          │
 │                 │        │   Inverse-distance    │
 │                 │        │   weighted sum of     │
 │                 │        │   neighbor states h_j │
 └────────┬────────┘        └──────────┬────────────┘
          │                            │
          └────────────┬───────────────┘
                       │  concat([h_i, social_ctx])
                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │              3-HEAD GRU DECODER                             │
 │                                                             │
 │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
 │    │   Head 1    │  │   Head 2    │  │   Head 3    │       │
 │    │ (straight)  │  │ (left turn) │  │(right turn) │       │
 │    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
 │           │                │                │               │
 │           ▼                ▼                ▼               │
 │       Path 1           Path 2           Path 3             │
 │      (T=6, 2)          (T=6, 2)         (T=6, 2)           │
 └─────────────────────────────────────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │         COMBINED LOSS (WTA + MSE Warmup)                    │
 │  L_total = L_WTA + α * L_MSE   (α decays 1.0 → 0.0)        │
 │  Prevents "dead head" problem in pure WTA training          │
 └─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### ✅ 3-Head Decoder (not CVAE)
A Conditional VAE carries a high risk of **posterior collapse** — the latent variable gets ignored and all three outputs converge to the same average path. Three independent GRU heads trained with Winner-Takes-All loss are simpler, more stable, and equally expressive.

#### ✅ WTA Loss with Warmup (not pure WTA)
Pure WTA from epoch 0 causes the **Dead Head problem**: if random init makes Head 1 slightly better, it wins every batch, collects all gradients, and Heads 2 & 3 never update. The fix: a decaying MSE warmup term forces all heads to learn basic trajectory mechanics before being forced to compete.

```python
# alpha = 1.0 (first N epochs: cooperative learning)
# alpha = 0.0 (final epochs:   competitive specialization)
L_total = L_WTA + alpha * L_MSE
```

#### ✅ 6 m Social Pooling Radius (not 2 m)
Average pedestrian speed is ~1.4 m/s. Over a 3-second horizon, an agent covers **4.2 m**. A 2 m radius means an oncoming pedestrian is structurally invisible to the model until avoidance is impossible. The fix: 6 m radius. Inverse-distance weighting still naturally deprioritises far agents.

#### ✅ Stable Heading Normalization
Heading is computed over the **full 2-second window** (first → last point), not the last two noisy frames. A `hypot < 0.1 m` guard skips rotation entirely for stationary agents, avoiding undefined `arctan2(0, 0)` behavior.

---

## 📦 Dataset

This project uses the **[nuScenes Prediction Challenge](https://www.nuscenes.org/prediction)** dataset.

| Property | Value |
|---|---|
| Dataset | nuScenes v1.0-trainval |
| Annotation rate | 2 Hz |
| Observed window | 2 seconds = **4 timesteps** |
| Prediction horizon | 3 seconds = **6 timesteps** |
| Agent types | Pedestrians, Cyclists |
| Coordinate system | Global (extracted via PredictHelper, then normalized) |

### ⚠️ Critical API Note

`get_prediction_challenge_split()` returns **token strings**, not coordinates. Coordinate extraction **must** go through `PredictHelper`:

```python
from nuscenes.prediction import PredictHelper

helper = PredictHelper(nusc)
past   = helper.get_past_for_agent(instance_token, sample_token,
                                    seconds=2.0, in_agent_frame=False)
future = helper.get_future_for_agent(instance_token, sample_token,
                                      seconds=3.0, in_agent_frame=False)
# Verify shapes:  past.shape == (4, 2)   future.shape == (6, 2)
```

Do **not** attempt to parse `(x, y)` from raw token strings — this is a common failure mode.

### Download nuScenes

1. Register at [https://www.nuscenes.org](https://www.nuscenes.org)
2. Download **v1.0-trainval** (Full dataset) + **Map expansion**
3. Extract to a local directory, e.g. `/data/nuscenes/`

Expected structure:
```
/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/
    ├── attribute.json
    ├── calibrated_sensor.json
    └── ...
```

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU training will be slow)
- ~50 GB disk space for nuScenes v1.0-trainval

### 1. Clone the Repository

```bash
git clone https://github.com/your-team/trajectory-prediction.git
cd trajectory-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0.0
numpy
nuscenes-devkit
pandas
matplotlib
tqdm
scikit-learn
```

### 4. Verify Installation

```bash
python -c "import torch; import nuscenes; print('Setup OK')"
```

---

## 🚀 How to Run

### Step 1 — Extract Trajectories from nuScenes

This step uses `PredictHelper` to extract past/future `(x, y)` sequences and saves them as `.npy` arrays. **Run this once before anything else.**

```bash
python data/extract_nuscenes.py --dataroot /data/nuscenes
```

Output files saved to `data/processed/`:
```
data/processed/
├── train_past.npy      # shape: (N_train, 4, 2)
├── train_future.npy    # shape: (N_train, 6, 2)
├── val_past.npy        # shape: (N_val,   4, 2)
└── val_future.npy      # shape: (N_val,   6, 2)
```

Expected runtime: ~15–30 minutes depending on hardware.

---

### Step 2 — Train the Model

```bash
python training/train.py \
  --dataroot  data/processed \
  --epochs    50 \
  --warmup    5 \
  --hidden    64 \
  --batch     64 \
  --lr        1e-3 \
  --save_dir  checkpoints/
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataroot` | `data/processed` | Path to extracted .npy files |
| `--epochs` | `50` | Total training epochs |
| `--warmup` | `5` | Epochs with alpha=1.0 (MSE warmup before WTA competition) |
| `--hidden` | `64` | GRU hidden state dimension |
| `--batch` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--radius` | `6.0` | Social pooling radius in metres |
| `--save_dir` | `checkpoints/` | Where to save model weights |

Training logs include per-epoch `minADE@3` and `minFDE@3` on the validation set.

---

### Step 3 — Evaluate on Validation Set

```bash
python evaluation/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --dataroot   data/processed
```

Expected output:
```
─────────────────────────────────────────
  Validation Results
─────────────────────────────────────────
  minADE@3  :  X.XX m
  minFDE@3  :  X.XX m
  Samples   :  XXXX
─────────────────────────────────────────
```

---

### Step 4 — Generate Submission Predictions

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --dataroot   /data/nuscenes \
  --output     submission/predictions.json
```

This generates the final prediction file in the nuScenes submission format.

---

### Optional — Visualise Predictions

```bash
python evaluation/visualise.py \
  --checkpoint checkpoints/best.pt \
  --dataroot   data/processed \
  --n_samples  20
```

Saves trajectory plots to `outputs/plots/` — past track in blue, three predicted futures in orange/green/red, ground truth in grey.

---

## 📊 Example Outputs & Results

### Sample Trajectory Visualisation

```
  Future (GT) ········►
  Past ───────►  [AGENT]
                        ╲── Prediction 1 (straight)
                        ├── Prediction 2 (slight left)
                        ╰── Prediction 3 (right turn)
```

The three decoder heads specialize through training:
- **Head 1** → Learns continuation of current heading (straight/mild curves)
- **Head 2** → Learns left-biased deviations (crossing, avoidance left)
- **Head 3** → Learns right-biased deviations (crossing, avoidance right)

### Evaluation Metrics

| Metric | Description | Goal |
|---|---|---|
| `minADE@3` | Mean displacement error across all timesteps, best of 3 modes | Lower is better |
| `minFDE@3` | Displacement error at final predicted timestep, best of 3 modes | Lower is better |

> **Note:** Single-mode ADE/FDE are intentionally **not** reported. Multi-modal models must be evaluated with best-of-K metrics. Reporting single-mode ADE for a 3-head model would understate performance.

### What Good Output Looks Like

- All 3 predicted paths are **visually distinct** — if any two paths are nearly identical, the warmup schedule or WTA loss needs debugging
- `minFDE@3 < minADE@3 * 1.5` — the model should be particularly confident about the general direction of travel
- Social pooling effect: agents near intersections or other pedestrians should show more conservative/spread predictions than isolated agents

---

## 📁 Project Structure

```
trajectory-prediction/
│
├── data/
│   ├── extract_nuscenes.py       # PredictHelper extraction pipeline
│   ├── preprocess.py             # normalize_trajectory(), add_velocity()
│   └── dataset.py                # PyTorch Dataset class with augmentation
│
├── models/
│   ├── encoder.py                # GRU encoder  (T=4, 4) -> (hidden_dim,)
│   ├── decoder.py                # 3-head GRU decoder -> (3, T=6, 2)
│   └── social_pooling.py         # Distance-weighted pooling, radius=6.0 m
│
├── training/
│   ├── loss.py                   # combined_loss() — WTA + MSE warmup
│   └── train.py                  # Full training loop with augmentation
│
├── evaluation/
│   ├── metrics.py                # min_ade(), min_fde()
│   ├── evaluate.py               # Validation script
│   └── visualise.py              # Trajectory plot generator
│
├── inference.py                  # Generates submission predictions
├── requirements.txt
└── README.md
```


---

## 👥 Team

Vishaal Pillay | Nikhil Balamurugan | Karur Nikhil
Built for **Hackathon 2026 — Problem Statement 1: Behavioral AI & Temporal Modeling**

---

<div align="center">

*Built with PyTorch · nuScenes · 72 hours*

</div>
