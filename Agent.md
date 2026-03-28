# Agent Context — Trajectory Prediction (Root)

## Project Overview
This is a **multi-modal trajectory prediction** model for the **Mahe Mobility Hackathon** (Deadline: March 31, 2026). We predict where pedestrians and cyclists will be in the next 3 seconds, based on 2 seconds of observed (x, y) history, in an L4 autonomous driving environment using the **nuScenes** dataset.

## What the Model Does
- **Input**: 2 seconds of past (x, y) coordinate history per agent (4 timesteps at 2 Hz)
- **Output**: 3 distinct predicted future paths over the next 3 seconds (6 timesteps at 2 Hz)
- **Agents**: Pedestrians and cyclists in urban environments
- **Architecture**: GRU encoder → Social Pooling → 3-Head GRU decoder, trained with Winner-Takes-All loss

## Evaluation Criteria
The competition ranks teams on two hard criteria + numerical accuracy:

| Criterion | What It Means | Our Approach |
|-----------|--------------|--------------|
| **Social Context** | Model must account for nearby agents and their influence | Distance-weighted Social Pooling (6m radius) |
| **Multi-Modality** | Model must generate multiple distinct plausible futures | 3-Head Decoder + WTA Loss |
| **ADE/FDE Score** | Raw numerical accuracy | Coordinate normalization + velocity features |

Metrics: **minADE@3** and **minFDE@3** — the best-of-3-modes metric.

---

## Repository Structure
```
trajectory-prediction/
├── data/                          # Data pipeline (Day 1)
│   ├── Agent.md                   # Context for this module
│   ├── extract_nuscenes.py        # Raw nuScenes → .npy arrays
│   ├── preprocess.py              # normalize_trajectory, add_velocity
│   └── dataset.py                 # PyTorch Dataset class
├── models/                        # Neural network architecture (Day 2)
│   ├── Agent.md                   # Context for this module
│   ├── encoder.py                 # GRU encoder
│   ├── decoder.py                 # 3-head GRU decoder
│   └── social_pooling.py          # Social context layer (6m radius)
├── training/                      # Training pipeline (Day 2-3)
│   ├── Agent.md                   # Context for this module
│   ├── loss.py                    # WTA loss + warmup
│   └── train.py                   # Full training loop
├── evaluation/                    # Metrics (Day 2)
│   ├── Agent.md                   # Context for this module
│   └── metrics.py                 # minADE@3, minFDE@3
├── Docs/                          # Hackathon documentation
│   └── Mobility_Hackathon_v2.docx # Technical brief
├── inference.py                   # Generates submission predictions
├── requirements.txt               # Python dependencies
├── Agent.md                       # THIS FILE — root project context
└── README.md                      # Setup + run instructions
```

## End-to-End Pipeline
```
[nuScenes Raw Data]
        │
        ▼
  data/extract_nuscenes.py     →  .npy files (past/future trajectories)
        │
        ▼
  data/preprocess.py           →  Normalized + velocity features (x, y, dx, dy)
        │
        ▼
  data/dataset.py              →  PyTorch DataLoader (with augmentation)
        │
        ▼
  models/encoder.py            →  GRU encodes 4 timesteps → hidden state
        │
        ▼
  models/social_pooling.py     →  Neighbor-aware context (6m radius)
        │
        ▼
  models/decoder.py            →  3-head GRU → 3 future trajectories (6 steps each)
        │
        ▼
  training/loss.py             →  WTA loss with alpha warmup
  training/train.py            →  Full training + validation loop
        │
        ▼
  evaluation/metrics.py        →  minADE@3, minFDE@3 validation scores
        │
        ▼
  inference.py                 →  Final predictions on test split → submission
```

## 72-Hour Execution Plan
| Day | Date | Goal | Deliverables |
|-----|------|------|-------------|
| **1** | March 28 | Data is the Bottleneck | Clean .npy arrays, verified shapes (4,2) and (6,2), normalize + velocity working |
| **2** | March 29 | Baseline Model | Working 3-head GRU, correct minADE@3, visually distinct trajectories, no dead heads |
| **3** | March 30-31 | Winning Features & Submission | Social pooling, augmentation, optional ensemble, code freeze, README, final commit |

## Critical Bugs & Fixes (All Pre-Identified)
| Bug ID | Problem | Fix |
|--------|---------|-----|
| **v1** | Pure WTA from epoch 0 → dead heads | `combined_loss()` with alpha warmup (1.0 → 0.0) |
| **v2** | Stationary agent → random rotation | Heading from full window + `hypot < 0.1` guard |
| **v3** | Social radius 2m → invisible agents | Expanded to 6.0m |
| **v4** | Parsing raw tokens as coordinates | Use `PredictHelper.get_past/future_for_agent()` |

## Key Commands
```bash
# Setup
pip install -r requirements.txt

# Training
python training/train.py --dataroot /path/to/nuscenes

# Inference
python inference.py --checkpoint checkpoints/best.pt
```

## Dependencies
```
torch>=2.0.0, numpy, nuscenes-devkit, pandas, matplotlib, tqdm, scikit-learn
```

## Design Philosophy
> We are not building the most complex model. We are building the model that correctly satisfies both evaluation criteria and actually compiles. Complexity that ships on Day 3 beats elegance that does not run.