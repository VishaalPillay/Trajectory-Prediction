# Trajectory-Prediction

A lightweight, multi-modal trajectory prediction model for urban L4 environments. Built with GRUs, Social Context Pooling, and Winner-Takes-All loss on the nuScenes dataset. Specifically built for the Mahe Mobility Hackathon.

---

## Quick Start (for teammates)

The preprocessed data (`.npy` files) is already included in the repo — **no dataset download needed**.

### 1. Clone & install

```bash
git clone https://github.com/VishaalPillay/Trajectory-Prediction.git
cd Trajectory-Prediction
pip install -r requirements.txt
```

### 2. Start training

```bash
python training/train.py
```

That's it. The training script loads data from `data/` using relative paths.

---

## Project Structure

```
Trajectory-Prediction/
├── data/
│   ├── train_past.npy          # Preprocessed training past trajectories
│   ├── train_future.npy        # Preprocessed training future trajectories
│   ├── val_past.npy            # Preprocessed validation past trajectories
│   ├── val_future.npy          # Preprocessed validation future trajectories
│   ├── *_raw.npy               # Raw (unnormalized) trajectories
│   ├── extract_nuscenes.py     # Step 1: Extract from nuScenes (optional)
│   ├── preprocess.py           # Step 2: Normalize & add velocity features
│   ├── dataset.py              # PyTorch Dataset class
│   └── preview_npy.py          # Utility to inspect .npy arrays
├── models/
│   ├── encoder.py              # GRU trajectory encoder
│   ├── decoder.py              # Multi-modal decoder
│   └── social_pooling.py       # Social context pooling module
├── training/
│   └── train.py                # Training loop
├── evaluation/
│   └── metrics.py              # ADE / FDE metrics
├── inference.py                # Generate submission predictions
├── requirements.txt
└── README.md
```

---

## Re-extracting data from nuScenes (optional)

Only needed if you want to re-extract from scratch with different parameters.

1. Download the [nuScenes mini split](https://www.nuscenes.org/nuscenes#download) (~4 GB)
2. Run extraction:
   ```bash
   python data/extract_nuscenes.py --dataroot /path/to/nuscenes
   ```
   Or set the environment variable:
   ```bash
   export NUSCENES_DATAROOT=/path/to/nuscenes   # Linux/Mac
   set NUSCENES_DATAROOT=C:\path\to\nuscenes    # Windows
   python data/extract_nuscenes.py
   ```
3. Then run preprocessing:
   ```bash
   python data/preprocess.py
   ```

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- See `requirements.txt` for full list
