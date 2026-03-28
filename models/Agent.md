# Agent Context — `models/` Module

## Purpose
This folder contains the **neural network architecture** — the GRU-based encoder, multi-head decoder, and social context layer that form the core prediction model. This is the **Day 2 deliverable**.

## Role in the Overall Architecture
```
data/ ──→ [models/] ──→ training/ ──→ evaluation/ ──→ inference.py
           │
           ├── encoder.py      → Past motion → hidden state
           ├── social_pooling.py → Neighbor-aware context
           └── decoder.py      → Hidden state → 3 future trajectories
```

The `training/` module instantiates these model components, and `inference.py` loads the trained checkpoint to generate predictions. The architecture is deliberately **GRU-based, not Transformer-based** — GRUs train in hours with stable gradients and perform near-identically for the short 4-timestep sequences in this task.

---

## Files

### `encoder.py` — GRU Encoder
- **What it does**: Processes the 4-dimensional input sequence `(x, y, dx, dy)` across 4 observed timesteps and outputs a fixed-size hidden state vector.
- **Architecture**:
  - Input: `(batch, T=4, 4)` — 4 timesteps of `(x, y, dx, dy)`
  - Module: `nn.GRU(input_size=4, hidden_size=hidden_dim, batch_first=True)`
  - Output: Hidden state `h_i` of shape `(batch, hidden_dim)`
- **Design choice**: GRU over LSTM — fewer parameters, faster training, equivalent performance for short temporal sequences.
- **The encoder hidden state is the "memory" of each agent's recent motion pattern.** It captures speed, heading, acceleration, and walking style in a single vector that downstream components consume.

### `decoder.py` — 3-Head GRU Decoder (MultiHeadDecoder)
- **What it does**: Takes the concatenated `[encoder_hidden + social_context]` vector and produces **3 independent future trajectory predictions**, each 6 timesteps long.
- **Architecture**:
  - Input: `context` of shape `(batch, 2 * hidden_dim)` — encoder hidden + social context
  - 3 independent `nn.GRU` heads, each with its own `nn.Linear` output layer
  - Each head autoregressively generates 6 steps: feed previous `(x, y)` prediction as next input
  - Output: `(batch, 3, 6, 2)` — 3 modes × 6 timesteps × (x, y)
- **Why 3 heads instead of CVAE**: CVAEs carry high risk of **posterior collapse** where the latent variable is ignored and all 3 outputs converge to the same path. The 3-head approach is simpler, more stable, and equally effective when paired with Winner-Takes-All loss.
- **Head specialization**: Through WTA training, the heads naturally specialize — one learns straight paths, another learns left turns, another learns right turns. This happens automatically; no manual assignment is needed.
- **⚠️ Important**: Each head must be initialized with a **different random seed state** (PyTorch handles this automatically via `nn.ModuleList` if you don't explicitly re-seed). If all heads start identical, WTA may not break symmetry.

### `social_pooling.py` — Social Context Layer
- **What it does**: For every agent, identifies all nearby agents within a fixed radius, computes a distance-weighted sum of their hidden states, and produces a social context vector.
- **Algorithm**:
  1. For agent `i`, compute Euclidean distance to all other agents `j`
  2. Select neighbors where `0 < distance < radius` (exclude self)
  3. Compute weights = `1.0 / distance`, then normalize to sum to 1
  4. Social context = weighted sum of neighbor hidden states
  5. If no neighbors: zero vector of shape `(hidden_dim,)`
  6. Concatenate: `combined = [h_ego, social_ctx]` → shape `(2 * hidden_dim,)`
- **⚠️ BUG v3 GUARD**: Radius **MUST be 6.0 m**, not 2.0 m. Average pedestrian speed is ~1.4 m/s × 3s prediction horizon = 4.2 m of travel. A 2 m radius makes oncoming agents invisible until collision is mathematically unavoidable. The inverse-distance weighting naturally deprioritizes distant agents, so a larger radius adds no noise.
- **Optional upgrade** (Day 3, if time permits): Replace distance-based weights with **single-layer dot-product attention**:
  ```python
  scores = (h_i @ h_j.T) / (hidden_dim ** 0.5)
  weights = torch.softmax(scores, dim=-1)
  social_ctx = weights @ h_j
  ```

---

## Data Flow Through the Model
```
Input: (B, T=4, 4)
        │
   ┌────▼────┐
   │ Encoder  │  GRU over 4 timesteps
   │ (GRU)   │
   └────┬────┘
        │ h_ego: (B, hidden_dim)
        │
   ┌────▼──────────┐
   │ Social Pooling │  distance-weighted neighbor aggregation
   │ (radius=6m)    │
   └────┬──────────┘
        │ social_ctx: (B, hidden_dim)
        │
   ┌────▼──────────────┐
   │ Concatenate        │  [h_ego, social_ctx]
   │ → (B, 2*hidden_dim)│
   └────┬──────────────┘
        │
   ┌────▼──────────┐
   │ 3-Head Decoder │  3 independent GRU heads
   │                │  each → (B, 6, 2)
   └────┬──────────┘
        │
   Output: (B, 3, 6, 2)  — 3 predicted trajectories
```

## Key Hyperparameters
| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `hidden_dim` | 64 or 128 | Encoder & decoder hidden state size |
| `input_dim` | 4 | (x, y, dx, dy) |
| `pred_len` | 6 | 3 seconds at 2 Hz |
| `num_heads` | 3 | Required by hackathon criteria |
| `social_radius` | 6.0 m | See BUG v3 guard |

## Dependencies
- `torch` (PyTorch) — `nn.Module`, `nn.GRU`, `nn.Linear`, `nn.ModuleList`
- `numpy` — for distance calculations in social pooling

## Common Mistakes to Avoid
1. **Never use a social radius below 6.0 m** — agents become invisible too early
2. **Never share weights between decoder heads** — they must be independent `nn.GRU` instances
3. **Always concatenate social context before feeding to decoder** — decoder expects `2 * hidden_dim` input
4. **Handle the "no neighbors" case** — return a zero vector, not NaN or an error
5. **Don't build a Transformer** — with 72 hours and 4-timestep sequences, a well-tuned GRU will outperform a half-baked Transformer every time
