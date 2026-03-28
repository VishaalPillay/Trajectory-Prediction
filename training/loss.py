# loss.py
# Winner-Takes-All (WTA) loss function with warmup.
# Computes loss only for the closest-of-3 predictions to ground truth.
# Includes decaying MSE warmup to prevent the "dead head" problem.
