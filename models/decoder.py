# decoder.py
# 3-Head GRU Decoder for multi-modal trajectory prediction.
# Each head is an independent GRU that decodes a 6-timestep future trajectory.
# Heads specialize through Winner-Takes-All loss — one learns straight paths,
# others learn turns, producing 3 distinct plausible futures.
