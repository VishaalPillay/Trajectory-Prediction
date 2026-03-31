import os
import numpy as np


def normalize_trajectory(past_xy, future_xy, stationary_threshold=0.1):
	"""
	Normalize one trajectory pair into an agent-centric frame.
	- Translate so last observed point is at origin.
	- Rotate so agent heading points along +x axis.
	"""
	past_xy = np.asarray(past_xy, dtype=np.float32)
	future_xy = np.asarray(future_xy, dtype=np.float32)

	origin = past_xy[-1]
	past_centered = past_xy - origin
	future_centered = future_xy - origin

	heading = past_xy[-1] - past_xy[-2]
	heading_norm = float(np.linalg.norm(heading))

	# Stationary guard: skip unstable rotation for nearly static agents.
	if heading_norm < stationary_threshold:
		return past_centered, future_centered

	theta = np.arctan2(heading[1], heading[0])
	c = np.cos(-theta)
	s = np.sin(-theta)
	rot = np.array([[c, -s], [s, c]], dtype=np.float32)

	past_rot = past_centered @ rot.T
	future_rot = future_centered @ rot.T
	return past_rot, future_rot


def add_velocity(past_xy):
	"""
	Convert past position trajectory [T,2] into [T,4] with (x,y,dx,dy).
	"""
	past_xy = np.asarray(past_xy, dtype=np.float32)
	vel = np.zeros_like(past_xy, dtype=np.float32)
	vel[1:] = past_xy[1:] - past_xy[:-1]
	return np.concatenate([past_xy, vel], axis=-1)


def _preprocess_split(raw_past, raw_future):
	if raw_past.size == 0 or raw_future.size == 0:
		return (
			np.zeros((0, 4, 4), dtype=np.float32),
			np.zeros((0, 6, 2), dtype=np.float32),
		)

	proc_past = []
	proc_future = []
	for past_xy, future_xy in zip(raw_past, raw_future):
		n_past, n_future = normalize_trajectory(past_xy, future_xy)
		feat_past = add_velocity(n_past)
		proc_past.append(feat_past.astype(np.float32))
		proc_future.append(n_future.astype(np.float32))

	return np.stack(proc_past, axis=0), np.stack(proc_future, axis=0)


def main():
	data_dir = os.path.dirname(os.path.abspath(__file__))

	train_past_raw = np.load(os.path.join(data_dir, "train_past_raw.npy"), allow_pickle=True)
	train_future_raw = np.load(os.path.join(data_dir, "train_future_raw.npy"), allow_pickle=True)
	val_past_raw = np.load(os.path.join(data_dir, "val_past_raw.npy"), allow_pickle=True)
	val_future_raw = np.load(os.path.join(data_dir, "val_future_raw.npy"), allow_pickle=True)

	train_past, train_future = _preprocess_split(train_past_raw, train_future_raw)
	val_past, val_future = _preprocess_split(val_past_raw, val_future_raw)

	np.save(os.path.join(data_dir, "train_past.npy"), train_past)
	np.save(os.path.join(data_dir, "train_future.npy"), train_future)
	np.save(os.path.join(data_dir, "val_past.npy"), val_past)
	np.save(os.path.join(data_dir, "val_future.npy"), val_future)

	print("Saved preprocessed arrays:")
	print(f"  train_past.npy: {train_past.shape}")
	print(f"  train_future.npy: {train_future.shape}")
	print(f"  val_past.npy: {val_past.shape}")
	print(f"  val_future.npy: {val_future.shape}")


if __name__ == "__main__":
	main()
