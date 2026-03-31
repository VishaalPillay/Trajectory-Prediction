import os
import sys
import argparse
import numpy as np
import concurrent.futures
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

# Configuration ── set via env var or --dataroot CLI flag
DATAROOT = os.environ.get("NUSCENES_DATAROOT", None)
VERSION = "v1.0-mini"


def _xy_from_record(record):
    """Extract (x, y) from PredictHelper outputs that may be dict or array-like."""
    if isinstance(record, dict):
        if "translation" in record:
            return np.asarray(record["translation"][:2], dtype=np.float32)
        return np.asarray([record.get("x", 0.0), record.get("y", 0.0)], dtype=np.float32)
    arr = np.asarray(record, dtype=np.float32).reshape(-1)
    return arr[:2]

def get_agent_data(scene_token):
    """
    Worker function to process a single scene.
    Initializes a lightweight connection to avoid Windows pickling issues with multiprocessing.
    """
    # Initialize inside each process to avoid pickling large NuScenes objects on Windows.
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    helper = PredictHelper(nusc)

    past_trajectories = []
    future_trajectories = []

    scene = nusc.get("scene", scene_token)
    current_sample_token = scene["first_sample_token"]

    while current_sample_token != "":
        sample = nusc.get("sample", current_sample_token)

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category = ann["category_name"]
            if not (category.startswith("vehicle.") or category.startswith("human.pedestrian.")):
                continue

            instance_token = ann["instance_token"]
            try:
                past_records = helper.get_past_for_agent(
                    instance_token,
                    current_sample_token,
                    seconds=2,
                    in_agent_frame=False,
                )
                future_records = helper.get_future_for_agent(
                    instance_token,
                    current_sample_token,
                    seconds=3,
                    in_agent_frame=False,
                )
            except Exception:
                continue

            # Need fixed windows: 4 past and 6 future at 2Hz.
            if len(past_records) < 4 or len(future_records) < 6:
                continue

            # Past is typically returned newest->oldest; reverse to oldest->newest for sequence modeling.
            past_records = list(reversed(past_records))
            past_xy = np.stack([_xy_from_record(r) for r in past_records[-4:]], axis=0)
            future_xy = np.stack([_xy_from_record(r) for r in future_records[:6]], axis=0)

            if past_xy.shape == (4, 2) and future_xy.shape == (6, 2):
                past_trajectories.append(past_xy.astype(np.float32))
                future_trajectories.append(future_xy.astype(np.float32))

        current_sample_token = sample["next"]

    return past_trajectories, future_trajectories

def main():
    global DATAROOT

    parser = argparse.ArgumentParser(description="Extract trajectories from nuScenes")
    parser.add_argument(
        "--dataroot",
        type=str,
        default=DATAROOT,
        help="Path to nuScenes dataset root (or set NUSCENES_DATAROOT env var)",
    )
    args = parser.parse_args()
    DATAROOT = args.dataroot

    if not DATAROOT:
        print("ERROR: nuScenes data root not set.")
        print("  Set the NUSCENES_DATAROOT environment variable, or pass --dataroot <path>")
        sys.exit(1)

    print(f"Initializing extraction from {DATAROOT}...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    scenes = nusc.scene
    
    np.random.seed(42)
    np.random.shuffle(scenes)
    split_idx = int(len(scenes) * 0.8)
    train_scenes = [s["token"] for s in scenes[:split_idx]]
    val_scenes = [s["token"] for s in scenes[split_idx:]]
    
    def process_split(scene_tokens, split_name):
        all_past = []
        all_future = []
        
        print(f"\nProcessing {split_name} split ({len(scene_tokens)} scenes)...")
        
        # Using ProcessPoolExecutor as requested for heavy parallel extraction
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Map the function across all scene tokens
            results = list(tqdm(executor.map(get_agent_data, scene_tokens), total=len(scene_tokens)))
            
        for pt, ft in results:
            all_past.extend(pt)
            all_future.extend(ft)

        past_np = np.asarray(all_past, dtype=np.float32)
        future_np = np.asarray(all_future, dtype=np.float32)
        
        out_dir = os.path.dirname(os.path.abspath(__file__))
        np.save(os.path.join(out_dir, f"{split_name}_past_raw.npy"), past_np)
        np.save(os.path.join(out_dir, f"{split_name}_future_raw.npy"), future_np)
        
        print(f"Extracted {past_np.shape[0]} valid trajectories for {split_name}.")
        print(f"  Past shape: {past_np.shape}")
        print(f"  Future shape: {future_np.shape}")
        
    process_split(train_scenes, "train")
    process_split(val_scenes, "val")
    print("\n✅ Step 1 Complete: Raw trajectories saved.")

if __name__ == "__main__":
    main()

