import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_columns(array: np.ndarray):
    shape = array.shape
    if array.ndim == 1:
        return ["value"]
    if array.ndim == 2:
        return [f"f{j}" for j in range(shape[1])]
    if array.ndim == 3:
        t, f = shape[1], shape[2]
        return [f"t{i}_f{j}" for i in range(t) for j in range(f)]
    raise ValueError(f"Unsupported ndim={array.ndim}. Expected 1D/2D/3D.")


def flatten_rows(array: np.ndarray):
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        n, t, f = array.shape
        return array.reshape(n, t * f)
    raise ValueError(f"Unsupported ndim={array.ndim}. Expected 1D/2D/3D.")


def main():
    parser = argparse.ArgumentParser(description="Preview .npy arrays as a table")
    parser.add_argument("file", type=str, help="Path to .npy file")
    parser.add_argument("--rows", type=int, default=8, help="Number of rows to preview")
    parser.add_argument("--start", type=int, default=0, help="Starting row index")
    args = parser.parse_args()

    path = Path(args.file)
    arr = np.load(path)

    print(f"File: {path}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}\n")

    flat = flatten_rows(arr)
    cols = build_columns(arr)

    start = max(0, args.start)
    end = min(flat.shape[0], start + max(1, args.rows))

    df = pd.DataFrame(flat[start:end], columns=cols)
    df.insert(0, "row", range(start, end))

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
