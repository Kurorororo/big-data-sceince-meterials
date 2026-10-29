import argparse
import csv
from pathlib import Path

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gaussian-mixture CSV instances.")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[10, 100, 1000, 2000, 4000, 6000, 8000, 9000, 10000])
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    means = np.array([[-4.0, 0.0], [0.0, 1.0], [4.0, 4.0], [13.0, 1.0]])
    scales = np.array([[0.85, 0.65], [0.65, 0.60], [0.70, 0.80], [0.50, 0.50]])
    weights = [0.33, 0.27, 0.25, 0.15]

    groups_rng, noise_rng = [np.random.default_rng(s)
                             for s in np.random.SeedSequence(args.seed).spawn(2)]
    groups = groups_rng.choice(len(weights), size=max(args.sizes), p=weights)
    points = np.round(means[groups] + noise_rng.standard_normal((max(args.sizes), 2))
                      * scales[groups], 4)
    args.output.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        path = args.output / f"gaussian_{n}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y"])
            writer.writerows((f"P{i + 1:04d}", *point) for i, point in enumerate(points[:n]))
        print(path)
