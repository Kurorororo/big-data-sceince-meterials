# Lecture slides and materials on Combinatorial Optimization in Introduction to Big Data Science

Each optimization script contains a SCIP model, a `read_data` function that
returns point IDs and coordinates from a CSV, and an `if __name__ == "__main__":`
block for parameter settings, preprocessing, solving, and output. The k-means example uses scikit-learn
and the same input function structure.

## Setup and usage

Run these commands from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python kcenter.py data/gaussian_100.csv
python kmedian.py data/gaussian_100.csv
python maximum_coverage.py data/gaussian_100.csv
python set_cover.py data/gaussian_100.csv
python kmeans.py data/gaussian_100.csv
```

## Input

The input is a CSV with a header and unique, nonempty point IDs:

```csv
id,x,y
A,-2.9,0.1
B,-2.0,-1.1
C,-1.4,0.4
```

## Objectives and models

| Script | Objective |
| --- | --- |
| `kcenter.py` | Minimize the maximum distance to a selected representative. |
| `kmedian.py` | Minimize the sum of distances to selected representatives, without squaring. |
| `maximum_coverage.py` | Maximize the number of points covered by at most `k` representatives. |
| `set_cover.py` | Minimize the number of representatives needed to cover every point. |
| `kmeans.py` | Minimize the sum of squared distances to continuous cluster centers using Lloyd's algorithm. |

## Instances and generation

The generated files contain 10, 100, 1000, 2000, 4000, 6000, 8000, 9000,
and 10000 points:

```bash
python generate_instances.py
python kcenter.py data/gaussian_100.csv
```

Use `--sizes`, `--seed`, and `--output` to choose instance sizes, the random seed,
and the output directory:

```bash
python generate_instances.py --sizes 10 100 1000 --seed 42 --output custom_data
```

By default, the generator uses all nine sizes above, seed `20261023`, and the
`data` directory next to the script. Sizes must be positive integers and the seed
must be a nonnegative integer. The output directory is created if needed.
Running the generator overwrites the corresponding generated CSV files.
The generator samples a four-component Gaussian mixture and rounds coordinates
to four decimal places. Smaller instances are prefixes of larger instances
generated with the same seed.

| Component | Mean (x, y) | Standard deviation (x, y) | Probability |
| --- | --- | --- | --- |
| Left | (-4, 0) | (0.85, 0.65) | 0.33 |
| Middle | (0, 1) | (0.65, 0.60) | 0.27 |
| Upper right | (4, 4) | (0.70, 0.80) | 0.25 |
| Remote | (13, 1) | (0.50, 0.50) | 0.15 |
