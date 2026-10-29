import argparse
import csv
import math

from sklearn.cluster import KMeans


def fit_kmeans(points, k, seed, n_init, max_iter):
    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        init="k-means++",
        algorithm="lloyd",
    )

    model.fit(points)

    selected = []

    for cluster, center in enumerate(model.cluster_centers_):
        members = [i for i, label in enumerate(model.labels_) if label == cluster]
        selected.append(min(members, key=lambda i: math.dist(points[i], center)))

    return model, selected


def read_data(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [row["id"] for row in rows]
    points = [(float(row["x"]), float(row["y"])) for row in rows]
    return ids, points


if __name__ == "__main__":
    k = 3
    radius = 1.8
    seed = 20261023
    n_init = 10
    max_iter = 300

    parser = argparse.ArgumentParser(
        description="Run k-means and select representative points."
    )
    parser.add_argument("input", help="CSV with id,x,y columns")
    args = parser.parse_args()

    ids, points = read_data(args.input)

    model, selected = fit_kmeans(points, k, seed, n_init, max_iter)
    nearest = [min(math.dist(p, points[i]) for i in selected) for p in points]

    covered = sum(distance <= radius for distance in nearest)
    print("selected_ids:", [ids[i] for i in selected])
    print("kmeans_objective:", model.inertia_)
    print("kcenter_objective:", max(nearest))
    print("kmedian_objective:", math.fsum(nearest))
    print("maximum_coverage_objective:", covered)
    print("set_cover_feasible:", covered == len(points))
    print("set_cover_objective:", len(selected) if covered == len(points) else None)
