import argparse
import csv
import math

import pyscipopt as scip


def build_model(n, k, d):
    model = scip.Model()

    x = [model.addVar(name=f"x_{i}", vtype="B") for i in range(n)]
    z = {
        (i, j): model.addVar(name=f"z_{i}_{j}", vtype="B")
        for i in range(n)
        for j in range(n)
    }
    radius = model.addVar(name="radius", vtype="C", obj=1)

    model.addCons(scip.quicksum(x) <= k)
    model.addConss(scip.quicksum(z[i, j] for i in range(n)) == 1 for j in range(n))
    model.addConss(z[i, j] <= x[i] for i in range(n) for j in range(n))
    model.addConss(
        radius >= scip.quicksum(d[i, j] * z[i, j] for i in range(n)) for j in range(n)
    )

    model.setMinimize()

    return model, x


def read_data(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [row["id"] for row in rows]
    points = [(float(row["x"]), float(row["y"])) for row in rows]
    return ids, points


if __name__ == "__main__":
    k = 3
    time_limit = 60.0

    parser = argparse.ArgumentParser(description="Solve discrete k-center with SCIP.")
    parser.add_argument("input", help="CSV with id,x,y columns")
    args = parser.parse_args()

    ids, points = read_data(args.input)
    n = len(points)
    d = {(i, j): math.dist(points[i], points[j]) for i in range(n) for j in range(n)}

    model, x = build_model(n, k, d)

    model.hideOutput()
    model.setParam("limits/time", time_limit)
    model.optimize()

    print("status:", model.getStatus())
    if model.getNSols() > 0:
        selected = [i for i, var in enumerate(x) if model.getVal(var) > 0.5]
        print("selected_ids:", [ids[i] for i in selected])
        print("objective:", model.getObjVal())
    print("dual_bound:", model.getDualbound())
