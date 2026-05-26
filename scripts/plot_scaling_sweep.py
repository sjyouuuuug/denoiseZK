#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/experiments/scaling_sweep/denoise_experiments_scaling_sweep.json"
OUTPUT = ROOT / "thesis/figures/scaling_sweep_constraints_variables.pdf"


def series(rows, backend, metric):
    points = [
        (row["n"], row[metric])
        for row in rows
        if row["backend"] == backend and row.get(metric) is not None
    ]
    points.sort()
    return [x for x, _ in points], [y for _, y in points]


def main():
    rows = json.loads(INPUT.read_text())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.4, 4.0))
    for backend, marker in [("MLP", "o"), ("Conv", "s")]:
        n, constraints = series(rows, backend, "primary_constraints")
        _, variables = series(rows, backend, "primary_variables")
        plt.plot(n, constraints, marker=marker, linewidth=2.2, label=f"{backend} constraints")
        plt.plot(n, variables, marker=marker, linewidth=1.8, linestyle="--", label=f"{backend} variables")

    plt.xlabel("State dimension N")
    plt.ylabel("Count")
    plt.title("BuildOnly circuit size scaling")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
