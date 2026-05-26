#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/experiments/steps_scaling/denoise_experiments_steps_scaling.json"
OUTPUT = ROOT / "thesis/figures/steps_scaling_constraints_variables.pdf"


def series(rows, backend, metric):
    points = [
        (row["total_iters"], row[metric])
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
        t, constraints = series(rows, backend, "primary_constraints")
        plt.plot(t, constraints, marker=marker, linewidth=2.2, label=f"{backend} constraints")

    plt.xlabel("Total denoise iterations T")
    plt.ylabel("Constraints (log scale)")
    plt.yscale("log")
    plt.title("Constraints as recursive trajectory length grows")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
