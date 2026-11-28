"""Render the fixed-seed TH08 capacity comparison from its recorded metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    protocol = json.loads((args.study / "protocol.json").read_bytes())
    comparison = json.loads((args.study / "comparison.json").read_bytes())
    order = ("spatial", "pooled-match", "pooled-wide")
    labels = ("Spatial\n235k", "Pooled, matched compute\n521k", "Pooled, wide\n923k")
    colors = ("#1e846f", "#3267ad", "#965273")
    fig, panels = plt.subplots(1, 2, figsize=(9.6, 4.45), layout="constrained")
    fig.suptitle("TH08 Normal  •  Matched architecture comparison", fontsize=14, fontweight="bold")
    for ax, name, scale, ylabel, gate in zip(panels, ("busy_accuracy_delta", "nll_delta"), (100., 1.),
                                             ("Busy-frame accuracy difference (pp)", "Direction NLL difference"), (1.5, .01)):
        for j, (arm, color) in enumerate(zip(order, colors)):
            values = np.array([r[name] for r in comparison[arm]["seeds"]])*scale
            ax.scatter(j+np.array((-.11,0,.11)), values, color=color, s=45, alpha=.85, zorder=3)
            ax.plot((j-.18,j+.18), [values.mean()]*2, color=color, linewidth=2.5, zorder=4)
        ax.axhline(0, color="#525b66", linewidth=.9)
        ax.axhline(gate, color="#838796", linestyle="--", linewidth=1.2, label="Predeclared threshold")
        ax.set_xticks(range(3), labels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title("Higher is better" if scale == 100 else "Lower is better", fontsize=10)
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.supxlabel(f"Each point is one training seed vs the 234k pooled baseline. Held {protocol['unique_held']:,} windows from two pilots.",
                  fontsize=9, color="#404c58")
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
