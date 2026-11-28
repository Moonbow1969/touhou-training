"""Plot matched held-fold resolution and localization evidence."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), layout="constrained")
    for axis, (key, title, gate) in zip(axes, (
            ("average_precision", "Average precision relative to logistic baseline", 1.5),
            ("event_coverage", "Fatal hits alerted ≥20 ticks ahead", .5))):
        for name, label, color in (("learned", "Camera-estimated position", "#087d82"),
                                   ("coordinates", "Native coordinates (diagnostic)", "#a35439")):
            arms = [f"{n}-{name}" for n in (96,192)]
            folds = []
            for row in report["folds"]:
                reference = row["clocks"][key] if key == "average_precision" else 1.
                values = [row["arms"][arm][key] / reference for arm in arms]
                folds.append(values)
                axis.plot([0,1], values, color=color, alpha=.28, linewidth=1)
            axis.plot([0,1], np.mean(folds, axis=0), "o-", color=color, label=label, linewidth=2.5)
        axis.set(title=title, xticks=[0,1], xticklabels=["96 × 112", "192 × 224"], xlim=(-.15,1.15))
        axis.axhline(gate, color="#525763", linestyle="--", linewidth=1)
        axis.spines[["right", "top"]].set_visible(False)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"TH08 Normal · {report['credits']} matched original credits · faint lines are held folds")
    fig.savefig(args.figure, dpi=170)


if __name__ == "__main__":
    main()
