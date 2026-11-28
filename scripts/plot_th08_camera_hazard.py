"""Plot held-fold hit-forecast metrics from a sealed TH08 training report."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    rows = [row for row in report["folds"] if row["guard"] == 8]
    x = np.arange(1, len(rows) + 1)
    fig, (ap, coverage, calibration) = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
    arms = (("clocks", "Clock, position and stock", "#717b8c", -.17),
            ("visual", "Frozen camera features", "#087d82", .17))
    for key, label, color, offset in arms:
        ap.bar(x+offset, [r[key]["average_precision"] for r in rows], width=.32, color=color, label=label)
        coverage.bar(x+offset, [r[key]["event_coverage"] for r in rows], width=.32, color=color)
        calibration.bar(x+offset, [r[key]["held_false_positive_rate"] for r in rows], width=.32, color=color)
    ap.scatter(x, [1.5*r["clocks"]["average_precision"] for r in rows], color="#aa4d38", marker="_", s=120)
    coverage.axhline(.5, color="#aa4d38", linestyle="--", linewidth=1)
    calibration.axhline(.05, color="#aa4d38", linestyle="--", linewidth=1)
    for axis, title in zip((ap, coverage, calibration), ("Average precision", "Early hit coverage", "False alarms on held credits")):
        axis.set(title=title, xlabel="Held fold", xticks=x)
        axis.spines[["right", "top"]].set_visible(False)
    ap.legend(fontsize=8, frameon=False)
    fig.suptitle(f"TH08 Normal hit forecast · {report['credits']} original credits · {report['primary_folds_passed']}/5 folds passed")
    fig.savefig(args.figure, dpi=170)


if __name__ == "__main__":
    main()
