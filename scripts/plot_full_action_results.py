"""Draw matched original full-credit outcomes. Each line connects one common controller seed on one host."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    rows = json.loads((args.folder/"credits.json").read_bytes())
    titles = sorted({row["title"] for row in rows})
    fig, axes = plt.subplots(len(args.candidates), len(titles), figsize=(4*len(titles), 2.95*len(args.candidates)+1.3),
                             squeeze=False, layout="constrained")
    for row, candidate in enumerate(args.candidates):
        report = json.loads((args.folder/(candidate+".json")).read_bytes())
        baseline = next(name for name, entry in next(iter(report.values())).items()
                        if name != "candidate" and isinstance(entry, dict) and "n" in entry)
        for col, title in enumerate(titles):
            stats = report[str(title)]
            ax = axes[row, col]
            values = []
            for pair in stats["pairs"]:
                lookup = {r["arm"]: r for r in rows if r["title"] == title and
                          r["host"] == pair["host"] and r["seed"] == pair["seed"]}
                pair_values = [lookup[arm]["decisions"] for arm in (baseline, candidate)]
                values.append(pair_values)
                ax.plot([0, 1], pair_values, color="#a0b5bb", alpha=.75, linewidth=1.1, marker="o", ms=3)
            avg = np.mean(values, axis=0)
            ax.plot([0, 1], avg, color="#006964", lw=3, marker="o", ms=6, zorder=3)
            stage = [(stats[key]["stage2"], stats[key]["n"]) for key in (baseline, "candidate")]
            stage_label='rate unverified' if any(k is None for k,n in stage) else f'{stage[0][0]}/{stage[0][1]} → {stage[1][0]}/{stage[1][1]}'
            ax.set_title(f"TH{title}    Stage 2   {stage_label}", fontsize=11)
            ax.set_xlim(-.17, 1.17)
            ax.set_ylim(bottom=0)
            ax.set_xticks([0, 1], [baseline.upper(), candidate.upper()])
            ax.grid(axis="y", alpha=.20)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Decisions to credit end")
    fig.suptitle("Original-game whole credits\nOne line per matched controller seed · thick line shows the mean",
                 fontsize=14)
    fig.savefig(args.folder/"paired-full-credits.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
