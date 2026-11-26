"""Summarize sealed, paired native TH10 curriculum evaluations with explicit censoring."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("sft", "normal_2048", "mixed_2048", "mixed_4096")
LABELS = ("Human\nSFT", "Normal\n2,048", "Mixed\n2,048", "Mixed\n4,096")


def score(directory):
    documents = []
    for path in sorted(directory.glob("*/report.json")):
        doc = json.loads(path.read_bytes())
        if "seed_blocks" in doc and "trials" in doc:
            if doc["status"] != "complete":
                raise ValueError(f"Unsealed trials in {path}")
            documents.append(doc)
    grouped = defaultdict(dict)
    for doc in documents:
        arms = doc.get("arms", list(ARMS))
        ranks = {1: "Normal", 2: "Hard"}
        name = (doc["horizon"], ranks[doc["difficulty"]])
        for trial in doc["trials"]:
            key = (doc["host"], trial["seed"], trial["arm"])
            if key in grouped[name] or trial["arm"] not in arms:
                raise ValueError(f"Duplicate or unexpected credit {key}")
            grouped[name][key] = trial
        if len(doc["trials"]) != len(arms) * len(doc["seed_blocks"]):
            raise ValueError("Missing credits in a sealed report")
    return grouped


def plot(directory, out):
    groups = score(directory)
    if not groups:
        raise ValueError("No frozen evaluation reports")
    horizons = sorted({h for h, _ in groups})
    fig, panels = plt.subplots(len(horizons), 2, figsize=(9.2, 3*len(horizons)+.8),
                               squeeze=False, layout="constrained")
    summary = []
    for row, horizon in enumerate(horizons):
        for col, rank in enumerate(("Normal", "Hard")):
            ax = panels[row, col]
            trials = groups[horizon, rank]
            blocks = sorted({(host, seed) for host, seed, _ in trials})
            arms = [a for a in ARMS if any(k[2] == a for k in trials)]
            if not blocks or "sft" not in arms:
                raise ValueError(f"Missing baseline for {rank} at {horizon}")
            values = defaultdict(list)
            for i, (host, seed) in enumerate(blocks):
                offset = (i - (len(blocks)-1)/2) * .025
                credits = [trials[host, seed, arm] for arm in arms]
                y = [c["decisions"] for c in credits]
                x = [j+offset for j in range(len(arms))]
                ax.plot(x, y, color=".70", lw=1, zorder=1)
                for j, (arm, c) in enumerate(zip(arms, credits)):
                    ax.scatter(x[j], y[j], marker="^" if c["outcome"] == "horizon" else "o",
                               s=42, color="#206593", edgecolors="white", linewidths=.45, zorder=2)
                    values[arm].append(y[j])
            for i, arm in enumerate(arms):
                outcomes = [trials[h, s, arm]["outcome"] for h, s in blocks]
                diff = [v-b for v, b in zip(values[arm], values["sft"])]
                summary.append(dict(difficulty=rank, horizon=horizon, arm=arm, blocks=blocks,
                                    decisions=values[arm], outcomes=outcomes,
                                    restricted_mean=mean(values[arm]), mean_difference_vs_sft=mean(diff),
                                    differences_vs_sft=diff, capped=outcomes.count("horizon")))
                ax.scatter(i, mean(values[arm]), marker="_", c="#A53E28", s=210, linewidths=2, zorder=3)
            ax.set_xticks(range(len(arms)), [LABELS[ARMS.index(a)] for a in arms])
            ax.set_ylim(0, horizon*1.10)
            ax.set_xlim(-.4, len(arms)-.6)
            ax.set_title(f"{rank}  ·  cap {horizon}  ·  {len(blocks)} matched blocks", fontsize=10)
            ax.grid(axis="y", alpha=.22)
            ax.spines[["top","right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Accepted control decisions")
    fig.suptitle("Original TH10 survival to first life loss\n"
                 "Triangles reach the decision limit. Lines pair controller seeds within each host. Red marks the restricted mean.",
                 fontsize=11)
    fig.savefig(out.with_suffix(".png"),dpi=165)
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    args=parser.parse_args()
    plot(args.directory,args.out)
