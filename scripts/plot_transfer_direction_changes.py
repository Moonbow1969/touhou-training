"""Plot frozen replay diagnostics for the two multi-title SFT sampling policies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source, output):
    models = json.loads(source.read_bytes())["models"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    variants = [
        ("shared-move-motion", "Uniform", "#355d77", -.18),
        ("shared-move-balance", "Change-balanced", "#c75b39", .18),
    ]
    sections = [("start_or_unobserved_prior","Start from rest*"), ("turn","Turn while moving"),
                ("all","All decisions")]
    for axis, (section, title) in zip(axes, sections):
        for key, label, color, shift in variants:
            by_title = models[key]["titles"]
            values = [100 * by_title[t][section]["correct"]/by_title[t][section]["n"]
                      for t in ("13","14","17")]
            bars = axis.bar([i+shift for i in range(3)], values, .33, label=label, color=color)
            axis.bar_label(bars,fmt="%.1f",fontsize=7,padding=1)
        axis.set_title(title,fontsize=10)
        axis.set_ylim(0,85 if section=="all" else 43)
        axis.set_xticks([0,1,2],["TH13","TH14","TH17"])
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y",alpha=.16)
    axes[0].set_ylabel("Correct greedy direction (%)")
    axes[0].legend(loc="upper left",fontsize=7)
    fig.suptitle("Human replay movement under two title-balanced SFT sampling schemes",fontsize=12)
    fig.supxlabel("* Includes frames with no usable previous displacement. Four held replays, 94,432 decisions.",fontsize=8)
    fig.savefig(output,dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    plot(args.source,args.output)
