"""Plot measured TH14 boss HP, policy firing, and resources from original-game credits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def measurements(path, stage, spell):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        columns = {key: index for index, key in enumerate(meta["native_field_order"])}
        before = data["native_before"]
        after = data["native_after"]
        keep = (before[:, columns["stage"]] == stage) & (before[:, columns["spell_id"]] == spell)
        indices = np.flatnonzero(keep)
        if not len(indices):
            return None
        if np.any(np.diff(indices) != 1):
            raise ValueError("The requested spell occurred in multiple disjoint intervals.")
        def field(name):
            return before[indices, columns[name]]
        return dict(frame=field("stage_frame"), hp=field("boss_hp"), live=field("boss_vulnerable") == 1,
                    shoot=data["action"][indices, 2], lives=field("lives"), bombs=field("bombs"),
                    after_frame=int(after[indices[-1], columns["stage_frame"]]),
                    result=json.loads(str(data["result_json"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credit", action="append", nargs=2, metavar=("LABEL", "NPZ"), required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--spell", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6.6), height_ratios=[2.4, 1, 1.3])
    for i, (label, path) in enumerate(args.credit):
        row = measurements(Path(path), args.stage, args.spell)
        if row is None:
            print(f"{label} did not reach this spell.")
            continue
        x = row["frame"] - row["frame"][0]
        color = f"C{i % 10}"
        axes[0].step(x, row["hp"], where="post", color=color, lw=1.7, label=label)
        # Each bin covers 60 native ticks. Weight held buttons by measured ticks.
        widths = np.diff(np.r_[row["frame"], row["after_frame"]])
        valid = (widths > 0) & (widths <= 8) & row["live"]
        bins = x // 60
        counts = np.bincount(bins, weights=widths * valid)
        firing = np.bincount(bins, weights=widths * valid * row["shoot"])
        fraction = np.divide(firing, counts, out=np.full_like(counts, np.nan), where=counts > 0)
        axes[1].step(np.arange(len(fraction))*60, fraction, where="post", color=color)
        axes[2].step(x, row["lives"], where="post", color=color)
        axes[2].step(x, row["bombs"], where="post", color=color, linestyle=":")
        print(label, len(x), "decisions;", int(row["hp"][0]), "to", int(row["hp"][-1]), "sampled HP;",
              int((row["shoot"] * row["live"]).sum()), "/", int(row["live"].sum()), "vulnerable fire requests")
    if not axes[0].lines:
        raise ValueError("No credit reached this spell.")
    axes[0].set_title(f"TH14 Normal Reimu A  ·  Stage {args.stage}, spell ID {args.spell}", loc="left")
    axes[0].legend(frameon=False)
    for ax, ylabel in zip(axes, ["Boss HP", "Fire held\n(vulnerable)", "Spare lives (solid)\nBombs (dotted)"]):
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["right", "top"]].set_visible(False)
    axes[1].set_ylim(-.08, 1.08)
    axes[2].set_xlabel("Native stage frames since first observation of the spell")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
