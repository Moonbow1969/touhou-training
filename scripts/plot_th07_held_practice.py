"""Plot the final, independently scored TH07 Letty comparison."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ARMS = (("parent", "Frozen parent", "#28608A"), ("practice-ppo", "Practice PPO", "#AB4F32"))
ANALYSES = (("strict_matched_pairs", "Strict"),
            ("all_observed_matched_pairs", "All observed"),
            ("censor_at_first_nonexact_matched_pairs", "Censor at first timing error"))


def draw(source, target):
    result = json.loads(source.read_bytes())
    if result["disposition_count"] != 2 * result["registered_pairs"]:
        raise ValueError("Plot only the completed held cohort")
    figure, (km, ratios) = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1.12, 1]})
    observed = result["all_observed_matched_pairs"]
    for arm, name, color in ARMS:
        points = observed["arms"][arm]["kaplan_meier"]
        if not points:
            continue
        ticks = [0, *(item["tick"] for item in points)]
        survival = [1., *(item["survival"] for item in points)]
        km.step(ticks, survival, where="post", color=color, linewidth=2.5, label=name)
        censored = [item for item in points if item["censored"]]
        km.scatter([item["tick"] for item in censored], [item["survival"] for item in censored],
                   marker="+", linewidths=1.4, s=55, color=color, zorder=3)
    km.set(xlim=(0, 6000), ylim=(0, 1.035), xlabel="Original playable ticks",
           ylabel="Probability of no first death", title=f"First native death, {observed['matched_pairs']} observed pairs")
    km.legend(loc="upper right", frameon=False)
    ratios.axvline(1., color="#738091", linestyle="--", linewidth=1.2, zorder=1)
    limits = [1.]
    labels = []
    for y, (key, label) in enumerate(ANALYSES):
        values = result[key]
        labels.append(f"{label} · {values['matched_pairs']} pairs")
        estimate, interval = values["death_rate_ratio_candidate_parent"], values["paired_bootstrap_percentile_95"]
        if estimate is None:
            continue
        if interval is not None:
            limits += interval
            ratios.hlines(y, interval[0], interval[1], color="#486577", linewidth=2.5, zorder=2)
            ratios.vlines(interval, y - .075, y + .075, color="#486577", linewidth=1.5, zorder=2)
        limits.append(estimate)
        ratios.scatter(estimate, y, color="#154F65", s=52, zorder=3)
    ratios.set(yticks=list(range(len(ANALYSES))), yticklabels=labels, ylim=(len(ANALYSES) - .5, -.5),
               xlabel="Death-rate ratio, PPO / parent", title="Paired estimates and 95% bootstrap intervals")
    width = max(limits) - min(limits)
    padding = max(width * .11, .07)
    ratios.set_xlim(max(0, min(limits) - padding), max(limits) + padding)
    for panel in (km, ratios):
        panel.grid(axis="x", color="#E2E8ED", linewidth=.7)
        panel.spines[["top", "right"]].set_visible(False)
        panel.title.set_fontsize(11)
    figure.suptitle("TH07 Normal Letty practice", x=.07, ha="left", fontsize=16, fontweight="bold")
    figure.text(.07, .025, "Native miss counters and playable tick exposure. The analysis includes invulnerability time.",
                fontsize=9.5, color="#50545B")
    figure.subplots_adjust(left=.07, right=.97, top=.82, bottom=.17, wspace=.77)
    figure.savefig(target, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("target", type=Path)
    args = cli.parse_args()
    draw(args.source, args.target)
