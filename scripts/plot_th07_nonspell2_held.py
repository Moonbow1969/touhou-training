"""Plot paired Normal Letty Nonspell 2 section clears and original first-loss survival."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ARMS = (("parent", "Frozen parent", "#28608A"), ("practice-ppo", "Nonspell 2 PPO", "#AE4D31"))
ESTIMATES = (("primary_all_assigned", "All assigned"), ("known_physical_pairs_sensitivity", "Known original outcomes"),
             ("strict_matching_sensitivity", "Strict control"))


def plot(source, target):
    result = json.loads(source.read_bytes())
    if result["dispositions"] != 2 * result["registered_pairs"]:
        raise ValueError("Plot only the completed registered original cohort")
    original = result["primary_all_assigned"]["arms"]
    figure, (clear, difference, km) = plt.subplots(1, 3, figsize=(15.3, 4.8),
                                                   gridspec_kw={"width_ratios": [.9, 1.1, 1.18]})
    for y, (arm, label, color) in enumerate(ARMS):
        item = original[arm]
        rate = item["fraction"]
        lower, upper = item["exact_binomial_95"]
        clear.errorbar(rate, y, xerr=[[rate - lower], [upper - rate]], color=color, fmt="o", capsize=4)
        clear.text(.03, y + .22, f"{item['clears']}/{item['denominator']} verified", fontsize=9, color=color)
    clear.set(xlim=(0, 1), yticks=[0, 1], yticklabels=[arm[1] for arm in ARMS], ylim=(1.52, -.5),
              xlabel="Verified clear fraction", title="All assigned originals")
    clear.xaxis.set_major_formatter(PercentFormatter(1.0))

    labels = []
    limits = [-.2, .2]
    for y, (key, label) in enumerate(ESTIMATES):
        item = result[key]
        labels.append(f"{label}\n{item['matched_pairs']} pairs")
        center = item["candidate_minus_parent_rate"]
        if center is None:
            continue
        low, high = item["conservative_exact_difference_95"]
        bootstrap = item["paired_bootstrap_percentile_95"]
        limits.extend((low, high, center))
        difference.hlines(y, low, high, color="#A2B2BC", linewidth=2)
        difference.hlines(y, *bootstrap, color="#356779", linewidth=3.5)
        difference.scatter(center, y, color="#184B62", s=37, zorder=3)
    missing = result["missing_endpoint_sensitivity"]["candidate_minus_parent_bounds"]
    limits.extend(missing)
    if missing[0] == missing[1]:
        difference.scatter(missing[0], .23, color="#AE4D31", marker="|", s=90, linewidth=2)
    else:
        difference.hlines(.23, *missing, color="#AE4D31", linewidth=2.2)
    difference.axvline(0., color="#677989", linewidth=1., linestyle="--")
    difference.axvline(.20, color="#AE4D31", linewidth=.9, linestyle=":")
    difference.xaxis.set_major_formatter(PercentFormatter(1.0))
    span = max(limits) - min(limits)
    difference.set(xlim=(max(-1.04, min(limits) - .07 * span), min(1.04, max(limits) + .07 * span)),
                   yticks=list(range(len(labels))), yticklabels=labels, ylim=(2.5, -.48),
                   xlabel="PPO − parent clear rate", title="Paired clear-rate difference")

    physical = result["native_mortality_and_resources"]["known_physical_matched"]
    for arm, label, color in ARMS:
        points = physical["arms"][arm]["kaplan_meier"]
        if not points:
            continue
        km.step([0, *(p["tick"] for p in points)], [1, *(p["survival"] for p in points)], color=color,
                where="post", linewidth=2.5, label=label)
        censored = [p for p in points if p["censored"]]
        km.scatter([p["tick"] for p in censored], [p["survival"] for p in censored], marker="+", color=color, s=50)
    km.set(xlim=(0, result["max_playable_native_elapsed_ticks"]), ylim=(0, 1.03),
           xlabel="Original playable ticks", ylabel="Probability of no first loss",
           title=f"Native first loss, {physical['matched_pairs']} pairs")
    if physical["matched_pairs"]:
        km.legend(frameon=False, loc="upper right")
    for axes in (clear, difference, km):
        axes.grid(axis="x", color="#DFE6EB", linewidth=.7)
        axes.spines[["top", "right"]].set_visible(False)
        axes.title.set_fontsize(11)
    figure.suptitle("TH07 Normal · Letty Nonspell 2 → Final Spell", x=.045, ha="left", fontweight="bold", fontsize=16)
    figure.text(.045, .052, "The left panel shows exact 95% binomial intervals. Dark and light lines in the middle show "
                "paired bootstrap and conservative exact intervals, respectively.", fontsize=9, color="#4B555F")
    figure.text(.045, .024, "Orange shows the bounds for unresolved endpoints. The dotted line marks the registered "
                "20-point gain. First loss comes from the original miss counter.", fontsize=9, color="#4B555F")
    figure.subplots_adjust(left=.105, right=.972, top=.81, bottom=.19, wspace=.91)
    figure.savefig(target, dpi=185, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("target", type=Path)
    args = cli.parse_args()
    plot(args.source, args.target)
