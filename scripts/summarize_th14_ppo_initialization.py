"""Verify and summarize the matched stock TH14 SFT-vs-random PPO pilot."""

import argparse
import json
from pathlib import Path
from statistics import median

from scripts.checkpoint_storage import compatibility


BUDGETS = (0, 8192, 16384)
INITIALIZATIONS = ("sft", "random")
SAMPLING_SEEDS = tuple(range(901, 909))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_training(report, initialization, budget, prior, evaluated):
    expected_initial = budget - 8192
    start = report["starting_steps"]
    checkpoint = report["checkpoints"]
    if (report["status"] != "complete" or report.get("initialization", "sft") != initialization
            or start != expected_initial or report["trained_steps"] != budget
            or report["checkpoint_steps"] != budget or checkpoint["ppo_sha256"] != evaluated["weights_sha256"]
            or evaluated["native_training_decisions"] != budget
            or evaluated["training_configuration"] != compatibility(report)):
        raise ValueError("The training report does not match an evaluated checkpoint and budget.")
    if expected_initial and checkpoint["resume_sha256"] != prior["checkpoints"]["ppo_sha256"]:
        raise ValueError("The two training segments did not preserve checkpoint lineage.")
    if [row["steps"] for row in report["updates"]] != list(range(start + 256, budget + 1, 256)):
        raise ValueError("A training segment is missing PPO updates.")


def episode_cells(report):
    cells = {}
    for item in report["completed"]:
        trial = item["trial"]
        key = (trial["arm"], trial["policy_seed"])
        if key in cells:
            raise ValueError("A matched evaluation arm repeats a sampling seed.")
        cells[key] = item["result"]
    expected = {(f"{init}-{budget}", seed) for init in INITIALIZATIONS
                for budget in BUDGETS for seed in SAMPLING_SEEDS}
    if (report["status"] != "complete" or report["matched_blocks"] != len(SAMPLING_SEEDS)
            or set(cells) != expected):
        raise ValueError("Expected all 48 frozen-policy original-game evaluations.")
    return cells


def summarize(cells):
    arms = {}
    for init in INITIALIZATIONS:
        by_budget = {}
        previous = None
        for budget in BUDGETS:
            rows = [cells[f"{init}-{budget}", seed] for seed in SAMPLING_SEEDS]
            frames = [row["stage_frame"] for row in rows]
            center = median(frames)
            initial = by_budget[0]["median_furthest_stage_frame"] if budget else center
            by_budget[budget] = {
                "median_furthest_stage_frame": center,
                "median_change_from_zero": center - initial,
                "frames_gained_per_1000_decisions_since_previous_checkpoint": (
                    (center - previous) / 8.192 if previous is not None else None
                ),
                "per_seed_furthest_stage_frames": dict(zip(SAMPLING_SEEDS, frames, strict=True)),
                "clears": sum(row["outcome"] == "advanced_stage" for row in rows),
                "bombs_used": sum(row["bombs_used"] for row in rows),
            }
            previous = center
        arms[init] = by_budget

    paired = {}
    for budget in BUDGETS:
        deltas = [cells[f"sft-{budget}", seed]["stage_frame"] - cells[f"random-{budget}", seed]["stage_frame"]
                  for seed in SAMPLING_SEEDS]
        paired[budget] = {
            "sft_minus_random_median_frame": arms["sft"][budget]["median_furthest_stage_frame"]
            - arms["random"][budget]["median_furthest_stage_frame"],
            "sft_minus_random_median_paired_frame": median(deltas),
            "sft_minus_random_median_gain_from_own_zero": arms["sft"][budget]["median_change_from_zero"]
            - arms["random"][budget]["median_change_from_zero"],
            "per_seed_sft_minus_random_frames": dict(zip(SAMPLING_SEEDS, deltas, strict=True)),
            "sft_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "random_wins": sum(delta < 0 for delta in deltas),
        }
    return {"by_initialization": arms, "paired_by_budget": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--warm-traces", type=Path, nargs=2, metavar="REPORT", required=True)
    parser.add_argument("--random-traces", type=Path, nargs=2, metavar="REPORT", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = read_json(args.evaluation)
    cells = episode_cells(report)
    specification = report["specification"]
    if (any(specification["controller"][name] for name in ("geometry", "aim_correction", "automatic_bomb"))
            or specification["task"]["decision_limit"] != 3300 or specification["task"]["max_misses"] != 1):
        raise ValueError("The frozen-policy evaluation used a different protocol.")
    policies = specification["policies"]
    template = policies["sft-0"]["weights_sha256"]
    warm = [read_json(path) for path in args.warm_traces]
    random = [read_json(path) for path in args.random_traces]
    if policies["random-0"]["weights_sha256"] != random[0]["checkpoints"]["initial_actor_sha256"]:
        raise ValueError("The evaluated random actor differs from the actual training initialization.")
    for init, traces in zip(INITIALIZATIONS, (warm, random), strict=True):
        for budget, trace in zip(BUDGETS[1:], traces, strict=True):
            verify_training(trace, init, budget, traces[0], policies[f"{init}-{budget}"])
            settings = compatibility(trace)
            settings.pop("initialization", None)
            if settings != compatibility(warm[0]) or settings["base_sha256"] != template:
                raise ValueError("The training arms used different base transformations or PPO settings.")
    report["analysis"] = {
        "training_seed": warm[0]["seed"],
        "sampling_seeds": SAMPLING_SEEDS,
        "online_decision_budgets": BUDGETS,
        "primary_metric": "Median furthest recorded Stage 1 frame across the eight evaluation episodes.",
        **summarize(cells),
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["analysis"], indent=2))


if __name__ == "__main__":
    main()
