"""Final reporting on sealed TH14 replays for three fixed SFT cohorts."""

import argparse
import json
from collections import defaultdict
from hashlib import file_digest, sha256
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import (
    batch_block,
    index_split,
    movement_direction,
    predict_heads,
    read_shard,
)

SEEDS = (61, 97, 137)
TH14_PLAN = Path(__file__).resolve().parents[1] / "results/2026-11-23-th14-full-policy/replay-capture-split.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def group_hash(row):
    group = row.get("player_group") or row.get("player", "").casefold()
    if not group:
        raise ValueError("Replay metadata lacks a player group.")
    return sha256(group.encode("utf-8")).hexdigest()


def test_shards(directory, split_plan):
    expected = {row["replay"]: row for row in split_plan["test"]}
    task = split_plan["task"]
    expected_stages = {int(stage) for stage in task["stages"]}
    shards = index_split(directory, "test")
    collected = set()
    evidence = []
    for path, cell in shards:
        row = load_json(path.with_suffix(".json"))
        replay = cell["replay"]
        stage_key = (replay, cell["stage"])
        if stage_key in collected or replay not in expected:
            raise ValueError("Test data contains an unplanned replay or a duplicate stage.")
        reservation = expected[replay]
        if (row["sha256"] != reservation["replay_sha256"] or group_hash(row) != reservation["player_group_sha256"]
                or f'TH{row["title"]}' != task["title"] or row["version"] != task["build"]
                or row["difficulty"] != task["difficulty"] or row["shot"].replace(" ", "") != task["shot"].replace(" ", "")
                or row["stride"] != 4 or row["moving_alignment"] < .95 or row["coverage"] < .9):
            raise ValueError(f"Test replay metadata does not clear the frozen gates for {replay} stage {cell['stage']}.")
        collected.add(stage_key)
        evidence.append({
            "replay": replay, "stage": cell["stage"], "pairs": cell["samples"],
            "replay_sha256": row["sha256"], "player_group_sha256": group_hash(row),
            "metadata_sha256": fingerprint(path.with_suffix(".json")), "arrays_sha256": fingerprint(path),
            "moving_alignment": row["moving_alignment"], "frame_coverage": row["coverage"],
        })
    if collected != {(name, stage) for name in expected for stage in expected_stages}:
        raise ValueError("Test data must contain all frozen replay stages.")
    return shards, evidence


def run_paths(legacy_root, expanded_root):
    return {
        "legacy_clock": {seed: legacy_root / f"overnight-title-equal-{seed}" / "sft" / "policy.pt" for seed in SEEDS},
        "expanded_clock": {seed: expanded_root / f"sft-expanded-512-{seed}" / "policy.pt" for seed in SEEDS},
        "expanded_no_clock": {seed: expanded_root / f"sft-expanded-no-clock-512-{seed}" / "policy.pt" for seed in SEEDS},
    }


def verify_runs(paths, evidence, training_snapshot):
    test_groups = {row["player_group_sha256"] for row in evidence}
    test_names = {row["replay"] for row in evidence}
    recorded = defaultdict(dict)
    validation = set()
    for arm, seeds in paths.items():
        for seed, path in seeds.items():
            sidecar = path.with_name("metrics.json")
            metrics = load_json(sidecar)
            for split in ("training_replays", "validation_replays"):
                if not metrics.get(split) or any(
                        row["replay"] in test_names or group_hash(row) in test_groups for row in metrics[split]):
                    raise ValueError(f"{arm} seed {seed} contains the test replay/player or lacks {split} evidence.")
                sources = tuple(sorted((row["replay"], row["stage"], row["samples"]) for row in metrics[split]))
                if arm != "legacy_clock" and sources != training_snapshot[split]:
                    raise ValueError(f"{arm} seed {seed} used a different expanded snapshot for {split}.")
                if split == "validation_replays":
                    validation.add(sources)
            if (metrics["config"]["seed"] != seed or metrics["hidden"] != 512 or not metrics["motion_history"]
                    or metrics.get("motion_spans") or metrics.get("retain_recent_motion")
                    or metrics["factorized_movement"] or metrics["stage_clock"] != (arm != "expanded_no_clock")):
                raise ValueError(f"{arm} seed {seed} does not match the fixed cohort configuration.")
            recorded[arm][seed] = {
                "checkpoint_sha256": fingerprint(path),
                "metrics_sha256": fingerprint(sidecar),
                "selected_validation_epoch": metrics["best"]["epoch"],
                "validation_nll": metrics["best"]["val_loss"],
                "training_decisions": sum(row["samples"] for row in metrics["training_replays"]),
                "validation_decisions": sum(row["samples"] for row in metrics["validation_replays"]),
                "validation_replays": sorted({row["replay"] for row in metrics["validation_replays"]}),
            }
    if len(validation) != 1:
        raise ValueError("The fixed cohorts have different validation replay sets.")
    return recorded


def counts(predicted, probabilities, actions, first, previous, bomb_onset):
    change = (first != 4) & (first != previous)
    cases = {
        "movement": (actions[:, 0], np.ones(len(first), dtype=bool)),
        "movement_while_moving": (actions[:, 0], actions[:, 0] != 4),
        "immediate_movement": (first, np.ones(len(first), dtype=bool)),
        "immediate_start_direction": (first, change & (previous == 4)),
        "immediate_turn_direction": (first, change & (previous != 4)),
    }
    values = {name: [int((predicted[subset] == target[subset]).sum()), int(subset.sum())]
              for name, (target, subset) in cases.items()}
    truth_bomb = actions[:, 3] == 1
    predicted_bomb = probabilities[:, 1] > probabilities[:, 0]
    tp = int((truth_bomb & predicted_bomb).sum())
    fp = int((~truth_bomb & predicted_bomb).sum())
    positives = int(truth_bomb.sum())
    values.update({
        "bomb_prevalence": [positives, len(first)],
        "bomb_precision": [tp, tp + fp],
        "bomb_recall": [tp, positives],
        "bomb_false_positive_rate": [fp, len(first) - positives],
        "bomb_observed_onset_recall": [int((predicted_bomb & bomb_onset).sum()), int(bomb_onset.sum())],
        "bomb_probability_on_positive": [float(probabilities[truth_bomb, 1].sum()), positives],
        "bomb_probability_on_negative": [float(probabilities[~truth_bomb, 1].sum()), len(first) - positives],
    })
    return values


def add_counts(rows):
    combined = defaultdict(lambda: [0, 0])
    for row in rows:
        for name, (numerator, denominator) in row.items():
            combined[name][0] += numerator
            combined[name][1] += denominator
    return dict(combined)


def stats(counts_by_name):
    return {name: {"numerator": x, "denominator": n, "value": x / n if n else None}
            for name, (x, n) in counts_by_name.items()}


@torch.inference_mode()
def measure(shards, policies):
    rows = []
    for shard in shards:
        path, meta = shard
        with np.load(path, allow_pickle=False) as source:
            first = movement_direction(source["expert_action_chunk"][:, 0])
            previous = movement_direction(source["native"])
            frames = source["stage_frame"]
        features = {clock: read_shard(shard, motion_history=True, stage_clock=clock) for clock in (False, True)}
        actions = features[False][2]
        bomb = actions[:, 3] > 0
        bomb_onset = np.concatenate(([False], (np.diff(frames) == 4) & ~bomb[:-1] & bomb[1:]))
        measurements = {}
        for arm, seeds in policies.items():
            measurements[arm] = {}
            for seed, policy in seeds.items():
                move_batches, bomb_batches = [], []
                for bullets, context, _ in batch_block(features[policy.stage_clock], 256, "cpu", np.arange(len(first))):
                    movement, _, _, bomb_logits = predict_heads(policy, bullets, context)
                    move_batches.append(movement.argmax(1).numpy())
                    bomb_batches.append(bomb_logits.softmax(dim=1).numpy())
                measurements[arm][seed] = counts(np.concatenate(move_batches), np.concatenate(bomb_batches),
                                                  actions, first, previous, bomb_onset)
        rows.append((meta["replay"], meta["stage"], measurements))
    return rows


def range_summary(values):
    available = [value for value in values if value is not None]
    return ({"seeds_with_support": len(available), "mean": mean(available), "min": min(available),
             "max": max(available), "seed_sample_sd": stdev(available) if len(available) > 1 else None}
            if available else {"seeds_with_support": 0, "mean": None, "min": None, "max": None, "seed_sample_sd": None})


def summarize(rows, policies):
    replay_names = sorted({replay for replay, _, _ in rows})
    result = {}
    for arm in policies:
        per_seed = {}
        for seed in SEEDS:
            by_replay = {}
            supported = defaultdict(list)
            for replay in replay_names:
                stages = {stage: cells[arm][seed] for name, stage, cells in rows if name == replay}
                metrics = stats(add_counts(stages.values()))
                for name, metric in metrics.items():
                    if metric["value"] is not None:
                        supported[name].append(metric["value"])
                by_replay[replay] = {"metrics": metrics,
                                     "stages": {stage: stats(stage_counts) for stage, stage_counts in stages.items()}}
            macro = {name: mean(supported[name]) if supported[name] else None
                     for name in next(iter(by_replay.values()))["metrics"]}
            per_seed[seed] = {"macro_replay_metrics": macro, "replays": by_replay}
        result[arm] = {"by_seed": per_seed, "across_seeds": {
            name: range_summary([row["macro_replay_metrics"][name] for row in per_seed.values()])
            for name in per_seed[SEEDS[0]]["macro_replay_metrics"]
        }}
    return result


def paired_deltas(summary, candidate, reference):
    metrics = summary[candidate]["by_seed"][SEEDS[0]]["macro_replay_metrics"]
    rows = {}
    for seed in SEEDS:
        left = summary[candidate]["by_seed"][seed]["macro_replay_metrics"]
        right = summary[reference]["by_seed"][seed]["macro_replay_metrics"]
        rows[seed] = {name: left[name] - right[name] if left[name] is not None and right[name] is not None else None
                      for name in metrics}
    return {"by_seed": rows, "across_seeds": {
        name: range_summary([row[name] for row in rows.values()]) for name in metrics
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--expanded-root", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, default=TH14_PLAN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh report path.")
    torch.set_num_threads(1)
    shards, evidence = test_shards(args.test_data, load_json(args.split_plan))
    paths = run_paths(args.legacy_root, args.expanded_root)
    training_manifest = args.expanded_root / "frozen-training-snapshot.manifest.json"
    entries = load_json(training_manifest)["entries"]
    artifacts = verify_runs(paths, evidence, {
        f"{split}_replays": tuple(sorted((entry["replay"], entry["stage"], entry["pairs"])
                                       for entry in entries if entry["split"] == split_name))
        for split, split_name in (("training", "train"), ("validation", "validation"))
    })
    policies = {arm: {seed: read_policy(path, "cpu", motion_history=True) for seed, path in seeds.items()}
                for arm, seeds in paths.items()}
    summary = summarize(measure(shards, policies), policies)
    report = {
        "version": 1,
        "purpose": "Final locked-test reporting. Epochs selected on validation before opening the test. No ranking or selection.",
        "scope": "Offline agreement on recorded expert trajectories. Native survival, action success and clears require native evaluation.",
        "test_design": {
            "split_plan_sha256": fingerprint(args.split_plan),
            "expanded_training_manifest_sha256": fingerprint(training_manifest),
            "player_groups": len({row["player_group_sha256"] for row in evidence}),
            "replays": len({row["replay"] for row in evidence}),
            "stages": len(evidence), "paired_decisions": sum(row["pairs"] for row in evidence),
            "aggregation": "Equal replay weights within supported metric, then equal seed weights.",
            "uncertainty": "Range and sample SD across matched training seeds are descriptive. Stage rows and decisions are correlated within a replay.",
            "player_population_interval": None,
        },
        "definitions": {
            "movement": "Greedy exact direction against the four-tick training label.",
            "immediate_events": "Starts begin from native idle; turns begin moving and change to another moving direction at the first future tick.",
            "bomb": "Positive if any of four expert input ticks sets Bomb; predictions use the greedy Bomb head.",
            "bomb_onset": "First positive window following an observed contiguous negative window in the same stage.",
            "counts": "Null rates indicate zero denominator. Button labels and onset counts do not measure actual Bomb consumption.",
        },
        "inputs": evidence, "checkpoints": artifacts, "arms": summary,
        "paired_deltas": {
            "expanded_clock_minus_legacy_clock": paired_deltas(summary, "expanded_clock", "legacy_clock"),
            "expanded_no_clock_minus_expanded_clock": paired_deltas(summary, "expanded_no_clock", "expanded_clock"),
        },
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"Wrote sealed-test report for {report['test_design']['replays']} replay(s).", flush=True)


if __name__ == "__main__":
    main()
