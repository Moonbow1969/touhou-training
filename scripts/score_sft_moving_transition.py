"""Score the matched TH14 development replay in the movement-weight ablation."""

import argparse
import json
from hashlib import file_digest
from pathlib import Path
from statistics import mean

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


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


@torch.inference_mode()
def score(checkpoint, features, slices):
    policy = read_policy(checkpoint, "cpu", motion_history=True)
    movement = np.concatenate([
        predict_heads(policy, bullets, context)[0].argmax(dim=-1).numpy()
        for bullets, context, _ in batch_block(features, 256, "cpu", np.arange(len(features[2])))
    ])
    return {
        name: {
            "numerator": int((movement[select] == labels[select]).sum()),
            "denominator": int(select.sum()),
            "accuracy": float((movement[select] == labels[select]).mean()),
        }
        for name, (select, labels) in slices.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)

    shards = [
        shard for shard in index_split(args.root / "data", "validation")
        if shard[1]["title"] == 14 and shard[1]["stage"] == 1
    ]
    if len(shards) != 1:
        raise ValueError("The fixed development split must contain one TH14 Stage 1 replay.")
    path, metadata = shards[0]
    with np.load(path, allow_pickle=False) as arrays:
        previous = movement_direction(arrays["native"])
        first = movement_direction(arrays["expert_action_chunk"][:, 0])
    features = read_shard(shards[0], motion_history=True, stage_clock=True)
    expert_keeps_direction = previous == first
    target = features[2][:, 0]
    slices = {
        "four_tick_movement": (np.ones(len(first), dtype=bool), target),
        "immediate_starts": ((previous == 4) & (first != 4), first),
        "immediate_turns": ((previous != 4) & (first != 4) & ~expert_keeps_direction, first),
        "immediate_idle_holds": (expert_keeps_direction & (first == 4), first),
        "immediate_active_holds": (expert_keeps_direction & (first != 4), first),
    }

    arms = {}
    for weight in (1, 4):
        by_seed = {}
        for seed in SEEDS:
            name = f"width-512-{seed}" if weight == 1 else f"transition-4-512-{seed}"
            base = args.root / "runs" / name
            sidecar = base / "metrics.json"
            checkpoint = base / "policy.pt"
            metrics = json.loads(sidecar.read_text(encoding="utf-8"))
            if (metrics["moving_transition_weight"] != weight or metrics["config"]["seed"] != seed
                    or metrics["hidden"] != 512 or not metrics["stage_clock"]
                    or sum(metrics["training_samples_by_title"].values()) != 232697):
                raise ValueError("A run differs from the fixed comparison.")
            best = metrics["best"]
            by_seed[seed] = {
                "checkpoint_sha256": digest(checkpoint), "metrics_sha256": digest(sidecar),
                "selected_epoch": best["epoch"], "validation_nll": best["val_loss"],
                "stage1": score(checkpoint, features, slices),
                "epoch_trace": metrics["epochs"] if weight == 4 else None,
            }
        arms[weight] = {
            "mean_validation_nll": mean(row["validation_nll"] for row in by_seed.values()),
            "mean_stage1_accuracy": {
                name: mean(row["stage1"][name]["accuracy"] for row in by_seed.values()) for name in slices
            },
            "by_seed": by_seed,
        }

    report = {
        "protocol": {
            "snapshot_sha256": digest(args.root / "frozen-training-snapshot.manifest.json"),
            "development_replay": metadata["replay"],
            "development_pairs": len(first),
            "development_arrays_sha256": digest(path),
            "seeds": SEEDS,
            "selection": "Lowest shared validation NLL. Report diagnostic slices from the same development split.",
            "weighted_examples": "Four-tick modal moving direction differs from prior measured displacement direction.",
            "immediate_labels": "Previous native input and first upcoming tick of the aligned human action chunk.",
            "independent_test_used": False,
            "native_gameplay_measured": False,
        },
        "weights": arms,
        "paired_weight_4_minus_1": {
            seed: {
                name: arms[4]["by_seed"][seed]["stage1"][name]["accuracy"]
                - arms[1]["by_seed"][seed]["stage1"][name]["accuracy"] for name in slices
            } for seed in SEEDS
        },
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    for weight, row in arms.items():
        print(weight, row["mean_validation_nll"], row["mean_stage1_accuracy"])


if __name__ == "__main__":
    main()
