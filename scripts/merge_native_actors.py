"""Interpolate two native PPO checkpoints that share a verified SFT ancestor."""

import argparse
import json
from pathlib import Path

import torch

from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import load_arm, load_ppo_arm
from scripts.train_full_policy import publish_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--weight-second", type=float, default=.5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.weight_second <= 1:
        parser.error("Interpolation weight must be between zero and one.")
    torch.set_num_threads(1)
    template = load_arm(args.template)
    parents = [load_ppo_arm(path, template) for path in (args.first, args.second)]
    left, right = (arm.policy.state_dict() for arm in parents)
    if left.keys() != right.keys():
        raise ValueError("The parent state dictionaries differ.")
    weights = {}
    for key, value in left.items():
        other = right[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"Incompatible tensor {key}.")
        if value.is_floating_point():
            weights[key] = torch.lerp(value, other, args.weight_second)
            if not torch.isfinite(weights[key]).all():
                raise ValueError(f"Nonfinite merged tensor {key}.")
        elif torch.equal(value, other):
            weights[key] = value
        else:
            raise ValueError(f"Discrete states differ for {key}.")
    template.policy.load_state_dict(weights, strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.out / "policy.pt"
    torch.save(weights, checkpoint)
    metadata = {key: template.provenance[key] for key in (
        "hidden", "encoder", "motion_history", "retain_recent_motion", "stage_clock", "motion_spans",
        "factorized_movement")}
    metadata.update(method="PPO checkpoint weight interpolation", weight_second=args.weight_second,
                    parent_sha256=[p.provenance["weights_sha256"] for p in parents],
                    ancestor_sha256=template.provenance["weights_sha256"], weights_sha256=digest(checkpoint),
                    new_native_training_steps=0)
    publish_json(args.out / "metrics.json", metadata)
    restored = load_arm(checkpoint)
    for key, value in restored.policy.state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
