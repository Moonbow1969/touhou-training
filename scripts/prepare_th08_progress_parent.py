"""Initialize the common TH08 Progress/Bomb teacher and zero critic."""

import argparse
import json
from pathlib import Path

import torch

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_th08_normal_current_ppo import CurrentActorCritic, EXPECTED_PARENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("parent", "calibration", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    calibration = json.loads(args.calibration.read_bytes())
    source = digest(args.parent)
    if (source != EXPECTED_PARENT or source != calibration["actor"] or calibration["sample_size"] != 2364 or
            abs(calibration["resulting_mean_probability"]-.01) > 1e-7):
        raise ValueError("Unexpected registered common-policy calibration")
    initial = torch.jit.load(str(args.parent), map_location="cpu").eval()
    parent = CurrentActorCritic().eval()
    parent.actor.load_state_dict(initial.state_dict())
    inputs = torch.rand((3,2,112,96), generator=torch.Generator().manual_seed(809620))
    delta = calibration["bomb_log_odds_shift"]
    with torch.no_grad():
        parent.actor.prior.bias[-1] += delta
        baseline = initial(inputs)
        output = parent(inputs)
        if not all(torch.equal(old, new) for old, new in zip(baseline[:3], output[:3])):
            raise ValueError("Expected identical non-Bomb action logits")
        torch.testing.assert_close(output[3] - baseline[3], torch.tensor([0., delta]).expand(3,2))
        assert output[4].count_nonzero() == 0
        args.out.mkdir(parents=True, exist_ok=False)
        path = args.out / "policy.ts"
        torch.jit.trace(parent, inputs).save(str(path))
        loaded = torch.jit.load(str(path), map_location="cpu").eval()
        assert all(torch.equal(x,y) for x,y in zip(parent(inputs[:1]), loaded(inputs[:1])))
    torch.save(parent.state_dict(), args.out / "policy.pt")
    freeze(args.out / "protocol.json", dict(original_sft=source, parent=digest(path), bias=delta,
           calibration=digest(args.calibration), same_initial_policy_for_both_arms=True,
           calibration_mean_before=calibration["current_sampled_mean_probability"],
           calibration_mean_after=calibration["resulting_mean_probability"], critic="zero"))


if __name__ == "__main__":
    main()
