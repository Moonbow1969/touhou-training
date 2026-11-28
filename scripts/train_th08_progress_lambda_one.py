"""Run one bounded on-policy PPO iteration for the frozen TH08 spare-potential reward."""

import argparse
import json
from pathlib import Path

import torch

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_th08_progress_pair import CurrentActorCritic, GAMMA, KL_CAP, STAGE_TICKS, fit, trajectory


PARENT = "69c44e2ec484e5c5a915722fb1259a174da67b375cd1d4e786c54bf2589d3089"
SEEDS = tuple(range(80963001,80963033))


def admission(path):
    guard = json.loads(path.read_bytes())
    rows = sorted(guard["rows"], key=lambda r:r["seed"])
    if (guard["passed"] is not True or tuple(r["seed"] for r in rows) != SEEDS[:8] or
            any(r["exact_four_tick_fraction"] < .995 or r["max_observer_rtt_ms"] > 100 or
                r["every_retained_mask_consumed"] is not True or r["sealed_original_video"] is not True for r in rows)):
        raise ValueError("The first eight strict original qualification credits did not all pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "intake", "behavior", "state", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    admission(args.intake)
    if digest(args.behavior) != PARENT:
        raise ValueError("Expected the sealed first λ1 native PPO checkpoint")
    data = json.loads(args.manifest.read_bytes())
    rows = sorted(data["rows"], key=lambda r:r["seed"])
    if tuple(r["seed"] for r in rows) != SEEDS:
        raise ValueError("Required all 32 fresh registered original credits")
    native = torch.jit.load(str(args.behavior), map_location="cpu").eval()
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    binding = native.state_dict()
    if binding.keys() != state.keys() or any(not torch.equal(v,binding[k]) for k,v in state.items()):
        raise ValueError("Native TorchScript does not match the learner weights")
    model = CurrentActorCritic().eval()
    model.load_state_dict(state)
    sources = []
    for row in rows:
        record = trajectory(args.manifest,row,model,"cpu",actor=PARENT,min_exact=.995,require_zero_value=False)
        sources.append(record)
        print("Verified",record[2],flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    freeze(args.out/"protocol.json",dict(parent=PARENT, source=digest(args.manifest), intake=digest(args.intake),
           gamma=GAMMA, lambda_gae=.985, stage_ticks=STAGE_TICKS, life_weight=1, kl_cap=KL_CAP,
           seeds=SEEDS, maximum_iterations=1, credits=[entry[2] for entry in sources]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit(args,state,sources,"progress-plus-spares",1,device)


if __name__ == "__main__":
    main()
