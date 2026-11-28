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
TECHNICAL_EXCLUSIONS = (80963021, 80963022, 80963023)


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
    parser.add_argument("--accept-sealed-29", action="store_true",
                        help="Apply the reviewed amendment using the sealed 32-credit diagnostic.")
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
    exclusions = ()
    if args.accept_sealed_29:
        exclusions = tuple(row["seed"] for row in rows if row["qualified"] is not True)
        if exclusions != TECHNICAL_EXCLUSIONS or data["physical_originals"] != 32 or data["qualified"] != 29:
            raise ValueError("Diagnostic disagrees with the sealed 29-credit amendment")
        rows = [row for row in rows if row["qualified"] is True]
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
    if args.accept_sealed_29 and sum(entry[2]["decisions"] for entry in sources) != 25627:
        raise ValueError("Decision count disagrees with the reviewed 29-credit cohort")
    args.out.mkdir(parents=True,exist_ok=False)
    freeze(args.out/"protocol.json",dict(parent=PARENT, source=digest(args.manifest), intake=digest(args.intake),
           gamma=GAMMA, lambda_gae=.985, stage_ticks=STAGE_TICKS, life_weight=1, kl_cap=KL_CAP,
           registered_seeds=SEEDS, seeds=[row["seed"] for row in rows], excluded_seeds=exclusions,
           maximum_iterations=1, credits=[entry[2] for entry in sources]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit(args,state,sources,"progress-plus-spares",1,device)


if __name__ == "__main__":
    main()
