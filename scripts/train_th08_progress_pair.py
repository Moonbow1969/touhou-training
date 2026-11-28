"""Make the first matched TH08 Progress PPO update on one shared on-policy cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.th08_camera_history import reconstruct
from scripts.th08_progress_reward import GAMMA, progress_reward
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import chosen, logits_on
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_th08_normal_current_ppo import CurrentActorCritic, WIDTHS, ORIGINAL_EXE, optimize


PAIR_SEEDS = tuple(range(80962001, 80962017))
EXPECTED_ACTOR = "ef5f34532a1c986f10088ee202ada14a9a8d2083499648772233d0218cca42a9"
STAGE_TICKS = {1: 15652, 2: 18624}
KL_CAP = .01


def trajectory(manifest, row, model, device):
    path = manifest.parent / row["file"]
    evidence_path = manifest.parent / row["receipt"]
    if (row["role"] != "train" or row["policy_sha256"] != EXPECTED_ACTOR or
            row["rank"] != "Normal" or row["team"] != "Ghost" or
            digest(path) != row["sha256"] or digest(evidence_path) != row["receipt_sha256"]):
        raise ValueError(f"Training provenance changed for {row['seed']}")
    receipt = json.loads(evidence_path.read_bytes())
    if (receipt["seed"] != row["seed"] or receipt["game_sha256"] != ORIGINAL_EXE or
            receipt["policy_sha256"] != EXPECTED_ACTOR or receipt["sha256"] != row["sha256"] or
            receipt["outcome"] != "game_over" or receipt["exact_four_tick_fraction"] < .99 or
            receipt["action_masks_with_native_consumption"] != receipt["decisions"] or receipt["continues"] != 0 or
            receipt["normal_rank_samples"] < .97*receipt["decisions"] or receipt["qualified"] is not True):
        raise ValueError(f"Original lifecycle evidence failed for {row['seed']}")
    with np.load(path, allow_pickle=False) as data:
        observations = data["frames"]
        action = data["action"].astype(np.int64)
        old_logp, value, done = data["logp"], data["value"].reshape(-1), data["terminated"]
        stage, tick, spares, bombs = (data[key] for key in ("native_stage", "native_tick", "spares", "bombs"))
        next_stage, next_tick, next_spares, next_bombs = (data[key] for key in ("native_next_stage", "native_next_tick", "next_spares", "next_bombs"))
        fatalities, casts = data["life_loss"], data["bomb_spent"]
    n = len(action)
    if (n < 2 or n != receipt["decisions"] or observations.dtype != np.uint8 or observations.shape != (n,112,96) or
            action.shape != (n,4) or np.any((action < 0) | (action >= np.asarray(WIDTHS))) or
            old_logp.shape != (n,4) or not np.isfinite(old_logp).all() or value.shape != (n,) or
            done.shape != (n,) or done.dtype != np.bool_ or done[:-1].any() or not done[-1] or np.any(value != 0)):
        raise ValueError(f"Invalid camera, action, critic, or endpoint in {path.name}")
    for prior, later in ((stage, next_stage), (tick, next_tick), (spares, next_spares), (bombs, next_bombs)):
        if prior.shape != (n,) or later.shape != (n,) or not np.array_equal(prior[1:], later[:-1]):
            raise ValueError(f"Discontinuous original gameplay state for {path.name}")
    if (stage[0] != 1 or spares[0] != 2 or bombs[0] != 3 or next_spares[-1] not in (0,-1) or casts.shape != (n,) or
            fatalities.shape != (n,) or np.any((casts < 0) | (casts > np.maximum(bombs-next_bombs,0))) or
            np.any((fatalities < 0) | (fatalities > 1))):
        raise ValueError(f"Changed original stock or event ledger for {path.name}")
    spare_delta = np.maximum(next_spares,0)-np.maximum(spares,0)
    verified_losses = (spare_delta < 0) | done
    if not np.array_equal(verified_losses, fatalities.astype(bool)):
        raise ValueError(f"Fatality ledger differs from original spare stock: {path.name}")
    stages = np.r_[stage, next_stage[-1]]
    ticks = np.r_[tick, next_tick[-1]]
    stocks = np.r_[spares, next_spares[-1]]
    state = np.full(n+1, "play", dtype="U16")
    state[-1] = "game_over"
    args = dict(stage=stages, tick=ticks, spares=stocks, state_kind=state, continues=np.zeros(n+1, dtype=int),
                spare_delta=spare_delta, stage_ticks=STAGE_TICKS,
                boundaries={int(k):tuple(v) for k,v in receipt.get("native_stage_boundaries",{}).items()})
    rewards = [progress_reward(**args, life_weight=k) for k in (0, 1)]
    weights = np.power(GAMMA, np.arange(n))
    if abs(float(weights @ (rewards[1].reward-rewards[0].reward)) + spares[0]+1) > 1e-7:
        raise ValueError("Potential shaping did not telescope to the known starting stock")
    frames = torch.from_numpy(reconstruct(observations, 0, n, 1))
    actions = torch.from_numpy(action)
    *heads, critic = logits_on(model, frames, device)
    error = float((chosen(heads, actions)-torch.from_numpy(old_logp)).abs().max())
    if error > 8e-4 or critic.count_nonzero():
        raise ValueError(f"New PPO data are not from the frozen behavior actor {path.name}")
    common = (frames, actions, torch.from_numpy(old_logp.sum(1)), torch.cat(heads, dim=1))
    return common, rewards, dict(seed=row["seed"], decisions=n, last_stage=int(next_stage[-1]),
                                 confirmed_bomb_stock_events=int(np.count_nonzero(casts)), bombs_spent=int(casts.sum()),
                                 confirmed_deaths=int(fatalities.sum()),
                                 exact_four=receipt["exact_four_tick_fraction"],
                                 max_logp_error=error, summed_r_base=float(rewards[0].reward.sum()),
                                 summed_r_shaped=float(rewards[1].reward.sum()))


def arm_data(sources, weight):
    rows = []
    for sample, rewards, _ in sources:
        frames, actions, old, heads = sample
        observed = rewards[weight]
        adv = calculate_gae(observed.reward.astype(np.float32), np.zeros(len(actions),dtype=np.float32), GAMMA, .985)
        rows.append((frames, actions, old, torch.from_numpy(adv), torch.from_numpy(adv.copy()), heads))
    return tuple(torch.cat(field) for field in zip(*rows))


def fit(args, state, sources, arm, weight, device):
    torch.manual_seed(80962099)
    model = CurrentActorCritic().to(device)
    model.load_state_dict(state)
    packed = arm_data(sources, weight)
    epochs = optimize(model, packed, device, kl_cap=KL_CAP)
    out = args.out / arm
    out.mkdir()
    traced = torch.jit.trace(model.cpu().eval(), packed[0][:2].float()/255)
    actor = out / "policy.ts"
    traced.save(str(actor))
    torch.save(model.state_dict(), out / "policy.pt")
    freeze(out / "report.json", dict(arm=arm, life_weight=weight, optimizer_epochs=epochs,
           total_training_decisions=len(packed[0]), policy=digest(actor), optimizer_seed=80962099))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "behavior", "state", "out"):
        parser.add_argument("--"+arg, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if digest(args.behavior) != EXPECTED_ACTOR:
        raise ValueError("Expected the frozen 1%-calibrated parent policy")
    manifest = json.loads(args.manifest.read_bytes())
    rows = manifest["rows"] if isinstance(manifest, dict) else manifest
    by_seed = sorted(rows, key=lambda row: row["seed"])
    if tuple(row["seed"] for row in by_seed) != PAIR_SEEDS:
        raise ValueError("Expected all 16 preassigned independent native TRAIN seeds")
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    model = CurrentActorCritic().to(device).eval()
    model.load_state_dict(state)
    anchor = torch.jit.load(str(args.behavior), map_location=device).eval()
    test = torch.rand(2,2,112,96,device=device)
    with torch.no_grad():
        if not all(torch.equal(a,b) for a,b in zip(model(test),anchor(test))):
            raise ValueError("Actor state and original behavior TorchScript differ")
    sources = [trajectory(args.manifest, row, model, device) for row in by_seed]
    del model, anchor
    args.out.mkdir(parents=True,exist_ok=False)
    freeze(args.out / "protocol.json", dict(source=digest(args.manifest), parent=EXPECTED_ACTOR,
           stage_ticks=STAGE_TICKS, gamma=GAMMA, lambda_gae=.985, kl_cap=KL_CAP, seeds=PAIR_SEEDS,
           credits=[row[2] for row in sources], design="Same behavior, same minibatches, same progress and Bomb initialization; spare potential lambda_L=0 versus 1."))
    for name, weight in (("progress",0), ("progress-plus-spares",1)):
        fit(args, state, sources, name, weight, device)


if __name__ == "__main__":
    main()
