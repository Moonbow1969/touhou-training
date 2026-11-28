"""Fit the mixed original-start and restored-start TH08 PPO pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.th08_camera_history import frame_pair, reconstruct
from scripts.th08_fragment_controls import control_evidence
from scripts.th08_progress_reward import GAMMA, progress_reward
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import chosen, logits_on
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_th08_camera_hazard import ENCODER
from scripts.train_th08_normal_current_ppo import CurrentActorCritic, ORIGINAL_EXE, WIDTHS, optimize
from scripts.train_th08_progress_pair import KL_CAP, STAGE_TICKS, trajectory


TRAIN_SEED = 80965261
LAMBDA = .985


def reward_at_boundary(*, stages, ticks, spares, terminated, shaped_boundary_value, boundaries):
    states = np.full(len(ticks), "play", dtype="U16")
    states[-1] = "game_over" if terminated else "play"
    base_value = None if terminated else float(shaped_boundary_value) + int(spares[-1]) + 1
    reward = progress_reward(stage=stages,tick=ticks,spares=spares,state_kind=states,
        continues=np.zeros(len(ticks),dtype=np.int64),spare_delta=np.diff(np.maximum(spares,0)),
        stage_ticks=STAGE_TICKS,boundaries=boundaries,life_weight=1,base_bootstrap=base_value)
    return reward


def as_optimizer_input(frames, action, old, heads, values, reward):
    values = torch.as_tensor(values)
    advantage = calculate_gae(reward.reward.astype(np.float32),values.numpy(),GAMMA,LAMBDA,bootstrap=reward.bootstrap)
    result = torch.from_numpy(advantage)
    return frames,action,old,result,values+result,heads


def original_fragment(manifest, row, actor):
    data_path = manifest.parent / row["file"]
    receipt_path = manifest.parent / row["receipt"]
    if digest(data_path) != row["sha256"] or digest(receipt_path) != row["receipt_sha256"]:
        raise ValueError(f"Fragment file or source receipt changed for {row['seed']}")
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt["seed"] != row["seed"] or receipt["game_sha256"] != ORIGINAL_EXE or receipt["sha256"] != row["sha256"]
            or receipt["policy_sha256"] != ENCODER or receipt["qualified"] is not True or receipt["continues"] != 0
            or receipt["wholly_stale_windows"] != 0):
        raise ValueError(f"Fragment was not physically qualified: {row['seed']}")
    if receipt["outcome"] not in ("game_over","life_loss_cutoff","native_tick_cutoff"):
        raise ValueError(f"Unregistered fragment endpoint: {receipt['outcome']}")
    with np.load(data_path,allow_pickle=False) as trace:
        frames = trace["frames"]
        previous = trace["predecessor_frame"]
        action = trace["action"].astype(np.int64)
        old_logp = trace["logp"]
        values = trace["value"].reshape(-1)
        stages = trace["native_stage"]
        ticks = trace["native_tick"]
        reserves = trace["spares"]
        bombs = trace["bombs"]
        native_terminal = trace["terminated"]
        native_slots = trace["native_word_slots"]
        native_duration = trace["native_ticks_under_command"]
        later = (trace["native_next_stage"],trace["native_next_tick"],trace["next_spares"],trace["next_bombs"])
        boundary_image = None if receipt["outcome"] == "game_over" else trace["boundary_frame"]
    n = len(action)
    true_end = receipt["outcome"] == "game_over"
    if (n < 1 or n != receipt["decisions"] or receipt["action_masks_with_native_consumption"] != n
            or frames.shape != (n,112,96) or frames.dtype != np.uint8
            or previous.shape != (112,96) or previous.dtype != np.uint8
            or action.shape != (n,4) or np.any((action < 0) | (action >= np.asarray(WIDTHS)))
            or values.shape != (n,) or old_logp.shape != (n,4) or not np.isfinite(values).all() or not np.isfinite(old_logp).all()
            or native_terminal.shape != (n,) or native_terminal[:-1].any() or bool(native_terminal[-1]) != true_end
            or stages[0] != 1 or reserves[0] < 1):
        raise ValueError(f"Invalid policy history, clock, or stocks for fragment {row['seed']}")
    complete_state = []
    for before,after in zip((stages,ticks,reserves,bombs),later):
        if before.shape != (n,) or after.shape != (n,) or not np.array_equal(before[1:],after[:-1]):
            raise ValueError(f"Disconnected original fragment states for {row['seed']}")
        complete_state.append(np.r_[before,after[-1]])
    gaps = later[1]-ticks
    if np.array_equal(stages,later[0]) and not np.array_equal(native_duration,gaps):
        raise ValueError(f"Native command window does not match the accepted camera clock for fragment {row['seed']}")
    control = control_evidence(action,native_slots,native_duration,true_end)
    if not control["qualified"] or abs(control["fraction"]-receipt["exact_four_tick_fraction"]) > 1e-8:
        raise ValueError(f"Original clock or native inputs failed independent recheck for fragment {row['seed']}")
    images = reconstruct(frames,0,n,1)
    images[0] = frame_pair(frames[0],previous)
    images = torch.from_numpy(images)
    taken = torch.from_numpy(action)
    old_values = torch.from_numpy(values)
    *heads, critic = logits_on(actor,images,"cpu")
    lp_difference = float((chosen(heads,taken)-torch.from_numpy(old_logp)).abs().max())
    critic_difference = float((critic[:,0]-old_values).abs().max())
    if lp_difference > 8e-4 or critic_difference > 3e-4:
        raise ValueError(f"Physical fragment actor outputs could not be reproduced for {row['seed']}")
    shaped_boundary = 0.
    if not true_end:
        if (boundary_image.shape != (112,96) or boundary_image.dtype != np.uint8
                or receipt["boundary_camera_sha256"] != hashlib.sha256(boundary_image.tobytes()).hexdigest()):
            raise ValueError(f"Missing exact next policy camera for fragment {row['seed']}")
        next_image = torch.from_numpy(frame_pair(boundary_image,frames[-1])[None])
        with torch.no_grad():
            shaped_boundary = float(actor(next_image.float()/255)[-1][0,0])
    reward = reward_at_boundary(stages=complete_state[0],ticks=complete_state[1],spares=complete_state[2],
        terminated=true_end,shaped_boundary_value=shaped_boundary,
        boundaries={int(k):tuple(v) for k,v in receipt.get("native_stage_boundaries",{}).items()})
    original = as_optimizer_input(images,taken,torch.from_numpy(old_logp.sum(1)),torch.cat(heads,1),old_values,reward)
    return original,dict(seed=row["seed"],decisions=n,first_tick=int(ticks[0]),final_tick=int(later[1][-1]),
        outcome=receipt["outcome"],bootstrap=reward.bootstrap,max_logp_error=lp_difference,max_critic_error=critic_difference,
        control=control)


def fit_data(manifest, actor):
    data = json.loads(manifest.read_bytes())
    grouped = {"ordinary": [], "anchor": []}
    verified = []
    seeds = set()
    for row in sorted(data["rows"],key=lambda r:r["seed"]):
        kind = row["source_kind"]
        if row["role"] != "train" or row["seed"] in seeds or kind not in grouped:
            raise ValueError("The sealed optimizer manifest must contain unique training originals only")
        seeds.add(row["seed"])
        if kind == "ordinary":
            common,rewards,details = trajectory(manifest,row,actor,"cpu",actor=ENCODER,min_exact=.995,require_zero_value=False)
            packed = as_optimizer_input(*common,rewards[1])
        else:
            if row["anchor_sha256"] != data["anchor_sha256"]:
                raise ValueError("Archive snapshot identity changed across the registered anchor fragments")
            packed,details = original_fragment(manifest,row,actor)
        grouped[kind].append(packed)
        verified.append(dict(source_kind=kind,**details))
        print("Verified PPO source",verified[-1],flush=True)
    if not all(grouped.values()):
        raise ValueError("Both ordinary and restored original-game sources are required")
    n_anchor = len(grouped["anchor"])
    attempted = data["anchor_attempt_count"]
    controls = [r["control"] for r in verified if r["source_kind"]=="anchor"]
    exact_share = sum(c["exact"]for c in controls)/sum(c["windows"]for c in controls)
    if not n_anchor <= attempted <= 60 or n_anchor / attempted < .90 or exact_share < .995:
        raise ValueError("Restored original cohort failed aggregate admission")
    fields = {kind: tuple(torch.cat(items) for items in zip(*rows)) for kind,rows in grouped.items()}
    sizes = {kind: len(items[0]) for kind,items in fields.items()}
    quota = min(sizes.values())
    generator = torch.Generator().manual_seed(TRAIN_SEED)
    balanced = []
    for kind,items in fields.items():
        indices = torch.randperm(sizes[kind],generator=generator)[:quota]
        balanced.append(tuple(t[indices] for t in items))
    packed = tuple(torch.cat(parts) for parts in zip(*balanced))
    return packed,dict(sizes=sizes,sampled_per_source=quota,credits=verified,anchor_sha256=data["anchor_sha256"],
        anchor_attempt_count=attempted,anchor_qualified=n_anchor,anchor_combined_exact_fraction=exact_share)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","parent","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if digest(args.parent) != ENCODER:
        raise ValueError("The starting PPO policy differs from the registered rollout controller")
    torch.manual_seed(TRAIN_SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = CurrentActorCritic().eval()
    actor.load_state_dict(torch.jit.load(str(args.parent),map_location="cpu").state_dict())
    args.out.mkdir(parents=True,exist_ok=False)
    packed, inputs = fit_data(args.manifest,actor)
    protocol = args.out/"protocol.json"
    freeze(protocol,dict(manifest_sha256=digest(args.manifest),parent=ENCODER,seed=TRAIN_SEED,gamma=GAMMA,
          lambda_gae=LAMBDA,kl_cap=KL_CAP,inputs=inputs,
          mixture="Equal decision counts per source drawn without replacement; whole-credit GAE computed before sampling.",
          cutoff="Nonterminal truncations use the exact retained next camera and frozen original critic."))
    epochs = optimize(actor.to(device),packed,device,kl_cap=KL_CAP)
    example = packed[0][:2].float()/255
    export = args.out/"policy.ts"
    frozen = torch.jit.trace(actor.cpu().eval(),example)
    frozen.save(str(export))
    torch.save(actor.state_dict(),args.out/"policy.pt")
    freeze(args.out/"report.json",dict(protocol_sha256=digest(protocol),policy_sha256=digest(export),
           total_training_decisions=len(packed[0]),optimizer_epochs=epochs))


if __name__ == "__main__":
    main()
