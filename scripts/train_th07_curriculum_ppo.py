"""Fit a camera-only TH07 retail PPO actor on two signed original-state sources."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.check_five_title_critic_returns import ridge
from scripts.checkpoint_storage import digest
from scripts.th08_camera_history import frame_pair, reconstruct
from scripts.th08_progress_reward import GAMMA, progress_reward
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import CameraActorCritic, CameraPolicy, chosen, logits_on
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_th08_normal_current_ppo import WIDTHS, optimize


ACTOR = "a6e1dec1625af50e214f141a4e405795fbdae905d78c4b887e423de0bca6a77e"
ORIGINAL = "35467eaf8dc7fc85f024f16fb2037255f151cefda33cf4867bc9122aaa2e80ca"
SEED = 2174951
STAGE_LENGTH = {1: 10694, 2: 15128}
FOLDS = 4
CRITIC_REFITS = 2
LAMBDA = .985
KL_CAP = .01
PRACTICE_SCENES = ("letty_full", "letty_low", "cirno", "letty_final")
SCENES = ("easy", "normal", *PRACTICE_SCENES)


def frozen_features(model, frames):
    with torch.inference_mode():
        parts = [model.camera.policy[:-1](model.camera.spatial(block.float() / 255))
                 for block in frames.split(96)]
        latent = torch.cat(parts)
        heads = model.camera.policy[-1](latent).split(WIDTHS, 1)
        value = model.value(latent)[:, 0]
    return latent, heads, value


def returns(reward, bootstrap):
    return calculate_gae(reward, np.zeros(len(reward), dtype=np.float32), GAMMA, 1., bootstrap=bootstrap)


def load_episode(root, entry, actor):
    path = root / entry["file"]
    receipt_path = root / entry["receipt"]
    if digest(path) != entry["sha256"] or digest(receipt_path) != entry["receipt_sha256"]:
        raise ValueError(f"Signed training source changed: {path}")
    receipt = json.loads(receipt_path.read_bytes())
    scene = entry["scene"]
    outcome = receipt["outcome"]
    practice = scene in PRACTICE_SCENES
    source_kind = "practice" if practice else "game_start"
    if (entry["role"] != "train" or receipt["role"] != "train" or receipt["qualified"] is not True
            or receipt["policy_sha256"] != ACTOR or receipt["game_sha256"] != ORIGINAL
            or receipt["seed"] != entry["seed"] or receipt["continues"] != 0
            or receipt["sha256"] != entry["sha256"] or scene not in SCENES
            or receipt["source_kind"] != source_kind or receipt["native_rank"] != (0 if scene == "easy" else 1)):
        raise ValueError(f"Ineligible original policy source: {path}")
    endpoints = ("game_over", "practice_stage_clear", "native_tick_cutoff") if practice else ("game_over", "stage2_entry")
    if outcome not in endpoints:
        raise ValueError(f"Unregistered original endpoint: {path}")
    if scene == "letty_final" and receipt.get("scenario") != "letty-final-2-3":
        raise ValueError(f"Late practice window lacks the qualified final-spell entrance: {path}")
    native_terminal = outcome in ("game_over", "practice_stage_clear")
    success = outcome in ("stage2_entry", "practice_stage_clear")
    timeout = outcome == "native_tick_cutoff"
    with np.load(path, allow_pickle=False) as data:
        images = data["frames"]
        predecessor = data["predecessor_frame"]
        actions = data["action"].astype(np.int64)
        old_logp = data["logp"].astype(np.float32)
        archived_value = data["value"].astype(np.float32).reshape(-1)
        states = [data[name] for name in ("native_stage", "native_tick", "spares", "bombs")]
        next_states = [data[name] for name in ("native_next_stage", "native_next_tick", "next_spares", "next_bombs")]
        done = data["terminated"]
        eligible = data.get("native_action_eligible", np.ones(len(actions), dtype=bool))
        boundary_frame = data.get("boundary_frame")
        boundary_value = data.get("boundary_value")
    n = len(actions)
    if (n < 2 or n != receipt["decisions"] or images.shape != (n, 112, 96) or images.dtype != np.uint8
            or predecessor.shape != (112, 96) or predecessor.dtype != np.uint8
            or actions.shape != (n, 4) or np.any((actions < 0) | (actions >= WIDTHS))
            or old_logp.shape != (n, 4) or not np.isfinite(old_logp).all()
            or archived_value.shape != (n,) or not np.isfinite(archived_value).all()
            or done.shape != (n,) or done.dtype != np.bool_ or done[:-1].any()
            or bool(done[-1]) != native_terminal
            or eligible.shape != (n,) or eligible.dtype != np.bool_ or not eligible.any()):
        raise ValueError(f"Invalid policy camera, actions, values or terminal mask: {path}")
    ledger = []
    for old, new in zip(states, next_states, strict=True):
        if old.shape != (n,) or new.shape != (n,) or not np.array_equal(old[1:], new[:-1]):
            raise ValueError(f"Discontinuous original clock or inventory: {path}")
        ledger.append(np.r_[old, new[-1]])
    stages, ticks, spares, bombs = ledger
    first_stock = (0, 1) if scene == "letty_low" else (2, 3)
    if (stages[0] != 1 or tuple((spares[0], bombs[0])) != first_stock
            or np.any(stages[:-1] != 1) or (stages[-1] == 2) != (outcome == "stage2_entry")):
        raise ValueError(f"Original stage or starting inventory mismatch: {path}")
    frames = reconstruct(images, 0, n, 1)
    frames[0] = frame_pair(images[0], predecessor)
    frames = torch.from_numpy(frames)
    act = torch.from_numpy(actions)
    latent, heads, value = frozen_features(actor, frames)
    lp_error = float((chosen(heads, act) - torch.from_numpy(old_logp)).abs().max())
    value_error = float((value - torch.from_numpy(archived_value)).abs().max())
    if lp_error > 8e-4 or value_error > 3e-4:
        raise ValueError(f"Behavior log probabilities or values do not replay: {path}")
    bootstrap = None
    boundary = None
    if not native_terminal:
        if boundary_frame is None or boundary_frame.shape != (112, 96) or boundary_frame.dtype != np.uint8:
            raise ValueError(f"Playable cutoff lacks its real next camera: {path}")
        observation = torch.from_numpy(frame_pair(boundary_frame, images[-1])[None])
        next_latent, _, native_value = frozen_features(actor, observation)
        if boundary_value is None or not np.isfinite(boundary_value).all() or abs(float(native_value.item()) - float(boundary_value.item())) > 3e-4:
            raise ValueError(f"Playable cutoff lacks its verified next-camera value: {path}")
        if timeout:
            bootstrap = float(native_value.item())
            boundary = next_latent.numpy().astype(np.float64)
    state = np.full(n + 1, "play", dtype="U20")
    if not timeout:
        state[-1] = "objective_terminal" if success else "game_over"
    reward = progress_reward(stage=stages, tick=ticks, spares=spares,
        continues=np.zeros(n + 1, dtype=np.int64), state_kind=state,
        spare_delta=np.diff(np.maximum(spares, 0)), stage_ticks=STAGE_LENGTH, life_weight=1,
        boundaries={int(k): tuple(v) for k, v in receipt["native_stage_boundaries"].items()},
        base_bootstrap=bootstrap)
    shaped = reward.reward.astype(np.float32)
    if outcome == "practice_stage_clear":
        shaped[-1] += 1.
    if success:
        shaped[-1] += GAMMA * (int(spares[-1]) + 1)
    target = returns(shaped, reward.bootstrap)
    scene_bits = np.eye(len(SCENES))[SCENES.index(scene)]
    privileged = np.tile(scene_bits, (n, 1))
    vision = latent.numpy().astype(np.float64)
    conditioned = np.column_stack((vision, privileged, spares[:-1], bombs[:-1], ticks[:-1] / STAGE_LENGTH[1]))
    if boundary is not None:
        conditioned_boundary = np.column_stack((boundary, scene_bits[None], [spares[-1]], [bombs[-1]],
                                                [ticks[-1] / STAGE_LENGTH[1]]))
        boundary = dict(vision=boundary, conditioned=conditioned_boundary)
    return dict(seed=entry["seed"], scene=scene, vision=vision, conditioned=conditioned, y=target,
        reward=shaped, boundary=boundary, active=torch.from_numpy(eligible),
        tensors=(frames, act, torch.from_numpy(old_logp.sum(1)), torch.cat(heads, dim=1)),
        audit=dict(seed=entry["seed"], scene=scene, outcome=outcome, decisions=n,
                   eligible=int(eligible.sum()), lp_error=lp_error, value_error=value_error,
                   initial_critic_bootstrap=reward.bootstrap, native_stage_final=int(stages[-1]),
                   final_spares=int(spares[-1]), final_bombs=int(bombs[-1]),
                   reward_sum=float(shaped.sum())))


def fit_value_fold(train, held, kind):
    # Initialize censored returns with the archived critic, then refit on this fold's originals.
    train = [row.copy() for row in train]
    censored = [row for row in train if row["boundary"] is not None]
    if censored:
        boundaries = [row["boundary"] for row in censored]
        for _ in range(CRITIC_REFITS):
            for row, prediction in zip(censored, ridge(train, kind, boundaries), strict=True):
                row["y"] = returns(row["reward"], float(prediction.item()))
    censored_held = [row for row in held if row["boundary"] is not None]
    predictions = ridge(train, kind, held + [row["boundary"] for row in censored_held])
    for row, value in zip(held, predictions[:len(held)], strict=True):
        row[kind + "_value"] = value.astype(np.float32)
        row[kind + "_bootstrap"] = 0.
    for row, value in zip(censored_held, predictions[len(held):], strict=True):
        row[kind + "_bootstrap"] = float(value.item())


def crossfit(episodes):
    by_scene = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scene"]].append(episode)
    rng = np.random.default_rng(SEED)
    fold_rows = [[] for _ in range(FOLDS)]
    for scene, rows in sorted(by_scene.items()):
        if len(rows) < 2:
            raise ValueError(f"Too few independent originals for held-train comparison in {scene}")
        for index, j in enumerate(rng.permutation(len(rows))):
            fold_rows[index % FOLDS].append(rows[j])
    metrics = []
    for index, held in enumerate(fold_rows):
        if not held:
            continue
        train = [row for j, fold in enumerate(fold_rows) if j != index for row in fold]
        for kind in ("vision", "conditioned"):
            fit_value_fold(train, held, kind)
        metrics.append(dict(train=[r["seed"] for r in train], held=[r["seed"] for r in held]))
    rmse = {}
    scene_rmse = {scene: {} for scene in by_scene}
    complete = [r for r in episodes if r["boundary"] is None]
    for kind in ("vision", "conditioned"):
        errors = [(r[kind + "_value"] - r["y"])[r["active"].numpy()] for r in complete]
        rmse[kind] = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        for scene in by_scene:
            grouped = [error for row, error in zip(complete, errors, strict=True) if row["scene"] == scene]
            scene_rmse[scene][kind] = float(np.sqrt(np.mean(np.concatenate(grouped) ** 2))) if grouped else None
    selected = min(rmse, key=rmse.get)
    for episode in episodes:
        value = episode[selected + "_value"]
        bootstrap = episode[selected + "_bootstrap"]
        advantage = calculate_gae(episode["reward"], value, GAMMA, LAMBDA, bootstrap=bootstrap)
        episode["audit"]["ppo_bootstrap"] = bootstrap
        obs, action, old, heads = episode["tensors"]
        episode["packed"] = (obs, action, old, torch.from_numpy(advantage), torch.from_numpy(value + advantage), heads)
    return dict(rmse=rmse, by_scene=scene_rmse, selected=selected, folds=metrics,
                complete_credits_for_rmse=len(complete), timeout_critic_refits=CRITIC_REFITS)


def equal_sources(episodes, mode):
    layouts = {"easy-normal": ("easy", "normal"), "normal-practice": ("normal", "practice"),
               "last-spell": ("letty_final",)}
    sources = defaultdict(list)
    for episode in episodes:
        scene = episode["scene"]
        group = "practice" if scene in ("letty_full", "letty_low", "cirno") else scene
        sources[group].append(tuple(field[episode["active"]] for field in episode["packed"]))
    if set(sources) != set(layouts[mode]):
        raise ValueError("The inputs must match the registered sources before fitting")
    if mode == "last-spell" and not any(e["audit"]["outcome"] == "practice_stage_clear" for e in episodes):
        raise ValueError("The final-spell fit requires a witnessed section completion")
    merged = {key: tuple(torch.cat(part) for part in zip(*rows, strict=True)) for key, rows in sources.items()}
    available = {key: len(value[0]) for key, value in merged.items()}
    quota = min(available.values())
    generator = torch.Generator().manual_seed(SEED)
    balanced = []
    for key in layouts[mode]:
        index = torch.randperm(available[key], generator=generator)[:quota]
        balanced.append(tuple(field[index] for field in merged[key]))
    return tuple(torch.cat(field) for field in zip(*balanced, strict=True)), available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if digest(args.parent) != ACTOR:
        raise ValueError("Expected the registered original TH07 parent")
    frozen = torch.jit.load(str(args.parent), map_location="cpu").eval()
    actor = CameraActorCritic(CameraPolicy()).eval()
    actor.load_state_dict(frozen.state_dict())
    probe = torch.rand((2, 2, 112, 96))
    with torch.inference_mode():
        if not all(torch.equal(a, b) for a, b in zip(actor(probe), frozen(probe), strict=True)):
            raise ValueError("Learner model differs from its archived actor")
    manifest = json.loads(args.manifest.read_bytes())
    rows = sorted(manifest["rows"], key=lambda entry: entry["seed"])
    if (len({r["seed"] for r in rows}) != len(rows)
            or any(r["seed"] in manifest["held_seeds"] for r in rows)):
        raise ValueError("Duplicated or reserved original seeds in optimizer input")
    episodes = []
    for row in rows:
        episode = load_episode(args.manifest.parent, row, actor)
        episodes.append(episode)
        print("Admitted original", episode["audit"], flush=True)
    value_review = crossfit(episodes)
    packed, available = equal_sources(episodes, manifest["mode"])
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = args.out / "protocol.json"
    freeze(protocol, dict(manifest_sha256=digest(args.manifest), actor=ACTOR, mode=manifest["mode"],
        learner_sha256=digest(Path(__file__)), seed=SEED, stage_ticks=STAGE_LENGTH, gamma=GAMMA,
        lambda_gae=LAMBDA, joint_kl_cap=KL_CAP, value_diagnostic=value_review,
        available_decisions=available, used_decisions=len(packed[0]), credits=[e["audit"] for e in episodes]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = optimize(actor.to(device), packed, device, kl_cap=KL_CAP)
    *fitted_logits, _ = logits_on(actor, packed[0], device)
    action_diagnostic = {}
    for label, width, old, new in zip(("move", "focus", "shoot", "bomb"), WIDTHS,
                                     packed[-1].split(WIDTHS, dim=1), fitted_logits, strict=True):
        old_prob, new_prob = old.softmax(1), new.softmax(1)
        action_diagnostic[label] = {
            "joint_kl_component": float((old_prob * (old.log_softmax(1) - new.log_softmax(1))).sum(1).mean()),
            "parent_mean_probability": old_prob.mean(0).tolist(),
            "fitted_mean_probability": new_prob.mean(0).tolist(),
        }
        assert len(action_diagnostic[label]["fitted_mean_probability"]) == width
    actor = actor.cpu().eval()
    export = args.out / "policy.ts"
    example = packed[0][:2].float() / 255
    with torch.inference_mode():
        torch.jit.trace(actor, example).save(str(export))
        reloaded = torch.jit.load(str(export), map_location="cpu")
        if not all(torch.equal(a, b) for a, b in zip(actor(example), reloaded(example), strict=True)):
            raise ValueError("Archived policy camera outputs changed on reload")
    torch.save(actor.state_dict(), args.out / "policy.pt")
    freeze(args.out / "report.json", dict(protocol_sha256=digest(protocol), policy_sha256=digest(export),
        device=device, optimizer_epochs=epochs, training_camera_action_diagnostic=action_diagnostic,
        accepted_updates=sum(e["optimizer_steps"] for e in epochs if e["accepted"])))


if __name__ == "__main__":
    main()
