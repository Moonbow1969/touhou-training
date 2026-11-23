import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
import torch
from torch.nn import functional

from scripts.replay_risk import replay_risk_terms
from scripts.train_spell_qr import AttentiveThreatEncoder, ThreatEncoder


ACTION_WIDTHS = (9, 2, 2, 2)
ACTION_NAMES = ("movement", "focus", "shoot", "bomb")
LEVELS = ("Easy", "Normal", "Hard", "Lunatic", "Extra")
CHARACTERS = ("Reimu", "Marisa", "Sakuya", "Sanae", "Youmu", "Cirno", "Aya")
CONFIGS = ("A", "B", "Spring", "Summer", "Autumn", "Winter", "Wolf", "Otter", "Eagle")
CONTEXT_SIZE = 10 + 1 + len(CHARACTERS) + len(CONFIGS)
MOTION_INTERVALS = 3
STAGE_CONTEXT_SIZE = 3


def network_kwargs(hidden, encoder="pool"):
    return {"features_extractor_class": AttentiveThreatEncoder if encoder == "attention" else ThreatEncoder,
            "net_arch": {"pi": [hidden, hidden], "vf": [hidden, hidden]}}


def observation_space(motion_history=False, retain_recent_motion=False, stage_clock=False):
    context_size = CONTEXT_SIZE + 3 * MOTION_INTERVALS * (int(motion_history) + int(retain_recent_motion))
    context_size += int(stage_clock) * STAGE_CONTEXT_SIZE
    return gym.spaces.Dict({
        "bullets": gym.spaces.Box(-1, 1, (96, 7), dtype=np.float32),
        "context": gym.spaces.Box(-1, 1, (context_size,), dtype=np.float32),
    })


def action_space():
    return gym.spaces.MultiDiscrete(ACTION_WIDTHS)


def condition_context(context, difficulty, shot):
    shot = shot.replace(" ", "")
    character = next((name for name in CHARACTERS if shot.startswith(name)), "")
    config = shot.removeprefix(character)
    extras = np.asarray([difficulty / 4] + [name == character for name in CHARACTERS]
                        + [name == config for name in CONFIGS], dtype=np.float32)
    extras = np.broadcast_to(extras, (*context.shape[:-1], len(extras)))
    return np.concatenate((context, extras), axis=-1)


def append_stage_context(context, frames, stage):
    seconds = np.asarray(frames, dtype=np.float32) / 60
    fields = np.stack((np.full_like(seconds, stage / 7), (seconds / 240).clip(0, 1),
                       (np.log1p(seconds) / 6).clip(0, 1)), axis=-1)
    return np.concatenate((context, fields.astype(context.dtype, copy=False)), axis=-1)


def append_motion_history(context, frames, spans=None, retain_recent_motion=False):
    if spans is not None:
        if retain_recent_motion:
            context = append_motion_history(context, frames)
        return append_multiscale_motion(context, frames, spans)
    elapsed = np.diff(frames)
    uninterrupted = (elapsed > 0) & (elapsed <= 6) & (np.diff(context[:, 3]) == 0)
    displacement = np.diff(context[:, :2], axis=0) * np.array([192, 448], dtype=np.float32)
    velocity = displacement / (8 * elapsed.clip(min=1)[:, None])
    intervals = np.zeros((len(context), 3), dtype=np.float32)
    intervals[1:, :2] = np.where(uninterrupted[:, None], velocity.clip(-1, 1), 0)
    intervals[1:, 2] = uninterrupted
    history = np.zeros((len(context), 3 * MOTION_INTERVALS), dtype=np.float32)
    for lag in range(min(MOTION_INTERVALS, len(context))):
        history[lag:, 3 * lag : 3 * (lag + 1)] = intervals[:len(context) - lag]
    return np.concatenate((context, history), axis=-1)


def append_multiscale_motion(context, frames, spans):
    scale = np.array([192, 448], dtype=np.float32)
    history = np.zeros((len(context), 3 * len(spans)), dtype=np.float32)
    for index, span in enumerate(spans):
        previous = np.searchsorted(frames, frames - span, side="right") - 1
        starts = previous.clip(min=0)
        elapsed = frames - frames[starts]
        valid = (previous >= 0) & (elapsed <= span + 3) & (context[:, 3] == context[starts, 3])
        displacement = (context[:, :2] - context[starts, :2]) * scale
        velocity = displacement / (8 * elapsed.clip(min=1)[:, None])
        history[:, index * 3:index * 3 + 2] = np.where(valid[:, None], velocity.clip(-1, 1), 0)
        history[:, index * 3 + 2] = valid
    return np.concatenate((context, history), axis=-1)


def movement_direction(keys):
    dx = ((keys & 128) != 0).astype(np.int64) - ((keys & 64) != 0)
    dy = ((keys & 32) != 0).astype(np.int64) - ((keys & 16) != 0)
    return (dy + 1) * 3 + dx + 1


def movement_distribution(chunks):
    return (movement_direction(chunks)[..., None] == np.arange(9)).mean(axis=1, dtype=np.float32)


def label_actions(chunks):
    # Pick the most common direction/focus/shoot combination over a decision window.
    keys = chunks & 0xF9
    counts = (keys[:, :, None] == keys[:, None, :]).sum(axis=-1)
    primary = keys[np.arange(len(keys)), counts.argmax(axis=1)]
    return np.stack((movement_direction(primary), (primary & 8) != 0, (primary & 1) != 0,
                     ((chunks & 2) != 0).any(axis=1)), axis=-1).astype(np.int64)


def index_split(directory, split):
    shards = []
    for path in sorted(directory.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["split"] != split:
            continue
        shards.append((path.with_suffix(".npz"), {
            "title": entry["title"], "replay": entry["name"], "stage": entry["stage"],
            "player": entry["player"], "player_group": entry.get("player_group", entry["player"].casefold()),
            "difficulty": entry["difficulty"], "shot": entry["shot"], "samples": entry["pairs"],
        }))
    if not shards:
        raise ValueError(f"No paired complete replay stages for split {split}.")
    return shards


def read_shard(shard, motion_history=False, soft_movement=False, motion_spans=None, retain_recent_motion=False,
               stage_clock=False):
    path, meta = shard
    with np.load(path) as data:
        chunks = data["expert_action_chunk"]
        labels = label_actions(chunks)
        if len(labels) != meta["samples"]:
            raise ValueError(f"Metadata size does not match {path.name}.")
        context = condition_context(data["context"], LEVELS.index(meta["difficulty"]), meta["shot"])
        if motion_history or stage_clock:
            frames = data["stage_frame"]
            if motion_history:
                context = append_motion_history(context, frames, motion_spans, retain_recent_motion)
            if stage_clock:
                context = append_stage_context(context, frames, meta["stage"])
        fields = data["bullets"], context, labels
        return (*fields, movement_distribution(chunks)) if soft_movement else fields


def batch_block(arrays, batch_size, device, order):
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield tuple(torch.from_numpy(field[indices]).to(device) for field in arrays)


def training_batches(shards, block_size, batch_size, device, rng, title_weights, motion_history=False,
                     soft_movement=False, motion_spans=None, retain_recent_motion=False, player_weights=None,
                     stage_clock=False):
    order = rng.permutation(len(shards))
    for start in range(0, len(shards), block_size):
        block = []
        for index in order[start:start + block_size]:
            shard = shards[index]
            fields = read_shard(shard, motion_history, soft_movement, motion_spans, retain_recent_motion, stage_clock)
            meta = shard[1]
            player_weight = player_weights[(meta["title"], meta["player_group"])] if player_weights is not None else 1.
            weight = np.full(len(fields[2]), title_weights[meta["title"]] * player_weight, dtype=np.float32)
            block.append((*fields, weight))
        fields = tuple(np.concatenate(items) for items in zip(*block))
        yield from batch_block(fields, batch_size, device, rng.permutation(len(fields[2])))


def label_counts(shards):
    counts = [np.zeros(width, dtype=np.int64) for width in ACTION_WIDTHS]
    for path, _ in shards:
        with np.load(path) as data:
            labels = label_actions(data["expert_action_chunk"])
        for index, values in enumerate(counts):
            values += np.bincount(labels[:, index], minlength=len(values))
    return counts


def weights_by_player(replays, strength):
    counts = Counter()
    for replay in replays:
        counts[replay["title"], replay["player_group"]] += replay["samples"]
    weighted_totals = Counter()
    totals = Counter()
    for (title, _), samples in counts.items():
        totals[title] += samples
        weighted_totals[title] += samples ** (1 - strength)
    return {(title, player): samples ** -strength * totals[title] / weighted_totals[title]
            for (title, player), samples in counts.items()}


def predict_heads(policy, bullets, context):
    features = policy.extract_features({"bullets": bullets, "context": context})
    latent = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent).split(ACTION_WIDTHS, dim=-1)


def split_movement(logits):
    directions = torch.cat((logits[:, :4], logits[:, 5:]), dim=1)
    return torch.logsumexp(directions, dim=1) - logits[:, 4], directions


def decode_movement(logits, factorized=False):
    if not factorized:
        return logits.argmax(dim=1)
    gate, directions = split_movement(logits)
    selected = directions.argmax(dim=1)
    selected = selected + (selected >= 4)
    return torch.where(gate > 0, selected, 4)


def factorized_movement_loss(logits, targets, sample_weights, class_weights):
    gate, directions = split_movement(logits)
    moving_targets = torch.cat((targets[:, :4], targets[:, 5:]), dim=1)
    moving_probability = moving_targets.sum(dim=1)
    gate_loss = functional.binary_cross_entropy_with_logits(gate, moving_probability, reduction="none")
    gate_loss = (gate_loss * sample_weights).sum() / sample_weights.sum()
    direction_weights = torch.cat((class_weights[:4], class_weights[5:]))
    weighted_targets = moving_targets * direction_weights * sample_weights[:, None]
    direction_loss = -(functional.log_softmax(directions, dim=1) * weighted_targets).sum()
    return gate_loss + direction_loss / weighted_targets.sum().clamp_min(1e-8)


def mirror_batch(bullets, context, actions, movement_targets, rows, motion_fields):
    mapping = actions.new_tensor((2, 1, 0, 5, 4, 3, 8, 7, 6))
    bullets[rows, :, 0] *= -1
    bullets[rows, :, 2] *= -1
    context[rows, 0] *= -1
    context[rows, 6] *= -1
    context[rows, CONTEXT_SIZE:CONTEXT_SIZE + motion_fields:3] *= -1
    actions[rows, 0] = mapping[actions[rows, 0]]
    if movement_targets is not None:
        movement_targets[rows] = movement_targets[rows][:, mapping]


class MovementPolicy(MultiInputActorCriticPolicy):
    factorized_movement = False

    def _predict(self, observation, deterministic=False):
        if not deterministic or not self.factorized_movement:
            return super()._predict(observation, deterministic=deterministic)
        movement, *buttons = predict_heads(self, observation["bullets"], observation["context"])
        return torch.stack((decode_movement(movement, True), *(head.argmax(1) for head in buttons)), dim=1)


@torch.inference_mode()
def validate(policy, shards, majority, batch_size, device, motion_history=False, motion_spans=None,
             retain_recent_motion=False, stage_clock=False):
    policy.eval()
    correct_total = np.zeros(len(ACTION_WIDTHS), dtype=np.int64)
    baseline_total = correct_total.copy()
    sample_total = 0
    moving_total = 0
    correct_moving = 0
    predicted_moving_total = 0
    moving_true_positive = 0
    loss = 0.0
    per_replay = {}
    for shard in shards:
        data = read_shard(shard, motion_history, motion_spans=motion_spans,
                          retain_recent_motion=retain_recent_motion, stage_clock=stage_clock)
        labels = data[2]
        predicted = []
        for bullets, context, truth in batch_block(data, batch_size, device, np.arange(len(labels))):
            logits = predict_heads(policy, bullets, context)
            movement, *buttons = logits
            decoded = torch.stack((decode_movement(movement, getattr(policy, "factorized_movement", False)),
                                   *(head.argmax(1) for head in buttons)), dim=1)
            predicted.append(decoded.cpu().numpy())
            for index, head in enumerate(logits):
                loss += functional.cross_entropy(head, truth[:, index], reduction="sum").item()
        predicted = np.concatenate(predicted)
        correct = predicted == labels
        baseline = labels == majority
        moving = labels[:, 0] != 4
        predicted_moving = predicted[:, 0] != 4
        moving_total += int(moving.sum())
        predicted_moving_total += int(predicted_moving.sum())
        moving_true_positive += int((moving & predicted_moving).sum())
        correct_moving += int(correct[moving, 0].sum())
        per_replay[shard[0].stem] = {
            "accuracy": dict(zip(ACTION_NAMES, correct.mean(axis=0).tolist())),
            "majority_baseline": dict(zip(ACTION_NAMES, baseline.mean(axis=0).tolist())),
            "movement_accuracy_on_moving_frames": float(correct[moving, 0].mean()) if moving.any() else None,
            "predicted_moving_fraction": float((predicted[:, 0] != 4).mean()),
        }
        correct_total += correct.sum(axis=0)
        baseline_total += baseline.sum(axis=0)
        sample_total += len(labels)
    metrics = {"val_loss": loss / sample_total,
               "accuracy": dict(zip(ACTION_NAMES, correct_total / sample_total)),
               "predicted_moving_fraction": predicted_moving_total / sample_total,
               "moving_precision": moving_true_positive / predicted_moving_total if predicted_moving_total else None,
               "moving_recall": moving_true_positive / moving_total if moving_total else None,
               "active_direction_accuracy": correct_moving / moving_total if moving_total else None}
    return metrics, dict(zip(ACTION_NAMES, baseline_total / sample_total)), per_replay


def main():
    parser = argparse.ArgumentParser(description="Behavioral cloning from aligned native game replays.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many epochs without better validation loss. Zero disables early stopping.")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--title-balance", type=float, default=0.0,
                        help="Exponent for inverse-frequency title weights, from 0 to 1.")
    parser.add_argument("--shards-per-block", type=int, default=12)
    parser.add_argument("--player-balance", type=float, default=0.0,
                        help="Equalize replay contributors within each title. Use 0 for sample weighting and 1 for equal players.")
    parser.add_argument("--motion-history", action="store_true")
    parser.add_argument("--motion-spans", nargs=MOTION_INTERVALS, type=int, metavar="FRAMES")
    parser.add_argument("--retain-recent-motion", action="store_true")
    parser.add_argument("--stage-clock", action="store_true")
    parser.add_argument("--encoder", choices=("pool", "attention"), default="pool")
    parser.add_argument("--soft-movement", action="store_true")
    parser.add_argument("--factorized-movement", action="store_true",
                        help="Train separate movement intent and conditional direction losses.")
    parser.add_argument("--geometry-loss-weight", type=float, default=0.,
                        help="Penalize probability on predicted collisions when the demonstration is geometrically safe.")
    parser.add_argument("--mirror-probability", type=float, default=0.,
                        help="Probability of reflecting each training observation and direction horizontally.")
    parser.add_argument("--moving-transition-weight", type=float, default=1.0,
                        help="Extra direction loss weight when the expert starts moving or turns.")
    parser.add_argument("--resume", type=Path, help="Initialize from a prior SFT checkpoint of matching width.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--retain-resume", action="store_true",
                        help="Include the initialization in validation-based checkpoint selection.")
    parser.add_argument("--anchor-movement", type=float, default=0.,
                        help="KL weight for retaining movement predictions from the initial checkpoint.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    if not 0 <= args.player_balance <= 1:
        parser.error("Player balance must be between zero and one.")
    if not 0 <= args.geometry_loss_weight < float("inf"):
        parser.error("Geometry loss weight must be finite and nonnegative.")
    if not 0 <= args.mirror_probability <= 1:
        parser.error("Mirror probability must be between zero and one.")
    if args.motion_spans and (not args.motion_history or args.motion_spans != sorted(set(args.motion_spans))
                             or not 4 <= args.motion_spans[0] <= args.motion_spans[-1] <= 120):
        parser.error("Motion spans require motion history and three distinct increasing spans from 4 to 120 frames.")
    if args.retain_recent_motion and not args.motion_spans:
        parser.error("Retaining the most recent positions requires explicitly requested longer spans.")
    if min(args.epochs, args.hidden, args.batch_size, args.shards_per_block) < 1:
        parser.error("Epochs, hidden width, batch size and shards per block must be positive.")
    if args.patience < 0:
        parser.error("Early-stopping patience cannot be negative.")
    if not 0 < args.learning_rate < float("inf") or args.retain_resume and not args.resume:
        parser.error("Use a finite positive learning rate and supply a checkpoint when retaining the initialization.")
    if not 0 <= args.anchor_movement < float("inf") or args.anchor_movement and not args.resume:
        parser.error("Use a finite nonnegative anchor weight and supply its initial checkpoint.")
    if not 0 <= args.title_balance <= 1:
        parser.error("Title balance must be between zero and one.")
    if not 1 <= args.moving_transition_weight < float("inf"):
        parser.error("Moving transition weight must be finite and at least one.")
    if args.moving_transition_weight != 1 and not args.motion_history:
        parser.error("Moving transition weights require motion history.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train = index_split(args.data, "train")
    val = index_split(args.data, "validation")
    training_replays = [meta for _, meta in train]
    validation_replays = [meta for _, meta in val]
    if {row["player_group"] for row in training_replays} & {
            row["player_group"] for row in validation_replays}:
        raise RuntimeError("Player overlap between training and validation.")
    args.out.mkdir(parents=True, exist_ok=False)
    population = Counter()
    for row in training_replays:
        population[row["title"]] += row["samples"]
    title_weights = {title: (max(population.values()) / count) ** args.title_balance
                     for title, count in population.items()}
    player_weights = weights_by_player(training_replays, args.player_balance)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    policy = MovementPolicy(observation_space(args.motion_history, args.retain_recent_motion, args.stage_clock),
                            action_space(), lambda _: args.learning_rate,
                            **network_kwargs(args.hidden, args.encoder)).to(device)
    policy.factorized_movement = args.factorized_movement
    if args.resume:
        policy.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    teacher = deepcopy(policy).eval().requires_grad_(False) if args.anchor_movement else None
    counts = label_counts(train)
    majority = np.asarray([count.argmax() for count in counts])
    weights = [torch.from_numpy(np.maximum(count, 1).astype(np.float32)).rsqrt().to(device) for count in counts]
    history = []
    best = None
    initial_validation = None
    if args.retain_resume:
        metrics, _, per_replay = validate(policy, val, majority, args.batch_size * 2, device,
                                          args.motion_history, args.motion_spans, args.retain_recent_motion,
                                          args.stage_clock)
        initial_validation = {"epoch": 0, **metrics}
        best = dict(initial_validation, per_replay=per_replay)
        torch.save(policy.state_dict(), args.out / "policy.pt")
        print(json.dumps({"initial_validation": initial_validation}), flush=True)
    motion_fields = 3 * MOTION_INTERVALS * (int(args.motion_history) + int(args.retain_recent_motion))
    for epoch in range(1, args.epochs + 1):
        policy.train()
        geometry_totals = torch.zeros(2, device=device)
        for bullets, context, truth, *extra, title_weight in training_batches(
                train, args.shards_per_block, args.batch_size, device, rng, title_weights,
                args.motion_history, args.soft_movement, args.motion_spans, args.retain_recent_motion,
                player_weights, args.stage_clock):
            if args.mirror_probability:
                mirrored = torch.rand(len(truth), device=device) < args.mirror_probability
                mirror_batch(bullets, context, truth, extra[0] if args.soft_movement else None, mirrored, motion_fields)
            logits = predict_heads(policy, bullets, context)
            direction_weight = title_weight
            if args.moving_transition_weight != 1:
                motion = context[:, CONTEXT_SIZE : CONTEXT_SIZE + 3]
                axes = motion[:, :2].sign() * (motion[:, :2].abs() >= 0.01)
                previous_direction = (axes[:, 1] + 1) * 3 + axes[:, 0] + 1
                changed = (motion[:, 2] > 0) & (truth[:, 0] != 4) & (truth[:, 0] != previous_direction)
                direction_weight = title_weight * (1 + (args.moving_transition_weight - 1) * changed)
            loss = 0.0
            for index, head in enumerate(logits):
                if index == 0 and args.factorized_movement:
                    targets = extra[0] if args.soft_movement else functional.one_hot(truth[:, 0], 9).float()
                    loss += factorized_movement_loss(head, targets, direction_weight, weights[index])
                    continue
                if index == 0 and args.soft_movement:
                    target_weights = extra[0] * weights[index] * direction_weight[:, None]
                    loss -= (functional.log_softmax(head, dim=-1) * target_weights).sum() / target_weights.sum()
                    continue
                target = truth[:, index]
                sample_weight = (direction_weight if index == 0 else title_weight) * weights[index][target]
                losses = functional.cross_entropy(head, target, reduction="none")
                loss += (losses * sample_weight).sum() / sample_weight.sum()
            if args.geometry_loss_weight:
                negative_mass, eligible, _ = replay_risk_terms(bullets, context, truth, logits[0], logits[1])
                loss += args.geometry_loss_weight * negative_mass / eligible.clamp_min(1)
                geometry_totals += torch.stack((negative_mass.detach(), eligible))
            if teacher is not None:
                with torch.no_grad():
                    target_movement = predict_heads(teacher, bullets, context)[0].softmax(dim=1)
                divergence = functional.kl_div(logits[0].log_softmax(dim=1), target_movement,
                                               reduction="none").sum(dim=1)
                loss += args.anchor_movement * (divergence * title_weight).sum() / title_weight.sum()
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy.optimizer.step()
        metrics, baseline, per_replay = validate(policy, val, majority, args.batch_size * 2, device,
                                                 args.motion_history, args.motion_spans, args.retain_recent_motion,
                                                 args.stage_clock)
        entry = {"epoch": epoch, **metrics}
        if args.geometry_loss_weight:
            mass_sum, risky_examples = geometry_totals.tolist()
            entry.update(geometry_eligible_decisions=int(risky_examples),
                         geometry_collision_probability=mass_sum / risky_examples if risky_examples else None)
        print(json.dumps(entry), flush=True)
        with (args.out / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        history.append(entry)
        if best is None or entry["val_loss"] < best["val_loss"]:
            best = dict(entry, per_replay=per_replay)
            torch.save(policy.state_dict(), args.out / "policy.pt")
        if args.patience and epoch - best["epoch"] >= args.patience:
            print(json.dumps({"early_stop_epoch": epoch, "best_epoch": best["epoch"],
                              "patience": args.patience}), flush=True)
            break
    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "hidden": args.hidden,
        "motion_history": args.motion_history,
        "motion_spans": args.motion_spans,
        "retain_recent_motion": args.retain_recent_motion,
        "stage_clock": args.stage_clock,
        "encoder": args.encoder,
        "soft_movement": args.soft_movement,
        "factorized_movement": args.factorized_movement,
        "geometry_loss_weight": args.geometry_loss_weight,
        "moving_transition_weight": args.moving_transition_weight,
        "training_replays": training_replays, "validation_replays": validation_replays,
        "training_samples_by_title": population,
        "training_title_loss_weights": title_weights,
        "player_balance": args.player_balance,
        "training_player_loss_weights": {f"{title}/{player}": weight for (title, player), weight in player_weights.items()},
        "validation_samples": sum(row["samples"] for row in validation_replays),
        "epochs_requested": args.epochs, "epochs_completed": len(history),
        "majority_baseline": baseline, "initial_validation": initial_validation, "best": best, "epochs": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
