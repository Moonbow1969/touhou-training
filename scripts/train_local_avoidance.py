"""Fit a neural four-tick movement residual against recorded exogenous TH14 fields."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, HASHES, PPO397, TEMPLATE
from scripts.evaluate_sft_native import load_arm, load_ppo_arm
from scripts.pretrain_policy import (CONTEXT_SIZE, append_motion_history, append_stage_context,
                                     movement_direction, predict_heads, read_shard)
from scripts.stage_risk import _DIRECTIONS
from scripts.train_full_policy import publish_json
from scripts.train_world import average_precision


REPLAYS = DATA / "sft-native-data-261123/frozen-training-with-hard"
NATIVE = (DATA / "th14-stock-value-story-261125/transition-shards",
          DATA / "th14-demo-alice-story-2001-261125/transition-shards")
SHOTS = ("ReimuA", "MarisaB", "SakuyaA")
TICKS = 4
PLAYER_SCALE = (192., 448.)
BULLET_SCALE = (384., 448., 16., 16., 32., 1., 1.)
CONTACT_RADIUS = 1.
SMOOTHING = 2.5
POLICY_TEMPERATURE = 3.5


@dataclass
class Transitions:
    bullets: np.ndarray
    context: np.ndarray
    next_bullets: np.ndarray
    next_context: np.ndarray
    actions: np.ndarray
    shot: int
    hit: np.ndarray | None = None


def replay_index(root):
    for path in sorted(root.glob("th14*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["split"] in ("train", "validation") and meta["shot"].replace(" ", "") in SHOTS:
            yield path.with_suffix(".npz"), meta


def replay_rows(path):
    with np.load(path, allow_pickle=False) as source:
        words = source["expert_action_chunk"][:-1]
        raw_context = source["context"]
        active = source["player_state"] == 1
        valid = ((np.diff(source["stage_frame"]) == TICKS) & active[:-1] & active[1:]
                 & ((words & 2) == 0).all(axis=1) & ((source["native"][:-1] & 2) == 0)
                 & (raw_context[:-1, 4] == raw_context[1:, 4]))
        controls = words & 0xF8
        constant = (controls == controls[:, :1]).all(axis=1)
    return np.flatnonzero(valid), constant[valid], words[valid]


def replay_transitions(path, meta, anchor):
    args = (path, dict(meta, samples=meta["pairs"]))
    bullets, context, actions = read_shard(args, motion_history=anchor.history,
                                           motion_spans=anchor.policy.motion_spans,
                                           retain_recent_motion=anchor.policy.retain_recent_motion,
                                           stage_clock=anchor.policy.stage_clock)
    indices, constant, words = replay_rows(path)
    shot = SHOTS.index(meta["shot"].replace(" ", ""))
    return Transitions(bullets[indices], context[indices], bullets[indices + 1], context[indices + 1],
                       actions[indices], shot), constant, words


def native_transitions(path, anchor):
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(source["metadata_json"].item())
        if (meta["schema"] != "th14-stock-story-actor-transitions-v1"
                or meta["task"]["shot"].replace(" ", "") != "ReimuA"):
            raise ValueError(f"Unexpected native credit observation contract: {path.name}")
        names = {name: i for i, name in enumerate(meta["native_field_order"])}
        before, after, actions = source["native_before"], source["native_after"], source["action"]
        valid = (source["four_tick_stage_transition"] & (before[:, names["player_state"]] == 1)
                 & (before[:, names["previous_x"]] == 0) & (actions[:, 3] == 0)
                 & (source["actual_bombs_consumed"] == 0))
        # Credit shards hold the next decision's bullets. Terminal successors have no such row.
        valid[-1] = False
        valid[:-1] &= before[1:, names["global_tick"]] == after[:-1, names["global_tick"]]
        selected = np.flatnonzero(valid)
        bullets, recorded_context = source["bullets"], source["context"]
        # Comparators use a common PPO transform even for credits recorded by a smaller actor.
        parts = []
        expected = anchor.policy.observation_space["context"].shape[0]
        if recorded_context.shape[1] < CONTEXT_SIZE:
            raise ValueError(f"Native credit lacks the common matchup context: {path.name}")
        for stage, start, end in source["stage_spans"]:
            context = recorded_context[start:end, :CONTEXT_SIZE]
            frames = before[start:end, names["stage_frame"]]
            if anchor.history:
                context = append_motion_history(context, frames, anchor.policy.motion_spans,
                                                anchor.policy.retain_recent_motion)
            if anchor.policy.stage_clock:
                context = append_stage_context(context, frames, stage)
            parts.append(context)
        context = np.concatenate(parts)
        if context.shape != (len(bullets), expected):
            raise ValueError(f"Could not build the matched anchor transform for {path.name}.")
        return Transitions(bullets[selected], context[selected], bullets[selected + 1], context[selected + 1],
                           actions[selected].astype(np.int64), 0, after[selected, names["player_state"]] == 4)


def fit_dynamics(replays):
    observations = [[] for _ in SHOTS]
    bounds = [[] for _ in SHOTS]
    directions = _DIRECTIONS
    for path, meta in replays:
        shot = SHOTS.index(meta["shot"].replace(" ", ""))
        indices, constant, words = replay_rows(path)
        with np.load(path, allow_pickle=False) as source:
            context = source["context"]
        position = (context[:, :2] * np.array(PLAYER_SCALE, dtype=np.float32))
        bounds[shot].append(np.stack((position.min(axis=0), position.max(axis=0))))
        xy = position[indices]
        interior = (np.abs(xy[:, 0]) < 162) & (xy[:, 1] > 54) & (xy[:, 1] < 409)
        chosen = constant & interior
        index, keys = indices[chosen], words[chosen, 0]
        movement = movement_direction(keys)
        command = directions[movement]
        delta = (position[index + 1] - position[index]) / TICKS
        travel = (command * delta).sum(axis=1)
        observations[shot].append((context[index, 2] > .5, (keys & 8) > 0, movement, travel))
    speeds = np.empty((len(SHOTS), 2, 2, 9), dtype=np.float32)
    limits = np.empty((len(SHOTS), 2, 2), dtype=np.float32)
    evidence = {}
    for shot, name in enumerate(SHOTS):
        prev, focus, movement, travel = (np.concatenate(group) for group in zip(*observations[shot]))
        corners = np.stack(bounds[shot])
        limits[shot] = np.round(np.stack((corners[:, 0].min(axis=0), corners[:, 1].max(axis=0))), 2)
        counts = np.empty((2, 2, 9), dtype=np.int64)
        for previous in range(2):
            for focused in range(2):
                pool = (focus == focused) & (movement != 4)
                if pool.sum() < 16:
                    raise ValueError(f"Too few empirical movements for {name}, focus {focused}.")
                fallback = np.median(travel[pool])
                for move in range(9):
                    match = (prev == previous) & (focus == focused) & (movement == move)
                    count = int(match.sum())
                    speeds[shot, previous, focused, move] = (0. if move == 4 else
                                                            np.median(travel[match]) if count >= 16 else fallback)
                    counts[previous, focused, move] = count
        evidence[name] = dict(bounds=limits[shot].tolist(), speed_units_per_tick=speeds[shot].tolist(),
                              n_by_previous_focus_commanded_focus_move=counts.tolist())
    return speeds, limits, evidence


def local_context(context, shot):
    label = torch.full((len(context),), shot, dtype=torch.long, device=context.device)
    return torch.cat((context[:, :3], F.one_hot(label, len(SHOTS)).to(context.dtype)), dim=1)


@torch.inference_mode()
def modeled_contacts(bullets, context, next_bullets, next_context, speeds, bounds):
    position = context[:, :2] * context.new_tensor(PLAYER_SCALE)
    next_position = next_context[:, :2] * context.new_tensor(PLAYER_SCALE)
    previous_focus = (context[:, 2] > .5).long()
    time = torch.arange(1, TICKS + 1, dtype=context.dtype, device=context.device)
    direction = context.new_tensor(_DIRECTIONS)
    movement = speeds[previous_focus, :, :, None, None] * direction[None, None, :, None, :]
    path = position[:, None, None, None, :] + movement * time[None, None, None, :, None]
    path = path.clamp(min=bounds[0], max=bounds[1])

    scale = bullets.new_tensor(BULLET_SCALE)
    raw, future = bullets * scale, next_bullets * scale
    centers = raw[:, None, :, :2] + position[:, None, None, :]
    velocity = raw[:, None, :, 2:4]
    interior = centers + velocity * time[None, :-1, None, None]
    endpoint = future[:, None, :, :2] + next_position[:, None, None, :]
    locations = torch.cat((interior, endpoint), dim=1)
    present = (raw[..., 6] > 0) & (raw[..., 5] > 0) & (raw[..., 4] > 0)
    next_present = (future[..., 6] > 0) & (future[..., 5] > 0) & (future[..., 4] > 0)
    present = torch.cat((present[:, None].expand(-1, TICKS - 1, -1), next_present[:, None]), dim=1)
    radius = torch.cat((raw[:, None, :, 4].expand(-1, TICKS - 1, -1), future[:, None, :, 4]), dim=1)
    relative = path[:, :, :, :, None, :] - locations[:, None, None]
    margin = torch.linalg.vector_norm(relative, dim=-1) - radius[:, None, None] - CONTACT_RADIUS
    closest = margin.masked_fill(~present[:, None, None], 512.).amin(dim=(-2, -1))
    return closest, path[:, :, :, -1] - position[:, None, None]


class LocalAvoidance(nn.Module):
    """One neural forward pass gives costs and a residual for each focus/movement pair."""

    def __init__(self, width=48):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(12, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.contact = nn.Linear(width, 19)
        self.residual = nn.Sequential(nn.Linear(width + 19 + 6, width), nn.SiLU(), nn.Linear(width, 18))
        self.contact.bias.data.fill_(-4.)

    def forward(self, bullets, local):
        scale = bullets.new_tensor((6., 7., 2., 2., 4., 1.))
        points = bullets[..., :6] * scale
        features = self.point(torch.cat((points, local[:, None].expand(-1, bullets.shape[1], -1)), dim=-1))
        mask = ((bullets[..., 6] <= 0) | (bullets[..., 5] <= 0) | (bullets[..., 4] <= 0))[..., None]
        pool = features.masked_fill(mask, -32.).amax(dim=1).clamp(min=-8.)
        logits = self.contact(features).masked_fill(mask, -12.).amax(dim=1)
        confidence = logits[:, :18].sigmoid().amax(dim=1, keepdim=True)
        delta = 3. * confidence * torch.tanh(self.residual(torch.cat((pool, logits, local), dim=1)))
        return logits[:, :18].view(-1, 2, 9), logits[:, 18], delta.view(-1, 2, 9)


def fields(rows, anchor, speeds, limits, batch_size, device):
    columns = (rows.bullets, rows.context, rows.next_bullets, rows.next_context, rows.actions)
    speed = torch.as_tensor(speeds[rows.shot], device=device)
    bounds = torch.as_tensor(limits[rows.shot], device=device)
    for start in range(0, len(rows.bullets), batch_size):
        bullets, context, next_bullets, next_context, action = (
            torch.from_numpy(arr[start:start + batch_size]).to(device) for arr in columns
        )
        with torch.inference_mode():
            margin, displacement = modeled_contacts(bullets, context, next_bullets, next_context, speed, bounds)
            movement, focus, _, _ = predict_heads(anchor.policy, bullets, context)
            labels = torch.sigmoid(-margin / SMOOTHING)
            actual_delta = (next_context[:, :2] - context[:, :2]) * context.new_tensor(PLAYER_SCALE)
        yield bullets, local_context(context, rows.shot), labels.clone(), movement.clone(), focus.clone(), action, (
            displacement.clone(), actual_delta.clone(), margin.clone(),
        )


def training_tensors(replays, anchor, speeds, limits, device):
    blocks = []
    for path, meta in replays:
        rows, _, _ = replay_transitions(path, meta, anchor)
        blocks.extend((b.cpu().half(), c.cpu(), y.cpu(), m.cpu(), f.cpu())
                      for b, c, y, m, f, _, _ in fields(rows, anchor, speeds, limits, 256, device))
    return tuple(torch.cat(parts) for parts in zip(*blocks))


def train_epoch(model, optimizer, arrays, indices, batch_size, device):
    totals = torch.zeros(3, device=device)
    for subset in indices.split(batch_size):
        bullets, local, labels, movement, focus = (array[subset].to(device).float() for array in arrays)
        risk, marginal, delta = model(bullets, local)
        weight = 1 + 3 * labels.amax(dim=(1, 2))
        contact = F.binary_cross_entropy_with_logits(risk, labels, reduction="none").mean(dim=(1, 2))
        agnostic = F.binary_cross_entropy_with_logits(marginal, labels.mean(dim=(1, 2)), reduction="none")
        target = (movement[:, None] - POLICY_TEMPERATURE * labels).softmax(dim=-1)
        improved = (movement[:, None] + delta).log_softmax(dim=-1)
        kl = F.kl_div(improved, target, reduction="none").sum(dim=-1)
        kl = (kl * focus.softmax(dim=1)).sum(dim=1)
        losses = torch.stack(((weight * contact).mean(), (weight * agnostic).mean(), (weight * kl).mean()))
        optimizer.zero_grad(set_to_none=True)
        (losses[0] + .25 * losses[1] + 2. * losses[2]).backward()
        optimizer.step()
        totals += losses.detach() * len(subset)
    return dict(zip(("action_contact_bce", "action_agnostic_bce", "actor_target_kl"), (totals / len(indices)).tolist()))


@torch.inference_mode()
def evaluate(model, rows, anchor, speeds, limits, device):
    records = []
    for b, c, y, m, f, action, diagnostics in fields(rows, anchor, speeds, limits, 256, device):
        pred, agnostic, delta = model(b, c)
        index = torch.arange(len(y), device=device)
        move, focus = action[:, 0].long(), action[:, 1].long()
        anchor_move = m.softmax(dim=-1)
        focus_prob = f.softmax(dim=-1)
        anchor_joint = anchor_move[:, None, :] * focus_prob[:, :, None]
        improved = (m[:, None, :] + delta).softmax(dim=-1)
        improved_joint = improved * focus_prob[:, :, None]
        risk_prob = pred.sigmoid()
        displacement, actual_delta, margin = diagnostics
        chosen = (index, focus, move)
        expected = lambda cost, behavior: (cost * behavior).sum(dim=(1, 2))
        record = dict(actual_risk=risk_prob[chosen], risk_agnostic=agnostic.sigmoid(),
                      risk_focus_mean=risk_prob[index, focus, :].mean(dim=1),
                      risk_anchor_mean=expected(risk_prob, anchor_joint),
                      risk_teacher_factual=y[chosen], risk_teacher_agnostic=y.mean(dim=(1, 2)),
                      modeled_contact=margin[chosen] <= 0,
                      anchor_action_surprisal=-anchor_move[index, move].clamp(min=1e-8).log(),
                      delta_mae=(displacement[chosen] - actual_delta).abs().mean(dim=1),
                      stationary_mae=actual_delta.abs().mean(dim=1),
                      anchor_cost=expected(y, anchor_joint), learned_cost=expected(y, improved_joint),
                      anchor_hard_contact=expected((margin <= 0).float(), anchor_joint),
                      learned_hard_contact=expected((margin <= 0).float(), improved_joint),
                      movement_tv=(improved_joint - anchor_joint).abs().sum(dim=(1, 2)) / 2,
                      anchor_direction=anchor_joint.sum(dim=1), learned_direction=improved_joint.sum(dim=1),
                      teacher_max=y.amax(dim=(1, 2)), teacher_span=y.amax(dim=(1, 2)) - y.amin(dim=(1, 2)))
        records.append({name: values.cpu().numpy() for name, values in record.items()})
    return {name: np.concatenate([batch[name] for batch in records]) for name in records[0]}


def score(data, hit=None):
    risky = data["teacher_span"] >= .2
    quiet = data["teacher_max"] < .05
    anchor_dir = data["anchor_direction"].mean(axis=0)
    learned_dir = data["learned_direction"].mean(axis=0)
    mean = lambda key, selector: float(data[key][selector].mean()) if np.any(selector) else None
    summary = dict(rows=len(risky), modeled_action_sensitive_rows=int(risky.sum()), quiet_rows=int(quiet.sum()),
                   expected_soft_contact=dict(anchor=mean("anchor_cost", risky), learned=mean("learned_cost", risky)),
                   expected_hard_contact=dict(anchor=mean("anchor_hard_contact", risky),
                                              learned=mean("learned_hard_contact", risky)),
                   quiet_movement_tv=mean("movement_tv", quiet), risky_movement_tv=mean("movement_tv", risky),
                   whole_direction_total_variation=float(np.abs(learned_dir - anchor_dir).sum() / 2),
                   largest_direction_mass_increase=float((learned_dir - anchor_dir).max()),
                   anchor_direction_frequency=anchor_dir.tolist(), learned_direction_frequency=learned_dir.tolist())
    scores = ("actual_risk", "risk_agnostic", "risk_focus_mean", "risk_anchor_mean",
              "risk_teacher_factual", "risk_teacher_agnostic", "anchor_action_surprisal")
    if data["modeled_contact"].any():
        summary["modeled_factual_contact_ap"] = average_precision(data["modeled_contact"], data["actual_risk"])
    if hit is not None:
        summary.update(native_hit_onsets=int(hit.sum()), native_hit_rate=float(hit.mean()),
                       native_hit_ap={name: average_precision(hit, data[name]) for name in scores} if hit.any() else {},
                       hit_action_risk_above_same_focus_mean=float(np.mean(
                           data["actual_risk"][hit] > data["risk_focus_mean"][hit])) if hit.any() else None)
    stable = np.ones(len(risky), dtype=bool) if hit is None else ~hit
    summary["four_tick_player_displacement_mae"] = dict(empirical=mean("delta_mae", stable),
                                                        stationary=mean("stationary_mae", stable))
    return summary


def promotion_gate(replay_validation, native, per_credit):
    native_ap = native["native_hit_ap"]
    wins = sum(row["native_hit_ap"].get("actual_risk", 0) > row["native_hit_ap"].get("risk_agnostic", 0)
               for row in per_credit)
    proxy = native["expected_soft_contact"]
    dev_proxy = replay_validation["expected_soft_contact"]
    checks = dict(
        native_ap_above_agnostic_and_focus_mean=native_ap["actual_risk"] >= 1.10 * max(
            native_ap["risk_agnostic"], native_ap["risk_focus_mean"]),
        native_majority_credit_agnostic_wins=wins > len(per_credit) / 2,
        native_and_development_proxy_reduction=proxy["learned"] <= .90 * proxy["anchor"]
        and dev_proxy["learned"] <= .90 * dev_proxy["anchor"],
        away_from_hazards_tv_at_most_008=native["quiet_movement_tv"] <= .08,
        no_direction_mass_increase_above_010=native["largest_direction_mass_increase"] <= .10,
    )
    return {"passed": all(checks.values()), "criteria": checks, "per_credit_action_aware_ap_wins": wins,
            "native_trials": len(per_credit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=REPLAYS)
    parser.add_argument("--native", type=Path, nargs="+", default=NATIVE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2631)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("Use a positive epoch count and batch size.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    anchor = load_ppo_arm(PPO397, load_arm(TEMPLATE))
    if anchor.provenance["weights_sha256"] != HASHES["ppo397"]:
        raise ValueError("Expected the published PPO397 movement anchor.")
    anchor.policy.to(device)
    train, dev = [], []
    for path, meta in replay_index(args.replays):
        (train if meta["split"] == "train" else dev).append((path, meta))
    if not train or not dev or {row["sha256"] for _, row in train} & {row["sha256"] for _, row in dev}:
        raise ValueError("Replay train and development must use disjoint original replay files.")
    native_paths = sorted({path for root in args.native for path in root.glob("credit-*.npz")})
    if not native_paths:
        raise ValueError("Provide recorded original-game credit shards for held-out validation.")
    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    speeds, limits, dynamics = fit_dynamics(train)
    report = dict(status="training", seed=args.seed, device=str(device), epochs=args.epochs,
                  batch_size=args.batch_size, runner_sha256=digest(Path(__file__)), ticks=TICKS,
                  factual_native_hit="Exact 4-tick active transition from native player_state=1 to 4; exclude X held or commanded",
                  counterfactual=dict(
                      projectile_field="Recorded behavior's exogenous world-coordinate field. Ticks 1–3 constant velocity then actual tick 4 field.",
                      unknown="Player-aimed trajectory changes, canceled bullets and state changes need action interventions. Lasers and enemy bodies are absent.",
                      contact_prior="Empirical clipped player trajectories, active visible circular bullets, 1-unit assumed player radius, 2.5-unit soft margin.",
                      limitations="Up to 96 nearest snapshot bullets with clipped velocity; within-step onset/speed changes and bump timing are unresolved.",
                      train_policy="Expected soft contact optimization of a one-decision model, KL-anchored to original PPO397 at each focus.",
                  ), anchor=anchor.provenance, fitted_player_dynamics=dynamics,
                  replay_train=[dict(file=path.name, replay_sha256=meta["sha256"], shard_sha256=digest(path))
                                for path, meta in train],
                  replay_validation=[dict(file=path.name, replay_sha256=meta["sha256"], shard_sha256=digest(path))
                                     for path, meta in dev],
                  heldout_native=[dict(file=f"{p.parents[1].name}/{p.name}", sha256=digest(p)) for p in native_paths],
                  epoch_log=[])
    path = args.out / "report.json"
    publish_json(path, report)
    arrays = training_tensors(train, anchor, speeds, limits, device)
    report["training_rows"] = len(arrays[0])
    print(json.dumps(dict(status="rollouts_ready", training_rows=report["training_rows"],
                          modeled_action_rows=18 * report["training_rows"])), flush=True)
    model = LocalAvoidance().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        losses = train_epoch(model, optimizer, arrays, torch.randperm(len(arrays[0]), generator=rng),
                             args.batch_size, device)
        entry = dict(epoch=epoch + 1, elapsed_seconds=perf_counter() - started, **losses)
        report["epoch_log"].append(entry)
        publish_json(path, report)
        print(json.dumps(entry), flush=True)
    model.eval()
    replay_scores = [evaluate(model, replay_transitions(path, meta, anchor)[0], anchor, speeds, limits, device)
                     for path, meta in dev]
    dev_score = score({k: np.concatenate([row[k] for row in replay_scores]) for k in replay_scores[0]})
    native_rows = [native_transitions(path, anchor) for path in native_paths]
    native_scores = [evaluate(model, rows, anchor, speeds, limits, device) for rows in native_rows]
    combined = {k: np.concatenate([row[k] for row in native_scores]) for k in native_scores[0]}
    per_credit = [dict(credit=f"{path.parents[1].name}/{path.name}", **score(metrics, rows.hit))
                  for path, metrics, rows in zip(native_paths, native_scores, native_rows)]
    native = score(combined, np.concatenate([row.hit for row in native_rows]))
    gate = promotion_gate(dev_score, native, per_credit)
    artifact = args.out / ("native-candidate.pt" if gate["passed"] else "offline-model.pt")
    temporary = artifact.with_suffix(".tmp")
    torch.save(dict(state_dict=model.cpu().state_dict(), source_anchor_sha256=HASHES["ppo397"],
                    promotion_gate_passed=gate["passed"], schema="th14-neural-avoidance-v1",
                    local_features=["x", "y", "previous_focus", *SHOTS]), temporary)
    temporary.replace(artifact)
    report["model_artifact"] = dict(file=artifact.name, sha256=digest(artifact))
    report.update(status="complete", development_score=dev_score, heldout_native_score=native,
                  per_native_credit=per_credit, promotion_gate=gate, seconds=perf_counter() - started)
    publish_json(path, report)
    print(json.dumps(dict(status="complete", training_rows=report["training_rows"],
                          native_hit_ap=native["native_hit_ap"], promotion_gate=gate)), flush=True)


if __name__ == "__main__":
    main()
