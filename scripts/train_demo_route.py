"""Train and evaluate a pure neural tracker of one cleared stock TH14 demonstration."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.pretrain_policy import movement_direction
from scripts.train_full_policy import publish_json


ALICE = "th14_ud0613"
ROUTE_SHA256 = "4356b957df23cb19234e8a51fed8723e173f07c269658e662733bb2b0e804158"
HOLDOUTS = ("th14_ud06d6", "th14_ud068f")
SCALE = np.array([192, 448], dtype=np.float32)
SCHEMA = "th14-alice-neural-route-v1"


def open_shard(data, replay, stage):
    stem = data / f"{replay}-stage{stage}"
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if (meta["title"], meta["version"], meta["difficulty"], meta["shot"].replace(" ", ""), meta["stride"]) != (
            14, "1.00b", "Normal", "ReimuA", 4):
        raise ValueError("The native movement shard has a different observation contract.")
    path = stem.with_suffix(".npz")
    with np.load(path, allow_pickle=False) as arrays:
        record = dict(frame=arrays["stage_frame"], positions=arrays["context"][:, :2] * SCALE,
                      lives=arrays["context"][:, 3], boss=arrays["context"][:, 8] > 0,
                      keys=arrays["expert_action_chunk"])
    if len(record["frame"]) != meta["pairs"] or np.any(np.diff(record["frame"]) <= 0):
        raise ValueError("A replay contains missing support or unordered frames.")
    return record, meta, digest(path)


def eligible(record):
    frame, positions, keys = (record[key] for key in ("frame", "positions", "keys"))
    focus = ((keys & 8) != 0).astype(np.int64)
    movement = movement_direction(keys)
    action = focus * 9 + movement
    stable = (action == action[:, :1]).all(1)
    base = ((np.diff(frame) == 4) & (np.diff(record["lives"]) == 0)
            & stable[:-1] & ~((keys[:-1] & 2) != 0).any(1))
    unclamped = (np.abs(positions[:, 0]) < 173) & (positions[:, 1] > 43) & (positions[:, 1] < 420)
    return base, base & unclamped[1:] & unclamped[:-1], action[:-1, 0]


def fit_action_deltas(records, fallback=None):
    labels, displacements = [], []
    for record in records:
        _, clean, action = eligible(record)
        labels.append(action[clean])
        displacements.append(np.diff(record["positions"], axis=0)[clean])
    labels, displacements = np.concatenate(labels), np.concatenate(displacements)
    count = np.bincount(labels, minlength=18)
    if fallback is None and count.min() < 10:
        raise ValueError("The training credit does not have enough support for all move/focus combinations.")
    delta = np.stack([
        np.median(displacements[labels == index], axis=0) if count[index] >= 10 else fallback[index]
        for index in range(18)
    ]).astype(np.float32)
    return delta, count


def features(position, target, focus, shoot, bounds):
    lo, hi = bounds
    difference = np.clip((target - position) / 16, -6, 6)
    room = np.clip(np.concatenate((position - lo, hi - position), axis=-1) / 32, 0, 1)
    return np.concatenate((difference, room, focus[..., None], shoot[..., None]), axis=-1).astype(np.float32)


class RouteTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.movement_focus = nn.Linear(64, 18)
        self.shoot = nn.Linear(64, 2)

    def forward(self, value):
        hidden = self.body(value)
        return self.movement_focus(hidden), self.shoot(hidden)


def synthesize(records, delta, bounds, rng):
    vectors, movement, firing = [], [], []
    for record in records:
        nominal = record["positions"][:-1]
        live = (np.diff(record["frame"]) == 4) & (np.diff(record["lives"]) == 0)
        position, target = nominal[live], record["positions"][1:][live]
        native = record["keys"][:-1][live, 0]
        focus, shoot = ((native & 8) != 0).astype(np.int64), ((native & 1) != 0).astype(np.int64)
        expert_movement = movement_direction(native)
        for deviation in (0., 4., 16., 45., 90.):
            sampled = np.clip(position + rng.normal(0., deviation, position.shape), *bounds).astype(np.float32)
            candidate = np.clip(sampled[:, None, :] + delta, *bounds)
            errors = np.linalg.norm(candidate - target[:, None, :], axis=-1)
            costs = errors + 1.8 * (focus[:, None] != np.arange(18)[None, :] // 9)
            # Prefer an observed expert direction among near-ties, while fitting actual 2D recovery.
            costs += .25 * (expert_movement[:, None] != np.arange(18)[None, :] % 9)
            vectors.append(features(sampled, target, focus, shoot, bounds))
            movement.append(costs.argmin(1))
            firing.append(shoot)
    return tuple(np.concatenate(column) for column in (vectors, movement, firing))


def encounters(record):
    frames = record["frame"][record["boss"]]
    groups = np.split(frames, np.flatnonzero(np.diff(frames) > 64) + 1)
    return [(int(rows[0]), int(rows[-1])) for rows in groups if len(rows) >= 16]


def forward_metrics(record, delta, bounds):
    valid, _, actions = eligible(record)
    position = np.clip(record["positions"][:-1][valid] + delta[actions[valid]], *bounds)
    error = np.linalg.norm(position - record["positions"][1:][valid], axis=1)
    moving = actions[valid] % 9 != 4
    return dict(rows=len(error), mean=float(error.mean()), p95=float(np.quantile(error, .95)),
                movement_rows=int(moving.sum()), movement_p95=float(np.quantile(error[moving], .95)))


@torch.inference_mode()
def counterfactual(model, record, heldout_delta, bounds):
    frames, position, keys = record["frame"], record["positions"], record["keys"][:, 0]
    starts = np.arange(0, len(frames) - 16, 3)
    starts = starts[(frames[starts + 16] - frames[starts] == 64)
                    & (record["lives"][starts + 16] == record["lives"][starts])]
    rows = {}
    for amount in (32, 96, "wall"):
        target_x = position[starts, 0]
        if amount == "wall":
            shifted = position[starts].copy()
            shifted[:, 0] = np.where(target_x < 0, bounds[1, 0], bounds[0, 0])
        else:
            shifted = np.clip(position[starts] + np.array([amount, -amount * .5]), *bounds).astype(np.float32)
        initial = np.linalg.norm(shifted - position[starts], axis=1)
        current = shifted.copy()
        for tick in range(16):
            time = starts + tick
            product = features(current, position[time + 1], (keys[time] & 8) != 0,
                               (keys[time] & 1) != 0, bounds)
            selection = model(torch.from_numpy(product))[0].argmax(1).numpy()
            current = np.clip(current + heldout_delta[selection], *bounds)
        target = position[starts + 16]
        errors = np.linalg.norm(current - target, axis=1)
        stationary = np.linalg.norm(shifted - target, axis=1)
        rows[str(amount)] = dict(starts=len(starts), initial_error_median=float(np.median(initial)),
                                 endpoint_median=float(np.median(errors)), endpoint_p95=float(np.quantile(errors, .95)),
                                 within_12=int((errors <= 12).sum()),
                                 stationary_within_12=int((stationary <= 12).sum()))
    return rows


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [open_shard(args.data, ALICE, stage) for stage in range(1, 7)]
    for stage, (example, meta, _) in enumerate(rows, 1):
        if (meta["split"] != "train" or meta["player"].casefold() != "alice" or meta["sha256"] != ROUTE_SHA256
                or set(map(int, meta["frames_by_stage"])) != set(range(1, 7))
                or meta["stage"] != stage or meta["moving_alignment"] < .95 or meta["coverage"] < .9
                or np.any(np.diff(example["lives"]) < 0)):
            raise ValueError("The preselected training clear is incomplete or failed alignment.")
    training = [record for record, _, _ in rows]
    positions = np.concatenate([record["positions"] for record in training])
    bounds = np.round(np.stack((positions.min(0), positions.max(0)))).astype(np.float32)
    delta, support = fit_action_deltas(training)
    inputs, movement, fire = map(torch.from_numpy, synthesize(training, delta, bounds, rng))
    model = RouteTracker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    logs = []
    for epoch in range(1, args.epochs + 1):
        loss_total = 0.
        for batch in torch.randperm(len(inputs)).split(512):
            motion, shoot = model(inputs[batch])
            loss = F.cross_entropy(motion, movement[batch]) + .2 * F.cross_entropy(shoot, fire[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(batch)
        logs.append(dict(epoch=epoch, training_examples=len(inputs), loss=loss_total / len(inputs)))
        print(json.dumps(logs[-1]), flush=True)
    model.eval()
    policy_path = args.out / "model.pt"
    torch.save(model.state_dict(), policy_path)
    route_path = args.out / "route.npz"
    route = {f"{key}_{stage}": record[key] for stage, record in enumerate(training, 1)
             for key in ("frame", "positions", "keys")}
    route.update({f"encounters_{stage}": np.asarray(encounters(record), dtype=np.int32).reshape(-1, 2)
                  for stage, record in enumerate(training, 1)})
    np.savez_compressed(route_path, **route, bounds=bounds, action_deltas=delta)
    report = dict(schema=SCHEMA, train_replay=ALICE, replay_sha256=ROUTE_SHA256,
                  route_selection="Preselected complete six-stage Alice Normal Reimu A training replay; no-miss metadata and no observed life-stock decrements.",
                  train_stage_shards={str(stage): fingerprint for stage, (_, _, fingerprint) in enumerate(rows, 1)},
                  training_seed=args.seed, epochs=logs, action_support=support.tolist(), bounds=bounds.tolist(),
                  dynamics_deltas=delta.tolist(), model_sha256=digest(policy_path), route_sha256=digest(route_path),
                  prior=dict(route="recorded expert screen position/Focus/Shoot indexed by native stage clock",
                             labels="nearest target under empirical per-action 2D displacement with disturbance recovery",
                             phase_alignment="first recorded boss presence of each matched encounter; no subphase retiming",
                             bomb="disabled", gameplay_assistance=False),
                  diagnostic_math="Counterfactual trajectory uses per-action movement calibrated on held-out native observations; no native bullets/collisions. Route from each held-out expert is a diagnostic target only.",
                  stage1_train_boss_encounters=encounters(training[0]), heldout={})
    # Validation happens after the training artifact is frozen.
    for name in HOLDOUTS:
        record, meta, fingerprint = open_shard(args.data, name, 1)
        if (meta["player"].casefold() not in ("toshi", "aria") or name == ALICE
                or fingerprint in report["train_stage_shards"].values()):
            raise ValueError("Held-out demonstration identity validation failed.")
        acquired, acquired_support = fit_action_deltas([record], fallback=delta)
        report["heldout"][name] = dict(shard_sha256=fingerprint, forward=forward_metrics(record, delta, bounds),
                                       calibration_actions=acquired_support.tolist(),
                                       recovery=counterfactual(model, record, acquired, bounds),
                                       boss_encounters=encounters(record))
    publish_json(args.out / "report.json", report)


class DemoPolicy:
    motion_spans = None
    retain_recent_motion = False
    stage_clock = False

    def __init__(self, checkpoint, route, *, resync=True, bomb_model=None):
        self.model = RouteTracker().eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with np.load(route, allow_pickle=False) as arrays:
            self.route = {key: arrays[key] for key in arrays.files}
        self.environment = None
        self.resync = resync
        self.credit_stage = None
        self.bomb_model = None
        if bomb_model is not None:
            from scripts.fit_randomized_bomb_value import LearnedBomb

            self.bomb_model = LearnedBomb(self, bomb_model).weights

    def phase_time(self, frame, stage, boss):
        credit_stage = (self.environment.reset_count, stage)
        if self.credit_stage != credit_stage:
            self.credit_stage = credit_stage
            self.time_offset, self.next_encounter, self.last_boss_frame = 0, 0, -1000
        encounters = self.route[f"encounters_{stage}"]
        if self.resync and boss and frame - self.last_boss_frame > 64 and self.next_encounter < len(encounters):
            next_start = int(encounters[self.next_encounter, 0])
            if abs(frame + self.time_offset - next_start) <= 1000:
                self.time_offset = next_start - frame
                self.next_encounter += 1
        if boss:
            self.last_boss_frame = frame
        return frame + self.time_offset

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        environment = self.environment
        stage = environment.snapshot.stage
        context = observation["context"]
        frame = self.phase_time(environment.last_frame, stage, bool(context[8] > 0))
        frames = self.route[f"frame_{stage}"]
        pos = self.route[f"positions_{stage}"]
        # The first upcoming expert tick aligns the route's preferred buttons to the held action.
        index = int(np.clip(np.searchsorted(frames, frame, side="right") - 1, 0, len(frames) - 1))
        key = int(self.route[f"keys_{stage}"][index, int(np.clip(frame - frames[index], 0, 3))])
        target = np.array([np.interp(frame + 4, frames, coordinates) for coordinates in pos.T], dtype=np.float32)
        product = features(context[:2] * SCALE, target, np.array(bool(key & 8)), np.array(bool(key & 1)),
                           self.route["bounds"])
        motion, shoot = self.model(torch.from_numpy(product))
        focus, move = divmod(int(motion.argmax()), 9)
        bomb = 0
        if self.bomb_model is not None:
            from scripts.fit_randomized_bomb_value import action_values, native_bomb_features

            values, _ = action_values(self.bomb_model, native_bomb_features(environment))
            bomb = int(values[0].argmax())
        return np.array([move, focus, int(shoot.argmax()), bomb], dtype=np.int64), None


def evaluate(args):
    from scripts import evaluate_credit_story as story
    from scripts.evaluate_sft_native import Arm, load_arms, output_lock
    from scripts.native_story_restart import GameOverMenu

    torch.set_num_threads(1)
    report = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    policy_path, route_path = args.artifact / "model.pt", args.artifact / "route.npz"
    if report["schema"] != SCHEMA or (digest(policy_path), digest(route_path)) != (
            report["model_sha256"], report["route_sha256"]):
        raise ValueError("The route model artifacts do not match their frozen report.")
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8")) if args.bomb_model else None
    policy = DemoPolicy(policy_path, route_path, resync=not args.no_resync, bomb_model=bomb)
    provenance = {key: report[key] for key in ("schema", "train_replay", "replay_sha256", "model_sha256", "route_sha256", "prior")}
    provenance.update(boss_encounter_resync=not args.no_resync, policy_decode="greedy")
    name = "alice_route_with_learned_bomb" if bomb is not None else "alice_route"
    if bomb is not None:
        provenance["prior"] = dict(provenance["prior"], bomb="frozen learned value head transferred from PPO397")
        provenance["bomb_transfer"] = dict(training_actor_sha256=bomb["anchor_sha256"],
                                           value_model_sha256=digest(args.bomb_model),
                                           training_stage=1, observation=bomb["observation"],
                                           method="frozen conditional value action on native player state/input/inventory")
    arms = {name: Arm(policy, False, provenance)}
    if args.compare_no_bomb:
        no_bomb = DemoPolicy(policy_path, route_path, resync=not args.no_resync)
        control = {key: value for key, value in provenance.items() if key != "bomb_transfer"}
        control["prior"] = report["prior"]
        arms["alice_route"] = Arm(no_bomb, False, control)
    other_arms = load_arms([], args.ppo_policy, args.ppo_template)
    if arms.keys() & other_arms.keys():
        raise ValueError("A comparison arm reuses a reserved route name.")
    arms.update(other_arms)
    args.limit = args.limit or story.LIMITS[args.target]
    args.restart_hook = None
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    menu = GameOverMenu(*refs, args.menu_max_error) if all(refs) else None
    args.crossover = args.timing_policy = []
    specification = story.run_specification(args, arms)
    specification["runtime"]["source_sha256"][__name__] = digest(Path(__file__))
    specification["controller"].update(kind="frozen_demo_conditioned_neural_tracker", source="train-only Alice")
    if bomb is not None:
        from scripts import fit_randomized_bomb_value, train_native_bomb_timing

        for module in (fit_randomized_bomb_value, train_native_bomb_timing):
            specification["runtime"]["source_sha256"][module.__name__] = digest(Path(module.__file__))
    transition_archive = None
    if getattr(args, "record_transitions", False):
        from scripts.native_story_transitions import configure
        transition_archive = configure(args, specification)
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        result = story.evaluate(args, arms, specification, args.out / "report.json", menu, None,
                                transition_archive=transition_archive)
    print(json.dumps(dict(status=result["status"], summary=result.get("matched_summary"), waiting=result.get("waiting_for_fresh_credit"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--seed", type=int, default=721)
    training.add_argument("--epochs", type=int, default=12)
    evaluating = sub.add_parser("evaluate")
    evaluating.add_argument("--artifact", type=Path, required=True)
    evaluating.add_argument("--out", type=Path, required=True)
    evaluating.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    evaluating.add_argument("--target", choices=("stage2", "stage3", "ending"), default="stage2")
    evaluating.add_argument("--start-frame", type=int, default=30)
    evaluating.add_argument("--limit", type=int)
    evaluating.add_argument("--focus-wait", type=int, default=30)
    evaluating.add_argument("--seeds", nargs="+", type=int, default=[1901, 1902])
    evaluating.add_argument("--no-resync", action="store_true")
    evaluating.add_argument("--resume", action="store_true")
    evaluating.add_argument("--record-transitions", action="store_true")
    evaluating.add_argument("--transition-budget-mib", type=int, default=256)
    evaluating.add_argument("--transition-reserve-mib", type=int, default=2048)
    evaluating.add_argument("--bomb-model", type=Path, help="Frozen PPO397-conditioned randomized Bomb value model.")
    evaluating.add_argument("--compare-no-bomb", action="store_true")
    evaluating.add_argument("--ppo-policy", action="append", default=[], metavar="NAME=ARCHIVE")
    evaluating.add_argument("--ppo-template", type=Path)
    evaluating.add_argument("--game-over-default-reference", type=Path)
    evaluating.add_argument("--game-over-retry-reference", type=Path)
    evaluating.add_argument("--menu-crop", action="append", nargs=4, type=int)
    evaluating.add_argument("--menu-max-error", type=float, default=.035)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or not 0 <= args.seed < 2**63:
            parser.error("Use a positive epoch budget and a valid seed.")
        train(args)
    else:
        if (args.limit is not None and args.limit < 1 or args.start_frame < 1 or args.focus_wait < 0
                or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds)):
            parser.error("Use valid native evaluation limits and unique policy seeds.")
        refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
        if any(refs) and (not all(refs) or len(args.menu_crop) < 2) or not 0 < args.menu_max_error < .2:
            parser.error("Supply both verified Game Over screenshots and at least two row crops.")
        if args.compare_no_bomb and not args.bomb_model or bool(args.ppo_policy) != bool(args.ppo_template):
            parser.error("A Bomb ablation requires a value model and PPO comparisons require their template.")
        evaluate(args)


if __name__ == "__main__":
    main()
