"""Replay the 12-tick contact model against recorded Alice decisions in TH14 Stages 1–2."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import evaluate_demo_route_phase as phase, evaluate_demo_route_stage2_phase as exact
from scripts import evaluate_phase_contact as route_audit, train_anticipatory_contact as contact
from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


STRENGTHS = (8, 16, 32, 64)
DEV_SEEDS = (2201, 2202)
FINAL_SEED = 2402
OPENING_LIMIT = 800
DEFAULT_MODEL = contact.local.DATA.parent / "th14-anticipatory-contact-12tick-261125/model.pt"
DEFAULT_TAPES = {
    2201: "th14-demo-alice-phase-2201-261125", 2202: "th14-demo-alice-phase-2202-261125",
    2402: "th14-demo-exact-stage2-2402-261125",
}


@dataclass
class Tape:
    path: Path
    arrays: dict
    names: dict
    metadata: dict
    route: object
    references: np.ndarray
    targets: np.ndarray
    buttons: np.ndarray
    logits: np.ndarray
    fields: dict

    def field(self, name, when="before"):
        return self.arrays[f"native_{when}"][:, self.names[name]]

    def sensor(self, row, name, column):
        index = self.fields.get(row)
        if index is None or self.arrays["sensor_status"][index, column] != 1:
            return None
        offsets = self.arrays[f"sensor_{name}_indptr"]
        return self.arrays[f"sensor_{name}"][offsets[index]:offsets[index + 1]]


def load_model(path):
    prior = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
    values = torch.load(path, map_location="cpu", weights_only=True)
    if (prior["status"] != "complete" or prior["model_sha256"] != digest(path)
            or values["schema"] != "th14-learned-contact-12tick-v1" or prior["horizon_native_ticks"] != 12):
        raise ValueError("Expected the trained, frozen, native 12-tick contact model.")
    model = contact.AnticipatoryContact().eval()
    model.load_state_dict(values["state_dict"], strict=True)
    movement = prior["dynamics"]["ReimuA"]
    return model, np.asarray(movement["speed_units_per_tick"], np.float32), np.asarray(movement["bounds"], np.float32)


@torch.inference_mode()
def read_tape(path, args):
    with np.load(path, allow_pickle=False) as tape:
        metadata = json.loads(tape["metadata_json"].item())
        names = {name: i for i, name in enumerate(metadata["native_field_order"])}
        arrays = {k: tape[k] for k in tape.files if not k.endswith("_json")}
    policy = metadata["actor"]
    task = metadata["task"]
    if (metadata["schema"] != "th14-stock-story-actor-transitions-v1" or task["difficulty"] != "Normal"
            or task["shot"] != "Reimu A" or task["frames_per_action"] != 4):
        raise ValueError("Expected a four-tick original-game TH14 story tape.")
    actor, source = route_audit.load_route(args.artifact, args.demo_stage1, "phase-aware")
    if policy["route_sha256"] != source["route_sha256"] or policy["model_sha256"] != source["model_sha256"]:
        raise ValueError("Route and actor fingerprints do not match the tape.")
    if "stage2_phase_intervals" in policy:
        symbols = SimpleNamespace(motion_spans=None, retain_recent_motion=False)
        stage2, _ = exact.stage2_knots(args.demo_stage2, source["train_stage_shards"]["2"], actor.route["frame_2"])
        if stage2 != policy["stage2_phase_intervals"]:
            raise ValueError("The 2402 Stage 2 timeline does not match the exact observed policy.")
        checkpoint, route_path = args.artifact / "model.pt", args.artifact / "route.npz"
        tracker = exact.ExactStage2Route(checkpoint, route_path, None, SimpleNamespace(policy=symbols), stage2)
        tracker.knots = actor.knots
        actor, module = tracker, exact
    elif policy.get("mode") == "phase-aware":
        module = phase
    else:
        raise ValueError("The tape does not come from a supported frozen phase runner.")
    if metadata["runtime"]["runner_sha256"] != digest(Path(module.__file__)):
        raise ValueError(f"The frozen prefix source fingerprint differs from the archived {module.__name__}.")
    actor.environment = SimpleNamespace(reset_count=1)
    n = len(arrays["action"])
    reference = np.full(n, -1, dtype=np.int32)
    target = np.zeros((n, 2), dtype=np.float32)
    buttons = np.zeros(n, dtype=np.uint16)
    xy = arrays["positions_before"][:, :2]
    for i, row in enumerate(arrays["native_before"]):
        get = lambda name: int(row[names[name]])
        stage = get("stage")
        if stage > 2:
            break
        boss = (SimpleNamespace(hp=get("boss_hp"), max_hp=get("boss_max_hp"), vulnerable=get("boss_vulnerable") == 1)
                if get("boss_pointer") >= 0 else None)
        spell = get("spell_id")
        actor.environment.snapshot = SimpleNamespace(boss=boss, stage=stage, spell_id=spell if spell >= 0 else None)
        elapsed = actor.phase_time(get("stage_frame"), stage, bool(arrays["context"][i, 8] > 0))
        frames, positions, keys = (actor.route[f"{name}_{stage}"] for name in ("frame", "positions", "keys"))
        reference[i] = elapsed
        target[i] = route_audit.sample_position(elapsed + 4, frames, positions)
        buttons[i] = route_audit.recorded_word(elapsed, frames, keys)
    evidence = route_audit.demo.features(xy, target, buttons & 8 != 0, buttons & 1 != 0, actor.route["bounds"])
    logits = actor.model(torch.from_numpy(evidence))[0].numpy()
    fields = {int(row): i for i, row in enumerate(arrays.get("sensor_decision_index", []))}
    return Tape(path, arrays, names, metadata, actor, reference, target, buttons, logits, fields)


@torch.inference_mode()
def learned_danger(model, bullets, context):
    b, c = torch.from_numpy(bullets), torch.from_numpy(context)
    rows = [model(b[i:i + 256], contact.local.local_context(c[i:i + 256], 0)).sigmoid()
            for i in range(0, len(b), 256)]
    return torch.cat(rows).numpy()


def choose(logits, danger, focus, strength):
    return (logits.reshape(-1, 2, 9)[np.arange(len(focus)), focus] - strength *
            danger[:, :18].reshape(-1, 2, 9)[np.arange(len(focus)), focus]).argmax(axis=1)


def step_positions(position, previous_focus, focus, move, speeds, bounds):
    v = speeds[previous_focus, focus, move] * contact.local._DIRECTIONS[move]
    path = (position + np.arange(5, dtype=np.float32)[:, None] * v).clip(*bounds)
    return path[:-1], path[-1]


def projectile_field(tape, row, use_extra):
    snapshot = tape.arrays["bullets"][row] * np.array(contact.local.BULLET_SCALE, dtype=np.float32)
    visible = (snapshot[:, 6] > 0) & (snapshot[:, 5] > 0) & (snapshot[:, 4] > 0)
    if use_extra:
        stored = tape.sensor(row, "projectiles", 0)
        if stored is not None and len(stored):
            matching = stored[:, 5] < len(snapshot)
            snapshot[stored[matching, 5].astype(int), :5] = stored[matching, :5]
            visible[stored[matching, 5].astype(int)] = True
            return np.concatenate((snapshot[visible, :5], stored[~matching, :5]))
    return snapshot[visible, :5]


def projected_gap(tape, row, player, use_extra):
    steps = np.arange(1, 5, dtype=np.float32)[:, None]
    projectiles = projectile_field(tape, row, use_extra)
    result = dict(projectile=None, body=None, laser=None)
    if len(projectiles):
        origin = projectiles[:, :2] + tape.arrays["positions_before"][row, :2]
        future = origin[None] + steps[..., None] * projectiles[None, :, 2:4]
        gaps = np.linalg.norm(player[:, None] - future, axis=-1) - projectiles[:, 4] - 1.
        result["projectile"] = float(gaps.min())
    if not use_extra:
        return result
    bodies = tape.sensor(row, "bodies", 1)
    if bodies is not None and len(bodies):
        rel = player[:, None] - (bodies[None, :, :2] + steps[..., None] * bodies[None, :, 2:4])
        cosine, sine = np.cos(bodies[:, 6]), np.sin(bodies[:, 6])
        x, y = rel[..., 0], rel[..., 1]
        local = np.stack((x * cosine + y * sine, -x * sine + y * cosine), axis=-1)
        centered = np.abs(local) - bodies[:, 4:6]
        rectangle = np.linalg.norm(np.maximum(centered, 0), axis=-1) + np.minimum(centered.max(axis=-1), 0)
        circle = np.linalg.norm(rel, axis=-1) - bodies[:, 4]
        result["body"] = float(np.where(bodies[:, 7] > 0, rectangle, circle).min() - 1)
    lasers = tape.sensor(row, "straight_lasers", 2)
    if lasers is not None and len(lasers):
        origin = lasers[None, :, :2] + steps[..., None] * lasers[None, :, 2:4]
        angle = lasers[:, 4] + steps * lasers[:, 5]
        along = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        rel = player[:, None] - origin
        target = np.clip((rel * along).sum(axis=-1), 0, lasers[:, 6])
        clearance = np.linalg.norm(rel - along * target[..., None], axis=-1) - lasers[:, 7] - 1.
        active = steps >= lasers[:, 8]
        if active.any():
            result["laser"] = float(clearance[active].min())
    return result


def nearest(gaps):
    return min((g for g in gaps.values() if g is not None), default=512.)


@torch.inference_mode()
def simulate(tape, first, hit_row, strength, model, speeds, bounds, use_extra):
    actions = tape.arrays["action"]
    position = tape.arrays["positions_before"][first, :2].copy()
    previous_focus = int(tape.arrays["context"][first, 2] > .5)
    lines = []
    stage_frames = tape.field("stage_frame")
    for row in range(first, hit_row + 1):
        bullet = tape.arrays["bullets"][row].copy()
        obs = tape.arrays["context"][row].copy()
        original_xy = tape.arrays["positions_before"][row, :2]
        bullet[:, :2] = (bullet[:, :2] + (original_xy - position) / np.array((384, 448), np.float32)).clip(-1, 1)
        obs[:2] = position / route_audit.demo.SCALE
        obs[2] = previous_focus
        focus, move = int(actions[row, 1]), int(actions[row, 0])
        if strength:
            word = int(tape.buttons[row])
            feature = route_audit.demo.features(position, tape.targets[row], np.array(bool(word & 8)),
                                                np.array(bool(word & 1)), tape.route.route["bounds"])
            logits = tape.route.model(torch.from_numpy(feature[None]))[0].numpy()
            scores = learned_danger(model, bullet[None], obs[None])
            move = int(choose(logits, scores, np.array([focus]), strength)[0])
        player, next_position = step_positions(position, previous_focus, focus, move, speeds, bounds)
        gaps = projected_gap(tape, row, player, use_extra)
        lines.append(dict(frame=int(stage_frames[row]), move=route_audit.MOVEMENTS[move], focus=focus,
                          raw=route_audit.MOVEMENTS[int(actions[row, 0])], minimum_margin=nearest(gaps),
                          margins=gaps, next_position=next_position.tolist(),
                          drift_from_logged_position=float(np.linalg.norm(next_position - tape.arrays["positions_after"][row, :2]))))
        position, previous_focus = next_position, focus
    return lines


def stage_results(tape, stage, model, speeds, bounds):
    actions, context = tape.arrays["action"], tape.arrays["context"]
    current, successor = tape.field("player_state"), tape.field("player_state", "after")
    frames = tape.field("stage_frame")
    ticks = tape.field("global_tick")
    same_stage = tape.field("stage") == stage
    valid = (same_stage & tape.arrays["four_tick_stage_transition"] & (current == 1)
             & (tape.field("previous_x") == 0) & (actions[:, 3] == 0) & (tape.arrays["actual_bombs_consumed"] == 0))
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError(f"No eligible decisions in Stage {stage}.")
    rank = tape.logits.argmax(axis=1)
    agree = (rank % 9 == actions[:, 0]) & (rank // 9 == actions[:, 1])
    if agree[ids].mean() < .995:
        raise ValueError(f"Stage {stage} recovered {agree[ids].mean():.1%} of recorded decisions in {tape.path.name}.")
    focus = actions[ids, 1]
    danger = learned_danger(model, tape.arrays["bullets"][ids], context[ids])
    focused = danger[:, :18].reshape(-1, 2, 9)[np.arange(len(ids)), focus]
    base = actions[ids, 0]
    baseline_danger = focused[np.arange(len(ids)), base]
    onsets = valid & (successor == 4)
    future_hit = np.zeros(len(ids), dtype=bool)
    for k, row in enumerate(ids):
        for target in range(row, min(row + 3, len(valid))):
            if not valid[target] or ticks[target] != ticks[row] + 4 * (target - row):
                break
            if onsets[target]:
                future_hit[k] = True
                break
    prior = (context[ids, 2] > .5).astype(int)
    position = tape.arrays["positions_before"][ids, :2]
    moves = contact.local._DIRECTIONS[None, :, :] * speeds[prior, focus, :, None]
    choice_position = (position[:, None] + 4 * moves).clip(*bounds)
    distance = np.linalg.norm(choice_position - tape.targets[ids, None], axis=-1)
    comparable = np.arange(len(ids))
    variants = {}
    for strength in STRENGTHS:
        proposed = choose(tape.logits[ids], danger, focus, strength)
        changed = proposed != base
        quieter = focused.max(axis=1) < .05
        highest = np.argsort(baseline_danger, kind="stable")[-max(1, len(ids)//10):]
        displacement = distance[comparable, proposed] - distance[comparable, base]
        variants[strength] = dict(changed=int(changed.sum()), changed_rate=float(changed.mean()),
                                  quiet_rows=int(quieter.sum()), quiet_changed_rate=route_audit.metric(changed, quieter),
                                  high_risk_decile_changed=float(changed[highest].mean()),
                                  changed_with_future_native_hit=int((changed & future_hit).sum()),
                                  track_error_delta_game_units=float(displacement.mean()),
                                  track_error_delta_on_changed=route_audit.metric(displacement, changed))

    events = []
    native_before, native_after = tape.arrays["native_before"], tape.arrays["native_after"]
    losses = tape.arrays["native_life_losses"]
    for hit in np.flatnonzero(onsets):
        this_fate = route_audit.fate(hit, native_before, native_after, tape.names, losses)
        event = dict(frame=int(frames[hit]), row=int(hit), fate=this_fate,
                     opening=stage == 2 and int(frames[hit]) <= OPENING_LIMIT,
                     resources=dict(lives=int(tape.field("lives")[hit]), bombs=int(tape.field("bombs")[hit])), leads=[])
        for start in range(max(0, hit - 2), hit + 1):
            if not valid[start:hit + 1].all() or ticks[hit] - ticks[start] != 4 * (hit - start):
                continue
            geometry = {}
            for extra in ([False, True] if tape.fields else [False]):
                fact = simulate(tape, start, hit, 0, model, speeds, bounds, extra)
                alternatives = {}
                for strength in STRENGTHS:
                    policy = simulate(tape, start, hit, strength, model, speeds, bounds, extra)
                    avoid = (fact[-1]["minimum_margin"] <= 0 and all(step["minimum_margin"] > 0 for step in policy))
                    alternatives[strength] = dict(modeled_avoided=avoid,
                                                   first_changed=next((row["frame"] for row in policy if row["move"] != row["raw"]), None),
                                                   steps=policy)
                geometry["with_raw_sensors" if extra else "actor_projectiles"] = dict(factual=fact, variants=alternatives)
            event["leads"].append(dict(ticks_before_hit=int(ticks[hit] - ticks[start]), geometry=geometry))
        events.append(event)
    return dict(observations=len(ids), route_action_agreement=float(agree[ids].mean()), hit_onsets=len(events),
                future_hit_windows=int(future_hit.sum()), actual_action_ap=contact.ap(future_hit, baseline_danger),
                same_focus_mean_ap=contact.ap(future_hit, focused.mean(axis=1)),
                agnostic_ap=contact.ap(future_hit, danger[:, -1]), variants=variants, events=events)


def openings(results, strength, label):
    observations = []
    for event in results[2]["events"]:
        if not (event["opening"] and event["fate"] == "life_lost"):
            continue
        tested = [lead["geometry"][label] for lead in event["leads"] if label in lead["geometry"]]
        observations.append(dict(frame=event["frame"], matching_geometry_has_contact=any(
            sample["factual"][-1]["minimum_margin"] <= 0 for sample in tested),
            modeled_avoided=any(sample["variants"][strength]["modeled_avoided"] for sample in tested)))
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--artifact", type=Path, default=route_audit.DEFAULT_ROUTE)
    parser.add_argument("--demo-stage1", type=Path, default=route_audit.DEFAULT_DEMO)
    parser.add_argument("--demo-stage2", type=Path, default=route_audit.DEFAULT_DEMO.with_name("th14_ud0613-stage2.npz"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, speeds, bounds = load_model(args.model)
    paths = {seed: next((contact.local.DATA / folder / "transition-shards").glob("credit-*.npz"))
             for seed, folder in DEFAULT_TAPES.items()}
    outcome = {}
    for seed in DEV_SEEDS:
        tape = read_tape(paths[seed], args)
        outcome[seed] = {stage: stage_results(tape, stage, model, speeds, bounds) for stage in (1, 2)}
    candidates = []
    for strength in STRENGTHS:
        events = [item for seed in DEV_SEEDS for item in openings(outcome[seed], strength, "actor_projectiles")]
        drift_limit = all(outcome[seed][stage]["variants"][strength]["track_error_delta_game_units"] <= .4
                          and outcome[seed][stage]["variants"][strength]["quiet_changed_rate"] <= .005
                          for seed in DEV_SEEDS for stage in (1, 2))
        candidates.append(dict(strength=strength, openings=events, drift_limit_pass=drift_limit,
                               eligible=bool(events) and all(item["modeled_avoided"] for item in events) and drift_limit))
    prospective = next((row["strength"] for row in candidates if row["eligible"]), None)
    tape = read_tape(paths[FINAL_SEED], args)
    outcome[FINAL_SEED] = {stage: stage_results(tape, stage, model, speeds, bounds) for stage in (1, 2)}
    final = {strength: openings(outcome[FINAL_SEED], strength, "with_raw_sensors") for strength in STRENGTHS}
    qualifies = (prospective is not None and bool(final[prospective])
                  and all(item["modeled_avoided"] for item in final[prospective]))
    report = dict(status="complete", model_sha256=digest(args.model), runner_sha256=digest(Path(__file__)),
                  tapes={seed: dict(file=path.name, sha256=digest(path)) for seed, path in paths.items()},
                  strength_selected_on_development=prospective, development_screen=candidates,
                  final_stage2_opening_sensor_estimates=final, recommended_native_pilot_strength=prospective if qualifies else None,
                  stages=outcome,
                  scope="Fixed observed focus and exogenous world fields. Latent approach starts 0/4/8 ticks before observed hits and re-queries the trained contact model under each shifted observation. Weights were never fit on 2402.",
                  caveats="Continuous shot response, target re-aiming, curved lasers, missing new bullets and live movement collision response cannot be recovered. Raw sensor estimates are retrospective only; sensor data never enter the actor. Prospectively screened coefficient is chosen before reading final outcomes.")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "report.json"
    if path.exists():
        raise FileExistsError(path)
    publish_json(path, report)
    print(json.dumps(dict(dev_strength=prospective, final_opening=final, recommended_native_pilot_strength=report["recommended_native_pilot_strength"])))


if __name__ == "__main__":
    main()
