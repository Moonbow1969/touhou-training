"""Audit learned contact-score reranking of Alice's frozen TH14 phase tracker."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import evaluate_demo_route_phase as phase
from scripts import train_demo_route as demo
from scripts import train_local_avoidance as local
from scripts.checkpoint_storage import digest
from scripts.diagnose_demo_route import recorded_word, sample_position
from scripts.train_full_policy import publish_json


DEFAULT_ROUTE = local.DATA / "th14-demo-alice-route-261125"
DEFAULT_RISK = local.DATA.parent / "th14-local-avoidance-offline-matched-261125/offline-model.pt"
DEFAULT_DEMO = local.REPLAYS / f"{demo.ALICE}-stage1.npz"
MOVEMENTS = ("up_left", "up", "up_right", "left", "idle", "right", "down_left", "down", "down_right")
STRENGTHS = (.5, 1., 2., 4., 8., 16., 32., 64., 128.)


def load_contact(path):
    report = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    artifact = report["model_artifact"]
    if (report["status"] != "complete" or artifact != dict(file=path.name, sha256=digest(path))
            or not report["promotion_gate"]["criteria"]["native_ap_above_agnostic_and_focus_mean"]
            or payload["schema"] != "th14-neural-avoidance-v1"):
        raise ValueError("The risk head needs a validated and fingerprinted offline contact report.")
    model = local.LocalAvoidance().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    dynamics = report["fitted_player_dynamics"]["ReimuA"]
    return (model, torch.tensor(dynamics["speed_units_per_tick"], dtype=torch.float32),
            torch.tensor(dynamics["bounds"], dtype=torch.float32))


def rerank(logits, danger, strength):
    """Keep the chosen focus and select only movement from learned risk-adjusted logits."""
    rows = torch.arange(len(logits), device=logits.device)
    chosen = logits.argmax(dim=-1)
    focus = chosen // 9
    fixed_focus = logits.reshape(-1, 2, 9)[rows, focus]
    cost = danger[rows, focus]
    return (fixed_focus - strength * cost).argmax(dim=-1), focus, chosen % 9, cost


def load_route(artifact, source, mode):
    checkpoint, route = artifact / "model.pt", artifact / "route.npz"
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    if (report["schema"] != demo.SCHEMA or digest(checkpoint) != report["model_sha256"]
            or digest(route) != report["route_sha256"]):
        raise ValueError("Mismatched Alice trajectory and follower weights.")
    tracker = phase.PhaseRoute(checkpoint, route, None,
                               SimpleNamespace(policy=SimpleNamespace(motion_spans=None, retain_recent_motion=False)),
                               mode)
    tracker.knots = phase.phase_knots(source, report["train_stage_shards"]["1"], tracker.route["frame_1"])
    return tracker, report


def fate(index, before, after, names, losses):
    clock, player, stage = (names[key] for key in ("global_tick", "player_state", "stage"))
    last_tick = before[index, clock] + 48
    for future in range(index, len(before)):
        if before[future, clock] > last_tick or before[future, stage] != before[index, stage]:
            return "unresolved"
        if losses[future] > 0:
            return "life_lost"
        if future > index and after[future, player] == 1:
            return "returned_to_active"
    return "unresolved"


@torch.inference_mode()
def tape(path, tracker, report, risk_model, speeds, bounds):
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(source["metadata_json"].item())
        task, policy = meta["task"], meta["actor"]
        if (meta["schema"] != "th14-stock-story-actor-transitions-v1" or task["frames_per_action"] != 4
                or task["shot"] != "Reimu A" or task["difficulty"] != "Normal"
                or policy["model_sha256"] != report["model_sha256"] or policy["route_sha256"] != report["route_sha256"]):
            raise ValueError("The tape needs verified four-tick Normal Reimu A Alice policy data.")
        if tracker.mode == "phase-aware" and (policy.get("mode") != "phase-aware"
                or meta["runtime"]["runner_sha256"] != digest(Path(phase.__file__))):
            raise ValueError("Phase-aware mode needs the tape of this fingerprinted frozen phase runner.")
        if tracker.mode == "shoot-only" and policy.get("mode") == "phase-aware":
            raise ValueError("Do not replay phase-aware tape with encounter-only alignment.")
        names = {name: i for i, name in enumerate(meta["native_field_order"])}
        before, after = source["native_before"], source["native_after"]
        context, bullets = source["context"], source["bullets"]
        action, losses = source["action"], source["native_life_losses"]
        valid = (source["four_tick_stage_transition"] & (before[:, names["stage"]] == 1)
                 & (before[:, names["player_state"]] == 1) & (action[:, 3] == 0)
                 & (before[:, names["previous_x"]] == 0) & (source["actual_bombs_consumed"] == 0))
        valid[-1] = False
        valid[:-1] &= before[1:, names["global_tick"]] == after[:-1, names["global_tick"]]

    tracker.environment = SimpleNamespace(reset_count=1)
    position = context[:, :2] * demo.SCALE
    inputs, targets, reference = [], [], []
    # Every Stage 1 row advances alignment, including rows outside the scoring mask.
    for i, n in enumerate(before):
        stage, frame = int(n[names["stage"]]), int(n[names["stage_frame"]])
        if stage != 1:
            break
        phase_hp = int(n[names["boss_max_hp"]])
        boss = SimpleNamespace(max_hp=phase_hp) if phase_hp >= 0 else None
        spell = int(n[names["spell_id"]])
        tracker.environment.snapshot = SimpleNamespace(boss=boss, stage=stage, spell_id=spell if spell >= 0 else None)
        route_frame = tracker.phase_time(frame, stage, bool(context[i, 8] > 0))
        frames, positions, keys = (tracker.route[f"{field}_{stage}"] for field in ("frame", "positions", "keys"))
        target = sample_position(route_frame + local.TICKS, frames, positions)
        word = recorded_word(route_frame, frames, keys)
        evidence = demo.features(position[i], target, np.array(bool(word & 8)), np.array(bool(word & 1)),
                                 tracker.route["bounds"])
        inputs.append(evidence)
        targets.append(target)
        reference.append(route_frame)
    inputs, targets, reference = np.stack(inputs), np.stack(targets), np.asarray(reference)
    logits, _ = tracker.model(torch.from_numpy(inputs))
    tracked_move = logits.argmax(1).numpy() % 9
    tracked_focus = logits.argmax(1).numpy() // 9
    verified = (tracked_move == action[:len(logits), 0]) & (tracked_focus == action[:len(logits), 1])
    indices = np.flatnonzero(valid)
    if not len(indices) or not len(logits) or int(indices.max()) >= len(logits):
        raise ValueError("The tape lacks playable Stage 1 four-tick observations.")

    risk, margins, deltas = [], [], []
    for selected in np.array_split(indices, (len(indices) + 255) // 256):
        b, c, b_after, c_after = (torch.from_numpy(array) for array in
                                  (bullets[selected], context[selected], bullets[selected + 1], context[selected + 1]))
        risk.append(risk_model(b, local.local_context(c, 0))[0].sigmoid())
        margin, delta = local.modeled_contacts(b, c, b_after, c_after, speeds, bounds)
        margins.append(margin)
        deltas.append(delta)
    onset = after[indices, names["player_state"]] == 4
    fates = {idx: fate(idx, before, after, names, losses) for idx in indices[onset].tolist()}
    return dict(indices=indices, clock=before[indices, names["stage_frame"]], route_frames=reference[indices],
                agreement=float(verified.mean()), eligible_agreement=float(verified[indices].mean()),
                mismatches=np.flatnonzero(~verified).tolist(), recorded_actions=action[indices],
                recorded_next=position[indices + 1], current_position=position[indices],
                onset=onset, fates=fates, target=targets[indices], logits=logits[indices],
                risk=torch.cat(risk), margin=torch.cat(margins).numpy(), delta=torch.cat(deltas).numpy(),
                bounds=bounds.numpy(), native_rows=len(before), phase_sessions=tracker.sessions, task=task)


def metric(condition, selector):
    values = np.asarray(condition)[selector]
    return float(values.mean()) if values.size else None


def analyze(data, strength):
    proposed, focus, base, risk = rerank(data["logits"], data["risk"], strength)
    proposed, focus, base, risk = (arr.numpy() for arr in (proposed, focus, base, risk))
    timestamps = data["clock"]
    rows = np.arange(len(timestamps))
    changed = proposed != base
    margin = data["margin"][rows, focus]
    actual = data["recorded_actions"][:, 0]
    actual_margin = margin[rows, actual]
    old, new = margin[rows, base], margin[rows, proposed]
    old_risk, new_risk = risk[rows, base], risk[rows, proposed]
    position, target = data["current_position"], data["target"]
    lowest, highest = data["bounds"]
    choices = position[:, None] + data["delta"][rows, focus]
    track = np.linalg.norm(choices - target[:, None], axis=-1)
    one_step_track_delta = track[rows, proposed] - track[rows, base]
    edge_room = lambda p: np.minimum(p - lowest, highest - p).min(axis=-1)
    edge = edge_room(position) <= 20
    old_edge = edge_room(choices[rows, base])
    new_edge = edge_room(choices[rows, proposed])
    quiet = margin.min(axis=1) > 8
    risk_order = np.argsort(old_risk, kind="stable")
    bins = []
    for group in np.array_split(risk_order, min(len(risk_order), 10)):
        bins.append(dict(rows=len(group), initial_selected_risk_min=float(old_risk[group].min()),
                         initial_selected_risk_max=float(old_risk[group].max()),
                         changed_action_ratio=metric(changed, group),
                         projected_contact_ratio_base=float((old[group] <= 0).mean()),
                         projected_contact_ratio_candidate=float((new[group] <= 0).mean())))
    hits = []
    for row in np.flatnonzero(data["onset"]):
        hits.append(dict(frame=int(timestamps[row]), route_frame=int(data["route_frames"][row]),
                         outcome=data["fates"][int(data["indices"][row])], focus=int(focus[row]),
                         logged_move=MOVEMENTS[actual[row]], base_move=MOVEMENTS[base[row]],
                         new_move=MOVEMENTS[proposed[row]], changed=bool(changed[row]),
                         base_risk=float(old_risk[row]), new_risk=float(new_risk[row]),
                         logged_margin=float(actual_margin[row]), base_margin=float(old[row]),
                         new_margin=float(new[row]), extra_projected_track_error=float(one_step_track_delta[row])))
    hard_by_move = {}
    for i, name in enumerate(MOVEMENTS):
        hard_by_move[name] = dict(all_rows_if_forced_contact_rate=float((margin[:, i] <= 0).mean()),
                                  base_selected=int((base == i).sum()), candidate_selected=int((proposed == i).sum()),
                                  base_contacts=int(((base == i) & (old <= 0)).sum()),
                                  candidate_contacts=int(((proposed == i) & (new <= 0)).sum()))
    rescued = int(((old <= 0) & (new > 0)).sum())
    introduced = int(((old > 0) & (new <= 0)).sum())
    fatalities = [row for row in hits if row["outcome"] == "life_lost"]
    saved = sum(row["base_margin"] <= 0 < row["new_margin"] for row in fatalities)
    promoted = dict(
        tape_movement_focus_agreement_at_least_995=data["eligible_agreement"] >= .995,
        modeled_contacts_reduce_at_least_15_pct=int((new <= 0).sum()) <= .85 * int((old <= 0).sum()),
        at_least_three_avoided_and_more_than_introduced=rescued >= 3 and rescued > introduced,
        at_least_one_native_fatal_onset_modeled_to_avoid=saved >= 1,
        quiet_change_rate_at_most_005=bool(quiet.any()) and float(changed[quiet].mean()) <= .005,
        mean_tracking_drift_at_most_04=float(one_step_track_delta.mean()) <= .4,
    )
    return dict(strength=strength, rows=len(rows), promotion_gate=dict(passed=all(promoted.values()), criteria=promoted),
                changed_actions=int(changed.sum()), changed_action_rate=float(changed.mean()),
                changed_action_by_baseline_risk_decile=bins, projected_contact_by_direction=hard_by_move,
                projected_base_contacts=int((old <= 0).sum()), projected_candidate_contacts=int((new <= 0).sum()),
                projected_contact_removed=rescued, projected_contact_introduced=introduced,
                life_costing_native_hit_onsets=len(fatalities), modeled_avoided_onsets_costing_a_life=saved,
                all_native_hit_onsets=hits, boost_away_from_all_modeled_hazards_rows=int(quiet.sum()),
                quiet_changed_action_rate=metric(changed, quiet),
                track=dict(native_factual_error=float(np.linalg.norm(data["recorded_next"] - target, axis=1).mean()),
                           rollout_base_error=float(track[rows, base].mean()),
                           rollout_candidate_error=float(track[rows, proposed].mean()),
                           mean_one_step_delta=float(one_step_track_delta.mean()),
                           mean_one_step_delta_changed=metric(one_step_track_delta, changed),
                           p95_one_step_delta_changed=float(np.percentile(one_step_track_delta[changed], 95))
                           if changed.any() else None,
                           p95_one_step_delta=float(np.percentile(one_step_track_delta, 95))),
                near_edge=dict(rows=int(edge.sum()), changed_rows=int((edge & changed).sum()),
                               candidate_projected_closer=int((edge & changed & (new_edge < old_edge - .5)).sum()),
                               candidate_projected_further=int((edge & changed & (new_edge > old_edge + .5)).sum()),
                               mean_projected_edge_room_change=metric(new_edge - old_edge, edge),
                               base_projected_contacts=int((edge & (old <= 0)).sum()),
                               candidate_projected_contacts=int((edge & (new <= 0)).sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--demo-stage1", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--risk-model", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--mode", choices=("phase-aware", "shoot-only"), default="phase-aware")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    tracker, frozen = load_route(args.artifact, args.demo_stage1, args.mode)
    contact, speeds, bounds = load_contact(args.risk_model)
    data = tape(args.tape, tracker, frozen, contact, speeds, bounds)
    variants = [analyze(data, strength) for strength in STRENGTHS]
    selection = next((row["strength"] for row in variants if row["promotion_gate"]["passed"]), None)
    report = dict(status="complete", mode=args.mode, policy_source_sha256=digest(Path(phase.__file__)),
                  analysis_source_sha256=digest(Path(__file__)),
                  tape=dict(file=args.tape.name, sha256=digest(args.tape), total_rows=data["native_rows"],
                            eligible_exact_stage1_observations=len(data["indices"]),
                            logged_movement_focus_agreement_all_stage1=data["agreement"],
                            logged_movement_focus_agreement_eligible=data["eligible_agreement"],
                            logit_reconstruction_mismatches=data["mismatches"], task=data["task"]),
                  contact_head_sha256=digest(args.risk_model), route_sha256=frozen["route_sha256"],
                  route_model_sha256=frozen["model_sha256"], variants=variants,
                  proposed_native_strength=selection if args.mode == "phase-aware" else None,
                  method="Within the original chosen focus, subtract the frozen neural contact score times a fixed coefficient from Alice's trained movement logits.",
                  limitations="Projected contact and track drift are four-tick counterfactuals on recorded exogenous fields with sampled nearest bullets and a training-time circular contact prior. Affected aimed bullets, native lasers and body collisions were not replayed. This is policy development on the recorded credit.")
    args.out.mkdir(parents=True, exist_ok=False)
    publish_json(args.out / "report.json", report)
    print(json.dumps(dict(rows=len(data["indices"]), alignment=data["eligible_agreement"],
                          proposed_native_strength=report["proposed_native_strength"],
                          variants=[{k: v[k] for k in ("strength", "changed_actions", "projected_base_contacts",
                                    "projected_candidate_contacts", "modeled_avoided_onsets_costing_a_life")}
                                    for v in variants])), flush=True)


if __name__ == "__main__":
    main()
