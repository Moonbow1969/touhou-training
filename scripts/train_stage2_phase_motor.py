"""Distill the six measured Stage 2 boss phases from a single designated human training clear."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.distill_state_route import ART, PAIR, SCALE
from scripts.evaluate_demo_route_stage2_phase import NATIVE_STAGE2_IDS, stage2_knots
from scripts.stage2_phase_motor import PhaseAge, PhaseMotor
from scripts.train_demo_route import ALICE, features
from scripts.train_full_policy import publish_json


def references():
    artifact = json.loads((ART / "report.json").read_bytes())
    source = PAIR / f"{ALICE}-stage2.npz"
    if digest(ART / "route.npz") != artifact["route_sha256"] or digest(ART / "model.pt") != artifact["model_sha256"]:
        raise ValueError("The human reference or trained motor changed.")
    with np.load(ART / "route.npz", allow_pickle=False) as z:
        frame, pos, keys, bounds = [z[k] for k in ("frame_2", "positions_2", "keys_2", "bounds")]
    intervals, brackets = stage2_knots(source, artifact["train_stage_shards"]["2"], frame)
    pairs = [(size, -1 if spell is None else spell) for size, spell in NATIVE_STAGE2_IDS]
    spans = np.asarray([intervals[name] for name in NATIVE_STAGE2_IDS.values()])
    provenance = dict(training_replay=ALICE, paired_replay_sha256=digest(source), motor_sha256=artifact["model_sha256"],
                      route_sha256=artifact["route_sha256"], brackets=brackets, intervals=intervals)
    return frame, pos, keys, bounds, pairs, spans, provenance


def targets(ref, phase, age):
    frame, pos, keys, _, _, spans, _ = ref
    start, end = spans[phase].T
    t = np.minimum(start + age, end - 8)
    idx = np.clip(np.searchsorted(frame, t, side="right") - 1, 0, len(frame)-1)
    word = keys[idx, np.clip(t - frame[idx], 0, 3)]
    goal = np.column_stack([np.interp(t + 4, frame, column) for column in pos.T]).astype(np.float32)
    pref = np.column_stack([(word & mask) != 0 for mask in (8, 1)]).astype(np.float32)
    return goal, pref


def train(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ref = references()
    _, _, _, bounds, pairs, spans, provenance = ref
    torch.manual_seed(41301)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseMotor(len(pairs)).to(device)
    raw = torch.load(ART / "model.pt", map_location=device, weights_only=True)
    model.motor.load_state_dict(raw["state_dict"] if "state_dict" in raw else raw)
    model.motor.requires_grad_(False)
    model.bounds.copy_(torch.as_tensor(bounds))
    # Each distinct native tick is an interpolation of the one training replay.
    # Long native suffixes use the last in-phase key and position as the teacher.
    phase, age = np.meshgrid(np.arange(len(pairs)), np.arange(8193), indexing="ij")
    phase, age = phase.ravel(), age.ravel()
    pos, pref = targets(ref, phase, age)
    active = age < spans[phase, 1] - spans[phase, 0]
    group = [torch.as_tensor(np.flatnonzero(mask), device=device) for mask in (active, ~active)]
    data = [torch.as_tensor(v, device=device) for v in (phase, age, pos / SCALE, pref)]
    opt = torch.optim.AdamW([*model.temporal.parameters(), *model.embedding.parameters()], lr=.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=.00004)
    report = dict(status="training", provenance=provenance, device=str(device), seed=41301,
                  source_replays=1, selected_checkpoint="final fixed optimization step",
                  physical_ticks_in_verified_phases=int(active.sum()), phase_keys=pairs,
                  tail_label="Last in-phase training waypoint and preferences.", history=[])
    for step in range(args.steps):
        ids = torch.cat([g[torch.randint(len(g), (n,), device=device)] for g, n in zip(group, (320, 64))])
        phase_t, age_t, goal, pref_t = (v[ids] for v in data)
        pred = model.goals(phase_t, age_t)
        loss = 32 * F.smooth_l1_loss(pred[:, :2], goal, beta=.008) + .7 * F.binary_cross_entropy_with_logits(pred[:, 2:], pref_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        if step % 400 == 0:
            report["history"].append(dict(step=step, loss=float(loss.detach())))
            publish_json(args.out / "training.json", report)
    dest = args.out / "phase.pt"
    torch.save(dict(schema="th14-stage2-phase-motor-v1", phase_keys=pairs, state_dict=model.cpu().state_dict()), dest)
    report.update(status="complete", steps=args.steps, sha256=digest(dest))
    publish_json(args.out / "training.json", report)


@torch.inference_mode()
def audit(args):
    ref = references()
    pairs = ref[4]
    report = json.loads((args.out / "training.json").read_bytes())
    weights = args.out / "phase.pt"
    if report["status"] != "complete" or digest(weights) != report["sha256"]:
        raise ValueError("Unverified Stage 2 motor.")
    model = PhaseMotor(len(pairs)).eval()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu")["state_dict"])
    torch.set_num_threads(2)
    tape = {}
    for shard in args.shards:
        with np.load(shard, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata_json"]))
            c = {k: i for i, k in enumerate(metadata["native_field_order"])}
            b, actual, ctx, a = (z[k] for k in ("native_before", "action", "context", "native_after"))
        clock, sample = PhaseAge(), []
        for i, row in enumerate(b):
            key = (int(row[c["boss_max_hp"]]), int(row[c["spell_id"]]))
            idx = pairs.index(key) if (row[c["stage"]]==2 and row[c["snapshot_valid"]]==1 and row[c["boss_hp"]]>0 and key in pairs) else None
            elapsed = clock.read((1, int(row[c["stage"]]), idx), int(row[c["stage_frame"]]))
            if idx is not None:
                sample.append((i, idx, elapsed))
        if not sample:
            continue
        i, phase, age = np.asarray(sample).T
        goal, pref = targets(ref, phase, age)
        pos = ctx[i, :2] * SCALE
        teacher = model.motor(torch.tensor(features(pos, goal, pref[:, 0], pref[:, 1], ref[3])))[0].argmax(1).numpy()
        proposal = model(torch.tensor(phase), torch.tensor(age), torch.tensor(pos)).argmax(1).numpy()
        g = model.goals(torch.tensor(phase), torch.tensor(age)).numpy()[:, :2] * SCALE
        distance = np.linalg.norm(g - goal, axis=1)
        original = actual[i, 0] + 9 * actual[i, 1]
        hit = (b[i, c["player_state"]] == 1) & (a[i, c["player_state"]] == 4)
        def group(m):
            return dict(rows=int(m.sum()), executed_agreement=float((original[m] == teacher[m]).mean()),
                        student_agreement=float((proposal[m] == teacher[m]).mean()),
                        predicted_changes=int((original[m] != proposal[m]).sum()),
                        median_goal_error=float(np.median(distance[m])), p90_goal_error=float(np.quantile(distance[m], .9)))
        arm = metadata.get("trial", {})
        tape[shard.name] = dict(sha256=digest(shard), all=group(np.ones(len(i), bool)),
                                onsets=group(hit) if hit.any() else None, trial=arm)
    qualifies = len(tape) >= 2 and all(t["all"]["median_goal_error"] < 5
        and t["all"]["student_agreement"] >= .85
        and t["all"]["student_agreement"] - t["all"]["executed_agreement"] > .15 for t in tape.values())
    out = dict(model_sha256=digest(weights), tapes=tape, offline_pilot_gate=qualifies,
               pilot_gate="On both preregistered frozen-parent credits, median waypoint error below 5 units, teacher agreement >=85%, gain >15 percentage points.",
               metric="Agreement with the privileged phase-aligned teacher on fixed native observations. No candidate action executed.")
    publish_json(args.out / "offline.json", out)
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "audit"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=5200)
    p.add_argument("--shards", nargs="+", type=Path)
    args = p.parse_args()
    if args.mode == "train":
        train(args)
    elif args.shards:
        audit(args)
    else:
        p.error("A held native credit is required.")
