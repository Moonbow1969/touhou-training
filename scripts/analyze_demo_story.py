"""Reconstruct native route decisions and phase drift from a recorded TH14 credit."""

import argparse
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
from scripts.checkpoint_storage import digest
from scripts.train_demo_route import DemoPolicy, SCALE
from scripts.train_full_policy import publish_json


def stats(values):
    return dict(n=int(len(values)), median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                max=float(values.max()))


def coarse_field(rows, player):
    coords = rows[:, :2] * [384., 448.] + player
    mask = (rows[:, 6] > 0) & (rows[:, 5] > 0)
    return np.histogram2d(*coords[mask].T, bins=(6, 7), range=((-192, 192), (0, 448)))[0]


def analyze(args):
    with np.load(args.credit, allow_pickle=False) as z:
        meta, result = (json.loads(str(z[name])) for name in ("metadata_json", "result_json"))
        before, after, actions, context, bullets, losses, consumed = (z[name] for name in (
            "native_before", "native_after", "action", "context", "bullets", "native_life_losses", "actual_bombs_consumed"
        ))
    with np.load(args.demo, allow_pickle=False) as z:
        demo_frames, demo_context, demo_counts, demo_bullets, demo_keys = (z[name] for name in (
            "stage_frame", "context", "bullet_count", "bullets", "expert_action_chunk"
        ))
    columns = {name: i for i, name in enumerate(meta["native_field_order"])}
    field = lambda name: before[:, columns[name]]
    frames = field("stage_frame")
    if len(actions) != result["decision_count"] or set(field("stage")) != {1}:
        raise ValueError("Expected a complete recorded Stage 1 credit.")
    actor = DemoPolicy(args.artifact / "model.pt", args.artifact / "route.npz")
    actor.environment = SimpleNamespace(reset_count=1)
    virtual = np.array([actor.phase_time(int(f), 1, bool(obs[8] > 0)) for f, obs in zip(frames, context)])
    reference_idx = np.clip(np.searchsorted(demo_frames, virtual, side="right") - 1, 0, len(demo_frames) - 1)
    reference_key = demo_keys[reference_idx, np.clip(virtual - demo_frames[reference_idx], 0, 3)]
    reference_pos = np.stack([np.interp(virtual, demo_frames, a * s) for a, s in zip(demo_context[:, :2].T, SCALE)], 1)
    positions = context[:, :2] * SCALE
    error = np.linalg.norm(positions - reference_pos, axis=1)
    phase_keys = before[:, [columns[name] for name in ("boss_pointer", "boss_max_hp", "spell_id")]]
    indices = np.r_[0, np.flatnonzero(np.any(phase_keys[1:] != phase_keys[:-1], 1)) + 1, len(actions)]
    hit_rows = np.flatnonzero((field("player_state") != 4) & (after[:, columns["player_state"]] == 4))
    phases, hits = [], []
    for start, stop in zip(indices[:-1], indices[1:]):
        span = slice(start, stop)
        active = field("boss_vulnerable")[span] == 1
        damage = field("boss_hp")[span] - after[span, columns["boss_hp"]]
        stable = active & (after[span, columns["boss_vulnerable"]] == 1) & (
            after[span, columns["boss_pointer"]] == field("boss_pointer")[span]) & (
            after[span, columns["spell_id"]] == field("spell_id")[span]) & (
            after[span, columns["boss_max_hp"]] == field("boss_max_hp")[span])
        expert_firing = (reference_key[span] & 1) != 0
        actual_active_counts = np.rint(context[span, 9] * 512).astype(int)
        phases.append(dict(
            native_frame_interval=[int(frames[start]), int(frames[stop - 1])],
            reference_frame_interval=[int(virtual[start]), int(virtual[stop - 1])],
            decisions=int(stop-start), max_hp=int(field("boss_max_hp")[start]), spell_id=int(field("spell_id")[start]),
            boss_vulnerable_decisions=int(active.sum()), fire_requests=int(actions[span, 2].sum()),
            fire_vulnerable=int(actions[span, 2][active].sum()), reference_fire=int(expert_firing.sum()),
            focus_agreement=int((actions[span, 1] == ((reference_key[span] & 8) != 0)).sum()),
            hp_start=int(field("boss_hp")[start]), hp_min=int(field("boss_hp")[span].min()),
            verified_damage=int(damage[stable & (damage > 0) & (damage <= 512)].sum()),
            median_native_projectiles=float(np.median(actual_active_counts)),
            median_reference_projectiles=float(np.median(demo_counts[reference_idx[span]])),
            median_absolute_projectile_count_diff=float(np.median(np.abs(actual_active_counts - demo_counts[reference_idx[span]]))),
            vulnerable_within_24x=int((active & (np.abs(context[span, 6]*384) <= 24)).sum()),
            deaths=int(losses[span].sum()), bombs_consumed=int(consumed[span].sum()),
            player_tracking_error=stats(error[span]),
        ))
    for death in np.flatnonzero(losses):
        hits_before = hit_rows[(hit_rows <= death) & (frames[hit_rows] >= frames[death] - 40)]
        row = int(hits_before[-1]) if len(hits_before) else int(death)
        expert = int(reference_idx[row])
        world = coarse_field(bullets[row], positions[row])
        demo_world = coarse_field(demo_bullets[expert], demo_context[expert, :2] * SCALE)
        field_divergence = np.abs(world / max(1, world.sum()) - demo_world / max(1, demo_world.sum())).sum() / 2
        hits.append(dict(native_frame=int(frames[row]), consumed_life_at=int(after[death, columns["stage_frame"]]),
                         reference_frame=int(virtual[row]), had_observed_hit_state=bool(len(hits_before)),
                         tracking_error=float(error[row]), player_xy=positions[row].tolist(), target_xy=reference_pos[row].tolist(),
                         spell_id=int(field("spell_id")[row]), boss_hp=int(field("boss_hp")[row]),
                         bomb_stock=int(field("bombs")[row]), life_stock=int(field("lives")[row]),
                         fire_requested=int(actions[row,2]), power=float(context[row,5]*400),
                         native_projectiles=int(round(context[row,9]*512)), reference_projectiles=int(demo_counts[expert]),
                         active_nearest_projectile_world_cell_tv=float(field_divergence)))
    report = dict(credit_sha256=digest(args.credit), demo_sha256=digest(args.demo), trial=meta["trial"],
                  native_terminal=result["native_terminal"], tick_exact=int((after[:, 1]-before[:, 1] == 4).sum()),
                  decision_count=len(actions), phase_offsets=[
                      [int(frames[i]), int(virtual[i]), int(virtual[i]-frames[i])]
                      for i in np.r_[0, np.flatnonzero(np.diff(virtual-frames) != 0)+1]
                  ], overall_tracking_error=stats(error), phases=phases, lethal_hits=hits,
                  projectile_metric_note="Native count can saturate at 512. World-cell divergence uses the nearest 96 observable projectile rows, not the complete field.",
                  diagnosis="Final native combat exceeded the expert's firing window; close route tracking at all lethal damage onsets does not establish that same-time projectile fields agree.")
    publish_json(args.out, report)
    print(json.dumps(dict(lethal_hits=hits, phases=phases[-4:])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("credit","demo","artifact","out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh report path.")
    analyze(args)


if __name__ == "__main__":
    main()
