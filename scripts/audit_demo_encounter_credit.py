"""Audit a recorded multi-stage TH14 PhaseRoute credit without operating the game."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.evaluate_demo_route_phase import PhaseRoute, phase_knots
from scripts.evaluate_sft_native import load_arm
from scripts.evaluate_bomb_crossover import TEMPLATE
from scripts.train_demo_route import ALICE, SCALE
from scripts.train_full_policy import publish_json


def metric(values):
    if len(values) == 0:
        return None
    return dict(n=len(values), median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                max=float(np.max(values)))


def profile(actions, before, after, c, frame, time, err, death, bombs, report_order):
    field = lambda name: before[:, report_order[name]]
    hp = field("boss_hp")
    vulnerable = field("boss_vulnerable") == 1
    hp_after = after[:, report_order["boss_hp"]]
    valid_damage = vulnerable & (after[:, report_order["boss_vulnerable"]] == 1)
    for key in ("boss_pointer", "boss_max_hp", "spell_id"):
        valid_damage &= field(key) == after[:, report_order[key]]
    drop = hp - hp_after
    return dict(stage=int(field("stage")[0]), decisions=len(actions), native_frames=[int(frame[0]), int(frame[-1])],
                reference_frames=[int(time[0]), int(time[-1])], spell=int(field("spell_id")[0]),
                max_hp=int(field("boss_max_hp")[0]), hp_at_start=int(hp[0]), hp_at_min=int(hp.min()),
                verified_damage=int(drop[valid_damage & (drop > 0) & (drop <= 512)].sum()),
                vulnerable_decisions=int(vulnerable.sum()),
                firing_while_vulnerable=int(actions[vulnerable, 2].sum()),
                focus_decisions=int(actions[:, 1].sum()), fire_decisions=int(actions[:, 2].sum()),
                deaths=int(death.sum()), bombs_consumed=int(bombs.sum()),
                life_stock=[int(field("lives")[0]),int(after[-1,report_order["lives"]])],
                bomb_stock=[int(field("bombs")[0]),int(after[-1,report_order["bombs"]])],
                tracking_error=metric(err), median_live_projectile_count=float(np.median(c[:,9] * 512)))


def analyze(args):
    with np.load(args.credit, allow_pickle=False) as data:
        meta, result = (json.loads(str(data[key])) for key in ("metadata_json", "result_json"))
        before, after, actions, context, deaths, bombs = (data[key] for key in (
            "native_before", "native_after", "action", "context", "native_life_losses", "actual_bombs_consumed",
        ))
    report_order = {name: i for i, name in enumerate(meta["native_field_order"])}
    field = lambda name: before[:, report_order[name]]
    actor = PhaseRoute(args.artifact / "model.pt", args.artifact / "route.npz",
                       None, load_arm(args.shoot_policy), "phase-aware")
    training = json.loads((args.artifact / "report.json").read_text(encoding="utf-8"))
    actor.knots = phase_knots(args.data / f"{ALICE}-stage1.npz", training["train_stage_shards"]["1"], actor.route["frame_1"])
    env = SimpleNamespace(reset_count=1, snapshot=None)
    actor.environment = env
    refs = np.zeros(len(actions), dtype=int)
    for i, row in enumerate(before):
        stage, frame, pointer, hp, maximum, spell, vulnerability = (
            row[report_order[key]] for key in ("stage", "stage_frame", "boss_pointer", "boss_hp", "boss_max_hp",
                                              "spell_id", "boss_vulnerable")
        )
        boss = SimpleNamespace(hp=hp, max_hp=maximum, vulnerable=vulnerability == 1) if pointer > 0 else None
        env.snapshot = SimpleNamespace(stage=stage, boss=boss, spell_id=spell if spell >= 0 else None)
        refs[i] = actor.phase_time(int(frame), int(stage), bool(context[i, 8] > 0))
    target = np.zeros((len(actions), 2), dtype=np.float32)
    demo_counts = np.zeros(len(actions), dtype=float)
    sources = {}
    for st in np.unique(field("stage")):
        selected = field("stage") == st
        path = args.data / f"{ALICE}-stage{st}.npz"
        sources[int(st)] = digest(path)
        if sources[int(st)] != training["train_stage_shards"][str(st)]:
            raise ValueError("The observed route must be the hashed Alice train replay.")
        with np.load(path, allow_pickle=False) as shard:
            frame = shard["stage_frame"]
            demos = shard["context"][:, :2] * SCALE
            for axis in range(2):
                target[selected, axis] = np.interp(refs[selected], frame, demos[:, axis])
            demo_counts[selected] = np.interp(refs[selected], frame, shard["bullet_count"])
    position = context[:, :2] * SCALE
    error = np.linalg.norm(position - target, axis=1)
    frames, stages = field("stage_frame"), field("stage")
    raw_phases = before[:, [report_order[key] for key in ("stage","boss_pointer","boss_max_hp","spell_id")]]
    bounds = np.r_[0, np.flatnonzero(np.any(raw_phases[1:] != raw_phases[:-1], axis=1)) + 1, len(actions)]
    phases = [profile(actions[a:b], before[a:b], after[a:b], context[a:b], frames[a:b], refs[a:b],
                      error[a:b], deaths[a:b], bombs[a:b], report_order) for a,b in zip(bounds[:-1], bounds[1:])]
    hit = np.flatnonzero((field("player_state") != 4) & (after[:,report_order["player_state"]] == 4))
    lethal = []
    for event in np.flatnonzero(deaths):
        nearby = hit[(stages[hit] == stages[event]) & (hit <= event) & (frames[hit] >= frames[event] - 48)]
        origin = int(nearby[-1]) if len(nearby) else int(event)
        stage = int(stages[origin])
        lethal.append(dict(stage=stage, native_hit_frame=int(frames[origin]),
                           native_loss_frame=int(after[event,report_order["stage_frame"]]),
                           reference_frame=int(refs[origin]), track_error=float(error[origin]),
                           position=position[origin].tolist(), target=target[origin].tolist(),
                           life_stock=int(field("lives")[origin]), bomb_stock=int(field("bombs")[origin]),
                           live_bullet_count=float(context[origin,9]*512), route_bullet_count=float(demo_counts[origin]),
                           shoot=int(actions[origin,2]), focus=int(actions[origin,1]),
                           native_boss_hp=int(field("boss_hp")[origin]), raw_spell=int(field("spell_id")[origin])))
    summary = {}
    for stage in np.unique(stages):
        select = stages == stage
        row = int(np.flatnonzero(select)[0])
        early = select & (frames <= 320)
        summary[int(stage)] = dict(
            tracking_error=metric(error[select]), early_tracking_error=metric(error[early]),
            first_control_frame=int(frames[row]), first_control_native_position=position[row].tolist(),
            first_control_target_position=target[row].tolist(), deaths=int(deaths[select].sum()),
            bombs=int(bombs[select].sum()), absolute_drift_ref_time_max=int(np.max(np.abs(refs[select]-frames[select]))),
        )
    evidence = dict(credit_sha256=digest(args.credit), train_route_source_sha256=sources, trial=meta["trial"],
                    terminal=result["native_terminal"], stage_records=result["stage_records"], stage_summaries=summary,
                    true_native_phases=phases, lethal_hits=lethal, reconstructed_stage1_clock=actor.sessions)
    publish_json(args.out, evidence)
    print(json.dumps(dict(terminal=evidence["terminal"], per_stage=summary, lethal_hits=lethal)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("credit", "data", "artifact", "out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--shoot-policy", type=Path, default=TEMPLATE)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh audit output.")
    analyze(args)


if __name__ == "__main__":
    main()
