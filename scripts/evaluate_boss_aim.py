import argparse
from pathlib import Path
from statistics import median
from traceback import format_exc

import numpy as np
from stable_baselines3 import PPO
import torch
from torch import nn

from scripts.evaluate_stage_policies import read_policy, stage_action
from scripts.probe_stage_risk import player_radius
from scripts.record_run import write_json
from scripts.stage_policy import NormalStageOne
from scripts.stage_risk import action_clearances, body_clearances, laser_clearances, player_paths, suggest_dodge
from scripts.train_stage_dodge import learned_action
from scripts.world_shadow import WorldShadow


def aim_geometry(observation, action, clearances, boss):
    move, focus = map(int, action[:2])
    ends = player_paths(observation, 4)[focus, :, -1]
    lateral = np.abs(ends[:, 0] - boss.x)
    too_close = np.maximum(boss.y + 65 - ends[:, 1], 0)
    costs = lateral + 2 * too_close
    safety_minimum = max(4.0, min(10.0, float(clearances[focus, move]) - 2.0))
    safety_delta = clearances[focus] - safety_minimum
    return move, costs, safety_delta


def aim_features(observation, action, clearances, boss):
    if boss is None or not boss.vulnerable or not action[2]:
        return None, int(action[0])
    move, costs, safety_delta = aim_geometry(observation, action, clearances, boss)
    acceptable = safety_delta >= 0
    best = int(np.where(acceptable, costs, np.inf).argmin())
    if not acceptable[best] or costs[best] + 2 >= costs[move]:
        best = move
    features = np.stack((np.clip((costs[move] - costs) / 40, -2, 2),
                         np.clip(safety_delta / 12, -2, 2), np.arange(9) == move,
                         np.full_like(costs, np.clip(costs[move] / 256, 0, 2))), axis=-1)
    return features.astype(np.float32), best


class AimHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))

    def forward(self, features):
        return self.score(features).squeeze(-1)


def mask_aim_logits(logits, features):
    permitted = (features[..., 1] >= 0) | (features[..., 2] > 0)
    return logits.masked_fill(~permitted, -torch.inf)


def same_phase(before, after):
    return (before.stage == after.stage and after.boss is not None and before.boss.pointer == after.boss.pointer
            and before.boss.max_hp == after.boss.max_hp and before.spell_id == after.spell_id)


def trial(stage, policy, head, aim, emit, examples, student, body_risk, laser_risk, emergency_bomb, shadow=None):
    observation, _ = stage.reset()
    start_frame = stage.last_frame
    radius = player_radius(stage)
    vulnerable_frames = 0
    aligned_frames = 0
    firing_frames = 0
    damage = 0
    excluded_hp_drop = 0
    interventions = 0
    lateral_sum = 0.0
    phases = {}
    samples = []
    impacts = []
    protective_bombs = 0
    if shadow is not None:
        shadow.reset()
    while True:
        old_frame = stage.last_frame
        before = stage.snapshot
        base = stage_action(policy, stage, observation)
        projectile_risk = action_clearances(observation, radius)
        hitboxes = stage.reader.read_enemy_hitboxes()
        enemy_risk = body_clearances(observation, radius, hitboxes)
        segments = stage.reader.read_straight_lasers()
        beam_risk = laser_clearances(observation, radius, segments)
        clearance = np.minimum(projectile_risk, enemy_risk) if body_risk else projectile_risk
        if laser_risk:
            clearance = np.minimum(clearance, beam_risk)
        candidate = (suggest_dodge(base, clearance)[0] if head is None else learned_action(head, base, clearance))
        features, target = aim_features(observation, candidate, clearance, before.boss)
        correction = target != candidate[0]
        if features is not None and examples is not None and (correction or stage.steps % 3 == 0):
            examples.append((features, target, int(candidate[0])))
        action = candidate.copy()
        if aim and features is not None:
            if student is not None:
                with torch.inference_mode():
                    features_tensor = torch.from_numpy(features)
                    logits = mask_aim_logits(student(features_tensor), features_tensor)
                    target = int(logits.argmax())
            action[0] = target
        move, focus = map(int, action[:2])
        if emergency_bomb and observation['context'][4] > 0 and 'x' not in stage.held:
            if before.player_state == 4 or before.player_state == 1 and clearance[focus, move] < 0:
                imminent = np.minimum(action_clearances(observation, radius, horizon=4),
                                      body_clearances(observation, radius, hitboxes, horizon=4))
                imminent = np.minimum(imminent, laser_clearances(observation, radius, segments, horizon=4))
                if before.player_state == 4 or imminent.max() < -.5:
                    action[3] = 1
                    protective_bombs += 1
                elif imminent[focus, move] < 0 and imminent.max() > 0:
                    focus, move = np.unravel_index(imminent.argmax(), imminent.shape)
                    action[:2] = move, focus
        interventions += int(action[0] != candidate[0])
        previous_observation = observation
        shadow_valid = (shadow is not None and before.player_state == 1
                        and stage.game.read_game_val('in_dialog') != -1)
        observation, _, done, truncated, info = stage.step(action)
        impact = before.player_state != 4 and stage.snapshot.player_state == 4
        if (shadow_valid and before.stage == stage.snapshot.stage and stage.last_frame - old_frame == 4
                and (impact or stage.steps % shadow.stride == 0)):
            shadow.observe(previous_observation, observation, action, impact)
        if impact:
            move, focus = map(int, action[:2])
            impacts.append({'stage': before.stage, 'frame': stage.last_frame, 'previous_frame': old_frame,
                            'position': (before.observation['context'][:2] * [192, 448]).tolist(),
                            'projectile_clearance': float(projectile_risk[focus, move]),
                            'body_clearance': float(enemy_risk[focus, move]),
                            'laser_clearance': float(beam_risk[focus, move]), 'laser_count': len(segments),
                            'movement': move, 'best_clearance': float(clearance.max())})
        boss = before.boss
        delta = min(12, max(0, stage.last_frame - old_frame))
        if boss is not None and boss.vulnerable:
            lateral = abs(float(before.observation['context'][6] * 384))
            vulnerable_frames += delta
            aligned_frames += delta * (lateral <= 24)
            firing_frames += delta * bool(action[2])
            lateral_sum += lateral * delta
            phase = phases.setdefault((before.stage, boss.pointer, boss.max_hp, before.spell_id), {
                'stage': before.stage, 'spell_id': before.spell_id, 'max_hp': boss.max_hp, 'first_frame': old_frame,
                'lowest_hp': boss.hp, 'vulnerable_frames': 0, 'damage': 0, 'excluded_hp_drop': 0,
            })
            phase['lowest_hp'] = min(phase['lowest_hp'], boss.hp)
            phase['vulnerable_frames'] += delta
            if same_phase(before, stage.snapshot):
                applied = max(0, boss.hp - stage.snapshot.boss.hp)
                if stage.snapshot.boss.vulnerable and applied <= 512:
                    damage += applied
                    phase['damage'] += applied
                else:
                    # Timeouts can clear a large remaining HP counter in one engine tick.
                    excluded_hp_drop += applied
                    phase['excluded_hp_drop'] += applied
            if not samples or before.stage != samples[-1]['stage'] or old_frame >= samples[-1]['frame'] + 120:
                samples.append({'stage': before.stage, 'frame': old_frame, 'hp': boss.hp,
                                'lateral_offset': round(lateral, 2),
                                'shoot': int(action[2]), 'spell_id': before.spell_id})
        if done or truncated or info['steps'] % 300 == 0:
            row = dict(info, start_frame=start_frame, aim_interventions=interventions, vulnerable_frames=vulnerable_frames,
                       lives_remaining=stage.game.read_game_val('lives'),
                       bombs_remaining=stage.game.read_game_val('bombs'),
                       damage=damage, excluded_hp_drop=excluded_hp_drop,
                       hp_per_vulnerable_second=damage * 60 / max(1, vulnerable_frames),
                       mean_lateral_offset=lateral_sum / max(1, vulnerable_frames),
                       aligned_fraction=aligned_frames / max(1, vulnerable_frames),
                       firing_fraction=firing_frames / max(1, vulnerable_frames),
                       phases=list(phases.values()), observations=samples, impacts=impacts,
                       emergency_bomb_commands=protective_bombs)
            if shadow is not None:
                row['world_shadow'] = shadow.summary()
            emit(row, done or truncated)
        if done or truncated:
            return row


def main():
    parser = argparse.ArgumentParser(description='Compare boss alignment against the frozen native PPO dodge policy.')
    parser.add_argument('--base', required=True, type=Path)
    dodge_options = parser.add_mutually_exclusive_group(required=True)
    dodge_options.add_argument('--ppo', type=Path)
    dodge_options.add_argument('--analytic-dodge', action='store_true', help='Use explicit clearance instead of a PPO head.')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--limit', type=int, default=3200)
    parser.add_argument('--candidate-first', action='store_true')
    parser.add_argument('--candidate-only', action='store_true')
    parser.add_argument('--last-stage', type=int, choices=range(1, 7), default=1)
    parser.add_argument('--first-stage', type=int, choices=range(1, 7), default=1)
    parser.add_argument('--prepared', action='store_true', help='Adopt the active stage on the first trial without Retry.')
    parser.add_argument('--stage-practice', action='store_true', help='Identify the native Stage Clear result in Practice Start.')
    parser.add_argument('--focus-wait', type=int, default=0, help='Seconds to hold the game paused if it loses focus.')
    parser.add_argument('--max-misses', type=int, default=2)
    parser.add_argument('--samples', type=Path, help='Store candidate features and teacher directions by episode.')
    parser.add_argument('--student', type=Path, help='Evaluate a learned aim head against the frozen avoidance policy.')
    parser.add_argument('--teacher-control', action='store_true', help='Use geometric aim as the comparison for a student.')
    parser.add_argument('--body-risk', action='store_true', help='Include enemy-body collisions in action clearances.')
    parser.add_argument('--laser-risk', action='store_true', help='Include straight lasers in action clearances.')
    parser.add_argument('--emergency-bomb', action='store_true', help='Permit emergency bombs and four-frame evasions.')
    parser.add_argument('--world-shadow', type=Path, help='Score a world checkpoint on actual rollouts without controlling actions.')
    parser.add_argument('--world-shadow-stride', type=int, default=8)
    args = parser.parse_args()
    if min(args.blocks, args.limit, args.max_misses, args.world_shadow_stride) <= 0:
        parser.error('Blocks, decision limit, and miss limit must be positive.')
    if args.teacher_control and not args.student:
        parser.error('The teacher comparison requires a student checkpoint.')
    if args.first_stage > args.last_stage:
        parser.error('The first stage cannot follow the last requested stage.')
    if args.stage_practice and args.first_stage != args.last_stage:
        parser.error('Stage Practice requires identical first and last stages.')
    if args.focus_wait < 0:
        parser.error('Focus wait cannot be negative.')
    if args.out.exists():
        parser.error('Use a fresh report path.')
    torch.set_num_threads(1)
    base = read_policy(args.base, torch.device('cuda'), motion_history=True)
    head = PPO.load(args.ppo, device='cpu').policy.action_net.eval() if args.ppo else None
    student = None
    if args.student:
        student = AimHead()
        student.load_state_dict(torch.load(args.student, map_location='cpu', weights_only=True))
        student.eval()
    shadow = WorldShadow(args.world_shadow, torch.device('cuda'), args.world_shadow_stride) if args.world_shadow else None
    stage = NormalStageOne(Path(__file__).resolve().parents[2] / 'parakit',
                           limit=args.limit, max_misses=args.max_misses, motion_history=True,
                           history_frames=max(base.motion_spans or (12,)),
                           first_stage=args.first_stage, last_stage=args.last_stage, prepared=args.prepared,
                           stage_practice=args.stage_practice, focus_wait=args.focus_wait)
    dodge_name = 'ppo_dodge' if head is not None else 'analytic_dodge'
    candidate_name = f'{dodge_name}_with_learned_aim' if student is not None else f'{dodge_name}_with_aim'
    control_name = f'{dodge_name}_with_aim' if args.teacher_control else dodge_name
    results = {candidate_name: []} if args.candidate_only else {control_name: [], candidate_name: []}
    report = {'base': str(args.base), 'ppo': str(args.ppo) if args.ppo else None, 'limit': args.limit,
              'student': str(args.student) if args.student else None,
              'blocks': args.blocks, 'max_misses': args.max_misses,
              'first_stage': args.first_stage, 'last_stage': args.last_stage, 'prepared': args.prepared,
              'stage_practice': args.stage_practice, 'focus_wait_seconds': args.focus_wait,
              'body_risk': args.body_risk, 'laser_risk': args.laser_risk,
              'world_shadow': str(args.world_shadow) if args.world_shadow else None,
              'emergency_bomb': args.emergency_bomb, 'results': results}
    try:
        for block in range(args.blocks):
            candidate_first = (block % 2 == 1) ^ args.candidate_first
            order = (True,) if args.candidate_only else (True, False) if candidate_first else (False, True)
            for candidate in order:
                name = candidate_name if candidate else control_name

                def emit(row, finished):
                    saved = dict(row, block=block + 1, strategy=name)
                    report['current'] = None if finished else saved
                    if finished:
                        results[name].append(saved)
                    report['summary'] = {
                        label: {
                            'trials': len(rows), 'clears': sum(r['outcome'] in ('advanced_stage', 'practice_clear', 'story_clear')
                                                              for r in rows),
                            'damage_per_vulnerable_second': median(r['hp_per_vulnerable_second'] for r in rows),
                            'mean_horizontal_offsets': [r['mean_lateral_offset'] for r in rows],
                        } for label, rows in results.items() if rows
                    }
                    write_json(args.out, report)
                    print(block + 1, name, {k: v for k, v in row.items() if k not in ('phases', 'observations')},
                          flush=True)

                examples = [] if args.samples else None
                trial(stage, base, head, candidate or args.teacher_control, emit, examples,
                      student if candidate else None, args.body_risk, args.laser_risk, args.emergency_bomb, shadow)
                if examples:
                    args.samples.mkdir(parents=True, exist_ok=True)
                    features, target, prior = (np.stack(field) for field in zip(*examples))
                    np.savez_compressed(args.samples / f'{block + 1}-{name}.npz', features=features,
                                        target=target, prior=prior)
    except Exception:
        report['error'] = {'traceback': format_exc(), 'native_stage': stage.snapshot.stage,
                           'stage_frame': getattr(stage, 'last_frame', None),
                           'steps': getattr(stage, 'steps', None)}
        write_json(args.out, report)
        raise
    finally:
        stage.close()


if __name__ == '__main__':
    main()
