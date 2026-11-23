"""Measure frozen policy transfer on a reserved native game."""

import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter, sleep
from traceback import format_exc

import keyboard
import numpy as np
from stable_baselines3 import PPO
import torch
import win32gui
import win32process

from environment.structured import StateReader
from replays.collect_stage import ReplayProcess
from replays.observe_native import game_step
from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import LEVELS, append_motion_history, append_stage_context, condition_context
from scripts.record_run import write_json
from scripts.stage_policy import action_keys
from scripts.stage_risk import action_clearances, body_clearances, suggest_dodge
from scripts.train_stage_dodge import learned_action


def foreground(pid):
    return win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())[1] == pid


def play(args, game, report, policy, dodge):
    reader = StateReader(game, game.offsets, 96)
    identities = game.offsets.environment
    history = deque(maxlen=122)
    held = set()
    expiry = perf_counter() + args.wait_start
    while perf_counter() < expiry:
        if foreground(game.pid) and game.read_game_val('game_screen') == 7:
            marker = game_step(game, reader)
            if (marker is not None and marker[0] == 1 and 0 < marker[1] <= args.start_by_frame
                    and reader.read_uint(game.offsets.player.player_pointer, relative=True)):
                break
        sleep(.01)
    game.freeze()
    try:
        marker = game_step(game, reader)
        if (marker is None or marker[0] != 1 or not 0 < marker[1] <= args.start_by_frame or not foreground(game.pid)):
            raise RuntimeError(f'Prepare foreground TH{args.title} {args.difficulty} Stage 1 before evaluation.')
        character = reader.read_uint(identities.character, relative=True)
        rank = reader.read_uint(identities.difficulty, relative=True)
        pointdevice = args.title == 15 and reader.read_uint(game.offsets.game_specific['modeflags'], relative=True) & 256
        if character != 0 or rank != LEVELS.index(args.difficulty) or pointdevice:
            raise RuntimeError(f'Expected {args.difficulty} with Reimu. Select Legacy Mode in TH15.')
        subshot = reader.read_uint(identities.subshot, relative=True)
        shot = 'Reimu' if args.title == 15 else f'Reimu{game.offsets.associations.subshots[subshot]}'
        report['shot'] = shot
        write_json(args.out, report)
        snapshot = reader.snapshot()
        use_clearance = dodge is not None or args.analytic_dodge
        radius = reader.player_hit_radius() if use_clearance else None
        first_score = snapshot.score
        start_frame = marker[1]
        frame = start_frame
        start_lives = snapshot.lives
        last_lives = start_lives
        first_miss = None
        misses = 0
        bombs_used = 0
        previous_bombs = game.read_game_val('bombs')
        movement_actions = 0
        for decision in range(1, args.decisions + 1):
            if not foreground(game.pid):
                for key in held:
                    keyboard.release(key)
                held.clear()
                print(f'Awaiting TH{args.title} foreground at frame {frame}.', flush=True)
                expiry = perf_counter() + args.focus_wait
                while not foreground(game.pid) and perf_counter() < expiry:
                    sleep(.05)
                if not foreground(game.pid):
                    raise RuntimeError(f'TH{args.title} did not regain input focus.')
            if policy is None:
                action = (4, 0, 1, 0)
            else:
                conditioned = condition_context(snapshot.observation['context'], rank, shot)
                history.append((frame, conditioned))
                times, contexts = zip(*history)
                context = append_motion_history(np.asarray(contexts), np.asarray(times), policy.motion_spans,
                                                policy.retain_recent_motion)[-1]
                if policy.stage_clock:
                    context = append_stage_context(context, frame, 1)
                observation = dict(snapshot.observation, context=context)
                action = policy.predict(observation, deterministic=True)[0]
                if use_clearance:
                    clearance = action_clearances(observation, radius)
                    if args.body_risk:
                        clearance = np.minimum(clearance, body_clearances(observation, radius, reader.read_enemy_hitboxes()))
                    if dodge is not None:
                        action = learned_action(dodge, action, clearance)
                    else:
                        action, _, _ = suggest_dodge(action, clearance)
            if not foreground(game.pid):
                raise RuntimeError(f'TH{args.title} lost focus before an action could be applied.')
            movement_actions += int(action[0] != 4)
            keys = action_keys(action)
            for key in held - keys:
                keyboard.release(key)
            for key in keys - held:
                keyboard.press(key)
            held = keys
            started = game.clock(reader)
            deadline = perf_counter() + 5
            game.resume()
            while game.clock(reader) < started + 4:
                if perf_counter() >= deadline:
                    raise TimeoutError(f'TH{args.title} native clock stopped advancing.')
                sleep(.001)
            game.freeze()
            marker = game_step(game, reader)
            stage = reader.read_uint(identities.stage, relative=True)
            outcome = None
            if stage > 1:
                outcome = 'stage_advanced'
            elif marker is None:
                outcome = 'native_end'
            else:
                frame = marker[1]
                snapshot = reader.snapshot()
                lost = max(0, last_lives - snapshot.lives)
                if lost and first_miss is None:
                    first_miss = frame
                misses += lost
                current_bombs = game.read_game_val('bombs')
                bombs_used += int(current_bombs < previous_bombs and not lost)
                previous_bombs = current_bombs
                last_lives = snapshot.lives
                if misses >= args.max_misses:
                    outcome = 'miss_limit'
            if decision == args.decisions and outcome is None:
                outcome = 'decision_limit'
            if outcome is not None or decision % 120 == 0:
                row = {'start_frame': start_frame, 'stage_frame': frame, 'native_stage': stage,
                       'steps': decision, 'outcome': outcome, 'first_miss_frame': first_miss, 'misses': misses,
                       'bombs_used': bombs_used, 'moving_decisions': movement_actions,
                       'initial_lives': start_lives, 'score_delta': snapshot.score - first_score}
                report['current'] = row
                write_json(args.out, report)
                print(row, flush=True)
            if outcome is not None:
                break
    finally:
        for key in held:
            keyboard.release(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title', type=int, choices=(15, 17), required=True)
    parser.add_argument('--difficulty', choices=LEVELS[:4], default='Normal')
    parser.add_argument('--base', type=Path, help='Frozen motion-history SFT checkpoint. Omit for stationary shooting.')
    dodge_options = parser.add_mutually_exclusive_group()
    dodge_options.add_argument('--ppo', type=Path, help='Frozen projectile-dodge head trained on another title.')
    dodge_options.add_argument('--analytic-dodge', action='store_true', help='Select dodge actions using geometric clearance.')
    parser.add_argument('--body-risk', action='store_true', help='Include native enemy hitboxes in dodge scoring.')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--decisions', type=int, default=1500)
    parser.add_argument('--max-misses', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--focus-wait', type=int, default=900)
    parser.add_argument('--wait-start', type=int, default=0, help='Wait for a prepared Stage 1 game to start.')
    parser.add_argument('--start-by-frame', type=int, default=120, help='Reject a trial started after this native frame.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.out.exists() or min(args.decisions, args.max_misses, args.start_by_frame) < 1
            or min(args.focus_wait, args.wait_start) < 0):
        parser.error('Use a new output path and valid positive decision and miss limits.')
    use_clearance = args.ppo is not None or args.analytic_dodge
    if use_clearance and not args.base or args.body_risk and not use_clearance:
        parser.error('Dodge mode requires the base policy. Body risk requires a dodge mode.')
    training_titles = None
    if args.base:
        metrics_path = args.base.with_name('metrics.json')
        if not metrics_path.exists():
            parser.error('Keep the SFT metrics sidecar next to the checkpoint to verify its training titles.')
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        training_titles = sorted(map(int, metrics.get('training_samples_by_title', {})))
        checked_rows = metrics.get('validation_replays', [])
        if (not training_titles or args.title in training_titles or not checked_rows
                or any(row['title'] == args.title for row in checked_rows)):
            parser.error('Use a checkpoint with training and validation metadata that excludes the requested title.')
    torch.set_num_threads(1)
    policy = read_policy(args.base, torch.device(args.device), motion_history=True) if args.base else None
    dodge = PPO.load(args.ppo, device='cpu').policy.action_net.eval() if args.ppo else None
    strategy = ('sft_ppo_dodge' if dodge is not None else 'sft_analytic_dodge' if args.analytic_dodge
                else 'frozen_sft' if policy is not None else 'stationary_shoot')
    mode = ' Legacy Mode' if args.title == 15 else ''
    report = {'task': f'TH{args.title}{mode} {args.difficulty} Reimu Stage 1', 'title': args.title,
              'difficulty': args.difficulty,
              'strategy': strategy,
              'checkpoint': str(args.base) if args.base else None, 'configured_ticks_per_step': 4,
              'dodge_checkpoint': str(args.ppo) if args.ppo else None, 'body_risk': args.body_risk,
              'checkpoint_training_titles': training_titles, 'decision_limit': args.decisions,
              'start_by_frame': args.start_by_frame,
              'max_misses': args.max_misses}
    write_json(args.out, report)
    game = ReplayProcess(args.title, args.pid)
    try:
        play(args, game, report, policy, dodge)
    except Exception:
        report['error'] = format_exc()
        write_json(args.out, report)
        raise
    finally:
        game.close()


if __name__ == '__main__':
    main()
