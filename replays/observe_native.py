import argparse
from collections import Counter
import json
from pathlib import Path
from struct import error as StructError, unpack
from time import perf_counter, sleep

import numpy as np

from .collect_stage import CONTROL_BITS, ReplayProcess, StateReader, has_control
from scripts.stage_risk import action_clearances, body_clearances, laser_clearances


def game_step(game, reader):
    offsets = reader.offsets
    if game.read_game_val('pause_state') != 2:
        return None
    thread = offsets.game_thread
    pointer = reader.read_uint(thread.game_thread_pointer, relative=True)
    if not pointer:
        return None
    stage = reader.read_uint(offsets.environment.stage, relative=True)
    frame = reader.read_uint(pointer + thread.zGameThread_stage_timer)
    return stage, frame, pointer


def projected_hazards(game, reader, observation):
    player = reader.offsets.player
    address = reader.read_uint(player.player_pointer, relative=True)
    radius, = unpack('<f', game._read_game_memory(address + player.zPlayer_hit_rad, 4, rel=False))
    if not 0 < radius < 16:
        raise ValueError('Invalid player hit radius while reading the native state.')
    return np.stack((action_clearances(observation, radius),
                     body_clearances(observation, radius, reader.read_enemy_hitboxes()),
                     laser_clearances(observation, radius, reader.read_straight_lasers())))


def capture(game, reader, duration, stride, risk=False):
    until = perf_counter() + duration
    seen = set()
    snapshots, controls = {}, {}
    while perf_counter() < until:
        try:
            sample = game_step(game, reader)
            if sample is None or sample[:2] in seen:
                sleep(.0015)
                continue
            stage, frame, thread = sample
            word = game.read_game_val('input') & CONTROL_BITS
            snapshot = reader.snapshot() if frame % stride == 0 else None
            controllable = (snapshot is not None and snapshot.stage == stage
                            and has_control(reader, thread, snapshot.player_state))
            hazards = projected_hazards(game, reader, snapshot.observation) if risk and controllable else None
            if game_step(game, reader) != sample:
                continue
            key = stage, frame
            controls[key] = word
            seen.add(key)
            if controllable:
                snapshots[key] = snapshot.observation, hazards
        except (OSError, RuntimeError, StructError, ValueError):
            # A live entity can expire during either read. Only consistent frames become examples.
            pass
        sleep(.0015)
    rows = []
    for (stage, frame), (observation, hazards) in sorted(snapshots.items()):
        sequence = [(stage, frame + offset) for offset in range(1, stride + 1)]
        if all(key in controls for key in sequence):
            rows.append((stage, frame, observation['bullets'], observation['context'],
                         [controls[key] for key in sequence], hazards))
    return rows, seen, len(snapshots)


def audit_reimu_a(frames, context, chunks):
    positions = context[:, :2] * [192, 448]
    active = ((np.diff(frames) == chunks.shape[1]) & (np.diff(context[:, 3]) == 0)
              & ((chunks[:-1] & 2) == 0).all(axis=1))
    if not active.any():
        return {}
    x = ((chunks & 128) != 0).astype(int) - ((chunks & 64) != 0)
    y = ((chunks & 32) != 0).astype(int) - ((chunks & 16) != 0)
    direction = np.stack((x, y), axis=-1)
    speed = np.where((chunks & 8) != 0, 2., 4.5)
    velocity = direction * (speed / np.maximum(1, np.linalg.norm(direction, axis=-1)))[..., None]
    expected = positions[:-1].copy()
    for tick in range(chunks.shape[1]):
        expected = np.clip(expected + velocity[:-1, tick], [-184, 32], [184, 432])
    error = np.linalg.norm(expected - positions[1:], axis=1)
    moving = active & (np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1)
    return {
        'transitions': int(active.sum()), 'moving_transitions': int(moving.sum()),
        'within_half_unit': float((error[active] <= .5).mean()),
        'moving_within_half_unit': float((error[moving] <= .5).mean()) if moving.any() else None,
        'median_error': float(np.median(error[active])), 'p95_error': float(np.quantile(error[active], .95)),
    }


def main():
    parser = argparse.ArgumentParser(description='Observe stage states and input words without attaching a debugger.')
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--risk', action='store_true')
    args = parser.parse_args()
    if min(args.seconds, args.stride) < 1:
        parser.error('Duration and stride must be positive.')
    game = ReplayProcess(14, args.pid)
    try:
        reader = StateReader(game, game.offsets, 96)
        env = game.offsets.environment
        identity = {field: reader.read_uint(getattr(env, field), relative=True)
                    for field in ('difficulty', 'character', 'subshot')}
        rows, frames, snapshots = capture(game, reader, args.seconds, args.stride, args.risk)
    finally:
        game.close()
    counts = Counter(row[0] for row in rows)
    print({'consistent_native_frames': len(frames), 'controllable_snapshots': snapshots,
           'complete_action_windows': len(rows), 'pairs_by_stage': dict(counts)}, flush=True)
    for stage in sorted(counts):
        values = [row[1:] for row in rows if row[0] == stage]
        frames, bullets, context, chunks, hazards = zip(*values)
        metadata = {'title': 14, 'stage': stage, 'pairs': len(frames), 'source': 'native-live-policy',
                    'native_profile': identity,
                    'stride': args.stride, 'first_frame': int(frames[0]), 'last_frame': int(frames[-1]),
                    'risk_channels': ['projectile', 'enemy_body', 'straight_laser'] if args.risk else []}
        args.out.mkdir(parents=True, exist_ok=True)
        prefix = args.out / f'th14-native-stage{stage}'
        if prefix.with_suffix('.npz').exists() or prefix.with_suffix('.json').exists():
            raise FileExistsError(prefix)
        arrays = {'stage_frame': np.asarray(frames), 'bullets': np.stack(bullets), 'context': np.stack(context),
                  'expert_action_chunk': np.asarray(chunks, dtype=np.uint16)}
        if args.risk:
            arrays['action_clearances'] = np.stack(hazards)
        if identity['character'] == identity['subshot'] == 0:
            metadata['movement_alignment'] = audit_reimu_a(arrays['stage_frame'], arrays['context'],
                                                           arrays['expert_action_chunk'])
        with prefix.with_suffix('.npz').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
