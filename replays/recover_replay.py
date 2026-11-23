import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from .collect_stage import AlignmentError, ROOT, ReplayProcess, StateReader, align, has_control, save_stage
from .inspect_rpy import DECODER_VERSION, PAIRED_DIRECTORY


def observe(game, reader):
    offsets = reader.offsets
    thread = offsets.game_thread
    player = offsets.player
    pointer = reader.read_uint(thread.game_thread_pointer, relative=True)
    player_pointer = reader.read_uint(player.player_pointer, relative=True)
    if not pointer or not player_pointer or game.read_game_val('pause_state') != 2:
        return None
    stage = reader.read_uint(offsets.environment.stage, relative=True)
    timer_address = pointer + thread.zGameThread_stage_timer
    frame = reader.read_uint(timer_address)
    state = reader.read_uint(player_pointer + player.zPlayer_state)
    controlled = has_control(reader, pointer, state)
    if (frame == reader.read_uint(timer_address)
            and stage == reader.read_uint(offsets.environment.stage, relative=True)):
        return stage, frame, controlled
    return None


def recover(replay, evidence, visited):
    stem = Path(replay).stem
    for path in sorted((ROOT / 'rejected').glob(f'{stem}-stage*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        stage = meta['stage']
        if stage in visited or stage not in evidence:
            continue
        visited.add(stage)
        destination = ROOT / PAIRED_DIRECTORY / path.stem
        if destination.with_suffix('.json').is_file() and destination.with_suffix('.npz').is_file():
            continue
        with np.load(path.with_suffix('.npz')) as raw:
            arrays = dict(raw)
        observations = evidence[stage]
        # Only mark frame IDs where multiple agreeing reads saw the original game disable control.
        disabled = {frame for frame, states in observations.items() if states == [2, 0]}
        frames = arrays['stage_frame']
        proven_disabled = np.isin(frames, list(disabled))
        if not proven_disabled.any():
            continue
        arrays['controllable'] = (arrays['player_state'] == 1) & ~proven_disabled
        with np.load(ROOT / f'{stem}-actions.npz') as raw:
            words = raw[f'stage_{stage}_input_words'].copy()
        try:
            paired, summary = align(replay, arrays, words, meta['stride'], meta['elapsed'])
        except AlignmentError:
            print(f'{replay} stage {stage} remains below the quality threshold.', flush=True)
            continue
        meta.pop('failure', None)
        meta.update(summary, removed_noninteractive_frames=int(proven_disabled.sum()), decoder=DECODER_VERSION)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_stage(destination, paired, meta)
        print(f'Recovered {replay} stage {stage} with {summary["pairs"]} verified pairs.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Capture gameplay authority alongside an active replay reader.')
    parser.add_argument('replay')
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    game = ReplayProcess(int(args.replay[2:4]), args.pid)
    evidence = defaultdict(dict)
    visited = set()
    next_check = perf_counter()
    silence = perf_counter()
    try:
        reader = StateReader(game, game.offsets, 96)
        while perf_counter() - silence < 25:
            try:
                sample = observe(game, reader)
            except OSError:
                sample = None
            if sample is not None:
                stage, frame, controlled = sample
                counts = evidence[stage].setdefault(frame, [0, 0])
                counts[int(controlled)] = min(2, counts[int(controlled)] + 1)
                silence = perf_counter()
            if perf_counter() >= next_check:
                recover(args.replay, evidence, visited)
                next_check = perf_counter() + 3
            sleep(.003)
        recover(args.replay, evidence, visited)
    finally:
        game.close()


if __name__ == '__main__':
    main()
