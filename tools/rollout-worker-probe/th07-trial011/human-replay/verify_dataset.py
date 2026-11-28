"""Check one downloadable TH07 capture, its native labels, and optional original replay."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np
from PIL import Image


def original_words(source, stage):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'replays'))
    from inspect_rpy import decompress

    raw = bytearray(source)
    assert raw[:6] == bytes.fromhex('543752501110')
    raw[16:] = bytes((value - raw[13] - 7 * i) % 256 for i, value in enumerate(raw[16:]))
    compressed, expanded, *starts = struct.unpack_from('<9I', raw, 20)
    assert len(raw) == compressed + 84
    payload = decompress(bytes(raw[84:]), expanded)
    slowdowns = struct.unpack_from('<7I', raw, 56)
    start = starts[stage - 1]
    end = min(value for value in [*starts, *slowdowns, len(payload) + 84] if value > start)
    return np.frombuffer(payload[start - 84 + 40:end - 84], '<u2').reshape(-1, 2)[:, 0]


def verify(dataset, source=None, expected_sha=None):
    if expected_sha:
        with dataset.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected_sha
    with zipfile.ZipFile(dataset) as archive:
        report = json.loads(archive.read('source-audit.json'))
        for name, digest in report['artifacts_sha256'].items():
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
        with np.load(io.BytesIO(archive.read('positions.npz')), allow_pickle=False) as positions:
            ticks = positions['native_tick']
            idx = positions['causal_pair_rows']
            xyz = positions['xy_native']
            words = positions['stage_replay_words']
            actions = positions['action_next_fourheads']
            safe = positions['xy_qualified']
            assert np.all(np.diff(ticks) > 0) and np.all(ticks % 4 == 0)
            assert len(ticks) == report['rows'] and len(idx) == report['causal_four_tick_pairs']
            assert np.all(safe[idx] & safe[idx - 1]) and np.all(ticks[idx] - ticks[idx - 1] == 4)
            assert np.array_equal(ticks[~safe], report['excluded_xy_ticks'])
            assert np.array_equal(positions['native_word'], words[ticks])
            assert np.array_equal(positions['previous_native_word'], words[ticks - 1])
            assert np.all((xyz >= 0) & (xyz <= [384, 448]))
            observation = ticks[idx]
            mask = words[observation + 1].astype(int)
            dx = ((mask >> 7) & 1) - ((mask >> 6) & 1)
            dy = ((mask >> 5) & 1) - ((mask >> 4) & 1)
            bomb = np.diff(((words >> 1) & 1).astype(int), prepend=0) > 0
            onset = np.any(bomb[observation[:, None] + np.arange(1, 5)], axis=1)
            expected = np.column_stack((4 + dx + 3 * dy, (mask >> 2) & 1, mask & 1, onset))
            assert np.array_equal(expected, actions)
            if source is not None:
                data = source.read_bytes()
                assert hashlib.sha256(data).hexdigest() == report['original_replay_sha256']
                assert np.array_equal(original_words(data, report['stage']), words)
            color_ticks = positions['full_color_ticks']
        gray = np.frombuffer(gzip.decompress(archive.read('gray.raw.gz')), 'u1').reshape(-1, 112, 96)
        assert len(gray) == len(ticks)
        with archive.open('full-color.bgra.gz') as packed, gzip.GzipFile(fileobj=packed) as frames:
            for tick in color_ticks:
                assert struct.unpack('<I', frames.read(4))[0] == tick
                raw = frames.read(640 * 480 * 4)
                image = Image.frombytes('RGB', (640, 480), raw, 'raw', 'BGRX')
                view = image.crop((32, 16, 416, 464)).convert('L').resize((96, 112), Image.Resampling.BOX)
                assert np.array_equal(gray[ticks.searchsorted(tick)], np.asarray(view))
            assert not frames.read(1)
        assert len(color_ticks) == report['full_color_to_policy_byte_identical']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    verified = verify(args.dataset, args.replay, args.sha256)
    print(verified['author'], verified['shot'], verified['stage'], verified['rows'], verified['causal_four_tick_pairs'])
