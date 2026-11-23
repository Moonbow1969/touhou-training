"""Read TH10 1.00a demo telemetry and verify local replay input blocks."""

import argparse
import ctypes
from ctypes import wintypes
from hashlib import file_digest
import json
from pathlib import Path
from struct import unpack_from
from time import perf_counter, sleep

import numpy as np
import psutil
import win32api
import win32con
import win32process

from replays.inspect_rpy import action_streams


# Layout checked on the installed Steam executable using the TH10 reconstruction.
STEAM_SHA256 = '3bdb72cf3d7c33c183359d368c801490dbcf54e6b3b2f060b95d72250b6866a3'
BULLET_STRIDE = 0x7F0
BULLET_SLOTS = 2000


def sample(process, duration):
    handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, process.pid)
    try:
        base = win32process.EnumProcessModules(handle)[0]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        read = kernel.ReadProcessMemory
        read.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        read.restype = wintypes.BOOL
        pool_size = BULLET_STRIDE * BULLET_SLOTS
        pool = ctypes.create_string_buffer(pool_size)
        counts = np.ndarray(BULLET_SLOTS, dtype='<u2', buffer=pool, offset=0x446, strides=(BULLET_STRIDE,))

        def get(address, format_code):
            buffer = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint32) * len(format_code))
            if not read(int(handle), address, buffer, len(buffer), None):
                raise OSError(ctypes.get_last_error(), 'Could not read TH10 state')
            return unpack_from('<' + format_code, buffer)

        rows = []
        until = perf_counter() + duration
        while perf_counter() < until:
            try:
                stage, = get(base + 0x74C7C, 'I')
                rank, = get(base + 0x74C74, 'I')
                player, replay = get(base + 0x77834, 'II')
                bullets, = get(base + 0x776F0, 'I')
                if stage in range(1, 8) and rank < 5 and min(player, bullets, replay) > 0x10000:
                    frame, = get(replay + 0x1C8, 'I')
                    x, y, _ = get(player + 0x3C0, 'fff')
                    if read(int(handle), bullets + 0x60, pool, pool_size, None):
                        active = np.flatnonzero(counts == 1)
                        end_frame, = get(replay + 0x1C8, 'I')
                        if frame > 0 and frame == end_frame and -200 <= x <= 200 and 0 <= y <= 450:
                            rows.append((stage, rank, frame, x, y, int(active.size)))
            except OSError:
                pass
            sleep(.045)
    finally:
        handle.Close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    processes = [p for p in psutil.process_iter(['name']) if p.info['name'].lower() == 'th10.exe']
    if len(processes) != 1:
        raise RuntimeError('Expected one active TH10 game.')
    process = processes[0]
    path = Path(process.exe())
    with path.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    if digest != STEAM_SHA256:
        raise RuntimeError('Executable fingerprint differs from the tested Steam build.')
    rows = sample(process, 8)
    if not rows or len({row[:3] for row in rows}) < 10 or max(row[-1] for row in rows) == 0:
        raise RuntimeError('Start the native demo or a replay for a readable gameplay span.')
    replay_counts = {path.name: {stage: len(stream) // 6 for stage, stream in action_streams(path, 10)}
                     for path in sorted((path.parent / 'replay').glob('*.rpy'))}
    report = {'title': 10, 'version': '1.00a Steam', 'executable_sha256': digest,
              'source': 'read-only native demo', 'coherent_samples': len(rows),
              'distinct_game_frames': len({row[:3] for row in rows}), 'peak_active_bullets': max(row[-1] for row in rows),
              'record_format': ['stage', 'difficulty', 'replay_frame', 'player_x', 'player_y', 'active_bullets'],
              'samples': rows, 'input_frames_by_local_replay': replay_counts}
    with args.out.open('x', encoding='utf-8') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print({key: value for key, value in report.items() if key != 'samples'}, flush=True)


if __name__ == '__main__':
    main()
