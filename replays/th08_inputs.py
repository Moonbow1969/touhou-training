"""Read TH08 retail input words, keeping the FPS streams and branch clocks separate."""
from pathlib import Path
import struct

import numpy as np
from replays.inspect_rpy import decompress, u32


TEAMS = ('Border Team', 'Magic Team', 'Scarlet Team', 'Ghost Team')
RANKS = ('Easy', 'Normal', 'Hard', 'Lunatic', 'Extra')
SLOTS = ('1', '2', '3', '4A', '4B', '5', '6A', '6B', 'Extra')


def read(path):
    raw = Path(path).read_bytes()
    header = 0x68
    if len(raw) < header or raw[:4] != b'T8RP' or raw[4:7] != bytes((6, 0, 0)):
        raise ValueError('Expected the retail non-debug T8RP input layout')
    end = u32(raw, 12)
    if not header < end <= len(raw) - 12 or raw[end:end + 4] != b'USER':
        raise ValueError('Missing user block at the declared boundary')
    clear = bytearray(raw[:end])
    encrypted = np.frombuffer(clear, np.uint8, offset=24)
    encrypted -= (np.arange(len(encrypted), dtype=np.uint32) * 7 + raw[21]).astype(np.uint8)
    if (0x3f000318 + sum(clear[21:])) % (2**32) != u32(clear, 16):
        raise ValueError('TH08 replay checksum mismatch')
    size, expanded = struct.unpack_from('<II', clear, 24)
    if size != end - header or not 0xcc <= expanded <= 8_000_000:
        raise ValueError('Invalid replay payload lengths')
    image = clear[:header] + decompress(clear[header:], expanded)
    if image[0xd9] or image[0x7b] or struct.unpack_from('<h', image, 0x7c)[0] != -1:
        raise ValueError('Slow mode, practice or spell practice replay')
    offsets = struct.unpack_from('<9I', image, 0x20)
    fps = struct.unpack_from('<9I', image, 0x44)
    slots = [i for i, n in enumerate(offsets) if n]
    if not slots or offsets[slots[0]] != 0x134 or any(fps[i] for i in range(9) if i not in slots):
        raise ValueError('Invalid stage directory')
    boundaries = sorted([offsets[i] for i in slots] + [fps[i] for i in slots] + [len(image)])
    if len(set(boundaries)) != len(boundaries) or boundaries[-1] != len(image):
        raise ValueError('Overlapping replay sections')
    streams = {}
    for i in slots:
        start = offsets[i] + 0x24
        # Multistage retail files can store every stage's keys before the first FPS block.
        stop = boundaries[boundaries.index(offsets[i]) + 1]
        if not start + 2 <= stop or (stop - start) % 2 or fps[i] < stop:
            raise ValueError('A stage overlaps its FPS data or the next stage')
        words = np.frombuffer(image[start:stop], dtype='<u2')
        if words[0] != 0:
            raise ValueError('Missing leading dummy input')
        inputs = words[1:].copy()
        if not len(inputs) or np.any(inputs > 0x7fff):
            raise ValueError('Unexpected input word in a normal stage')
        streams[SLOTS[i]] = inputs
    credits = {}
    user = raw[end + 12:].split(b'\0', 1)[0].decode('cp932').splitlines()
    for line in user:
        fields = line.split('\t', 1)
        if len(fields) == 2:
            credits[fields[0].strip()] = fields[1].strip()
    rank = image[0x6b]
    if rank >= len(RANKS) or credits.get('難易度') != RANKS[rank]:
        raise ValueError('Disagreement between native and credited difficulty')
    return dict(rank=RANKS[rank], shot=int(image[0x6a]), version=credits.get('ゲームのバージョン'),
                pilot=credits.get('プレイヤー名'), finish=credits.get('最終ステージ'),
                score=u32(image, 0xb0)*10, user_score=credits.get('スコア')), streams


def full_actions(words):
    """TH08 uses 0x04 for focus. Convert just the modeled bits before the shared vote."""
    from scripts.pretrain_policy import label_actions
    keys = (words & 0xf3) | ((words & 4) << 1)
    n = len(keys) // 4
    return label_actions(keys[:4*n].reshape(-1, 4))
