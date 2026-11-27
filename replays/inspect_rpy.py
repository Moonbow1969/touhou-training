import argparse
from collections import Counter
import os
from pathlib import Path
from struct import unpack_from


DECODER_VERSION = 'circular-dictionary-v1'
ROOT = Path(os.environ.get('TOUHOU_REPLAY_DIR', Path(__file__).resolve().parent / 'data')).resolve()
PAIRED_DIRECTORY = 'paired-ringfix'


def u32(data, offset):
    return unpack_from("<I", data, offset)[0]


def unshuffle(data, block_length, key, increment):
    result = bytearray(data)
    end = len(data) - (len(data) & 1)
    remainder = len(data) % block_length
    if remainder < block_length // 4:
        end -= remainder
    for start in range(0, end, block_length):
        stop = min(start + block_length, end)
        target = [*range(stop - 1, start - 1, -2), *range(stop - 2, start - 1, -2)]
        for source, destination in enumerate(target, start):
            result[destination] = data[source] ^ key
            key = (key + increment) & 255
    return result


def decompress(data, expected):
    padded = data + bytes(3)
    cursor = 0
    out = bytearray()
    window_mask = (1 << 13) - 1
    window = bytearray(window_mask + 1)

    def read_bits(count):
        nonlocal cursor
        if cursor + count > len(data) * 8:
            raise ValueError("Truncated replay bit stream.")
        index = cursor // 8
        chunk = int.from_bytes(padded[index:index + 3], "big")
        value = (chunk >> (24 - cursor % 8 - count)) & ((1 << count) - 1)
        cursor += count
        return value

    while len(out) < expected:
        if read_bits(1):
            chunk = (read_bits(8),)
        else:
            start = read_bits(13) - 1
            if start < 0:
                break
            # A backreference can both cross the dictionary boundary and reuse bytes
            # written by this run. Read each byte from the live circular dictionary.
            chunk = (window[(start + index) & window_mask] for index in range(read_bits(4) + 3))
        for value in chunk:
            window[len(out) & window_mask] = value
            out.append(value)
    if len(out) != expected:
        raise ValueError(f"Replay payload length was {len(out)}, expected {expected}.")
    return out


REPLAY_LAYOUTS = {
    10: (b"t10r", 0x4C, 0x64, 0x1C4),
    12: (b"t12r", 0x58, 0x70, 0xA0),
    13: (b"t13r", 0x58, 0x74, 0xC4),
    14: (b"t13r", 0x78, 0x94, 0xDC),
    16: (b"t16r", 0x80, 0xA0, 0x294),
    17: (b"t17r", 0x84, 0xA0, 0x158),
}


def read_gameplay_payload(path, title=14):
    data = Path(path).read_bytes()
    magic = REPLAY_LAYOUTS[title][0]
    compressed, expanded = u32(data, 28), u32(data, 32)
    if data[:4] != magic or not 0 < compressed <= len(data) - 36 or not 0 < expanded < 8_000_000:
        raise ValueError("Invalid replay size or magic.")
    if title == 10:
        # Mountain of Faith uses its own shuffle keys and second block size.
        data = unshuffle(data[36:36 + compressed], 0x400, 0xAA, 0xE1)
        data = unshuffle(data, 0x80, 0x3D, 0x7A)
    elif title == 12:
        data = unshuffle(data[36:36 + compressed], 0x800, 0x5E, 0xE1)
        data = unshuffle(data, 0x40, 0x7D, 0x3A)
    else:
        data = unshuffle(data[36:36 + compressed], 0x400, 0x5C, 0xE1)
        data = unshuffle(data, 0x100, 0x7D, 0x3A)
    return decompress(data, expanded)


def action_streams(path, title=14):
    payload = read_gameplay_payload(path, title)
    _, header_count, offset, stage_header = REPLAY_LAYOUTS[title]
    for _ in range(payload[header_count]):
        stage, frames, length = payload[offset], u32(payload, offset + 4), u32(payload, offset + 8)
        start = offset + stage_header
        if frames <= 0 or length != 6 * frames + (frames + 29) // 30 or start + length > len(payload):
            raise ValueError(f"Unexpected frame/data length for stage {stage}: {frames}, {length}.")
        yield stage, payload[start:start + 6 * frames]
        offset = start + length
    if offset != len(payload):
        raise ValueError("Unexpected trailing gameplay payload.")


def inspect(path, title=14):
    payload = read_gameplay_payload(path, title)
    _, count_offset, stage_offset, header_size = REPLAY_LAYOUTS[title]
    stage_count = payload[count_offset]
    print(path, len(payload), "stages", stage_count)
    for index in range(stage_count):
        size = u32(payload, stage_offset + 8)
        entry = payload[stage_offset:stage_offset + header_size]
        offset = stage_offset + header_size
        body = payload[offset:offset + size]
        print("stage", index, "header starts", entry[:32].hex(), "data bytes", size)
        print("data prefix", body[:90].hex(" "))
        print("frequent bytes", Counter(body).most_common(12))
        stage_offset = offset + size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect native replay stage blocks.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--title", type=int, choices=REPLAY_LAYOUTS, default=14)
    args = parser.parse_args()
    for path in args.paths:
        inspect(path, args.title)
