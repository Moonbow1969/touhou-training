"""Read the frozen TH07 retail Normal/Reimu A Stage 1–2 corpus one stage at a time.

For optimization use training_stages; call held_stages only after freezing the model.
Each source stage can be an extracted directory or its signed GitHub release ZIP.
"""
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"SHA256 mismatch for {path}")
    return json.loads(raw)


def direction(word):
    bits = word.astype(np.int32)
    return 4 + ((bits >> 7) & 1) - ((bits >> 6) & 1) + 3 * (((bits >> 5) & 1) - ((bits >> 4) & 1))


@dataclass
class Stage:
    source_id: int
    author: str
    stage: int
    images: np.ndarray  # [N, 2, 112, 96] uint8, current and absolute four-native-tick difference
    actions: np.ndarray  # [N, 4] movement, focus, fire, Bomb onset in next four ticks
    xy: np.ndarray  # [N, 2], in 384 by 448 native playfield, auxiliary target only
    changed_direction: np.ndarray
    native_tick: np.ndarray


def _stages(study, source_root, index_sha256, role):
    study, source_root = Path(study), Path(source_root)
    index = read_json(study / "index.json", index_sha256)
    protocol = read_json(study / "protocol.json", index["protocol_sha256"])
    roster = {author["id"]: author for author in protocol["roles"]}
    if len({q["author"].casefold() for q in roster.values()}) != len(roster):
        raise ValueError("A player appears in more than one role")
    expected = {(sid, stage) for sid in roster for stage in (1, 2)}
    pairs = [(r["source_id"], r["stage"]) for r in index["stages"]]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("Incomplete or duplicated retail corpus")
    for row in index["stages"]:
        sid, stage = row["source_id"], row["stage"]
        author = roster[sid]
        if row["role"] != author["role"] or row["author"] != author["author"]:
            raise ValueError("Frozen role/author mismatch")
        if row["role"] != role:
            continue
        folder = source_root / row["native_path"]
        archive = source_root / row["name"]
        if (folder / "source-audit.json").exists():
            content = {key: (folder / key).read_bytes() for key in ("source-audit.json", "positions.npz", "gray.raw.gz")}
        else:
            if file_digest(archive) != row["sha256"]:
                raise ValueError(f"Archive differs for {author['author']} stage {stage}")
            with zipfile.ZipFile(archive) as z:
                content = {key: z.read(key) for key in ("source-audit.json", "positions.npz", "gray.raw.gz")}
        if digest(content["source-audit.json"]) != row["source_audit_sha256"]:
            raise ValueError("Native receipt mismatch")
        audit = json.loads(content["source-audit.json"])
        if (not audit["admitted"] or audit["stage"] != stage or audit["role"] != role
                or audit["protocol_sha256"] != index["protocol_sha256"]
                or audit["original_replay_sha256"] != author["original_sha256"]):
            raise ValueError("Native receipt is outside the frozen acquisition")
        for name in ("gray.raw.gz", "positions.npz"):
            if digest(content[name]) != audit["artifacts_sha256"][name]:
                raise ValueError(f"Receipt mismatch for {name}")
        frames = np.frombuffer(gzip.decompress(content["gray.raw.gz"]), np.uint8).reshape(-1, 112, 96)
        with np.load(io.BytesIO(content["positions.npz"]), allow_pickle=False) as data:
            ticks = data["native_tick"]
            all_rows = data["causal_pair_rows"]
            cap = data["fixed_optimizer_window_indices"]
            exact_cap = np.linspace(0, len(all_rows)-1, min(2000, len(all_rows))).astype(np.int64)
            if not np.array_equal(cap, exact_cap):
                raise ValueError("Source cap differs from preregistration")
            j = all_rows[cap]
            t = ticks[j]
            words = data["stage_replay_words"]
            good = data["xy_qualified"]
            if (len(frames) != len(ticks) or np.any(j < 1) or np.any(t - ticks[j-1] != 4)
                    or not np.all(good[j] & good[j-1]) or not np.all(data["live_stage"] == stage)
                    or not np.array_equal(data["native_word"], words[ticks-1])
                    or not np.array_equal(data["previous_native_word"], words[ticks-2])):
                raise ValueError("Native input/camera chronology mismatch")
            actions = data["action_next_fourheads"][cap].astype(np.int64)
            focus = (words[t] >> 2) & 1
            fire = words[t] & 1
            edge = np.diff(((words >> 1) & 1).astype(np.int8), prepend=0) > 0
            bomb = np.any(edge[t[:, None] + np.arange(4)], axis=1)
            if not np.array_equal(actions, np.column_stack((direction(words[t]), focus, fire, bomb))):
                raise ValueError("Retail four-head target mismatch")
            xy = data["xy_native"][j].copy()
            change = actions[:, 0] != direction(words[t-4])
        current = frames[j]
        delta = np.abs(current.astype(np.int16) - frames[j-1]).astype(np.uint8)
        yield Stage(sid, author["author"], stage, np.stack((current, delta), axis=1), actions, xy, change, t)


def training_stages(study, source_root, index_sha256):
    return _stages(study, source_root, index_sha256, "train")


def held_stages(study, source_root, index_sha256):
    return _stages(study, source_root, index_sha256, "held")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("index_sha256")
    args = parser.parse_args()
    for role, load in (("train", training_stages), ("held", held_stages)):
        total = 0
        authors = set()
        for part in load(args.study, args.source_root, args.index_sha256):
            n = len(part.native_tick)
            assert part.images.shape == (n, 2, 112, 96) and part.actions.shape == (n, 4)
            assert part.xy.shape == (n, 2) and n <= 2000
            total += n
            authors.add(part.author)
        print(json.dumps(dict(role=role, authors=len(authors), capped_windows=total)))
