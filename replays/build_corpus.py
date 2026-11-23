from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

from .inspect_rpy import ROOT, action_streams, u32


ARCHIVE = "https://maribelhearn.com"
DIFFICULTIES = ("Easy", "Normal", "Hard", "Lunatic", "Extra")
PER_DIFFICULTY = 8
PILOT_IDS = {"Easy": ("08a5", "05ed"), "Normal": ("093f", "0864", "06d6")}
EXCLUDE = re.compile(r"\bTAS\b|no shoot|pacifist|speedrun|左右封印|ノーショット", re.I)


def candidates(index, difficulty):
    table = index.split(f'id="{difficulty}ReimuAt"', 1)[1].split("</table>", 1)[0]
    rows = {}
    for raw_row in re.findall(r"<tr>(.*?)</tr>", table, re.S):
        match = re.search(r'href="(/replays/royalflare/th14/th14_ud([\da-f]+)\.rpy)"', raw_row)
        cells = re.findall(r"<td[^>]*>(.*?)</td>", raw_row, re.S)
        if match and len(cells) == 9:
            comment = unescape(re.sub("<[^>]+>", "", cells[7])).strip()
            if not EXCLUDE.search(comment):
                rows[match[2]] = (difficulty, match[1], comment)
    pinned = PILOT_IDS.get(difficulty, ())
    return [rows[key] for key in (*pinned, *(key for key in rows if key not in pinned))]


def player_group(player):
    key = player.casefold().strip()
    for prefix in ("k@g", "umz", "ydh", "kpny", "myon", "makon", "gzz", "zvs"):
        if key.startswith(prefix):
            return prefix
    return key


def split_for(player):
    group = player_group(player)
    reserved = {"sci": "train", "umz": "train", "8rabbits": "train",
                "toshi": "validation", "shaoflan": "validation"}
    if group in reserved:
        return reserved[group]
    bucket = int.from_bytes(sha256(group.encode()).digest()[:4], "big") % 10
    return "test" if bucket == 9 else "validation" if bucket == 8 else "train"


def fetch(candidate):
    difficulty, relative_url, comment = candidate
    source = ARCHIVE + relative_url
    filename = Path(relative_url).name
    path = ROOT / filename
    try:
        if not path.exists():
            with urllib.request.urlopen(source, timeout=15) as response:
                raw = response.read(2_000_001)
            if not 36 < len(raw) <= 2_000_000 or raw[:4] != b"t13r":
                return None
            path.write_bytes(raw)
        raw = path.read_bytes()
        metadata = raw[u32(raw, 12):].decode("cp932", errors="replace").split("\0", 1)[0]
        metadata_offset = u32(raw, 12)
        user = raw[metadata_offset + 12:metadata_offset + 400].decode("cp932", errors="replace")
        required = ("Version 1.00b", "Chara ReimuA", f"Rank {difficulty}")
        if not metadata.startswith("USER") or not all(marker in user for marker in required):
            return None
        if not any(clear in user for clear in ("All Clear", "Extra Clear", "Extra Stage Clear")):
            return None
        player = next(line[5:].strip() for line in user.splitlines() if line.startswith("Name "))
        stages = {
            stage: np.frombuffer(body, dtype="<u2").reshape(-1, 3)[:, 0].copy()
            for stage, body in action_streams(path)
        }
        if sorted(stages) != ([7] if difficulty == "Extra" else list(range(1, 7))):
            return None
        np.savez_compressed(ROOT / f"{path.stem}-actions.npz",
                            **{f"stage_{stage}_input_words": keys for stage, keys in stages.items()})
        return {
            "name": filename, "difficulty": difficulty, "shot": "Reimu A", "version": "1.00b",
            "player": player, "source": source, "comment": comment, "split": split_for(player),
            "sha256": sha256(raw).hexdigest(),
            "frames_by_stage": {stage: len(keys) for stage, keys in stages.items()},
        }
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def main():
    index = (ROOT / "royalflare-index.html").read_text(encoding="utf-8")
    manifest = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        for difficulty in DIFFICULTIES:
            rows = candidates(index, difficulty)
            seen_players = set()
            accepted = 0
            for start in range(0, len(rows), 4):
                for entry in workers.map(fetch, rows[start:start + 4]):
                    if not entry or entry["player"] in seen_players:
                        continue
                    seen_players.add(entry["player"])
                    manifest.append(entry)
                    accepted += 1
                    if accepted == PER_DIFFICULTY:
                        break
                if accepted == PER_DIFFICULTY:
                    break
            print(difficulty, accepted, flush=True)
            (ROOT / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(dict(Counter(row["split"] for row in manifest)),
          sum(sum(row["frames_by_stage"].values()) for row in manifest), "action frames")


if __name__ == "__main__":
    main()
