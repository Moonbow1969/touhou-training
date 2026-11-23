from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

from .build_corpus import ARCHIVE, DIFFICULTIES, EXCLUDE, ROOT, player_group, split_for
from .inspect_rpy import REPLAY_LAYOUTS, action_streams, u32


def get(url, limit):
    with urllib.request.urlopen(url, timeout=15) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Archive download exceeded its size bound.")
    return data


def select_groups(title):
    path = ROOT / f"th{title}-index.html"
    if not path.exists():
        path.write_bytes(get(f"{ARCHIVE}/royalflare/th{title}", 4_000_000))
    groups = defaultdict(list)
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", path.read_text(encoding="utf-8"), re.S):
        link = re.search(r'href="(/replays/royalflare/th' + str(title) + r'/[^"]+\.rpy)"', row)
        cells = [unescape(re.sub("<[^>]+>", "", cell)).strip()
                 for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if link and len(cells) == 9 and cells[4] in DIFFICULTIES and not EXCLUDE.search(cells[7]):
            groups[(cells[4], cells[3])].append((title, cells[4], cells[3], link[1], cells[7]))
    for difficulty in DIFFICULTIES:
        shots = [shot for level, shot in groups if level == difficulty]
        main_characters = {}
        for shot in shots:
            char = re.match(r"Reimu|Marisa|Sakuya|Sanae|Youmu|Cirno|Aya", shot)
            if char:
                main_characters.setdefault(char[0], shot)
        chosen = list(main_characters.values())
        chosen.extend(shot for shot in shots if shot not in chosen)
        for shot in chosen[:6]:
            yield groups[(difficulty, shot)][:4]


def fetch_first(group):
    for title, difficulty, shot, relative, comment in group:
        filename = Path(relative).name
        path = ROOT / filename
        try:
            raw = path.read_bytes() if path.exists() else get(ARCHIVE + relative, 2_000_000)
            pointer = u32(raw, 12)
            if raw[:4] != REPLAY_LAYOUTS[title][0] or raw[pointer:pointer + 4] != b"USER":
                continue
            user = raw[pointer + 12:pointer + 400].decode("cp932", errors="replace")
            fields = dict(line.split(" ", 1) for line in user.splitlines() if " " in line)
            if fields.get("Rank", "").strip() != difficulty:
                continue
            clear = fields.get("Stage", "")
            if not any(marker in clear for marker in ("All Clear", "Extra Clear", "Extra Stage Clear")):
                continue
            if not path.exists():
                path.write_bytes(raw)
            keys = {stage: np.frombuffer(body, dtype="<u2").reshape(-1, 3)[:, 0].copy()
                    for stage, body in action_streams(path, title)}
            if sorted(keys) != ([7] if difficulty == "Extra" else list(range(1, 7))):
                continue
            player = fields["Name"].strip()
            np.savez_compressed(ROOT / f"{path.stem}-actions.npz",
                                **{f"stage_{stage}_input_words": values for stage, values in keys.items()})
            return {
                "title": title, "difficulty": difficulty, "shot": shot, "name": filename,
                "version": fields["Version"].strip(), "player": player, "split": split_for(player),
                "source": ARCHIVE + relative, "sha256": sha256(raw).hexdigest(), "comment": comment,
                "frames_by_stage": {stage: len(values) for stage, values in keys.items()},
            }
        except (OSError, ValueError, IndexError, KeyError):
            continue
    return None


def main():
    corpus_path = ROOT / "multigame-corpus.json"
    original = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
    entries = {row["name"]: dict(row, title=14) for row in original}
    with ThreadPoolExecutor(max_workers=2) as workers:
        for title in (14, 16, 13, 17):
            for entry in workers.map(fetch_first, select_groups(title)):
                if entry:
                    entries[entry["name"]] = entry
            subset = [row for row in entries.values() if row["title"] == title]
            print(f"TH{title}", len(subset), "verified files", sorted({row["shot"] for row in subset}), flush=True)
    for row in entries.values():
        row.update(player_group=player_group(row["player"]), split=split_for(row["player"]))
    corpus_path.write_text(json.dumps(list(entries.values()), indent=2) + "\n", encoding="utf-8")
    print(dict(Counter(row["title"] for row in entries.values())),
          sum(sum(row["frames_by_stage"].values()) for row in entries.values()), "action frames")


if __name__ == "__main__":
    main()
