"""Measure actual original input coverage for the six-title 171 training cohort."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def table(path):
    return np.genfromtxt(path, delimiter="\t", names=True, encoding="utf-8", dtype=None, ndmin=1)


def masks(actions, game=False):
    move, focus, shot, bomb = actions.astype(np.int32).T
    x, y = move % 3 - 1, move // 3 - 1
    if game:
        return 64*(x<0) + 128*(x>0) + 16*(y<0) + 32*(y>0) + 8*focus + shot + 2*bomb
    return (x<0) + 2*(x>0) + 4*(y<0) + 8*(y>0) + 16*focus + 32*shot + 64*bomb


def original(root, title, seed):
    if title == 8:
        receipt = root / "selected08" / f"{seed}.json"
        r = read(receipt)
        source = root / (r.get("qualified_source") or r["recovered_source"]) / "timing.jsonl"
        shard = root / "th08-training" / r["record"]["file"]
        assert checksum(shard) == r["record"]["sha256"]
        with np.load(shard) as z:
            a = z["action"]
        log = [json.loads(line) for line in source.read_bytes().splitlines()]
        played = [row for row in log if row["phase"] == "policy"]
        assert len(played) == len(a) and log[-1]["phase"] == "terminal"
        matched = masks(a) == [row["mask"] for row in played]
        assert all(row["x_applied_ns"] >= row["monotonic_ns"] for row in played)
        sources = [receipt, shard, source]
    elif title == 12:
        (receipt,) = (root / "native12/credits").glob(f"train-six-ppo-{seed}-*.json")
        r = read(receipt)
        shard = receipt.parent / r["trajectory"]
        raw = root / "native12/raw" / receipt.stem
        with np.load(shard) as z:
            a = z["action"]
        dec = table(raw / "engineering-decisions.tsv")
        n = table(raw / "engineering-native.tsv")
        game = defaultdict(set)
        for v in n[(n["stage"] == n["manager_stage"]) & (n["mode"] == 0) & (n["spares"] >= 0)]:
            game[int(v["stage"]), int(v["tick"])].add(int(v["keys"]) & 251)
        expected = masks(a, game=True)
        assert len(dec) == len(a) and np.all(dec["observed_x11"] == masks(a))
        matched = np.array([
            any(int(key) in game[stage, tick] for tick in range(int(d["apply_tick"]) + 1, int(d["post_tick"]) + 1))
            if (stage := int(d["apply_stage"])) == d["post_stage"] else False
            for d, key in zip(dec, expected)
        ])
        sources = [receipt, shard, raw / "engineering-native.tsv", raw / "engineering-decisions.tsv"]
    else:
        raw = root / "native17" / f"train-six-ppo-{seed}"
        source, shard = raw / "decisions.tsv", raw / "behavior.npz"
        dec = table(source)
        with np.load(shard) as z:
            a = z["action"]
            assert np.array_equal(z["observed_keys"], dec["observed_keys"])
        assert len(dec) == len(a)
        matched = masks(a) == dec["observed_keys"]
        sources = [raw / "receipt.json", source, shard]
    x = a[matched]
    directions = np.bincount(x[:, 0].astype(int), minlength=9)
    return dict(seed=seed, logged=len(a), verified=len(x), directions=directions.tolist(),
                focus_shot_bomb=x[:, 1:].sum(axis=0).astype(int).tolist(),
                source_sha256={p.relative_to(root).as_posix(): checksum(p) for p in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plan = read(args.study / "camera-protocol.json")
    originals, cells = {}, {}
    for title in (8, 12, 17):
        by_rank = defaultdict(list)
        for seed in plan["train_seeds"][str(title)]:
            row = original(args.study, title, seed)
            by_rank[int(plan["rank_by_seed"][str(title)][str(seed)])].append(row)
        originals[title] = [row for cohort in by_rank.values() for row in cohort]
        cells[title] = {}
        for rank, cohort in by_rank.items():
            dirs = np.array([r["directions"] for r in cohort])
            buttons = np.array([r["focus_shot_bomb"] for r in cohort]).sum(axis=0)
            counts = dirs.sum(axis=0)
            n = int(counts.sum())
            assert len(cohort) == 16 and n > 0
            cells[title][rank] = dict(credits=len(cohort), logged=sum(r["logged"] for r in cohort),
                confirmed=n, directions=counts.tolist(), samples_all_nine=int(np.all(dirs > 0, axis=1).sum()),
                vertical_fraction=float(counts[[0,1,2,6,7,8]].sum()/n),
                diagonal_fraction=float(counts[[0,2,6,8]].sum()/n),
                focus_shot_bomb_rates=(buttons/n).tolist())
    output = dict(policy_sha256=plan["behavior_sha256"], direction_order=[
        "up-left","up","up-right","left","neutral","right","down-left","down","down-right"],
        method={"8":"Real X11 keymap acknowledgement per policy step, compared to four action heads. The original stock and terminal HUD were verified separately.",
                "12":"Independent in-game seven-key register on at least one playable native tick after action application through the registered end tick. All X11 acknowledgements are cross-checked too.",
                "17":"Real X11 keymap immediately after applying each four-head action. The native game telemetry checks stage, clock and stock independently."},
        cells=cells,originals=originals)
    args.out.write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")
    print(json.dumps(cells))


if __name__ == "__main__":
    main()
