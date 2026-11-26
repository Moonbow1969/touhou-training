"""Count visible stock gains, losses and refills across real credits and the reference replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


def events(frames, previous, following, spent=None):
    delta = following - previous
    if spent is not None:
        delta = delta + spent
    return [dict(frame=int(frames[i]), count=int(delta[i])) for i in np.flatnonzero(delta > 0)]


def audit(credit, demo_dir, replay):
    with np.load(credit, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"]))
        result = json.loads(str(z["result_json"]))
        col = {name: i for i, name in enumerate(meta["native_field_order"])}
        before, after, lost, used = (z[k] for k in (
            "native_before", "native_after", "native_life_losses", "actual_bombs_consumed"))
    stages = {}
    for stage in np.unique(before[:, col["stage"]]):
        rows = before[:, col["stage"]] == stage
        a, b = before[rows], after[rows]
        f = a[:, col["stage_frame"]]
        deaths, bombs = lost[rows], used[rows]
        lives_gained = events(f, a[:, col["lives"]], b[:, col["lives"]], deaths)
        alive = deaths == 0
        bombs_gained = events(f[alive], a[alive, col["bombs"]], b[alive, col["bombs"]], bombs[alive])
        bm = col["bombs"]
        refills = events(f[~alive], a[~alive, bm], b[~alive, bm], bombs[~alive])
        human_path = demo_dir / f"{replay}-stage{stage}.npz"
        with np.load(human_path, allow_pickle=False) as h:
            hframe = h["stage_frame"]
            counts = np.rint(h["context"][:, 3:5] * 8).astype(int)
            life_delta, bomb_delta = np.diff(counts, axis=0).T
            positive = hframe[1:] <= f[-1]
            human = dict(file_sha256=digest(human_path), observation_start=int(hframe[0]),
                         last_frame=int(hframe[-1]), visible_life_stock_increases=int(life_delta.clip(0)[positive].sum()),
                         visible_bomb_stock_increases=int(bomb_delta.clip(0)[positive].sum()),
                         visible_life_stock_decreases=int((-life_delta).clip(0)[positive].sum()),
                         opening_stock=counts[0].tolist(), final_observed_stock=counts[-1].tolist())
        stages[int(stage)] = dict(credit_opening=[int(a[0,col["lives"]]), int(a[0,bm])],
                                  last_observable_stock=[int(b[-1,col["lives"]]), int(b[-1,bm])],
                                  native_decisions=len(f), last_controlled_frame=int(f[-1]),
                                  lives_lost=int(deaths.sum()), actual_bombs_used=int(bombs.sum()),
                                  observed_life_gains=lives_gained, observed_bomb_gains_without_death=bombs_gained,
                                  observed_bomb_refills_at_death=refills, human_reference=human)
    return dict(credit_sha256=digest(credit), trial=meta["trial"], outcome=result["outcome"], stages=stages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credits", type=Path, nargs="+", required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--replay", default="th14_ud0613")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    publish_json(args.out, dict(credits=[audit(c, args.demo_dir, args.replay) for c in args.credits],
        interpretation="Gross gains inferred from native counters. The replay has sampled stock and training alignment gaps. "
                       "Stock increases at the cap are unobservable; replay clock and live phase duration can differ."))
