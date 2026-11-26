"""Compare two frozen action distributions on independent, identical state sequences."""
import argparse
import json
from pathlib import Path

import torch

from scripts.train_portable_full_sft import FullActor
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline", "candidate", "batch", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    baseline = FullActor.from_bin(args.baseline).eval()
    candidate = FullActor.from_bin(args.candidate).eval()
    credits = json.loads((args.batch/"credits.json").read_bytes())
    rows = []
    with torch.inference_mode():
        for credit in credits:
            with np.load(args.batch/credit["file"]) as data:
                x = torch.from_numpy(data["features"].copy())
            left, _ = baseline(x)
            right, _ = candidate(x)
            before, after = left[3].softmax(1)[:,1], right[3].softmax(1)[:,1]
            diff = after-before
            rows.append(dict(title=credit["title"], seed=credit["seed"], decisions=len(x),
                             expected_sampled_bomb_before=before.sum().item(),
                             expected_sampled_bomb_after=after.sum().item(),
                             expected_different_bomb_decisions=diff.abs().sum().item(),
                             movement_total_variation=((left[0].softmax(1)-right[0].softmax(1)).abs().sum(1)/2).mean().item()))
    summaries = {}
    for title in (10, 12, 13):
        group = [r for r in rows if r["title"] == title]
        summaries[title] = {key: float(np.median([r[key] for r in group])) for key in
                           ("expected_sampled_bomb_before", "expected_sampled_bomb_after",
                            "expected_different_bomb_decisions", "movement_total_variation")}
    args.out.write_text(json.dumps(dict(measure="Medians over full fixed, independently collected observation sequences. Common uniforms on each binary Bomb head would yield sum(abs(p1-p2)) expected action differences. This does not measure game outcomes after trajectories diverge.",
                                        titles=summaries, credits=rows), indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
