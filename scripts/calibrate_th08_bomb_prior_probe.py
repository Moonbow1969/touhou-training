"""Diagnose the current Bomb head on the four previously registered TH08 TRAIN credits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.th08_camera_history import reconstruct
from scripts.train_five_original_games import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(str(args.parent), map_location=device).eval()
    parent = digest(args.parent)
    credit_rows = json.loads(args.manifest.read_bytes())
    old, casts = [], 0
    with torch.inference_mode():
        for credit in credit_rows:
            path = args.manifest.parent / credit["file"]
            if credit["role"] != "train" or credit["policy_sha256"] != parent or digest(path) != credit["sha256"]:
                raise ValueError(f"Training provenance changed for {path}")
            with np.load(path, allow_pickle=False) as raw:
                pictures = raw["frames"]
                casts += int((raw["action"][:,3] > 0).sum())
            for begin in range(0, len(pictures), 64):
                frames = reconstruct(pictures, begin, min(begin+64, len(pictures)), 1)
                logits = policy(torch.from_numpy(frames).to(device).float()/255)[3]
                old.append((logits[:,1] - logits[:,0]).cpu())
        logits = torch.cat(old)
        low, high = -20., 20.
        for _ in range(40):
            guess = (low+high)/2
            if (logits+guess).sigmoid().mean() > .01:
                high = guess
            else:
                low = guess
        delta = (low+high)/2
        base = logits.sigmoid()
        shifted = (logits+delta).sigmoid()
    freeze(args.out, dict(actor=parent, manifest=digest(args.manifest), sample_size=len(logits), casts=casts,
           current_sampled_mean_probability=float(base.mean()), mean_target=.01, bomb_log_odds_shift=delta,
           resulting_mean_probability=float(shifted.mean()), resulting_median=float(shifted.median()),
           resulting_p95=float(shifted.quantile(.95)), resulting_max=float(shifted.max()),
           scope="Diagnostic from the existing wall-clock TRAIN credits. No held credit, checkpoint change or original game command."))


if __name__ == "__main__":
    main()
