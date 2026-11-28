"""Compare frozen Bomb probabilities on the same admitted training observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.th08_camera_history import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "parent", "candidate", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [torch.jit.load(str(p), map_location=device).eval() for p in (args.parent, args.candidate)]
    rows = [row for row in json.loads(args.manifest.read_bytes())["rows"] if row["qualified"] is True]
    totals = {}
    with torch.inference_mode():
        for row in rows:
            source = args.manifest.parent / row["file"]
            if digest(source) != row["sha256"]:
                raise ValueError(f"Changed source {source.name}")
            with np.load(source, allow_pickle=False) as trace:
                frames, stock = trace["frames"], trace["bombs"]
                deaths = np.flatnonzero(trace["life_loss"] > 0)
            indices = np.arange(len(frames))
            near = ((deaths[:, None] - indices) >= 0) & ((deaths[:, None] - indices) <= 7)
            near = near.any(0)
            eligible = {"all": np.ones(len(frames), dtype=bool),
                        "bomb_available": stock > 0,
                        "bomb_available_within_8_decisions_of_a_loss": (stock > 0) & near}
            for low in range(0, len(frames), 128):
                high = min(low + 128, len(frames))
                image = torch.from_numpy(reconstruct(frames, low, high, 1)).to(device).float().div_(255)
                output = [model(image) for model in models]
                bomb = np.stack([heads[3].softmax(1)[:, 1].cpu().numpy() for heads in output])
                move = np.stack([(1 - heads[0].softmax(1)[:, 4]).cpu().numpy() for heads in output])
                for name, selection in eligible.items():
                    keep = selection[low:high]
                    n, bomb_sum, move_sum = totals.get(name, (0, np.zeros(2), np.zeros(2)))
                    totals[name] = (int(n + keep.sum()), bomb_sum + bomb[:, keep].sum(1), move_sum + move[:, keep].sum(1))
    result = dict(scope="Fixed on-policy training views with observed native stock and losses; descriptive policy probabilities, no new game outcomes.",
                  models={"parent": digest(args.parent), "candidate": digest(args.candidate)},
                  original_credits=len(rows), groups={name: dict(decisions=n, bomb_probability=(b / n).tolist(),
                  move_probability=(m / n).tolist()) for name, (n, b, m) in totals.items()})
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
