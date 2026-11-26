"""Measure how much each action distribution changes when the current screen features are permuted."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_portable_full_sft import FullActor, WIDTHS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies", type=Path, nargs="+", required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    rows = json.loads((a.batch/"credits.json").read_bytes())
    models = {folder.name: FullActor.from_bin(folder/"policy.bin").eval() for folder in a.policies}
    results = {}
    with torch.inference_mode():
        for title in (10, 12, 13):
            games = [r for r in rows if r["title"] == title]
            results[title] = {}
            for arm, model in models.items():
                records = []
                for game in games:
                    with np.load(a.batch / game["file"]) as archive:
                        x = torch.from_numpy(archive["features"].copy()).float()
                    shifted = x.clone()
                    shifted[:, :84] = torch.roll(x[:, :84], len(x)//2, dims=0)
                    heads, _ = model(x)
                    alt, _ = model(shifted)
                    tv = [(h.softmax(1)-g.softmax(1)).abs().sum(1)/2 for h, g in zip(heads, alt)]
                    records.append({"seed": game["seed"], "n": len(x),
                                    "mean_tv": [v.mean().item() for v in tv],
                                    "move_argmax_changed": (heads[0].argmax(1)!=alt[0].argmax(1)).float().mean().item()})
                results[title][arm] = {"credits": records,
                    "equal_credit_mean_tv": np.mean([r["mean_tv"] for r in records], axis=0).tolist(),
                    "equal_credit_modal_move_changed": float(np.mean([r["move_argmax_changed"] for r in records]))}
    detail = {"heads": ["movement", "focus", "shoot", "bomb"], "widths": WIDTHS, "intervention":
              "Shift the 84 current visual features by half a full original episode. Preserve that row's actual past actions and decision clocks. Each complete credit has equal weight. Measures policy sensitivity, not survival under intervention.",
              "titles": results}
    a.out.write_text(json.dumps(detail, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
