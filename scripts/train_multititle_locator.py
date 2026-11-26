"""Measure multiclass-game transfer of a pixel-only player detector on disjoint original credits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.probe_full_action_player_vision import metrics, read
from scripts.train_native_pixel_locator import TinyLocator, targets


def labeled_bank(path, title):
    with np.load(path) as bank:
        obs = bank["obs"]
        xy = bank["xy"].astype(np.float32)
        if title == 12:
            prev = bank["prev"]
            good = np.ones(len(xy), dtype=bool)
        else:
            prev = np.concatenate([obs[:1], obs[:-1]])
            xy[:, 0] += 192
            xy /= 4
            good = bank["stable"].astype(bool)
        good &= (xy[:, 0] >= 0) & (xy[:, 0] < 96) & (xy[:, 1] >= 0) & (xy[:, 1] < 112)
        diff = np.abs(obs.astype(np.int16) - prev).astype(np.uint8)
        return np.stack([obs, diff], axis=1)[good], xy[good]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--th10-training", type=Path, nargs="+", required=True)
    for name in ("th10-test", "th12-train", "th12-test", "th13-train", "th13-test", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(6194)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    th10 = [read(folder) for folder in args.th10_training]
    training = [(np.concatenate([r[0] for r in th10]), np.concatenate([r[2] for r in th10]))]
    training += [labeled_bank(args.th12_train, 12), labeled_bank(args.th13_train, 13)]
    evaluation = [targets(args.th10_test), labeled_bank(args.th12_test, 12), labeled_bank(args.th13_test, 13)]
    tensor_banks = [(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)) for x, y in training]
    model = TinyLocator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    yy, xx = torch.meshgrid(torch.arange(56, device=device), torch.arange(48, device=device), indexing="ij")
    for _ in range(1600):
        chunks, coords = [], []
        for frames, xy in tensor_banks:
            idx = torch.randint(len(frames), (16,), device=device)
            chunks.append(frames[idx])
            coords.append(xy[idx])
        x = torch.cat(chunks).float() / 255
        pos = torch.cat(coords).clone() / 2
        flip = torch.rand(len(pos), device=device) < .5
        x[flip] = x[flip].flip(3)
        pos[flip, 0] = 47.5 - pos[flip, 0]
        x *= .8 + .4 * torch.rand(len(x), 1, 1, 1, device=device)
        x[:, 0] += (torch.rand(len(x), 1, 1, device=device) - .5) * .08
        dist = (xx - pos[:, 0, None, None]) ** 2 + (yy - pos[:, 1, None, None]) ** 2
        loss = -((-dist.flatten(1) / 1.4).softmax(1) * model(x).log_softmax(1)).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    baseline = TinyLocator().to(device)
    baseline.load_state_dict(torch.load(args.baseline, weights_only=True, map_location=device))
    report = {"steps": 1600, "training_rows": [len(x) for x, y in training],
              "th10_train_credits": [f.name for f in args.th10_training],
              "source_hashes": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in vars(args).items()
                                if k not in ("output", "th10_training")}, "evaluation": {}}
    with torch.inference_mode():
        for name, net in (("th10_only", baseline), ("three_titles", model)):
            net.eval()
            results = {}
            for title, (x, xy) in zip((10, 12, 13), evaluation):
                ids = torch.cat([net(chunk.to(device).float() / 255).argmax(1).cpu()
                                 for chunk in torch.from_numpy(x).split(96)]).numpy()
                results[title] = metrics(np.column_stack([ids % 48, ids // 48]) * 2, xy)
            report["evaluation"][name] = results
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(model.cpu().state_dict(), args.output / "detector.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(report["evaluation"])


if __name__ == "__main__":
    main()
