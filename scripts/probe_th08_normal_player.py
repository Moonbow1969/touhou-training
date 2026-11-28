"""Probe TH08 Normal camera features with independently observed player coordinates."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy


SEED = 820208
UPDATES = 400
BATCH = 32
STAGE_ROWS = 512


def sample(data, split, rng):
    records = json.loads(split.read_bytes())["qualified"]
    for row in records:
        if row["difficulty"] != "Normal":
            continue
        if row["team"] != "Ghost Team" or row["game_version"] != "0.02_":
            raise ValueError("Expected original TH08 0.02 Ghost Team Normal")
        path = data / row["path"]
        if digest(path) != row["archive_sha256"]:
            raise ValueError(f"Changed replay archive {path}")
        replay = load_replay(path, include_player_position=True)
        xy = replay["player_xy"]
        eligible = np.flatnonzero(np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy < [96, 112]).all(1))
        if not len(eligible):
            raise ValueError(f"No qualified positions in {path}")
        ids = rng.choice(eligible, size=min(STAGE_ROWS, len(eligible)), replace=False)
        yield row["player_group"], row["stage"], replay["raw"][ids], xy[ids]


def summarize(predict, xy):
    # Coordinates in the training archive are quarter-scale; report original playfield pixels.
    error = np.linalg.norm(predict - xy, axis=1) * 4
    return {"median_px": float(np.median(error)), "p90_px": float(np.quantile(error, .9)),
            "within_8px": float((error <= 8).mean()), "within_24px": float((error <= 24).mean())}


def decode(grid):
    cell = grid.argmax(1)
    return torch.stack((cell % 48, cell // 48), 1) * 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    replay = args.data / "th08-trial002-human"
    train_path = replay / "train-qualified-33-stages-frozen.json"
    held_path = replay / "held-qualified-12-stages-frozen.json"
    train = list(sample(args.data, train_path, rng))
    held = list(sample(args.data, held_path, rng))
    if len(train) != 15 or len(held) != 6 or {r[0] for r in train} & {r[0] for r in held}:
        raise ValueError("Original player groups or stage counts changed")
    checkpoint = args.data / "th08-human-six-title-rehearsal/epoch-10/policy.ts"
    actor = CameraPolicy().to(device).eval().requires_grad_(False)
    actor.load_state_dict(torch.jit.load(str(checkpoint), map_location=device).state_dict())
    before_head = nn.Sequential(*list(actor.policy.children())[:-1])
    latent = nn.Linear(128, 2).to(device)
    # Fresh 25-parameter convolution sees the frozen high-resolution feature map.
    spatial = nn.Conv2d(24, 1, 1).to(device)
    initial_spatial = deepcopy(actor.player)
    optimizer = torch.optim.AdamW([*latent.parameters(), *spatial.parameters()], lr=.008)
    args.out.mkdir(parents=True, exist_ok=False)
    freeze(args.out / "protocol.json", dict(prior=digest(checkpoint), train_data=digest(train_path),
           held_data=digest(held_path), seed=SEED, updates=UPDATES, batch=BATCH,
           rows_per_stage=STAGE_ROWS, train_stages=15, held_stages=6,
           objective="Locate player only. Frozen 128-dimensional policy embedding and frozen pre-pooling spatial features receive separately trained linear readouts. Compare stage median and parent's previous locator. This is not a bullet probe."))
    reduce = torch.tensor([96, 112], device=device)
    x = torch.from_numpy(np.concatenate([r[2] for r in train]))
    y = torch.from_numpy(np.concatenate([r[3] for r in train])).to(device)
    history = []
    for step in range(1, UPDATES + 1):
        ids = torch.randint(len(x), (BATCH,))
        target = y[ids]
        with torch.no_grad():
            grid = actor.spatial(x[ids].to(device).float() / 255)
            fc = before_head(grid)
        position = (target / 2).round().long()
        cell = position[:, 1].clamp(0, 55) * 48 + position[:, 0].clamp(0, 47)
        location_loss = F.cross_entropy(spatial(grid).flatten(1), cell)
        latent_loss = F.smooth_l1_loss(latent(fc), target / reduce, beta=.04)
        optimizer.zero_grad(set_to_none=True)
        (location_loss + 10 * latent_loss).backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            curve = dict(step=step, spatial_ce=float(location_loss.detach()), latent_huber=float(latent_loss.detach()))
            history.append(curve)
            print(curve, flush=True)
    stage_median = {stage: np.median(np.concatenate([r[3] for r in train if r[1] == stage]), axis=0)
                    for stage in {r[1] for r in train}}
    predictions = {"train_stage_median": [], "parent_locator": [], "spatial_probe": [], "latent_probe": []}
    truths, stage_report = [], []
    with torch.inference_mode():
        for group, stage, frames, xy in held:
            data = {key: [] for key in predictions}
            for batch in torch.from_numpy(frames).split(BATCH):
                grid = actor.spatial(batch.to(device).float() / 255)
                data["latent_probe"].append((latent(before_head(grid)) * reduce).cpu().numpy())
                data["spatial_probe"].append(decode(spatial(grid).flatten(1)).cpu().numpy())
                data["parent_locator"].append(decode(initial_spatial(grid).flatten(1)).cpu().numpy())
            data["train_stage_median"].append(np.broadcast_to(stage_median[stage], xy.shape))
            row = {"pilot": group, "stage": stage, "n": len(xy)}
            for key, chunks in data.items():
                points = np.concatenate(chunks)
                row[key] = summarize(points, xy)
                predictions[key].append(points)
            stage_report.append(row)
            truths.append(xy)
    xy = np.concatenate(truths)
    overall = {key: summarize(np.concatenate(points), xy) for key, points in predictions.items()}
    torch.save(dict(latent=latent.cpu().state_dict(), spatial=spatial.cpu().state_dict()), args.out / "probes.pt")
    freeze(args.out / "report.json", dict(device=device, train_windows=len(x), held_windows=len(xy),
           stages=stage_report, overall=overall, optimizer_history=history, weights=digest(args.out / "probes.pt")))
    print(overall, flush=True)


if __name__ == "__main__":
    main()
