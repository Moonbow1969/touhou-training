"""Test whether native local color motion is recoverable from TH08 low-resolution cameras."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from replays.audit_th08_trial002_camera import FRAMES, POSITIONS, digest
from replays.audit_th08_trial002_motion import PATCHES
from scripts.train_five_original_games import freeze


def inputs(raw, audit):
    frames = np.fromfile(raw / "frames.bin", dtype=FRAMES)
    xy = np.fromfile(raw / "positions.bin", dtype=POSITIONS)["xy"]
    patches = np.fromfile(raw / "motion-patches.bin", dtype=PATCHES)
    for filename, key in (("frames.bin", "original_camera_sha256"), ("positions.bin", "positions_sha256"),
                          ("motion-patches.bin", "patch_sha256")):
        if digest(raw / filename) != audit[key]:
            raise ValueError(f"Changed verified original artifact {filename}")
    ticks = frames["state"][:, 0].astype(np.int64)
    at = np.searchsorted(ticks, patches["tick"])
    history = at[:, None] + np.arange(-4, 1)
    keep = (at >= 4) & (np.diff(ticks[history], axis=1) == 4).all(1)
    history = history[keep]
    colors = torch.from_numpy(patches["rgb"][keep].copy()).float() / 255
    frames = torch.from_numpy(frames["obs"][history].copy()).float() / 255
    yx = torch.from_numpy(xy[at[keep]].copy())
    step = (torch.arange(16, dtype=torch.float32) - 7.5) * 4
    yy, xx = torch.meshgrid(step, step, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None] + yx.round()[:, None, None]
    grid = grid / torch.tensor((192., 224.)) - 1
    sensor = F.grid_sample(frames, grid, align_corners=False)
    camera = torch.stack((sensor[:,1:], torch.diff(sensor, dim=1).abs()), dim=2).flatten(1, 2)
    target = (colors[:,1] - colors[:,0]).abs().permute(0,3,1,2).contiguous()
    peak = colors.amax(dim=1)
    color = (peak.amax(dim=-1) > .65) & ((peak.amax(dim=-1) - peak.amin(dim=-1)) > .25)
    changing_color = color & (target.mean(1) > .08)
    return camera, target, changing_color


def score(prediction, target, mask):
    error = (prediction - target).abs().mean(1)
    return {"mae_all": float(error.mean()), "mae_bright_color_motion": float(error[mask].mean()),
            "mae_other_pixels": float(error[~mask].mean()), "bright_color_motion_pixels": int(mask.sum())}


def predict(model, camera):
    simple = F.interpolate(camera[:, -1:], size=(64,64), mode="bilinear", align_corners=False).expand(-1,3,-1,-1)
    return simple, (simple + model(camera)).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "train-audit", "held", "held-audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(280811)
    torch.set_num_threads(2)
    train_audit = json.loads(args.train_audit.read_bytes())
    held_audit = json.loads(args.held_audit.read_bytes())
    if train_audit["replay_sha256"] == held_audit["replay_sha256"]:
        raise ValueError("The motion probe requires two separate original replay sources")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, mask = (value.to(device) for value in inputs(args.train, train_audit))
    vx, vy, valid_mask = (value.to(device) for value in inputs(args.held, held_audit))
    if min(len(x), len(vx)) < 128 or min(int(mask.sum()), int(valid_mask.sum())) < 100:
        raise ValueError("Insufficient independently aligned native color motion for a probe")
    model = nn.Sequential(nn.Conv2d(8, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 48, 3, padding=1),
                          nn.PixelShuffle(4)).to(device)
    nn.init.zeros_(model[2].weight)
    nn.init.zeros_(model[2].bias)
    steps, batch = 500, 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    args.out.mkdir(exist_ok=False)
    freeze(args.out / "protocol.json", dict(train=digest(args.train_audit), held=digest(args.held_audit),
           stages=[train_audit["stage"], held_audit["stage"]], train_rows=len(x), held_rows=len(vx), updates=steps,
           seed=280811, batch=batch, design="Diagnostic with independent physical native coordinates providing crop centers. The low-resolution input and bilinear baseline use identical crops. Bright/or color weighted native-motion MAE. This is not a deployed policy or an annotations-based bullet detector."))
    history = []
    for step in range(1, steps+1):
        idx = torch.randint(len(x), (batch,), device=device)
        _, estimate = predict(model, x[idx])
        error = (estimate - y[idx]).abs().mean(1)
        weight = 1+4*mask[idx]
        loss = (weight * error).sum()/weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append(dict(step=step, weighted_mae=float(loss.detach())))
    with torch.inference_mode():
        base, treatment = predict(model, vx)
        metrics = {"bilinear": score(base, vy, valid_mask), "learned": score(treatment, vy, valid_mask)}
    torch.save(model.cpu().state_dict(), args.out / "probe.pt")
    freeze(args.out / "report.json", dict(device=device, metrics=metrics, optimizer_history=history,
           weight=digest(args.out / "probe.pt")))
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
