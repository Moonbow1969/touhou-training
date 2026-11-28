"""Matched SFT runs of a camera policy with and without location supervision."""

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


WIDTHS = (9, 2, 2, 2)
SEED = 820209
BATCH = 48
UPDATES = 800


class LocationReadout(nn.Module):
    def __init__(self, location):
        super().__init__()
        self.location = deepcopy(location).requires_grad_(True)
        self.actions = nn.Sequential(nn.Linear(154, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 13))
        nn.init.zeros_(self.actions[-1].weight)
        nn.init.zeros_(self.actions[-1].bias)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 56), torch.linspace(-1, 1, 48), indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy), -1).reshape(-1, 2))

    def forward(self, grid, latent, prior):
        location = self.location(grid).flatten(1)
        attention = location.softmax(1)
        localized = (grid.flatten(2) * attention.unsqueeze(1)).sum(2)
        expected_xy = attention @ self.coordinates
        movement, shoot, bomb = self.actions(torch.cat((latent, localized, expected_xy), 1)).split((9, 2, 2), 1)
        return (prior[0] + movement, prior[1], prior[2] + shoot, prior[3] + bomb), location


class Actor(nn.Module):
    def __init__(self, base, readout):
        super().__init__()
        self.spatial = base.spatial
        self.latent = nn.Sequential(*list(base.policy.children())[:-1])
        self.prior = base.policy[-1]
        self.readout = readout

    def forward(self, image):
        grid = self.spatial(image)
        latent = self.latent(grid)
        actions, _ = self.readout(grid, latent, self.prior(latent).split(WIDTHS, 1))
        return actions


def read(data, path):
    for row in json.loads(path.read_bytes())["qualified"]:
        if row["difficulty"] != "Normal":
            continue
        filename = data / row["path"]
        if row["team"] != "Ghost Team" or digest(filename) != row["archive_sha256"]:
            raise ValueError(f"Unrecognized original replay {filename}")
        replay = load_replay(filename, include_player_position=True)
        xy = replay["player_xy"]
        keep = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy < [96, 112]).all(1)
        yield {"pilot": row["player_group"], "stage": row["stage"], **{
            key: torch.from_numpy(replay[key][keep]) for key in ("raw", "action", "prior", "player_xy")}}


def location_target(xy):
    position = (xy / 2).round().long()
    return position[:, 1].clamp(0, 55) * 48 + position[:, 0].clamp(0, 47)


@torch.inference_mode()
def evaluate(base, trunk, arms, held, device):
    reports = {name: [] for name in arms}
    for replay in held:
        n = len(replay["raw"])
        row = {"pilot": replay["pilot"], "stage": replay["stage"], "windows": n,
               "change_rows": int((replay["action"][:, 0] != replay["prior"][:, 0]).sum())}
        current = {name: dict(row, direction_loss=0., movement_correct=0, change_loss=0.,
                              change_correct=0, position_errors=[]) for name in arms}
        for begin in range(0, n, BATCH):
            end = begin + BATCH
            grid = base.spatial(replay["raw"][begin:end].to(device).float() / 255)
            latent = trunk(grid)
            prior = base.policy[-1](latent).split(WIDTHS, 1)
            target = replay["action"][begin:end, 0].to(device)
            change = target != replay["prior"][begin:end, 0].to(device)
            xy = replay["player_xy"][begin:end].to(device)
            for name, arm in arms.items():
                actions, positions = arm(grid, latent, prior)
                move = actions[0]
                loss = F.cross_entropy(move, target, reduction="none")
                correct = move.argmax(1) == target
                cell = positions.argmax(1)
                error = (torch.stack((cell % 48, cell // 48), 1) * 2 - xy).norm(dim=1) * 4
                dest = current[name]
                dest["direction_loss"] += float(loss.sum())
                dest["movement_correct"] += int(correct.sum())
                dest["change_loss"] += float(loss[change].sum())
                dest["change_correct"] += int(correct[change].sum())
                dest["position_errors"].extend(error.cpu().tolist())
        for name, dest in current.items():
            error = np.asarray(dest.pop("position_errors"))
            dest.update(median_position_px=float(np.median(error)), p90_position_px=float(np.quantile(error, .9)))
            reports[name].append(dest)
    return reports


def totals(rows):
    n = sum(row["windows"] for row in rows)
    switches = sum(row["change_rows"] for row in rows)
    return {"windows": n, "direction_nll": sum(r["direction_loss"] for r in rows) / n,
            "direction_accuracy": sum(r["movement_correct"] for r in rows) / n,
            "change_rows": switches, "change_nll": sum(r["change_loss"] for r in rows) / switches,
            "change_accuracy": sum(r["change_correct"] for r in rows) / switches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED + 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normal = args.data / "th08-trial002-human"
    train_source = normal / "train-qualified-33-stages-frozen.json"
    held_source = normal / "held-qualified-12-stages-frozen.json"
    checkpoint = args.data / "th08-human-six-title-rehearsal/epoch-10/policy.ts"
    train = list(read(args.data, train_source))
    held = list(read(args.data, held_source))
    if len(train) != 15 or len(held) != 6 or {r["pilot"] for r in train} & {r["pilot"] for r in held}:
        raise ValueError("Unexpected TH08 Normal pilot split")
    images = torch.cat([r.pop("raw") for r in train])
    targets = torch.cat([r["action"] for r in train]).to(device)
    xy = location_target(torch.cat([r["player_xy"] for r in train])).to(device)
    base = CameraPolicy().to(device).eval().requires_grad_(False)
    base.load_state_dict(torch.jit.load(str(checkpoint), map_location=device).state_dict())
    trunk = nn.Sequential(*list(base.policy.children())[:-1])
    control = LocationReadout(base.player).to(device)
    arms = {"imitation": control, "imitation-plus-location": deepcopy(control)}
    opt = {name: torch.optim.AdamW(arm.parameters(), lr=.001) for name, arm in arms.items()}
    args.out.mkdir(parents=True, exist_ok=False)
    freeze(args.out / "protocol.json", dict(parent=digest(checkpoint), train=digest(train_source), held=digest(held_source),
           unique_train=len(images), updates=UPDATES, seed=SEED, batch=BATCH, location_weight=.2,
           study=("Same frozen CNN, learned spatial readout, action loss, initialization and sample indices. "
                  "Only treatment uses native-position heatmap CE, during training. Focus unchanged.")))
    history = []
    for step in range(1, UPDATES + 1):
        ids = torch.randint(len(images), (BATCH,), generator=generator)
        with torch.no_grad():
            grid = base.spatial(images[ids].to(device).float() / 255)
            latent = trunk(grid)
            prior = base.policy[-1](latent).split(WIDTHS, 1)
        curves = {"step": step}
        for name, arm in arms.items():
            out, location = arm(grid, latent, prior)
            behavior = sum(F.cross_entropy(out[i], targets[ids, i]) for i in (0, 2, 3))
            locate = F.cross_entropy(location, xy[ids])
            loss = behavior + (.2 * locate if name == "imitation-plus-location" else 0)
            opt[name].zero_grad(set_to_none=True)
            loss.backward()
            opt[name].step()
            if step == 1 or step % 50 == 0:
                curves[name] = {"action": float(behavior.detach()), "location": float(locate.detach())}
        if len(curves) > 1:
            history.append(curves)
            print(curves, flush=True)
    scores = evaluate(base, trunk, arms, held, device)
    overall = {name: totals(rows) for name, rows in scores.items()}
    checkpoints = {}
    for name, arm in arms.items():
        path = args.out / f"{name}.ts"
        actor = Actor(deepcopy(base).cpu(), deepcopy(arm).cpu()).eval()
        with torch.inference_mode():
            example = held[0]["raw"][:4].float() / 255
            traced = torch.jit.trace(actor, example)
            traced.save(str(path))
            roundtrip = torch.jit.load(str(path), map_location="cpu")
            assert all(torch.equal(a, b) for a, b in zip(actor(example), roundtrip(example)))
        checkpoints[name] = digest(path)
    freeze(args.out / "report.json", dict(device=device, checkpoints=checkpoints,
           scores=scores, overall=overall, optimizer_history=history))
    print(overall, flush=True)


if __name__ == "__main__":
    main()
