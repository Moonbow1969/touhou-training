"""Matched TH08 Normal SFT of spatial readout and width at three fixed seeds."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay
from scripts.train_th08_normal_temporal import history_windows


WIDTHS = (9, 2, 2, 2)
SEEDS = (280801, 280802, 280803)
BATCH = 24
UPDATES = 500
AUX_LOCATION = .2


@dataclass(frozen=True)
class Design:
    channels: tuple[int, int]
    latent: int
    spatial: bool = False

    def macs(self):
        low, high = self.channels
        stem = 56*48*(25*2*low + 9*low*high + 25*high*high)
        project = (56*48 if self.spatial else 28*24)*25*high*high
        if self.spatial:
            readout = 14*12*(high*32 + 32*2) + (high+64)*1450
        else:
            readout = 7*6*high*self.latent
        return 4*(stem+project+readout) + 5*self.latent**2 + 31*self.latent


BASE = Design((20, 24), 128)
SPATIAL = Design((20, 24), 128, True)
WIDE = Design((40, 48), 256)


def match_design():
    candidates = (Design((a, b), z) for a in range(20, 41, 4) for b in range(24, 49, 4)
                  for z in range(144, 225, 16) if a <= b and (a, b) != BASE.channels)
    goal = SPATIAL.macs()
    return min(candidates, key=lambda candidate: (abs(candidate.macs()-goal), abs(candidate.latent-160)))


class SpatialReadout(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.keypoints = nn.Conv2d(channels, 32, 1)
        self.global_features = nn.Sequential(nn.Linear(channels, 1450), nn.SiLU(), nn.Linear(1450, 64))
        y, x = torch.meshgrid(torch.linspace(-1, 1, 14), torch.linspace(-1, 1, 12), indexing="ij")
        self.register_buffer("xy", torch.stack((x, y), dim=-1).reshape(-1, 2))

    def forward(self, grid):
        weights = (self.keypoints(F.adaptive_avg_pool2d(grid, (14, 12))).flatten(2) / .2).softmax(dim=2)
        points = (weights @ self.xy).flatten(1)
        return torch.cat((points, self.global_features(grid.mean((2, 3)))), dim=1)


class Policy(nn.Module):
    def __init__(self, design):
        super().__init__()
        low, high = design.channels
        z = design.latent
        self.encoder = nn.Sequential(nn.Conv2d(2, low, 5, stride=2, padding=2), nn.GroupNorm(4, low), nn.SiLU(),
                                     nn.Conv2d(low, high, 3, padding=1), nn.GroupNorm(4, high), nn.SiLU(),
                                     nn.Conv2d(high, high, 5, dilation=2, padding=4), nn.GroupNorm(4, high), nn.SiLU(),
                                     nn.Conv2d(high, high, 5, stride=1 if design.spatial else 2, padding=2), nn.SiLU())
        self.readout = SpatialReadout(high) if design.spatial else nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 6)), nn.Flatten(), nn.Linear(42*high, z))
        self.normalize = nn.Sequential(nn.LayerNorm(z), nn.SiLU())
        self.prior = nn.Linear(z, sum(WIDTHS))
        self.history = nn.Sequential(nn.Linear(4*z, z), nn.LayerNorm(z), nn.SiLU(), nn.Linear(z, sum(WIDTHS)))
        self.location = nn.Linear(z, 2)

    def forward(self, pixels):
        frame = pixels.reshape(-1, 2, 112, 96).contiguous(memory_format=torch.channels_last)
        values = self.normalize(self.readout(self.encoder(frame))).reshape(pixels.shape[0], 4, -1)
        current = values[:, -1]
        actions = self.prior(current) + self.history(values.flatten(1))
        return actions.split(WIDTHS, dim=1), current


class Deploy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, pixels):
        actions, _ = self.policy(pixels)
        return actions


def load(data, source):
    for row in json.loads(source.read_bytes())["qualified"]:
        if row["difficulty"] != "Normal":
            continue
        file = data / row["path"]
        if row["team"] != "Ghost Team" or row["game_version"] != "0.02_" or digest(file) != row["archive_sha256"]:
            raise ValueError(f"Unqualified original replay {file}")
        replay = load_replay(file, include_player_position=True)
        if replay["action"][:, 1].any():
            raise ValueError("Focus training coverage changed; re-register the experiment")
        blocks = history_windows(replay["native_frame"])
        xy = replay["player_xy"][blocks[:, -1]]
        eligible = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy < [96, 112]).all(1)
        blocks = blocks[eligible]
        newest = blocks[:, -1]
        yield dict(pilot=row["player_group"], stage=row["stage"], frames=torch.from_numpy(replay["raw"]),
                   blocks=torch.from_numpy(blocks), action=torch.from_numpy(replay["action"][newest]),
                   last=torch.from_numpy(replay["prior"][newest, 0]), location=torch.from_numpy(xy[eligible]))


def bank(examples):
    offset = 0
    blocks = []
    for row in examples:
        blocks.append(row["blocks"] + offset)
        offset += len(row["frames"])
    return (torch.cat([r["frames"] for r in examples]), torch.cat(blocks),
            torch.cat([r["action"] for r in examples]), torch.cat([r["location"] for r in examples]))


def minibatch(frames, blocks, device):
    return frames[blocks].reshape(-1, 8, 112, 96).to(device).float().div_(255)


def busy_windows(row):
    xy = row["location"].round().long()
    axis = torch.arange(-8, 9)
    x = (xy[:, 0:1] + axis).clamp(0, 95)
    y = (xy[:, 1:2] + axis).clamp(0, 111)
    time = row["blocks"][:, -1, None, None]
    energy = row["frames"][time, 1, y[:, :, None], x[:, None, :]].float().mean((1, 2))
    busy = torch.zeros(len(energy), dtype=torch.bool)
    busy[energy.argsort()[-max(1, len(energy)//4):]] = True
    return busy


def forward_rows(model, frames, blocks, device):
    logits, features = [], []
    with torch.inference_mode():
        for block in blocks.split(48):
            heads, state = model(minibatch(frames, block, device))
            logits.append(heads[0].cpu())
            features.append(state.cpu())
    return torch.cat(logits), torch.cat(features)


def locate_probe(model, train, device):
    rng = torch.Generator().manual_seed(820208)
    features, targets = [], []
    for row in train:
        selected = torch.randperm(len(row["blocks"]), generator=rng)[:512]
        _, z = forward_rows(model, row["frames"], row["blocks"][selected], device)
        features.append(z)
        targets.append(row["location"][selected])
    x, y = torch.cat(features).to(device), torch.cat(targets).to(device)
    norm = torch.tensor([96., 112.], device=device)
    readout = nn.Linear(x.size(1), 2).to(device)
    opt = torch.optim.AdamW(readout.parameters(), lr=.008)
    for _ in range(400):
        ids = torch.randint(len(x), (32,), generator=rng)
        loss = F.smooth_l1_loss(readout(x[ids]), y[ids]/norm, beta=.04)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (lambda z: (readout(z.to(device))*norm).detach().cpu())


def evaluate(model, train, held, device):
    model.eval()
    locate = locate_probe(model, train, device)
    scores, lengths, error = [], [], []
    for row in held:
        logits, features = forward_rows(model, row["frames"], row["blocks"], device)
        target = row["action"][:, 0]
        busy = row["busy"]
        switch = target != row["last"]
        good = logits.argmax(1) == target
        position_error = (locate(features)-row["location"]).norm(dim=1)*4
        error.extend(position_error.tolist())
        scores.append(dict(pilot=row["pilot"], stage=row["stage"], n=len(target), correct=int(good.sum()),
                           loss=float(F.cross_entropy(logits, target, reduction="sum")), busy=int(busy.sum()),
                           busy_correct=int(good[busy].sum()), switches=int(switch.sum()),
                           switch_correct=int(good[switch].sum()), median_location_px=float(position_error.median())))
        lengths.append(len(target))
    return scores, dict(n=sum(lengths), direction_nll=sum(r["loss"] for r in scores)/sum(lengths),
                        accuracy=sum(r["correct"] for r in scores)/sum(lengths),
                        busy_accuracy=sum(r["busy_correct"] for r in scores)/sum(r["busy"] for r in scores),
                        switch_accuracy=sum(r["switch_correct"] for r in scores)/sum(r["switches"] for r in scores),
                        median_location_px=float(np.median(error)))


def fit(design, seed, indices, train, packed, held, teacher, out, device):
    torch.manual_seed(seed)
    model = Policy(design).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    frames, blocks, action, xy = packed
    norm = torch.tensor([96., 112.], device=device)
    history = []
    clock = time.perf_counter()
    for step, ids in enumerate(indices, 1):
        pixels = minibatch(frames, blocks[ids], device)
        labels = action[ids].to(device)
        position = xy[ids].to(device)/norm
        with torch.no_grad():
            focus = teacher(pixels[:, -2:].contiguous())[1].softmax(1)
        heads, features = model(pixels)
        action_loss = (sum(F.cross_entropy(heads[k], labels[:, k]) for k in (0, 2, 3)) +
                       F.kl_div(heads[1].log_softmax(1), focus, reduction="batchmean"))
        location_loss = F.smooth_l1_loss(model.location(features), position, beta=.04)
        optimizer.zero_grad(set_to_none=True)
        (action_loss + AUX_LOCATION*location_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            state = dict(step=step, action_loss=float(action_loss.detach()), location_loss=float(location_loss.detach()))
            history.append(state)
            print(out.name, seed, state, flush=True)
    train_seconds = time.perf_counter()-clock
    per_stage, scores = evaluate(model, train, held, device)
    base = Deploy(model.cpu().eval())
    with torch.inference_mode():
        sample = minibatch(held[0]["frames"], held[0]["blocks"][:2], "cpu")
        actor = out / "policy.ts"
        traced = torch.jit.trace(base, sample)
        traced.save(str(actor))
        reloaded = torch.jit.load(str(actor), map_location="cpu")
        for batch in (sample, sample[:1]):
            assert all(torch.equal(a, b) for a, b in zip(base(batch), reloaded(batch)))
    results = dict(seed=seed, params=sum(p.numel() for p in model.parameters()), macs=design.macs(),
                   training_seconds=train_seconds, policy=digest(actor), history=history, scores=scores, stages=per_stage)
    freeze(out / "report.json", results)
    print(out.name, seed, scores, flush=True)
    return results


def compare(results):
    anchors = results["baseline"]
    summary = {}
    for arm, rows in results.items():
        seeds = []
        for baseline, observed in zip(anchors, rows):
            assert baseline["seed"] == observed["seed"]
            a, b = baseline["scores"], observed["scores"]
            points = np.array([(n["busy_correct"]-o["busy_correct"])/n["busy"]
                               for o, n in zip(baseline["stages"], observed["stages"])])
            rng = np.random.default_rng(92311)
            bootstrap = np.quantile(rng.choice(points,(12000,len(points))).mean(1),(.025,.975))
            seeds.append(dict(seed=observed["seed"], busy_accuracy_delta=b["busy_accuracy"]-a["busy_accuracy"],
                              stages_improved=int((points>0).sum()), nll_delta=b["direction_nll"]-a["direction_nll"],
                              median_location_px=b["median_location_px"], stage_bootstrap_interval=bootstrap.tolist()))
        gate = all(s["busy_accuracy_delta"] >= .015 and s["stages_improved"] >= 5 and s["nll_delta"] <= .01
                   and (arm != "spatial" or s["median_location_px"] <= 20) for s in seeds)
        summary[arm] = dict(seeds=seeds, architecture_gate=gate)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    designs = {"baseline": BASE, "spatial": SPATIAL, "pooled-match": match_design(), "pooled-wide": WIDE}
    assert abs(designs["pooled-match"].macs()/SPATIAL.macs()-1) <= .05
    rows = {name: dict(**asdict(design), estimated_macs=design.macs(),
                       params=sum(p.numel() for p in Policy(design).parameters())) for name, design in designs.items()}
    assert abs(rows["spatial"]["params"]/rows["baseline"]["params"]-1) <= .03
    folder = args.data / "th08-trial002-human"
    train_path = folder / "train-qualified-33-stages-frozen.json"
    held_path = folder / "held-qualified-12-stages-frozen.json"
    train, held = list(load(args.data, train_path)), list(load(args.data, held_path))
    if len(train) != 15 or len(held) != 6 or {r["pilot"] for r in train} & {r["pilot"] for r in held}:
        raise ValueError("Original player split changed")
    packed = bank(train)
    for row in held:
        row["busy"] = busy_windows(row)
    prior = args.data / "th08-normal-temporal/change-free/policy.ts"
    teacher = torch.jit.load(str(prior), map_location=device).eval()
    args.out.mkdir(parents=True, exist_ok=False)
    freeze(args.out / "protocol.json", dict(train=digest(train_path), held=digest(held_path), focus_teacher=digest(prior),
           designs=rows, unique_train=len(packed[1]), unique_held=sum(len(r["blocks"]) for r in held),
           seeds=SEEDS, batch=BATCH, updates=UPDATES, supervised_location_weight=AUX_LOCATION,
           scope="All arms train the complete candidate network from matched seed-specific initialization recipes. Four ordered original camera pairs; same sampled rows per seed. Parent teaches Focus; no positive native Focus samples. Busy subset per held stage. Approximate MACs count convolution/dense/coordinate projections; latency is a separate live gate."))
    print(rows, flush=True)
    results = {arm: [] for arm in designs}
    for seed in SEEDS:
        rng = torch.Generator().manual_seed(seed+7)
        indices = torch.randint(len(packed[1]), (UPDATES, BATCH), generator=rng)
        for arm, design in designs.items():
            out = args.out / f"{arm}-{seed}"
            out.mkdir()
            results[arm].append(fit(design, seed, indices, train, packed, held, teacher, out, device))
    freeze(args.out / "comparison.json", compare(results))


if __name__ == "__main__":
    main()
