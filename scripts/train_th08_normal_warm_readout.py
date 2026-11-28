"""Full-replay warm-start comparison of pooled and spatial TH08 Normal readouts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_replay_camera_encoder import CameraPolicy
from scripts.train_th08_normal_capacity import AUX_LOCATION, SEEDS, WIDTHS, SpatialReadout, busy_windows, load
from scripts.train_th08_normal_temporal import Actor, Readout


BATCH = 128
EPOCHS = 5


class Projection(nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.layers = prior.features[:3]

    def forward(self, camera):
        return F.adaptive_avg_pool2d(self.layers(camera), (14, 12))


class ReadoutPolicy(nn.Module):
    def __init__(self, spatial):
        super().__init__()
        self.readout = SpatialReadout(24) if spatial else nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 6)), nn.Flatten(), nn.Linear(24*42, 128))
        self.normalization = nn.Sequential(nn.LayerNorm(128), nn.SiLU())
        self.current_head = nn.Linear(128, sum(WIDTHS))
        self.history_head = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, sum(WIDTHS)))
        self.location = nn.Linear(128, 2)

    def forward(self, grid):
        mapped = self.normalization(self.readout(grid.reshape(-1, 24, 14, 12)))
        history = mapped.reshape(grid.shape[0], 4, 128)
        current = history[:, -1]
        return (self.current_head(current) + self.history_head(history.flatten(1))).split(WIDTHS, 1), current


class ActorFromCamera(nn.Module):
    def __init__(self, projection, policy):
        super().__init__()
        self.projection = projection
        self.policy = policy

    def forward(self, pixels):
        grid = self.projection(pixels.reshape(-1, 2, 112, 96)).half().float()
        grid = grid.reshape(pixels.shape[0], 4, 24, 14, 12)
        return self.policy(grid)[0]


def cached(rows, prior, projection, device):
    result, offset = [], 0
    for sample in rows:
        busy = busy_windows(sample)
        originals = sample.pop("frames")
        features, focus = [], []
        with torch.no_grad():
            for block in originals.split(128):
                pixels = block.to(device).float()/255
                features.append(projection(pixels).half())
                focus.append(prior(pixels)[1].softmax(1))
        blocks = sample["blocks"].to(device)
        result.append(dict(sample, busy=busy, recoded=torch.cat(features), focus=torch.cat(focus)[blocks[:, -1]],
                           global_blocks=blocks+offset))
        offset += len(originals)
    return result


def vectors(model, row, device, choose=None):
    blocks = row["blocks"] if choose is None else row["blocks"][choose]
    directions, locators = [], []
    with torch.no_grad():
        for ids in blocks.split(BATCH):
            heads, z = model(row["recoded"][ids.to(device)].float())
            directions.append(heads[0].cpu())
            locators.append(z.cpu())
    return torch.cat(directions), torch.cat(locators)


def evaluate(model, train, held, device):
    rng = torch.Generator().manual_seed(820208)
    vectors_xy = []
    for row in train:
        ids = torch.randperm(len(row["blocks"]), generator=rng)[:512]
        _, z = vectors(model, row, device, choose=ids)
        vectors_xy.append((z, row["location"][ids]))
    z, xy = [torch.cat(items).to(device) for items in zip(*vectors_xy)]
    norm = torch.tensor([96., 112.], device=device)
    probe = nn.Linear(z.shape[1], 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=.008)
    for _ in range(400):
        ids = torch.randint(len(z), (32,), generator=rng)
        loss = F.smooth_l1_loss(probe(z[ids]), xy[ids]/norm, beta=.04)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    details, errors = [], []
    for row in held:
        logits, z = vectors(model, row, device)
        labels = row["action"][:, 0]
        correct = logits.argmax(1) == labels
        busy = row["busy"]
        switches = labels != row["last"]
        with torch.no_grad():
            error = ((probe(z.to(device))*norm).cpu()-row["location"]).norm(dim=1)*4
        details.append(dict(stage=row["stage"], pilot=row["pilot"], n=len(labels), correct=int(correct.sum()),
                            busy=int(busy.sum()), busy_correct=int(correct[busy].sum()),
                            switches=int(switches.sum()), switch_correct=int(correct[switches].sum()),
                            direction_loss=float(F.cross_entropy(logits, labels, reduction="sum")),
                            median_location_px=float(error.median())))
        errors.extend(error.tolist())
    n = sum(row["n"] for row in details)
    return details, dict(n=n, accuracy=sum(r["correct"] for r in details)/n,
                         direction_nll=sum(r["direction_loss"] for r in details)/n,
                         busy_accuracy=sum(r["busy_correct"] for r in details)/sum(r["busy"] for r in details),
                         switch_accuracy=sum(r["switch_correct"] for r in details)/sum(r["switches"] for r in details),
                         median_location_px=float(np.median(errors)))


def train(model, optimizer, draws, packed, norm):
    grids, blocks, actions, location, focus = packed
    total = torch.zeros(2, device=grids.device)
    for selected in draws.split(BATCH):
        heads, z = model(grids[blocks[selected]].float())
        behavior = sum(F.cross_entropy(heads[j], actions[selected, j]) for j in (0, 2, 3))
        behavior = behavior + F.kl_div(heads[1].log_softmax(1), focus[selected], reduction="batchmean")
        location_loss = F.smooth_l1_loss(model.location(z), location[selected]/norm, beta=.04)
        optimizer.zero_grad(set_to_none=True)
        (behavior + AUX_LOCATION*location_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total += torch.stack((behavior.detach(), location_loss.detach()))*len(selected)
    return (total/len(draws)).cpu().tolist()


def fit(seed, draws, arm, dataset, held, packed, projection, example, out, device):
    torch.manual_seed(seed)
    model = ReadoutPolicy(spatial=arm == "spatial").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.0006)
    norm = torch.tensor([96., 112.], device=device)
    steps, training_log = 0, []
    for epoch, order in enumerate(draws, 1):
        begin = time.perf_counter()
        loss, loc = train(model, opt, order, packed, norm)
        steps += (len(order)+BATCH-1)//BATCH
        stats = dict(epoch=epoch, n=len(order), action_loss=loss, localization_loss=loc, seconds=time.perf_counter()-begin)
        training_log.append(stats)
        print(arm, seed, stats, flush=True)
    stages, metrics = evaluate(model.eval(), dataset, held, device)
    deployment = ActorFromCamera(projection, model).eval()
    with torch.no_grad():
        actor = torch.jit.trace(deployment, example)
        path = out / "policy.ts"
        actor.save(str(path))
        reload = torch.jit.load(str(path), map_location=device).eval()
        assert all(torch.equal(x, y) for x, y in zip(deployment(example[:1]), reload(example[:1])))
    result = dict(seed=seed, arm=arm, presentations=sum(entry["n"] for entry in training_log), steps=steps,
                  policy=digest(path), training=training_log, scores=metrics, stages=stages)
    freeze(out / "report.json", result)
    print(arm, seed, metrics, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = args.data / "th08-trial002-human"
    train_manifest = root / "train-qualified-33-stages-frozen.json"
    held_manifest = root / "held-qualified-12-stages-frozen.json"
    source = args.data / "th08-normal-temporal/change-free/policy.ts"
    prior = Actor(CameraPolicy(), Readout(), False).to(device).eval().requires_grad_(False)
    prior.load_state_dict(torch.jit.load(str(source), map_location=device).state_dict())
    projection = Projection(prior).eval()
    raw_held = list(load(args.data, held_manifest))
    example = raw_held[0]["frames"][raw_held[0]["blocks"][:2]].reshape(-1,8,112,96).to(device).float()/255
    train = cached(load(args.data, train_manifest), prior, projection, device)
    held = cached(raw_held, prior, projection, device)
    if len(train) != 15 or len(held) != 6 or {r["pilot"] for r in train} & {r["pilot"] for r in held}:
        raise ValueError("Unexpected original replay split")
    packed = tuple(torch.cat([r[name].to(device) for r in train]) for name in ("recoded", "global_blocks", "action", "location", "focus"))
    length = len(packed[1])
    args.out.mkdir(parents=True, exist_ok=False)
    freeze(args.out/"protocol.json", dict(prior=digest(source), train=digest(train_manifest), held=digest(held_manifest),
           unique_train=length, unique_held=sum(len(r["action"]) for r in held), epochs=EPOCHS, seeds=SEEDS, batch=BATCH,
           location_weight=AUX_LOCATION, common_input="Aligned 4-step image history, frozen original trunk through strided conv, shared 14x12 cache",
           matching="Fresh pooled and spatial readouts and action heads, shared training rows each epoch, inherited Focus via same teacher logits",
           gate="Spatial readout <=20px position at every seed, >=1.5pp busy gain and >=5/6 stages per seed, and overall direction accuracy within 0.5pp"))
    comparisons = []
    for seed in SEEDS:
        rng = torch.Generator().manual_seed(seed+7)
        draws = [torch.randperm(length, generator=rng).to(device) for _ in range(EPOCHS)]
        results = {}
        for arm in ("pooled", "spatial"):
            out = args.out/f"{arm}-{seed}"
            out.mkdir()
            results[arm] = fit(seed, draws, arm, train, held, packed, projection, example, out, device)
        p, s = results["pooled"], results["spatial"]
        stages = sum(b["busy_correct"] > a["busy_correct"] for a, b in zip(p["stages"], s["stages"]))
        comparisons.append(dict(seed=seed, busy_delta=s["scores"]["busy_accuracy"]-p["scores"]["busy_accuracy"],
                           accuracy_delta=s["scores"]["accuracy"]-p["scores"]["accuracy"], stages_improved=stages,
                           spatial_location_px=s["scores"]["median_location_px"], pooled_location_px=p["scores"]["median_location_px"]))
    gate = all(r["busy_delta"] >= .015 and r["accuracy_delta"] >= -.005 and r["stages_improved"] >= 5
               and r["spatial_location_px"] <= 20 for r in comparisons)
    freeze(args.out/"comparison.json", dict(gate=gate, seeds=comparisons))


if __name__ == "__main__":
    main()
