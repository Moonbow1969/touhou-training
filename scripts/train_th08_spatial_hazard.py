"""Test a player-centered temporal camera CNN on exact four-tick TH08 originals."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_th08_camera_hazard import SEED, admit, allocate, calibrate, evaluate, fit, load_credit, pack, read


UPDATES = 540
BATCH = 40
WINDOW = (8, 32)


class SpatialForecast(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.native_stride = 16 / scale
        self.trunk = nn.Sequential(nn.Conv2d(7, 24, 5, stride=2, padding=2), nn.GroupNorm(4, 24), nn.SiLU(),
                                   nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.locator = nn.Conv2d(32, 1, 1)
        self.risk = nn.Sequential(nn.Linear(32 * 50, 64), nn.SiLU(), nn.Linear(64, 1))
        height, width = 28 * scale, 24 * scale
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        self.register_buffer("cells", torch.stack((x, y), -1).flatten(0, 1).float())
        d_y, d_x = torch.meshgrid(torch.arange(-3, 4), torch.arange(-3, 4), indexing="ij")
        self.register_buffer("offset", torch.stack((d_x, d_y), -1).float() * scale)
        self.register_buffer("size_xy", torch.tensor([width, height], dtype=torch.float32))

    def forward(self, history, diagnostic_xy=None):
        # Signed differences are defined in float to keep lightening and darkening distinct.
        camera = history.float() / 255
        features = self.trunk(torch.cat((camera, camera[:, 1:] - camera[:, :-1]), 1))
        heat = self.locator(features).flatten(1).float()
        center = heat.softmax(1) @ self.cells
        crop_center = center if diagnostic_xy is None else diagnostic_xy / self.native_stride - .5
        nearby = crop_center[:, None, None, :] + self.offset
        grid = (nearby + .5) * (2 / self.size_xy) - 1
        local = F.grid_sample(features.float(), grid, align_corners=False)
        features = torch.cat((local.flatten(1), features.mean((2, 3))), 1)
        return self.risk(features).flatten().float(), heat, (center + .5) * self.native_stride


def restrict_to_history(credit):
    offsets = np.array([-12, -8, -4, 0])
    ticks = credit.tick[:, None] + offsets
    physical = np.searchsorted(credit.tick, ticks).clip(0, len(credit.tick)-1)
    exact = np.all(credit.tick[physical] == ticks, axis=1)
    valid = exact & np.all(credit.vulnerable[physical[:, 1:]], axis=1)
    credit.vulnerable = valid
    credit.visual = credit.visual[physical]
    return physical


def fit_spatial(train, calibration, held, train_xy, seed, *, use_diagnostic_coordinates=False):
    x, labels, _ = train
    torch.manual_seed(seed)
    model = SpatialForecast(scale=x.shape[-1] // 96).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.015)
    pos_weight = ((len(labels) - labels.sum()) / labels.sum()).sqrt()
    target_grid = (model.cells + .5) * model.native_stride
    train_heat = (-(train_xy[:, None] - target_grid).square().sum(-1) / (2 * 12**2)).softmax(1)
    log = []
    for step in range(1, UPDATES+1):
        # Weight the rare positives without changing the calibration distribution.
        ids = torch.randint(len(labels), (BATCH,), device=x.device)
        center = train_xy[ids] if use_diagnostic_coordinates else None
        logits, heat, _ = model(x[ids], center)
        risk_loss = F.binary_cross_entropy_with_logits(logits, labels[ids], pos_weight=pos_weight)
        location_loss = -(train_heat[ids] * heat.log_softmax(1)).sum(1).mean()
        loss = risk_loss + .25 * location_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step in (1, 60, 180, 360, UPDATES):
            log.append(dict(update=step, risk_bce=float(risk_loss.detach()), location_ce=float(location_loss.detach())))
            print(f"Spatial optimizer {step}/{UPDATES}: {log[-1]}", flush=True)
    model.eval()

    def predict(packed):
        images = packed[0]
        xy = coordinates(packed).to(images.device) if use_diagnostic_coordinates else None
        with torch.no_grad():
            groups = [model(images[i:i+48], None if xy is None else xy[i:i+48]) for i in range(0, len(images), 48)]
            return (torch.cat([g[0] for g in groups]).cpu().double() - pos_weight.cpu().double().log(),
                    torch.cat([g[2] for g in groups]).cpu())

    cal, _ = predict(calibration)
    test, location = predict(held)
    probabilities, cutoff, slope, bias = calibrate(cal, test, calibration[1].cpu().double())
    params = dict(model=model.cpu().state_dict(), slope=slope, bias=bias, weight_correction=pos_weight.cpu().log(), cutoff=cutoff)
    return probabilities, cutoff, location, log, params


def coordinates(packed):
    return torch.from_numpy(np.concatenate([credit.coordinates[keep] for credit, keep, *_ in packed[2]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "frozen-probe", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = admit(args.data)
    folds = allocate(entries)
    previous = args.frozen_probe / "protocol.json"
    primary = read(previous)
    if primary["fold_groups"] != folds or primary["windows_ticks"][0] != list(WINDOW):
        raise ValueError("The comparison changed the original fold assignment or primary window")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = args.out / "protocol.json"
    freeze(protocol, dict(previous_protocol_sha256=digest(previous), fold_groups=folds, seed=SEED,
           model="Four original grayscale frames at offsets [-12,-8,-4,0] plus signed adjacent differences. Stride-4 spatial trunk, supervised player heatmap, predicted soft-argmax center, bilinear 7x7 feature sampling, global average, 64-unit MLP.",
           controls="Exactly the frozen probe's privileged logistic features, re-fit and evaluated on the same eligible observations as the spatial model.",
           supervision="0.25 player heatmap cross-entropy plus weighted risk BCE. Native position is a training label and a separate baseline input; future state and privileged variables are never camera-network inputs.",
           input_eligibility="All four cameras occur exactly four native ticks apart. All three intervening vulnerable camera pairs must be individually validated.",
           gate="AP at least 1.5x logistic, at least half of true hits alerted >=20 native ticks early, ECE <=0.02, jointly on at least four score folds. Calibrate the threshold for 5% negative alarms separately.",
           window=list(WINDOW), updates=UPDATES, batch=BATCH, tf32_convolution=device.type=="cuda"))
    bank = {}
    for manifest, row in entries:
        credit = load_credit(manifest, row, None, device)
        restrict_to_history(credit)
        bank[credit.seed] = credit
        print(f"Verified {credit.seed}: {int(credit.vulnerable.sum())} exact four-frame histories", flush=True)
    if len(bank) != 45 or sum(len(c.tick) for c in bank.values()) != 40074:
        raise ValueError("Original credit or decision count changed")
    results = []
    for fold, held in enumerate(folds):
        inner = (fold+1) % len(folds)
        calibration = folds[inner]
        training = [seed for i, group in enumerate(folds) if i not in (fold, inner) for seed in group]
        pools = (training, calibration, held)
        print(f"Fitting original fold {fold} on {len(training)} credits", flush=True)
        control = [pack(bank, pool, WINDOW, device, "clocks") for pool in pools]
        prob, cutoff, log, params = fit(*control, False, SEED+fold)
        base = evaluate(prob, cutoff, control[-1], WINDOW[0])
        base["optimizer"] = log
        base_path = args.out / f"fold{fold}-baseline.pt"
        torch.save(params, base_path)
        base["weights_sha256"] = digest(base_path)
        visual = [pack(bank, pool, WINDOW, device, "visual") for pool in pools]
        prob, cutoff, location, log, params = fit_spatial(*visual, coordinates(visual[0]).to(device), SEED+fold)
        score = evaluate(prob, cutoff, visual[-1], WINDOW[0])
        distance = (location - coordinates(visual[-1])).norm(dim=1).numpy()
        score["locator_error_native_pixels"] = dict(mean=float(distance.mean()),
            p50_p90=np.quantile(distance, [.5, .9]).tolist(), within_16_pixels=float((distance <= 16).mean()))
        score["optimizer"] = log
        path = args.out / f"fold{fold}-spatial.pt"
        torch.save(params, path)
        score["weights_sha256"] = digest(path)
        ratio = score["average_precision"] / base["average_precision"]
        passed = ratio >= 1.5 and score["event_coverage"] >= .5 and score["ece_10"] <= .02
        row = dict(fold=fold, guard=WINDOW[0], horizon=WINDOW[1], training=training, calibration=calibration,
                   held=held, ap_ratio=ratio, gate_passed=passed, clocks=base, visual=score)
        freeze(args.out / f"fold{fold}.json", row)
        results.append(row)
        print(f"Spatial fold {fold}: AP ratio {ratio:.3f}, early coverage {score['event_coverage']:.3f}, ECE {score['ece_10']:.4f}, pass {passed}", flush=True)
    n_pass = sum(row["gate_passed"] for row in results)
    freeze(args.out / "report.json", dict(protocol_sha256=digest(protocol), credits=len(bank),
           device=str(device), primary_folds_passed=n_pass, gate_passed=n_pass>=4, folds=results))
    print(f"Final spatial fold passes: {n_pass}/5", flush=True)


if __name__ == "__main__":
    main()
