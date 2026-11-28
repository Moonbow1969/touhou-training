"""Fit and score a camera-only TH08 hit forecast on grouped original credits."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.checkpoint_storage import digest
from scripts.probe_six_game_loss_risk import average_precision
from scripts.th08_camera_history import reconstruct
from scripts.th08_hazard_coordinates import read_native_camera
from scripts.train_five_original_games import freeze
from scripts.train_th08_normal_current_ppo import CurrentActorCritic, ORIGINAL_EXE
from scripts.train_th08_progress_lambda_one import PARENT, SEEDS, TECHNICAL_EXCLUSIONS
from scripts.train_th08_progress_pair import EXPECTED_ACTOR, PAIR_SEEDS


ENCODER = "8a0e4ff3a8e6f552db9bc5d2cf34f15db9a2c12d41454e90a9e297f99099b0b0"
SEED = 80963973
WINDOWS = ((8, 32), (16, 40))
UPDATES = 240


@dataclass
class Credit:
    seed: int
    tick: np.ndarray
    vulnerable: np.ndarray
    hit_tick: np.ndarray
    visual: torch.Tensor
    clocks: torch.Tensor
    coordinates: np.ndarray


def read(path):
    return json.loads(path.read_bytes())


def admit(data):
    first = data / "th08-progress-reward/qualified-train.json"
    second = data / "th08-progress-lambda1-round2/qualification-diagnostic.json"
    initial = sorted(read(first)["rows"], key=lambda row: row["seed"])
    continuation = sorted(read(second)["rows"], key=lambda row: row["seed"])
    if tuple(r["seed"] for r in initial) != PAIR_SEEDS or tuple(r["seed"] for r in continuation) != SEEDS:
        raise ValueError("The registered training-credit lists changed")
    if tuple(r["seed"] for r in continuation if r["qualified"] is not True) != TECHNICAL_EXCLUSIONS:
        raise ValueError("The reviewed 29-credit admission changed")
    return [(first, r) for r in initial] + [(second, r) for r in continuation if r["qualified"] is True]


def allocate(entries):
    strata = defaultdict(list)
    for _, row in entries:
        strata[row["policy_sha256"]].append(row["seed"])
    if {policy: len(seeds) for policy, seeds in strata.items()} != {EXPECTED_ACTOR: 16, PARENT: 29}:
        raise ValueError("Unexpected original collection policies")
    groups = [[] for _ in range(5)]
    for offset, (policy, seeds) in enumerate(sorted(strata.items())):
        order = sorted(seeds, key=lambda s: hashlib.sha256(f"{SEED}:{policy}:{s}".encode()).digest())
        for index, seed in enumerate(order):
            groups[(index + offset) % 5].append(seed)
    return [sorted(group) for group in groups]


def load_credit(manifest, row, encoder, device):
    source, receipt_path = (manifest.parent / row[key] for key in ("file", "receipt"))
    if digest(source) != row["sha256"] or digest(receipt_path) != row["receipt_sha256"]:
        raise ValueError(f"Original source hash mismatch for {row['seed']}")
    receipt = read(receipt_path)
    if (row["role"] != "train" or receipt["seed"] != row["seed"] or receipt["sha256"] != row["sha256"]
            or receipt["policy_sha256"] != row["policy_sha256"] or receipt["game_sha256"] != ORIGINAL_EXE
            or receipt["continues"] != 0 or receipt["outcome"] != "game_over" or not receipt["qualified"]):
        raise ValueError(f"Invalid original lifecycle for {row['seed']}")
    with np.load(source, allow_pickle=False) as trace:
        tick = trace["native_tick"].astype(np.int64)
        following = trace["native_next_tick"].astype(np.int64)
        spares, bombs, done = (trace[field] for field in ("spares", "bombs", "terminated"))
        deaths = trace["life_loss"].astype(bool)
        frames = trace["frames"]
        if (not np.all(trace["native_stage"] == 1) or not np.all(trace["native_next_stage"] == 1)
                or np.any(np.diff(tick) <= 0) or not np.array_equal(following[:-1], tick[1:])
                or deaths.sum() != 3 or np.count_nonzero(done) != 1 or not done[-1]
                or not np.array_equal(deaths, (np.maximum(trace["next_spares"], 0) < spares) | done)
                or receipt["action_masks_with_native_consumption"] != len(tick)):
            raise ValueError(f"Invalid clock, stock ledger, or endpoint for {row['seed']}")
    native = read_native_camera(manifest, row, tick, following[deaths])
    xy = native["xy"].astype(np.float32)
    camera_tick = native["camera_ticks"]
    hits = native["hit_ticks"]
    if xy.shape != (len(tick), 2) or hits.shape != (3,) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid causal native camera context for {row['seed']}")
    centers = np.arange(0, 15653, 240)
    clock_bins = np.maximum(1 - np.abs(camera_tick[:, None] - centers) / 240, 0)
    prior = np.searchsorted(hits, camera_tick, side="right") - 1
    since = camera_tick - np.where(prior >= 0, hits[np.maximum(prior, 0)], 0)
    controls = np.column_stack((xy, xy ** 2, spares, bombs, np.log1p(since), prior < 0))
    clocks = torch.from_numpy(np.column_stack((clock_bins, controls)).astype(np.float32))
    if encoder is None:
        visual = torch.from_numpy(frames)
    else:
        with torch.no_grad():
            batches = (torch.from_numpy(reconstruct(frames, i, min(i+128, len(tick)), 1))
                       for i in range(0, len(tick), 128))
            visual = torch.cat([encoder(batch.to(device).float().div_(255)).cpu() for batch in batches])
    return Credit(row["seed"], camera_tick, native["vulnerable"], hits, visual, clocks, xy)


def samples(credit, guard, horizon):
    target = np.searchsorted(credit.hit_tick, credit.tick, side="right")
    future = target < len(credit.hit_tick)
    ahead = credit.hit_tick[np.minimum(target, len(credit.hit_tick)-1)] - credit.tick
    keep = credit.vulnerable & future & (ahead >= guard)
    return keep, (ahead <= horizon)[keep], target[keep], ahead[keep]


def pack(bank, seeds, window, device, feature):
    rows = [(bank[seed], *samples(bank[seed], *window)) for seed in seeds]
    x = torch.cat([getattr(c, feature)[keep] for c, keep, *_ in rows]).to(device)
    y = torch.from_numpy(np.concatenate([y for _, _, y, *_ in rows]).astype(np.float32)).to(device)
    return x, y, rows


def fit(train, calibration, holdout, visual, seed):
    x, y, _ = train
    torch.manual_seed(seed)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.05 if visual else .001)

    def normalize(v):
        return ((v - mean) / scale).clamp(-10, 10)

    model = (nn.Sequential(nn.Linear(x.shape[1], 64), nn.SiLU(), nn.Linear(64, 1)) if visual
             else nn.Linear(x.shape[1], 1)).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    x = normalize(x)
    pos_weight = ((len(y) - y.sum()) / y.sum()).sqrt()
    log = []
    for step in range(1, UPDATES+1):
        batch = torch.randint(len(y), (512,), device=x.device)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[batch]).flatten(), y[batch], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (1, 60, 120, 180, UPDATES):
            log.append({"update": step, "weighted_bce": float(loss.detach())})
    with torch.no_grad():
        cal = model(normalize(calibration[0])).flatten().cpu().double() - pos_weight.cpu().double().log()
        test = model(normalize(holdout[0])).flatten().cpu().double() - pos_weight.cpu().double().log()
    test_p, threshold, slope, bias = calibrate(cal, test, calibration[1].cpu().double())
    params = dict(model=model.cpu().state_dict(), mean=mean.cpu(), scale=scale.cpu(),
                  slope=slope, bias=bias, weight_correction=pos_weight.cpu().log(), cutoff=threshold)
    return test_p, threshold, log, params


def calibrate(cal, test, target):
    # A separate whole-credit fold calibrates the probability and alarm threshold.
    raw = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    calibrator = torch.optim.LBFGS([raw], max_iter=50, line_search_fn="strong_wolfe")

    def objective():
        calibrator.zero_grad(set_to_none=True)
        p = cal * raw[0].exp() + raw[1]
        loss = nn.functional.binary_cross_entropy_with_logits(p, target) + .001 * raw.square().sum()
        loss.backward()
        return loss

    calibrator.step(objective)
    with torch.no_grad():
        slope, bias = raw[0].exp(), raw[1]
        cal_p = (cal * slope + bias).sigmoid().numpy()
        test_p = (test * slope + bias).sigmoid().numpy()
    if not np.isfinite(cal_p).all() or not np.isfinite(test_p).all():
        raise ValueError("Probe calibration produced nonfinite probabilities")
    threshold = float(np.quantile(cal_p[target.numpy() == 0], .95, method="higher"))
    return test_p, threshold, slope, bias


def evaluate(prob, threshold, packed, guard):
    labels = packed[1].cpu().numpy().astype(bool)
    offset = 0
    hits = 0
    events = 0
    by_credit = []
    for credit, _, y, event_id, ahead in packed[2]:
        p = prob[offset:offset+len(y)]
        early = (p > threshold) & y & (ahead >= guard + 12)
        hit = len(np.unique(event_id[early]))
        ap = average_precision(y, p) if y.any() else None
        by_credit.append(dict(seed=credit.seed, observations=len(y), positives=int(y.sum()),
                              ap=ap, detected=hit, events=len(credit.hit_tick)))
        hits += hit
        events += len(credit.hit_tick)
        offset += len(y)
    buckets = np.minimum((prob * 10).astype(int), 9)
    ece = sum(abs(float((prob[buckets == b] - labels[buckets == b]).sum())) for b in range(10)) / len(labels)
    return dict(average_precision=average_precision(labels, prob), positive_rate=float(labels.mean()),
                held_false_positive_rate=float((prob[~labels] > threshold).mean()),
                early_events=hits, total_events=events, event_coverage=hits/events,
                ece_10=ece, brier=float(np.mean((prob - labels) ** 2)), cutoff=threshold, by_credit=by_credit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = admit(args.data)
    folds = allocate(entries)
    actor = args.data / "th08-progress-lambda1-round2/policy/progress-plus-spares/policy.ts"
    if digest(actor) != ENCODER:
        raise ValueError("The frozen original camera actor changed")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = args.out / "protocol.json"
    freeze(protocol, dict(encoder=ENCODER, fold_groups=folds, seed=SEED,
           training_manifest_hashes={p.name: digest(p) for p in sorted({p for p, _ in entries})},
           windows_ticks=[list(w) for w in WINDOWS],
           inner_allocation="Three folds fit, next whole fold calibrates and sets the 5% false alarm threshold; rotating held fold only scores.",
           baseline="Logistic on elapsed game clock, native pre-action x/y and squared terms, current stocks and since-last-hit clock.",
           visual="Frozen deployed 128-dimensional camera trunk; readout is a 64-unit SiLU MLP on raw grayscale and previous-frame absolute difference. No stock, coordinates or actions enter the visual arm.",
           label="Native fatal-hit entry matched to later stock loss or Game Over. Require both camera frames to be in normal player state. This excludes visible hit animations and most post-death immunity.",
           weights="Square-root inverse rare-event odds during both fits, then separate temperature/intercept fits on calibration credits.",
           gate="At least four primary folds satisfy AP >= 1.5x baseline, event coverage >= 0.5 and ECE <= .02. Held false-positive rate is reported. Wider guard is a prespecified sensitivity analysis.",
           coverage="Cutoff set on calibration at 5% controls. Alert must occur at least 12 native ticks before the exclusion guard. Every death remains in denominator.",
           independence="The supervised head is held out by full credit. The deployed trunk has prior PPO exposure to these credits. Before shaping, require a fresh independent native shadow batch.",
           updates=UPDATES, batch=512))
    model = CurrentActorCritic().eval()
    model.load_state_dict(torch.jit.load(str(actor), map_location="cpu").state_dict())
    encoder = model.actor.features.to(device).eval()
    bank = {}
    for manifest, row in entries:
        credit = load_credit(manifest, row, encoder, device)
        bank[credit.seed] = credit
        print(f"Verified original credit {credit.seed}: {len(credit.tick)} decisions, {len(credit.hit_tick)} hits", flush=True)
    total = sum(len(c.tick) for c in bank.values())
    if total != 40074 or sum(len(c.hit_tick) for c in bank.values()) != 135:
        raise ValueError("Original decision or death count changed")
    results = []
    for window in WINDOWS:
        for fold, held in enumerate(folds):
            inner = (fold + 1) % len(folds)
            calibration = folds[inner]
            training = [seed for i, group in enumerate(folds) if i not in (fold, inner) for seed in group]
            scores = {}
            for feature in ("clocks", "visual"):
                packs = [pack(bank, pool, window, device, feature) for pool in (training, calibration, held)]
                prob, threshold, log, params = fit(*packs, feature == "visual", SEED+fold)
                scores[feature] = evaluate(prob, threshold, packs[-1], window[0])
                scores[feature]["optimizer"] = log
                weights = args.out / f"guard{window[0]}-fold{fold}-{feature}.pt"
                torch.save(params, weights)
                scores[feature]["weights_sha256"] = digest(weights)
            visual = scores["visual"]
            ratio = visual["average_precision"] / scores["clocks"]["average_precision"]
            passed = ratio >= 1.5 and visual["event_coverage"] >= .5 and visual["ece_10"] <= .02
            result = dict(guard=window[0], horizon=window[1], fold=fold, training=training, calibration=calibration,
                          held=held, ap_ratio=ratio, gate_passed=passed, **scores)
            results.append(result)
            freeze(args.out / f"guard{window[0]}-fold{fold}.json", result)
            print(f"Guard {window[0]} fold {fold}: AP ratio {ratio:.4f}, coverage {visual['event_coverage']:.4f}, "
                  f"ECE {visual['ece_10']:.4f}, FPR {visual['held_false_positive_rate']:.4f}, pass {passed}", flush=True)
    primary = [r for r in results if r["guard"] == WINDOWS[0][0]]
    n_pass = sum(r["gate_passed"] for r in primary)
    freeze(args.out / "report.json", dict(protocol_sha256=digest(protocol), credits=len(bank),
           full_decisions=total, deaths=135, device=str(device), primary_folds_passed=n_pass,
           gate_passed=n_pass >= 4, folds=results))
    print(f"Final primary folds passed: {n_pass}/5", flush=True)


if __name__ == "__main__":
    main()
