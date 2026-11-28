"""Compare observation resolution and learned localization on the same new originals."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_th08_camera_hazard import ENCODER, SEED, evaluate, fit, load_credit, pack, read
from scripts.train_th08_spatial_hazard import BATCH, UPDATES, WINDOW, coordinates, fit_spatial, restrict_to_history


MINIMUM = 24
SEEDS = tuple(range(80964001, 80964033))


def paired_cameras(manifest, row, device):
    credit = load_credit(manifest, row, None, device)
    source = manifest.parent / row["highres"]
    if digest(source) != row["highres_sha256"]:
        raise ValueError(f"Changed high-resolution camera source {row['seed']}")
    indices = np.asarray(row["original_observation_indices"])
    high = np.load(source, allow_pickle=False)["gray"]
    if (high.dtype != np.uint8 or high.ndim != 3 or high.shape[1:] != (224,192)
            or indices.dtype.kind not in "iu" or indices.shape != credit.tick.shape
            or indices[0] < 0 or indices[-1] >= len(high) or np.any(np.diff(indices) <= 0)):
        raise ValueError(f"Invalid exact decision-to-image index for {row['seed']}")
    high = high[indices]
    # Each modality is sampled from the same raw XGetImage. Area averaging two
    # consecutive resampling stages can change integer luminance by at most one.
    low_from_high = (high.reshape(-1,112,2,96,2).sum((2,4), dtype=np.int32)+2)//4
    if np.abs(low_from_high - credit.visual.numpy()).max() > 1:
        raise ValueError(f"Different low- and high-resolution original cameras for {row['seed']}")
    history = restrict_to_history(credit)
    return credit, torch.from_numpy(high[history])


def allocate(rows):
    ordered = sorted((r["seed"] for r in rows), key=lambda seed: hashlib.sha256(f"resolution-v1:{seed}".encode()).digest())
    return [sorted(ordered[i::5]) for i in range(5)]


def select(manifest):
    registration = read(manifest.parent / "registration.json")
    if registration["policy_sha256"] != ENCODER or tuple(r["seed"] for r in registration["assignments"]) != SEEDS:
        raise ValueError("The fresh same-policy collection registration changed")
    final = read(manifest)
    if (final["collection_complete"] is not True or final["registered_credits"] != len(SEEDS)
            or final["minimum_qualified"] != MINIMUM or final["eligible_for_fitting"] is not True):
        raise ValueError("Fresh collection is incomplete or failed the smallest registered fit size")
    selected = final["rows"]
    unique = {row["seed"] for row in selected}
    if (len(unique) != len(selected) or any(row["policy_sha256"] != ENCODER or row["seed"] not in SEEDS
            or row["role"] != "train" or row["rank"] != "Normal" or row["team"] != "Ghost"
            or row["terminal_stage"] != 1 or row["qualified"] is not True for row in selected)):
        raise ValueError("Unregistered, duplicated, or changed behavior in fresh collection")
    if len(selected) < MINIMUM or final["qualified_credits"] != len(selected):
        raise ValueError(f"Only {len(selected)} qualified Stage 1 credits; need {MINIMUM}")
    return selected, final["stage2_confirmed_credits"]


def camera_score(bank, pools, device, fold, use_diagnostic, filename):
    packed = [pack(bank, pool, WINDOW, device, "visual") for pool in pools]
    result = fit_spatial(*packed, coordinates(packed[0]).to(device), SEED+fold,
                         use_diagnostic_coordinates=use_diagnostic)
    p, cutoff, location, log, params = result
    score = evaluate(p, cutoff, packed[-1], WINDOW[0])
    if not use_diagnostic:
        distance = (location - coordinates(packed[-1])).norm(dim=1).numpy()
        score["locator_native_pixels_p50_p90"] = np.quantile(distance, [.5,.9]).tolist()
    score["optimizer"] = log
    torch.save(params, filename)
    score["weights_sha256"] = digest(filename)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=0.)
    parser.add_argument("--verify-guard-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = args.collection / "qualified-train.json"
    if args.verify_guard_only:
        for seed in SEEDS[:2]:
            row = read(args.collection / f"native08/selected/{seed}.json")
            credit, high = paired_cameras(manifest, row, device)
            print(f"Verified original {seed}: {credit.visual.shape}, {high.shape}, {len(credit.hit_tick)} fatal hits", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = args.out / "protocol.json"
    freeze(protocol, dict(registered_seed_range=[SEEDS[0],SEEDS[-1]], collection_policy_sha256=ENCODER,
           collection_registration_sha256=digest(args.collection / "registration.json"),
           admission="All qualified independent Stage 1 original credits from the final collection manifest. Retain Stage 2 progress as a policy outcome; this analysis models Stage 1. Minimum 24 qualified Stage 1 credits.",
           grouping="Five whole-credit hash folds, with one test and one distinct calibration fold per fit.",
           comparisons="Factorial 96x112/192x224 crossed with learned/true-coordinate grid center. All four arms share samples, original ticks, target labels, splits, optimizer budget and 16-native-pixel crop lattice. True-coordinate arms are diagnostic only.",
           gate="Four held folds must pass AP >=1.5x the privileged logistic baseline, >=50% early event coverage and ECE <=.02. Learned branch also needs median locator error <=12 pixels and 80% of oracle AP gain in those folds for a proposed BC warm-start.",
           threshold="5% negative alarms on separate calibration credits. Early warning means at least 20 native ticks ahead of a stock-confirmed hit.",
           secondary="A stronger single diagnostic arm motivates a controlled follow-up. Forecast skill here does not certify gameplay improvement.",
           updates=UPDATES, batch=BATCH, window=list(WINDOW)))
    expiry = time.monotonic() + args.wait_seconds
    while not manifest.is_file():
        if time.monotonic() >= expiry:
            raise TimeoutError("Current collection has not emitted its final manifest")
        print("Waiting for sealed fresh original collection", flush=True)
        time.sleep(30)
    rows, progressed = select(manifest)
    folds = allocate(rows)
    freeze(args.out / "admitted.json", dict(manifest_sha256=digest(manifest), groups=folds,
           stage2_confirmed_original_credits=progressed,
           credits=[dict(seed=r["seed"], standard_sha256=r["sha256"], highres_sha256=r["highres_sha256"]) for r in rows]))
    bank = {}
    high_camera = {}
    for row in rows:
        credit, high = paired_cameras(manifest, row, device)
        bank[credit.seed] = credit
        high_camera[credit.seed] = high
        print(f"Qualified both modalities for {credit.seed}: {credit.vulnerable.sum()} exact histories", flush=True)
    standard_camera = {seed: credit.visual for seed, credit in bank.items()}
    folds_scored = []
    for fold, held in enumerate(folds):
        inner = (fold+1) % 5
        train = [seed for i, group in enumerate(folds) if i not in (fold, inner) for seed in group]
        pools = (train, folds[inner], held)
        packed = [pack(bank, pool, WINDOW, device, "clocks") for pool in pools]
        p, cutoff, log, weights = fit(*packed, False, SEED+fold)
        reference = evaluate(p, cutoff, packed[-1], WINDOW[0])
        reference["optimizer"] = log
        path = args.out / f"fold{fold}-baseline.pt"
        torch.save(weights, path)
        reference["weights_sha256"] = digest(path)
        arms = {}
        baseline_ap = reference["average_precision"]
        for resolution, cameras in ((96, standard_camera), (192, high_camera)):
            for seed, camera in cameras.items():
                bank[seed].visual = camera
            for name, native in (("learned", False), ("coordinates", True)):
                arm = f"{resolution}-{name}"
                print(f"Fitting fold {fold} arm {arm}", flush=True)
                score = camera_score(bank, pools, device, fold, native, args.out / f"fold{fold}-{arm}.pt")
                ap = score["average_precision"]
                score["gate_passed"] = ap >= 1.5*baseline_ap and score["event_coverage"] >= .5 and score["ece_10"] <= .02
                arms[arm] = score
            learned = arms[f"{resolution}-learned"]
            oracle_ap = arms[f"{resolution}-coordinates"]["average_precision"]
            learned["bc_eligibility_gate_passed"] = (learned["gate_passed"] and learned["locator_native_pixels_p50_p90"][0] <= 12
                and learned["average_precision"] - baseline_ap >= .8*max(oracle_ap-baseline_ap,0))
        result = dict(fold=fold, held=held, clocks=reference, arms=arms)
        freeze(args.out / f"fold{fold}.json", result)
        folds_scored.append(result)
    summarize = {arm: dict(folds_passed=sum(row["arms"][arm]["gate_passed"] for row in folds_scored),
                          folds_eligible_bc=sum(row["arms"][arm].get("bc_eligibility_gate_passed",False) for row in folds_scored))
                 for arm in folds_scored[0]["arms"]}
    freeze(args.out / "report.json", dict(protocol_sha256=digest(protocol), credits=len(rows),
           stage2_confirmed_original_credits=progressed, arms=summarize, folds=folds_scored))
    print(summarize, flush=True)


if __name__ == "__main__":
    main()
