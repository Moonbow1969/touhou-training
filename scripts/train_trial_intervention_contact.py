"""Learn held-action injury probabilities from randomized official TH14 trial windows."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from scripts import score_trial_contact_windows as trial, train_anticipatory_contact as prior
from scripts.checkpoint_storage import digest
from scripts.collect_trial_contact_windows import TRIAL_SHA256
from scripts.evaluate_sft_native import output_lock
from scripts.train_full_policy import publish_json


SCHEMA = "th14-trial-native-intervention-v1"
DATA = prior.local.DATA
PROTOCOL = DATA.parents[1] / "work/th14-native-intervention-protocol-261125.json"
PROTOCOL_SHA256 = "006ff59161dfdb0c459c71002306d11f75de8b5b582fd10cc086905c7d681986"
PRETRAINED = DATA.parent / "th14-anticipatory-contact-12tick-261125/model.pt"
PILOT = DATA / "th14-trial-held12-prospective-pilot-b-261125"
TRAINING = DATA / "th14-trial-held12-training-only-261125/manifest.json"
RESERVE = DATA / "th14-trial-held12-prospective-expanded-261125"
BASE_SEED = 41101
BATCH = 320
FITS = {"scratch_action": (False, False, .0007), "pretrained_action": (True, False, .0003),
        "pretrained_agnostic": (True, True, .0003)}
HOSTS = {f"{host}{lane}": host for host in ("momentum", "accelerator", "entropy") for lane in ("alpha", "beta")}


@dataclass
class Dataset:
    bullets: torch.Tensor
    context: torch.Tensor
    action: torch.Tensor
    hit: torch.Tensor
    host: np.ndarray
    seed: np.ndarray
    episode: np.ndarray
    source: list

    def __len__(self):
        return len(self.hit)


def safe_child(directory, path):
    child = (directory / path).resolve()
    if not child.is_relative_to(directory.resolve()):
        raise ValueError("A capture descriptor points outside its sealed parent directory.")
    return child


def validate_sampling(sampling):
    if (sampling["near_projectile_radius"] != 112 or sampling["eligible_start_frame"] != 105
            or sampling["shoot"] != 1 or sampling["bomb"] != 0
            or sampling["move_focus"] != "uniform independent choice among 18, held 3 × 4 native ticks"):
        raise ValueError("The trial's randomized action or eligibility protocol differs.")


def collection(directory, permitted):
    path = directory / "collection.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger["status"] not in permitted:
        raise ValueError(f"Incomplete collection ledger: {directory.name}")
    validate_sampling(ledger["sampling"])
    return [dict(row, relative_path=row["file"], physical_host=HOSTS.get(name, name),
                 ledger_sha256=digest(path)) for name, row in sorted(ledger["workers"].items())]


def training_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    specification = manifest["specification"]
    if manifest.get("status") != "complete" or specification["purpose"] != "training_only":
        raise ValueError("The independent training-only manifest is not sealed as complete.")
    validate_sampling(specification["sampling"])
    rows = [row for _, row in sorted(manifest["shards"].items())]
    if sum(row["counts"].get("accepted_windows", 0) for row in rows) < 20000:
        raise ValueError("The locked trial protocol requires at least 20,000 accepted training windows.")
    for row in rows:
        if (row["seed"] in specification["reserved_eval_seeds"]
                or row["shard_sha256"] in specification["reserved_eval_window_shas"]):
            raise ValueError("The sealed training manifest contains a reserved identity.")
        if row["physical_host"] != manifest["worker_host"][row["worker"]]:
            raise ValueError(f"The physical host grouping changed for {row['worker']}.")
        if safe_child(path.parent, row["relative_path"]).stat().st_size != row["shard_bytes"]:
            raise ValueError("Training shard length differs from the sealed manifest.")
    return rows, specification["reserved_eval_collection_sha256"]


def load_dataset(directory, records):
    arrays, source = [], []
    worker_seeds, shard_hashes = set(), set()
    for record in records:
        file = safe_child(directory, record["relative_path"])
        fingerprint = digest(file)
        seed = int(record["seed"])
        if (fingerprint != record["shard_sha256"] or fingerprint in shard_hashes or seed in worker_seeds
                or record["launch"]["executable_sha256"] != TRIAL_SHA256):
            raise ValueError(f"Duplicate, changed, or unqualified worker shard: {file.name}")
        worker_seeds.add(seed)
        shard_hashes.add(fingerprint)
        with gzip.open(file, "rt", encoding="utf-8") as archive:
            rows = [json.loads(line) for line in archive]
        trial.verify_rows(rows, record["counts"])
        if not rows:
            continue
        closed_episodes = set()
        for row in rows:
            if row["episode"] in closed_episodes or not any(
                    active == 1 and radius > 0 and x*x + y*y <= 112**2
                    for x, y, _, _, radius, active in row["start"]["bullets"]):
                raise ValueError("A window broke the preregistered near-bullet eligibility or first-injury support.")
            if row["hit_after_step"] is not None:
                closed_episodes.add(row["episode"])
        bullets, context = trial.observations(rows)
        if not torch.isfinite(bullets).all() or not torch.isfinite(context).all():
            raise ValueError("A randomized observation contains invalid numeric values.")
        action = torch.tensor([row["focus"] * 9 + row["move"] for row in rows], dtype=torch.long)
        hit = torch.tensor([row["hit_after_step"] is not None for row in rows], dtype=torch.bool)
        host = record["physical_host"]
        episode = np.asarray([f"{seed}:{row['episode']}" for row in rows])
        arrays.append((bullets.half(), context, action, hit, np.full(len(rows), host), np.full(len(rows), seed), episode))
        source.append(dict(seed=seed, worker=record["worker"], host=host, name=record["relative_path"],
                           windows=len(rows), sha256=fingerprint))
    if not arrays:
        raise ValueError("No accepted randomized intervention windows.")
    columns = [torch.cat(parts) if isinstance(parts[0], torch.Tensor) else np.concatenate(parts)
               for parts in zip(*arrays)]
    return Dataset(*columns, source)


def seed_split(data):
    by_host = {host: np.unique(data.seed[data.host == host]).tolist() for host in np.unique(data.host)}
    if max(map(len, by_host.values())) == 1:
        by_host = {"pooled": np.unique(data.seed).tolist()}
    dev_seeds = []
    for host, seeds in sorted(by_host.items()):
        if len(seeds) < 2:
            continue
        ranked = sorted(seeds, key=lambda seed: sha256(f"{SCHEMA}:{host}:{seed}".encode()).digest())
        dev_seeds.extend(ranked[:min(len(seeds) - 1, max(1, round(.2 * len(seeds))))])
    validation = np.isin(data.seed, dev_seeds)
    for mask in (validation, ~validation):
        target = data.hit[mask]
        if not len(target) or not target.any() or target.all():
            raise ValueError("A predeclared whole-seed split lacks both binary outcomes.")
    return torch.from_numpy(np.flatnonzero(~validation)), torch.from_numpy(np.flatnonzero(validation)), dev_seeds


def load_pretrained(path):
    report = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
    state = torch.load(path, weights_only=True, map_location="cpu")
    if (report["status"] != "complete" or report["model_sha256"] != digest(path)
            or state["schema"] != "th14-learned-contact-12tick-v1" or report["horizon_native_ticks"] != 12):
        raise ValueError("The frozen 12-tick prior failed artifact validation.")
    return state["state_dict"], report["dynamics"]["ReimuA"]


def observed_logit(logits, action, agnostic):
    return logits[:, 18] if agnostic else logits.gather(1, action[:, None]).squeeze(1)


def model_batch(model, dataset, indices, device):
    return model(dataset.bullets[indices].to(device).float(), prior.local.local_context(
        dataset.context[indices].to(device), 0))


@torch.inference_mode()
def infer(model, dataset, indices, device):
    model.eval()
    return torch.cat([model_batch(model, dataset, part, device).cpu() for part in indices.split(512)])


def fit_variant(name, data, train, validation, initial, device, progress):
    warm, agnostic, rate = FITS[name]
    torch.manual_seed(BASE_SEED + 1)
    model = prior.AnticipatoryContact().to(device)
    if warm:
        model.load_state_dict(initial, strict=True)
    optim = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0001)
    ran = torch.Generator().manual_seed(BASE_SEED + 2)
    best, best_epoch, best_state, ledger = float("inf"), 0, None, []
    for epoch in range(1, 25):
        model.train()
        running = 0.
        for batch in train[torch.randperm(len(train), generator=ran)].split(BATCH):
            action = data.action[batch].to(device)
            label = data.hit[batch].to(device).float()
            logits = observed_logit(model_batch(model, data, batch, device), action, agnostic)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            running += loss.detach().item() * len(batch)
        selected = observed_logit(infer(model, data, validation, device), data.action[validation], agnostic)
        held_bce = float(F.binary_cross_entropy_with_logits(selected, data.hit[validation].float()))
        epoch_row = dict(epoch=epoch, train_bce=running / len(train), grouped_validation_bce=held_bce)
        ledger.append(epoch_row)
        progress(name, epoch_row)
        if held_bce < best:
            best, best_epoch = held_bce, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 6:
            break
    model.load_state_dict(best_state, strict=True)
    return best_state, infer(model, data, validation, device), dict(selected_epoch=best_epoch, epochs=ledger)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64).clip(-45, 45)))


def intercept(logits, label):
    lo, hi, target = -40., 40., float(np.mean(label))
    for _ in range(48):
        middle = (lo + hi) / 2
        if sigmoid(logits + middle).mean() > target:
            hi = middle
        else:
            lo = middle
    return float((lo + hi) / 2)


@torch.inference_mode()
def velocity_geometry(data, indices, dynamics):
    speed = torch.tensor(dynamics["speed_units_per_tick"], dtype=torch.float32)
    bounds = torch.tensor(dynamics["bounds"], dtype=torch.float32)
    scale = torch.tensor(prior.local.BULLET_SCALE, dtype=torch.float32)
    directions = torch.from_numpy(prior.local._DIRECTIONS)
    ticks = torch.arange(12, dtype=torch.float32)
    result = []
    for part in indices.split(512):
        bullets, ctx, action = data.bullets[part].float() * scale, data.context[part], data.action[part]
        focus, move = action // 9, action % 9
        prev = (ctx[:, 2] > .5).long()
        first = directions[move] * speed[prev, focus, move, None]
        steady = directions[move] * speed[focus, focus, move, None]
        position = ctx[:, :2] * torch.tensor(prior.local.PLAYER_SCALE)
        path = position[:, None] + first[:, None] * ticks[None, :, None].clamp(max=4)
        path = (path + steady[:, None] * (ticks[None, :, None] - 4).clamp(min=0)).clamp(*bounds)
        projected = bullets[:, None, :, :2] + (ticks[None, :, None, None] + 1) * bullets[:, None, :, 2:4]
        gap = torch.linalg.vector_norm(projected - (path - position[:, None])[:, :, None], dim=-1)
        present = (bullets[:, :, 5] > 0) & (bullets[:, :, 6] > 0) & (bullets[:, :, 4] > 0)
        min_gap = (gap - bullets[:, None, :, 4] - 1).masked_fill(~present[:, None], 512).amin((1, 2))
        result.append(-min_gap.numpy() / 2.5)
    return np.concatenate(result)


def fit(args):
    protocol_fingerprint = digest(args.protocol)
    if protocol_fingerprint != PROTOCOL_SHA256:
        raise ValueError("The locked trial intervention protocol changed.")
    if args.smoke:
        data = load_dataset(args.pilot, collection(args.pilot, ("complete", "partial")))
        manifest_fingerprint = reserve_fingerprint = None
    else:
        records, reserve_fingerprint = training_manifest(args.training_manifest)
        data = load_dataset(args.training_manifest.parent, records)
        manifest_fingerprint = digest(args.training_manifest)
    initial, dynamics = load_pretrained(args.pretrained)
    train, validation, dev_seeds = seed_split(data)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    log = dict(status="fitting", schema=SCHEMA, smoke=args.smoke, protocol_sha256=protocol_fingerprint,
               runner_sha256=digest(Path(__file__)), source_model_sha256=digest(args.pretrained),
               training_manifest_sha256=manifest_fingerprint, reserved_ledger_sha256=reserve_fingerprint,
               training_sources=data.source,
               training_rows=len(train), validation_rows=len(validation), development_seeds=dev_seeds,
               training_injuries=int(data.hit[train].sum()), validation_injuries=int(data.hit[validation].sum()),
               fitted_epochs={}, device=str(device))
    report_path = args.out / "fit.json"
    publish_json(report_path, log)
    started = perf_counter()

    def progress(name, epoch):
        log["fitted_epochs"].setdefault(name, []).append(epoch)
        publish_json(report_path, log)

    model = prior.AnticipatoryContact().to(device)
    model.load_state_dict(initial, strict=True)
    target = data.hit[validation].numpy()
    observed = data.action[validation]
    scores = observed_logit(infer(model, data, validation, device), observed, False).numpy()
    biases = {"frozen_pretrained_dev_intercept": intercept(scores, target),
              "constant_velocity_diagnostic": intercept(velocity_geometry(data, validation, dynamics), target)}
    states, fits = {}, {}
    for name, (_, agnostic, _) in FITS.items():
        state, validated, evidence = fit_variant(name, data, train, validation, initial, device, progress)
        states[name], fits[name] = state, evidence
        biases[name] = intercept(observed_logit(validated, observed, agnostic).numpy(), target)
    archive = args.out / "trial-only-models.pt"
    temporary = archive.with_suffix(".tmp")
    torch.save(dict(schema=SCHEMA, states=states, intercepts=biases, development_prevalence=float(np.mean(target)),
                    source_model_sha256=log["source_model_sha256"], smoke=args.smoke), temporary)
    temporary.replace(archive)
    log.update(status="complete", weights_sha256=digest(archive), fits=fits, development_intercepts=biases,
               seconds=perf_counter() - started)
    publish_json(report_path, log)
    print(json.dumps(dict(status="complete", trial_only=True, smoke=args.smoke, windows=len(data),
                          dev_epochs={name: value["selected_epoch"] for name, value in fits.items()})), flush=True)


def scores(data, payload, initial, dynamics, device):
    action = data.action.numpy()
    all_rows = torch.arange(len(data))
    row = np.arange(len(action))
    bias = payload["intercepts"]
    torch.manual_seed(BASE_SEED + 1)
    model = prior.AnticipatoryContact().to(device)
    random = infer(model, data, all_rows, device).numpy()
    model.load_state_dict(initial, strict=True)
    original = infer(model, data, all_rows, device).numpy()
    out = dict(frozen_random=sigmoid(random[row, action]), frozen_pretrained_raw=sigmoid(original[row, action]),
               frozen_pretrained_dev_intercept=sigmoid(original[row, action] + bias["frozen_pretrained_dev_intercept"]),
               development_prevalence=np.full(len(action), payload["development_prevalence"]),
               constant_velocity_diagnostic=sigmoid(velocity_geometry(data, all_rows, dynamics)
                                                    + bias["constant_velocity_diagnostic"]))
    for name, (_, agnostic, _) in FITS.items():
        model.load_state_dict(payload["states"][name], strict=True)
        predicted = sigmoid(infer(model, data, all_rows, device).numpy() + bias[name])
        out[name] = predicted[:, 18] if agnostic else predicted[row, action]
        if name == "pretrained_action":
            alternatives = predicted[:, :18]
            out["pretrained_action_uniform_marginal"] = alternatives.mean(axis=1)
            out["pretrained_action_same_focus_marginal"] = alternatives.reshape(-1, 2, 9)[row, action // 9].mean(axis=1)
    return out, alternatives


def measure(hit, risk):
    if not len(hit):
        return dict(AP=None, Brier=None, negative_log_likelihood=None, mean_probability=None)
    ap = prior.ap(hit, risk)
    bound = risk.clip(1e-9, 1. - 1e-9)
    return dict(AP=float(ap) if ap is not None else None, Brier=float(np.mean((hit - risk) ** 2)),
                negative_log_likelihood=float(np.mean(np.where(hit, -np.log(bound), -np.log1p(-bound)))),
                mean_probability=float(risk.mean()))


def reliability(hit, prediction):
    return [dict(windows=len(part), injuries=int(hit[part].sum()), observed_rate=float(hit[part].mean()),
                 mean_prediction=float(prediction[part].mean()), score_min=float(prediction[part].min()),
                 score_max=float(prediction[part].max()))
            for part in np.array_split(np.argsort(prediction, kind="stable"), 10)]


def paired_intervals(data, prediction, repetitions=600):
    by_host = defaultdict(lambda: defaultdict(list))
    for index, (host, episode) in enumerate(zip(data.host, data.episode)):
        by_host[host][episode].append(index)
    grouped = [list(episodes.values()) for episodes in by_host.values()]
    names = ("frozen_pretrained_dev_intercept", "pretrained_agnostic", "pretrained_action_uniform_marginal",
             "pretrained_action_same_focus_marginal", "scratch_action")
    contrasts = {name: [] for name in names}
    uncertainty = {name: [] for name in prediction}
    ran = np.random.default_rng(41109)
    labels = data.hit.numpy()
    for _ in range(repetitions):
        indices = np.concatenate([episodes[i] for episodes in grouped for i in ran.integers(0, len(episodes), len(episodes))])
        y = labels[indices]
        if not y.any() or y.all():
            continue
        scores = {name: (float(prior.ap(y, values[indices])), float(np.mean((y - values[indices]) ** 2)))
                  for name, values in prediction.items()}
        ap, brier = scores["pretrained_action"]
        for name in names:
            contrasts[name].append((ap - scores[name][0], scores[name][1] - brier))
        for name, item in scores.items():
            uncertainty[name].append(item)

    def interval(rows, names):
        values = np.quantile(rows, [.025, .975], axis=0)
        return {name: values[:, i].tolist() for i, name in enumerate(names)}

    return dict(episodes=int(sum(map(len, grouped))), episodes_by_host={name: len(value) for name, value in by_host.items()},
                draws=len(uncertainty["pretrained_action"]),
                paired_fine_tuned_advantage={name: interval(value, ("AP_delta", "Brier_reduction"))
                                             for name, value in contrasts.items()},
                intervals={name: interval(value, ("AP", "Brier")) for name, value in uncertainty.items()})


def evaluate(args):
    fit_report = args.fit / "fit.json"
    fitted = json.loads(fit_report.read_text(encoding="utf-8"))
    weights = args.fit / "trial-only-models.pt"
    if (fitted["status"] != "complete" or fitted["smoke"] or fitted["weights_sha256"] != digest(weights)
            or fitted["protocol_sha256"] != digest(args.protocol) or fitted["runner_sha256"] != digest(Path(__file__))
            or fitted["source_model_sha256"] != digest(args.pretrained)):
        raise ValueError("Reserved evaluation requires the unchanged sealed production fit and locked protocol.")
    ledger = args.reserve / "collection.json"
    key = dict(fit_sha256=digest(fit_report), reserved_ledger_sha256=digest(ledger))
    if key["reserved_ledger_sha256"] != fitted["reserved_ledger_sha256"]:
        raise ValueError("The evaluation ledger differs from the reserve sealed before training.")
    payload = torch.load(weights, weights_only=True, map_location="cpu")
    if (payload["schema"] != SCHEMA or payload["smoke"]
            or payload["source_model_sha256"] != fitted["source_model_sha256"]):
        raise ValueError("The model package is not the sealed trial-only production fit.")
    initial, dynamics = load_pretrained(args.pretrained)
    report_path = args.fit / "reserved-evaluation.json"
    claim = args.fit / "reserved-evaluation-claim.json"
    with output_lock(args.fit):
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["sealed_evaluation"] != key:
                raise ValueError("The fixed held-out evaluation already consumed a different data fingerprint.")
            print(json.dumps(dict(status="already_scored", metrics=report["pooled_metrics"])), flush=True)
            return
        records = collection(args.reserve, ("complete",))
        used_seeds = {row["seed"] for row in fitted["training_sources"]}
        used_hashes = {row["sha256"] for row in fitted["training_sources"]}
        if any(row["seed"] in used_seeds or row["shard_sha256"] in used_hashes for row in records):
            raise ValueError("A reserved shard or sampler seed was used in the fit.")
        if sum(row["counts"].get("accepted_windows", 0) for row in records) != 4272:
            raise ValueError("The reserved cohort does not contain the 4,272 predeclared windows.")
        if claim.is_file() and json.loads(claim.read_text(encoding="utf-8")) != key:
            raise ValueError("Held-out recovery requires identical fit and reserve fingerprints.")
        publish_json(claim, key)
        data = load_dataset(args.reserve, records)
        device = torch.device(args.device)
        predicted, matrix = scores(data, payload, initial, dynamics, device)
        y, assigned = data.hit.numpy(), data.action.numpy()
        pooled = {name: measure(y, risk) for name, risk in predicted.items()}
        hosts = {host: dict(windows=int(np.sum(data.host == host)), injuries=int(y[data.host == host].sum()),
                            metrics={name: measure(y[data.host == host], risk[data.host == host])
                                     for name, risk in predicted.items()}) for host in np.unique(data.host)}
        strata = []
        for action in range(18):
            group = assigned == action
            strata.append(dict(move=action % 9, focus=action // 9, windows=int(group.sum()),
                               injuries=int(y[group].sum()), observed_injury_rate=float(y[group].mean()) if group.any() else None,
                               actor=measure(y[group], predicted["pretrained_action"][group]),
                               agnostic=measure(y[group], predicted["pretrained_agnostic"][group])))
        report = dict(status="complete", schema=SCHEMA, sealed_evaluation=key, reserved_sources=data.source,
                      design="Randomized factual held-action risk in the official 0.01b trial. Stock transfer untested.",
                      windows=len(y), observed_injuries=int(y.sum()), injury_prevalence=float(y.mean()),
                      pooled_metrics=pooled, by_physical_host=hosts, by_randomized_action=strata,
                      reliability={name: reliability(y, values) for name, values in predicted.items()},
                      action_dependence=dict(average_per_state_risk_range=float(np.ptp(matrix, axis=1).mean()),
                                             factual_Brier=pooled["pretrained_action"]["Brier"],
                                             uniformly_averaged_Brier=pooled["pretrained_action_uniform_marginal"]["Brier"],
                                             factual_AP=pooled["pretrained_action"]["AP"],
                                             uniformly_averaged_AP=pooled["pretrained_action_uniform_marginal"]["AP"],
                                             note="Action averages and risk ranges are model estimates; outcomes for the other 17 actions are unobserved."),
                      paired_episode_cluster_uncertainty=paired_intervals(data, predicted))
        publish_json(report_path, report)
    print(json.dumps(dict(status="complete", windows=len(y), injuries=int(y.sum()), metrics=pooled)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_args = commands.add_parser("fit")
    fit_args.add_argument("--out", type=Path, required=True)
    fit_args.add_argument("--smoke", action="store_true")
    fit_args.add_argument("--pilot", type=Path, default=PILOT)
    fit_args.add_argument("--training-manifest", type=Path, default=TRAINING)
    reserved = commands.add_parser("evaluate")
    reserved.add_argument("--fit", type=Path, required=True)
    reserved.add_argument("--reserve", type=Path, default=RESERVE)
    for subparser in (fit_args, reserved):
        subparser.add_argument("--protocol", type=Path, default=PROTOCOL)
        subparser.add_argument("--pretrained", type=Path, default=PRETRAINED)
        subparser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(3)
    (fit if args.command == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()
