"""Score held-action trial injuries with the frozen TH14 anticipatory contact model."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.collect_trial_contact_windows import action_mask, TRIAL_SHA256
from scripts.train_anticipatory_contact import AnticipatoryContact, ap
from scripts.train_full_policy import publish_json
from scripts.train_local_avoidance import BULLET_SCALE, local_context


def observations(rows):
    bullets = np.zeros((len(rows), 96, 7), dtype=np.float32)
    context = np.zeros((len(rows), 3), dtype=np.float32)
    for i, row in enumerate(rows):
        start = row["start"]
        raw = np.asarray(start["bullets"], dtype=np.float32)
        if not len(raw) or len(raw) != min(96, start["bullet_count"]):
            raise ValueError("The recorded randomized state does not contain its projectile observations.")
        bullets[i, :len(raw), :6] = raw / BULLET_SCALE[:6]
        bullets[i, :len(raw), 6] = 1
        x, y = start["player"]
        context[i] = (x/192, y/448, int(start["prior_focus"]))
    return torch.from_numpy(bullets.clip(-1, 1)), torch.from_numpy(context.clip(-1, 1))


def verify_rows(rows, counts):
    if len(rows) != counts.get("accepted_windows", 0):
        raise ValueError("Shard row count differs from the worker completion ledger.")
    losses = 0
    last = (-1, -100)
    for row in rows:
        episode, start = row["episode"], row["frame"]
        if row["held_mask"] != action_mask(row["move"], row["focus"]) or row["lives"] != 2 or start < 105:
            raise ValueError("Unexpected startup state or held action in a sampled window.")
        if episode < last[0] or episode == last[0] and start < last[1] + 12:
            raise ValueError("The sample contains overlapping randomized windows.")
        last = episode, start
        transitions = row["transitions"]
        if len(transitions) != 3 or any(
                transition[0] - start != 4 * step or (transition[1] - row["tick"]) % (2**32) != 4 * step
                for step, transition in enumerate(transitions, start=1)):
            raise ValueError("Each scoring window needs a verified four-tick triplet.")
        hits = [i for i, transition in enumerate(transitions, start=1) if transition[2] == 4]
        if (hits[0] if hits else None) != row["hit_after_step"]:
            raise ValueError("The injury endpoint differs from the observed native onset.")
        if row["confirmed_lives"] != 2 - int(bool(hits)):
            raise ValueError("The native life stock did not corroborate injury.")
        losses += bool(hits)
    if losses != counts.get("injuries", 0):
        raise ValueError("The reported number of confirmed native injuries differs from the shard.")
    return losses


def score_metrics(y, pred):
    positive = pred[y]
    negative = pred[~y]
    auroc = None
    if len(positive) and len(negative):
        comparison = positive[:, None] - negative
        auroc = float(((comparison > 0) + .5 * (comparison == 0)).mean())
    ap_score = ap(y, pred)
    return dict(ap=float(ap_score) if ap_score is not None else None, auroc=auroc,
                brier=float(((pred - y) ** 2).mean()))


def bins(y, prob):
    result = []
    for indices in np.array_split(np.argsort(prob, kind="stable"), min(4, len(y))):
        hits = int(y[indices].sum())
        result.append(dict(n=len(indices), injuries=hits, injury_rate=hits/len(indices),
                           min_score=float(prob[indices].min()), max_score=float(prob[indices].max()),
                           mean_score=float(prob[indices].mean())))
    return result


def action_counts(rows, y, prob):
    matrix = defaultdict(list)
    for i, row in enumerate(rows):
        matrix[(row["focus"], row["move"])].append(i)
    return [dict(focus=key[0], movement=key[1], n=len(ids), injuries=int(y[ids].sum()),
                 mean_score=float(prob[ids].mean()) if ids else None)
            for key in [(focus, move) for focus in range(2) for move in range(9)] for ids in [matrix[key]]]


def movement_check(rows):
    moves = ([], [])
    for row in rows:
        x, y = row["start"]["player"]
        first = row["transitions"][0]
        if row["move"] == 4 or first[2] != 1 or not (-169 < x < 169 and 47 < y < 417):
            continue
        dy, dx = divmod(row["move"], 3)
        direction = np.array((dx-1, dy-1), dtype=float)
        direction /= np.linalg.norm(direction)
        actual = np.asarray(first[-1]) - (x, y)
        moves[row["focus"]].append(float(actual @ direction / 4))
    return [dict(focus=i, n=len(group), positive=sum(x > .05 for x in group),
                 median_speed=float(np.median(group)) if group else None)
            for i, group in enumerate(moves)]


def confidence(rows, y, action, average, agnostic):
    episodes = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        episodes[row["host"]][(row["worker"], row["episode"])].append(i)
    episode_count = sum(len(group) for group in episodes.values())
    if episode_count < 4 or y.sum() < 3:
        return None
    rng = np.random.default_rng(12601)
    groups = [list(group.values()) for group in episodes.values()]
    estimates = []
    for _ in range(600):
        subset = np.concatenate([group[k] for group in groups for k in rng.integers(0, len(group), size=len(group))])
        if y[subset].any() and (~y[subset]).any():
            levels = [ap(y[subset], value[subset]) for value in (action, average, agnostic)]
            estimates.append((levels[0]-levels[1], levels[0]-levels[2]))
    interval = np.quantile(estimates, [.025,.975], axis=0)
    return dict(episodes=episode_count, stratified_by_host=True, draws=len(estimates),
                action_minus_same_focus_ap_95=interval[:,0].tolist(),
                action_minus_agnostic_ap_95=interval[:,1].tolist())


def grouped_scores(rows, key, y, action, average, agnostic):
    result = {}
    for name in sorted({row[key] for row in rows}):
        selected = np.array([row[key] == name for row in rows])
        local_y = y[selected]
        result[name] = dict(windows=int(selected.sum()), injuries=int(local_y.sum()),
                            action=score_metrics(local_y, action[selected]),
                            same_focus_average=score_metrics(local_y, average[selected]),
                            agnostic=score_metrics(local_y, agnostic[selected]))
    return result


@torch.no_grad()
def infer_all(model_path, rows):
    model_fingerprint = digest(model_path)
    trained = json.loads(model_path.with_name("report.json").read_text())
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if (trained["status"] != "complete" or trained["model_sha256"] != model_fingerprint
            or payload["schema"] != "th14-learned-contact-12tick-v1"):
        raise ValueError("The prospective trial requires the exact frozen 12-tick actor.")
    model = AnticipatoryContact().eval()
    model.load_state_dict(payload["state_dict"])
    bullets, context = observations(rows)
    chunks = [model(bullets[i:i+256], local_context(context[i:i+256],0)).sigmoid().numpy()
              for i in range(0,len(rows),256)]
    return np.concatenate(chunks), model_fingerprint


def infer(args, rows):
    pred, model_fingerprint = infer_all(args.model, rows)
    move, focus = (np.asarray([r[key] for r in rows]) for key in ("move", "focus"))
    i = np.arange(len(rows))
    return pred[i,focus*9+move], pred[:,:18].reshape(-1,2,9)[i,focus].mean(1), pred[:,18], model_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-host", action="append", default=[], metavar="WORKER=HOST")
    args = parser.parse_args()
    torch.set_num_threads(1)
    capture = json.loads((args.capture / "collection.json").read_text())
    if capture["status"] not in ("complete", "partial") or len(capture["workers"]) < 2:
        raise ValueError("The prospective measurement needs two independently completed workers.")
    assignment = dict(item.split("=", 1) for item in args.worker_host)
    if len(assignment) != len(args.worker_host) or not set(assignment) <= set(capture["workers"]):
        parser.error("Use each completed worker at most once when assigning host groups.")
    rows = []
    for name, worker in capture["workers"].items():
        path = args.capture / worker["file"]
        if digest(path) != worker["shard_sha256"] or worker["launch"]["executable_sha256"] != TRIAL_SHA256:
            raise ValueError("A worker shard or the original trial version changed.")
        with gzip.open(path,"rt",encoding="utf-8") as shard:
            wave = [dict(json.loads(line), worker=name, host=assignment.get(name, name)) for line in shard]
        verify_rows(wave, worker["counts"])
        rows.extend(wave)
    if not rows:
        raise ValueError("No eligible randomized windows survived validation.")
    y = np.array([row["hit_after_step"] is not None for row in rows])
    action, average, agnostic, model_fingerprint = infer(args, rows)
    by_worker = grouped_scores(rows, "worker", y, action, average, agnostic)
    by_host = grouped_scores(rows, "host", y, action, average, agnostic)
    report = dict(capture=capture, model_sha256=model_fingerprint, evaluation_sha256=digest(Path(__file__)),
                  windows=len(y), injuries=int(y.sum()), observed_injury_rate=float(y.mean()),
                  evaluation_scope="Independent randomized held actions in official 0.01b trial only. No full-build clear inference.",
                  action=score_metrics(y,action), same_focus_average=score_metrics(y,average), agnostic=score_metrics(y,agnostic),
                  action_risk_quartiles=bins(y,action), by_action=action_counts(rows,y,action),
                  by_worker=by_worker, by_host=by_host,
                  recorded_movement_response=movement_check(rows),
                  episode_clustered_confidence=confidence(rows,y,action,average,agnostic))
    publish_json(args.out, report)
    print(json.dumps({k:val for k,val in report.items() if k not in ("capture","by_action")}, indent=2))


if __name__ == "__main__":
    main()
