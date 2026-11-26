"""Estimate one-window injury under a fixed policy using known uniform action propensities."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.collect_trial_contact_windows import TRIAL_SHA256
from scripts.score_trial_contact_windows import infer_all, verify_rows
from scripts.train_full_policy import publish_json


def lowest_four_probabilities(scores):
    cutoff = np.partition(scores, 3, axis=1)[:, 3:4]
    below, tied = scores < cutoff, scores == cutoff
    fraction = (4 - below.sum(1, keepdims=True)) / tied.sum(1, keepdims=True)
    return (below + tied * fraction) / 4


def estimate(hit, weight):
    observed = hit.mean()
    ips = np.mean(weight * hit)
    return dict(uniform_risk=float(observed), target_ips=float(ips), difference=float(ips - observed),
                self_normalized=float(np.average(hit, weights=weight)),
                effective_samples=float(weight.sum()**2 / (weight**2).sum()),
                expected_weight_one=float(weight.mean()))


def evaluate(capture, model_path):
    manifest = json.loads((capture / "collection.json").read_bytes())
    if manifest["status"] != "complete" or "uniform independent" not in manifest["sampling"]["move_focus"]:
        raise ValueError("Uniform joint action assignment with a completed collection is required.")
    rows = []
    lane_episodes = defaultdict(lambda: defaultdict(list))
    for lane, meta in sorted(manifest["workers"].items()):
        path = capture / meta["file"]
        if digest(path) != meta["shard_sha256"] or meta["launch"]["executable_sha256"] != TRIAL_SHA256:
            raise ValueError("Trial version or randomized samples changed.")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            shard = [json.loads(line) for line in stream]
        verify_rows(shard, meta["counts"])
        for row in shard:
            lane_episodes[lane][row["episode"]].append(len(rows))
            rows.append(row)
    pred, model_hash = infer_all(model_path, rows)
    i = np.arange(len(rows))
    move, focus = (np.asarray([r[k] for r in rows]) for k in ("move", "focus"))
    risks = pred[:, :18].reshape(-1, 2, 9)[i, focus]
    probabilities = lowest_four_probabilities(risks)
    np.testing.assert_allclose(probabilities.sum(1), 1)
    weight = 9 * probabilities[i, move]
    hit = np.asarray([r["hit_after_step"] is not None for r in rows])
    result = estimate(hit, weight)
    # Resample entire episodes within their original lane. The logged action changes subsequent states.
    rng = np.random.default_rng(41855)
    chunks = [[np.asarray(idx) for idx in ep.values()] for ep in lane_episodes.values()]
    diffs = []
    for _ in range(2000):
        idx = np.concatenate([g[j] for g in chunks for j in rng.integers(len(g), size=len(g))])
        diffs.append(np.mean((weight[idx] - 1) * hit[idx]))
    result.update(windows=len(rows), confirmed_hits=int(hit.sum()), episodes=sum(map(len, chunks)),
                  difference_95=np.quantile(diffs, [.025, .975]).tolist(), model_sha256=model_hash,
                  capture_sha256=digest(capture / "collection.json"),
                  policy="Uniform focus. Sample one of the four lowest learned-risk moves at that focus with uniform cutoff ties.",
                  estimand="Observed uniform-policy state distribution, one held 12-tick intervention. Stage 1 official trial 0.01b.",
                  uncertainty="2000 paired lane-stratified episode bootstrap draws. Historical secondary analysis.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(args.capture, args.model)
    publish_json(args.out, result)
    print(json.dumps(result))
