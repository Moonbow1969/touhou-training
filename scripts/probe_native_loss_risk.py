"""Can the portable actor's observations identify a near-future native life loss?"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


HORIZON = 8


def loss_targets(data, title, horizon=HORIZON):
    if title == 10:
        loss = data["reward_life"] < 0
    elif title == 12:
        parts = data["reward_parts"]
        loss = (parts[:, 1] < 0) | (parts[:, 4] < 0)
    else:
        loss = data["reward_components"][:, 0] > 0
    cum = np.r_[0, np.cumsum(loss)]
    end = np.minimum(np.arange(len(loss)) + horizon, len(loss))
    return (cum[end] - cum[:-1]) > 0, loss


def read(path, title):
    with np.load(path) as data:
        y, loss = loss_targets(data, title)
        # A pre-action feature predicts this or the next seven action outcomes.
        # Only windows with a full known horizon or a recorded loss have a label.
        known = (np.arange(len(loss)) + HORIZON <= len(loss)) | y
        return data["features"][known].astype(np.float64), y[known].astype(float), int(loss.sum())


def fit(x, y):
    mu, scale = x.mean(0), np.maximum(x.std(0), .01)
    xx = torch.from_numpy(np.column_stack([np.ones(len(x)), (x-mu)/scale]))
    labels = torch.from_numpy(y)
    w = torch.zeros(xx.shape[1], dtype=xx.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=250, tolerance_grad=1e-8, line_search_fn="strong_wolfe")
    def objective():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xx @ w, labels) + .005 * w[1:].square().sum()
        loss.backward()
        return loss
    optimizer.step(objective)
    weights = w.detach().numpy()
    # Ranking for AP only requires the untransformed linear score.
    return lambda tx: weights[0] + ((tx-mu)/scale) @ weights[1:]


def ap(y, scores):
    order = np.argsort(-scores, kind="stable")
    ranks = np.flatnonzero(np.r_[scores[order][1:] != scores[order][:-1], True])
    positives = np.cumsum(y[order])[ranks]
    precision = positives/(ranks+1)
    return float((np.diff(np.r_[0, positives])*precision).sum()/y.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {}
    for title in (10, 12, 13):
        paths = [path for folder in args.training for path in sorted(folder.glob(f"th{title}-*.npz"))]
        test_paths = sorted(args.test.glob(f"th{title}-*.npz"))
        if not paths or not test_paths or ({p.name for p in paths} & {p.name for p in test_paths}):
            raise ValueError("Empty or overlapping whole-credit splits")
        fit_data = [read(path, title) for path in paths]
        test_data = [read(path, title) for path in test_paths]
        x, y = (np.concatenate([r[k] for r in fit_data]) for k in (0, 1))
        results = {"training_credits": [p.stem for p in paths], "test_credits": [p.stem for p in test_paths],
                   "training_rows": len(x), "held_life_losses": sum(r[2] for r in test_data),
                   "held_positive_rate": float(np.concatenate([r[1] for r in test_data]).mean()), "probes": {}}
        for name, cols in (("history_clock", slice(84, 110)), ("history_clock_and_grid", slice(0, 110))):
            pred = fit(x[:, cols], y)
            per_credit = [{"credit": p.stem, "n": len(labels),
                           "average_precision": ap(labels, pred(frame[:, cols])),
                           "positive_rate": float(labels.mean())} for p, (frame, labels, _) in zip(test_paths, test_data)]
            results["probes"][name] = {"per_credit": per_credit, "mean_credit_ap": float(np.mean(
                [r["average_precision"] for r in per_credit]))}
        a, b = [np.array([v["average_precision"] for v in results["probes"][k]["per_credit"]])
                for k in ("history_clock", "history_clock_and_grid")]
        bootstrap = np.random.default_rng(20261125+title).choice(b-a, (20000, len(a)), replace=True).mean(1)
        results["paired_grid_ap_delta"] = float((b-a).mean())
        results["paired_grid_ap_delta_bootstrap95"] = np.quantile(bootstrap, [.025, .975]).tolist()
        result[title] = results
    args.out.write_text(json.dumps(dict(horizon_decisions=HORIZON, titles=result), indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
