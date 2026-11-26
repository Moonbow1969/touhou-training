"""Aggregate terminal stock story credits, retaining policy identity and actual entry resources."""

import argparse
from collections import defaultdict
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import median

from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


def interval(successes, n):
    z = 1.95996398454
    fraction = successes / n
    midpoint = fraction + z*z / (2*n)
    radius = z * sqrt(fraction*(1-fraction)/n + z*z/(4*n*n))
    scale = 1 + z*z/n
    return [max(0, (midpoint-radius)/scale), min(1, (midpoint+radius)/scale)]


def aggregate(paths):
    cohorts = defaultdict(list)
    seen = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        spec = report["specification"]
        expected = dict(title="TH14", difficulty="Normal", shot="Reimu A", version="1.00b",
                        target="ending", stock_story=True, all_native_lives=True)
        if report["status"] != "complete" or any(spec["task"][k] != v for k, v in expected.items()):
            raise ValueError(f"{path} is outside the completed stock full-story cohort.")
        for trial in report["completed"]:
            result, arm = trial["result"], trial["trial"]["arm"]
            if result["outcome"] not in ("one_credit_clear", "game_over") or result["native_continues"]:
                raise ValueError(f"{path} has no eligible zero-Continue terminal outcome.")
            identity = {k: spec[k] for k in ("task", "controller", "runtime")}
            identity["policy"] = spec["policies"][arm]
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            label = f"{arm}-{key[:12]}"
            episode = (key, trial["trial"]["policy_seed"], trial["trial"]["block"])
            if episode in seen:
                raise ValueError(f"{path} duplicates a policy, policy seed and block.")
            seen.add(episode)
            entries = {stage["stage"]: stage["entry"]["resources"] for stage in result["stage_records"]}
            if list(entries) != list(range(1, max(entries)+1)) or (
                    entries[1]["lives"], entries[1]["bombs"]) != (2, 3):
                raise ValueError("Invalid original credit or stage sequence.")
            cohorts[label].append(dict(report_sha256=digest(path), seed=trial["trial"]["policy_seed"],
                stage_entries=entries, outcome=result["outcome"], terminal=result["native_terminal"],
                lives_lost=result["deaths"], bombs_consumed=result["actual_bombs_consumed"]))
    summary = {}
    for label, rows in cohorts.items():
        n = len(rows)
        stages = {}
        for stage in range(2, 7):
            stocks = [row["stage_entries"][stage] for row in rows if stage in row["stage_entries"]]
            stages[stage] = dict(entries=len(stocks), credits=n, rate=len(stocks)/n,
                                 wilson95=interval(len(stocks), n),
                                 spare_lives_median=median(s["lives"] for s in stocks) if stocks else None,
                                 bombs_median=median(s["bombs"] for s in stocks) if stocks else None)
        summary[label] = dict(credits=n, one_credit_clears=sum(r["outcome"] == "one_credit_clear" for r in rows),
                              stages=stages, trials=rows)
    return dict(native_game_rng_controlled=False, interval_assumption="Independent credits under a fixed controller.",
                resources="Native entry stocks among entrants only.", cohorts=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    publish_json(args.out, aggregate(args.reports))
