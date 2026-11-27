"""Summarize matched whole original-game credits and verify per-credit action traces."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np


def wilson(k, n):
    z = 1.959963984540054
    center = (k + z*z/2) / (n + z*z)
    half = z*np.sqrt(k*(n-k)/n + z*z/4) / (n + z*z)
    return [max(0., float(center-half)), min(1., float(center+half))]


def one(rows):
    stages = [qualified_stage2(row) for row in rows]
    complete = None not in stages
    reached = [row for row,stage in zip(rows,stages) if stage is True]

    def average(field, group=rows):
        values = [row[field] for row in group if row.get(field) is not None]
        return mean(values) if values else None

    movement = np.asarray([row["action_counts"][0] for row in rows]).sum(axis=0)
    decisions = sum(row["decisions"] for row in rows)
    if int(movement.sum()) != decisions:
        raise ValueError("Action histogram and credited decision count differ")
    return dict(n=len(rows), stage2=len(reached) if complete else None,
                stage2_verified_n=sum(value is not None for value in stages),
                stage2_confirmed=len(reached),stage2_unknown=stages.count(None),
                stage2_rate_observation_bounds=[len(reached)/len(rows),(len(reached)+stages.count(None))/len(rows)],
                stage2_rate=len(reached)/len(rows) if complete else None,
                stage2_wilson95=wilson(len(reached), len(rows)) if complete else None,
                total_confirmed_bombs=sum(row["effective_bombs"] for row in rows),
                terminal_game_overs=sum(row["terminal"]=="game_over" for row in rows),
                mean_decisions=average("decisions"), mean_seconds=average("wall_seconds"),
                mean_terminal_spares=average("terminal_spares"), mean_terminal_bombs=average("terminal_bombs"),
                mean_terminal_power=average("terminal_power"), mean_stage2_spares=average("stage2_spares", reached),
                mean_stage2_bombs=average("stage2_bombs", reached), mean_stage2_power=average("stage2_power", reached),
                total_decisions=decisions, nine_way_counts=movement.tolist(),
                vertical_fraction=float((movement[:3].sum()+movement[6:].sum())/decisions),
                diagonal_fraction=float(movement[[0, 2, 6, 8]].sum()/decisions))


def qualified_stage2(row):
    # TH10 0.02a's historical stage channel stayed at 1 during a filmed Stage 2.
    if row['title']==10 and row.get('game_sha256')=='5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8':
        if not row.get('stage2_evidence_sha256'):
            return None
    if row['title']==12 and row.get('game_sha256')=='c6d15044394a33d6116c09eac44b33fdec814541c78518e29f60d54f49bdbd3a' and not row.get('stage2_evidence_sha256'):
        return None
    return row['stage2_reached']


def summarize(rows, protocol, candidate):
    baseline = protocol.get("reference_arm", "sft" if candidate in ("ppo1", "ppo2") else "ppo2")
    results = {}
    for title, seeds in protocol["eval_seeds"].items():
        title = int(title)
        host = None
        if title == 12 and baseline == "sft" and candidate in ("ppo1", "ppo2"):
            host = "steamdeck" if candidate == "ppo1" else "entropy"
        by_arm = {arm: {} for arm in (baseline, candidate)}
        for row in rows:
            if row["title"] != title or row["arm"] not in by_arm or (host and row["host"] != host):
                continue
            seed = row["seed"]
            records = by_arm[row["arm"]]
            if seed in records:
                raise ValueError(f"Duplicate eligible TH{title} credit on seed {seed}")
            records[seed] = row
        counts = {arm: len(records) for arm, records in by_arm.items()}
        if any(set(records) != set(seeds) for records in by_arm.values()):
            raise ValueError(f"Unfinished TH{title} {candidate} evaluation {counts}")
        pairs = []
        for seed in seeds:
            left = by_arm[baseline][seed]
            right = by_arm[candidate][seed]
            if left["host"] != right["host"] or not left["game_sha256"] or left["game_sha256"] != right["game_sha256"]:
                raise ValueError("Host/build mismatch within a pair")
            pairs.append(dict(seed=seed, host=left["host"], decision_difference=right["decisions"]-left["decisions"],
                              seconds_difference=right["wall_seconds"]-left["wall_seconds"],
                              reference_stage2=qualified_stage2(left), candidate_stage2=qualified_stage2(right)))
        rng = np.random.default_rng(216720 + title)
        differences = np.asarray([p["seconds_difference"] for p in pairs])
        draws = differences[rng.integers(len(pairs), size=(20000, len(pairs)))].mean(axis=1)
        sampled_by_host = []
        for physical_host in sorted({p["host"] for p in pairs}):
            within = np.array([p["seconds_difference"] for p in pairs if p["host"] == physical_host])
            sampled_by_host.append(rng.choice(within, size=(20000, len(within)), replace=True))
        stratified = np.concatenate(sampled_by_host, axis=1).mean(axis=1)
        results[title] = {
            baseline: one(list(by_arm[baseline].values())), "candidate": one(list(by_arm[candidate].values())),
            "pairs": pairs, "stage2_discordances": {
                "reference_only": sum(p["reference_stage2"] and not p["candidate_stage2"] for p in pairs),
                "candidate_only": sum(p["candidate_stage2"] and not p["reference_stage2"] for p in pairs)}
                if all(p['reference_stage2'] is not None and p['candidate_stage2'] is not None for p in pairs) else None,
            "mean_paired_wall_difference": float(differences.mean()),
            "mean_paired_wall_difference_bootstrap95": np.quantile(draws, [.025, .975]).tolist(),
            "mean_paired_wall_difference_fixed_host_bootstrap95": np.quantile(stratified, [.025, .975]).tolist()}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("credits", "protocol", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.credits.read_bytes()), json.loads(args.protocol.read_bytes()), args.candidate)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
