"""Recompute the Normal Ghost paired Stage 1 hazard from the four extracted original-game archives."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_th08_live_tick4 import sha
from scripts.report_th08_stock_exposure import postprompt_stage1_tick, terminal_stage_evidence


def verify(study):
    held = study / "held-original"
    report = json.loads((held / "results.json").read_bytes())
    assignment = json.loads((study / "held-registration.json").read_bytes())
    binding = json.loads((study / "held-review-release.json").read_bytes())
    if sha(study / "held-registration.json") != binding["registration_sha256"]:
        raise ValueError("Assignment has changed since the two actors were bound")
    results = {}
    for pair in assignment["pairs"]:
        for arm in pair["arms"]:
            seed = pair["seed"]
            folder = held / arm
            source = folder / "sealed" / f"{seed}.json"
            row = json.loads(source.read_bytes())
            scored = next(x for x in report["summaries"][arm]["by_seed"] if x["seed"] == seed)
            raw = folder / "raw" / str(seed)
            if (row["role"] != "held" or row["seed"] != seed or row["policy_sha256"] != binding["arms"][arm]
                    or sha(source) != scored["source_receipt_sha256"] or sha(raw / "clock.tsv") != row["stage_clock_trace_sha256"]
                    or sha(raw / "gates.tsv") != row["input_gates_sha256"] or sha(raw / "timing.jsonl") != row["timing_sha256"]):
                raise ValueError(f"Original source mismatch for {seed} {arm}")
            endpoint = terminal_stage_evidence(raw, row)
            if endpoint["stage"] != 1:
                raise ValueError(f"This completed comparison requires a separately witnessed Stage 1 endpoint for {seed} {arm}")
            tick = row["terminal_tick"]
            if endpoint["native_terminal_stage"] != 1:
                supplemental = postprompt_stage1_tick(raw, row)
                if not supplemental["verified"]:
                    raise ValueError("No supported original frozen terminal tick")
                tick = supplemental["first_native_stopped_tick"]
            observations = []
            for line in (raw / "timing.jsonl").read_text().splitlines():
                sample = json.loads(line)
                if sample["phase"] == "policy":
                    observations.append(sample)
            events = row["life_events"]
            available = sum(observations[e["decision"]]["bombs"] > 0 for e in events)
            gates = np.genfromtxt(raw / "gates.tsv", names=True, dtype=None, encoding="ascii", ndmin=1)
            first = int(gates["apply_target"][0]) + 1
            exposure = tick - first + 1
            if (first != row["first_tick"] or exposure != scored["stage1_at_risk_ticks"] or
                    len(events) != scored["stage1_lost_lives"] or available != scored["stage1_positive_bomb_deaths"]):
                raise ValueError(f"Original exposure or resource ledger mismatch for {seed} {arm}")
            results[arm, seed] = exposure, len(events), available, row["qualified"]

    arms = ["progress", "progress-plus-spares"]
    all_seeds = [pair["seed"] for pair in assignment["pairs"]]
    strict = [s for s in all_seeds if all(results[a,s][3] for a in arms)]
    checked = {}
    for seeds, key in ((all_seeds,"all_verified_original_endpoints_hazard_sensitivity"),(strict,"strict_matched_hazard_sensitivity")):
        rows = np.array([[results[a,s][:3] for s in seeds] for a in arms], dtype=np.int64)
        total = rows.sum(axis=1)
        hazards = total[:,1] / total[:,0]
        result = report[key]
        if total[:,0].tolist() != result["stage1_at_risk_native_ticks"] or total[:,2].tolist() != result["stage1_positive_bomb_deaths"]:
            raise ValueError(f"Arm-level ledgers differ in {key}")
        ratio = float(hazards[1] / hazards[0])
        if abs(ratio - result["ratio"]) > 1e-12:
            raise ValueError(f"Paired hazard mismatch in {key}")
        checked[key] = dict(pairs=len(seeds), actor_order=arms, totals_ticks_deaths_positive_bomb=total.tolist(), ratio=ratio)
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Directory containing held-registration.json and held-original/ after extracting all four archives")
    print(json.dumps(verify(parser.parse_args().study), indent=2))
