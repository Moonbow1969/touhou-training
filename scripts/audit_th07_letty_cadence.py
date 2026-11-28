"""Measure actual native decision-window discrepancies by TH07 rollout lane."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


STUDIES = ("letty-nonspell2", "letty-nonspell2-extension", "letty-final", "letty-final-extension")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def counts(trace, receipt):
    move, focus, shoot, bomb = trace["action"].T
    words = trace["native_words"]
    lengths = trace["native_ticks_under_command"]
    request = np.array([16, 0, 32])[move//3] | np.array([64, 0, 128])[move % 3] | 4*focus | shoot | 2*bomb
    result = Counter(complete_native_windows=receipt["original_windows"],
                     exact_complete_native_windows=receipt["exact_native_windows"],
                     control_qualified_originals=int(receipt["qualified"]),
                     native_commands=len(request), missing_entire_requested_mask=len(receipt["unconsumed_action_indices"]))
    indices = []
    losses = trace["next_spares"] < trace["spares"]
    for i in receipt["nonexact_action_indices"]:
        length = int(lengths[i])
        different = words[i] != request[i]
        if length > 4:
            category = "overlong"
        elif length < 4:
            category = "short"
        elif i > 0 and different[0] and not different[1:].any() and words[i][0] == request[i-1]:
            category = "previous_word_in_first_slot"
        else:
            category = "other_four_tick_word_mismatch"
        result[category] += 1
        result["mismatches_on_spare_loss"] += int(losses[i])
        indices.append(dict(action_index=i, disposition=category, native_length=length,
                            native_start_tick=int(trace["native_tick"][i]),
                            native_end_tick=int(trace["native_next_tick"][i]),
                            near_native_spare_loss=bool(losses[i])))
    return result, indices


def make(root, output):
    by_lane = defaultdict(Counter)
    sources = {}
    unknown = []
    mismatches = []
    for study in STUDIES:
        index = root / study / "intake-index.json"
        study_proof = read(index)
        if not study_proof["complete"]:
            raise ValueError(f"Awaiting signed sources in {study}")
        sources[study] = sha(index)
        for entry in study_proof["rows"]:
            if "receipt" not in entry:
                unknown.append(entry["seed"])
                continue
            receipt_path = index.parent / entry["receipt"]
            trace_path = index.parent / entry["file"]
            if sha(receipt_path) != entry["receipt_sha256"] or sha(trace_path) != entry["sha256"]:
                raise ValueError(f"Signed normalized source changed: {entry['seed']}")
            receipt = read(receipt_path)
            with np.load(trace_path, allow_pickle=False) as trace:
                windows, indices = counts(trace, receipt)
            lane = by_lane[f"{receipt['host']}/{receipt['lane']}"]
            lane["originals_with_normalized_timing"] += 1
            lane.update(windows)
            if indices:
                mismatches.append(dict(seed=entry["seed"], host=receipt["host"], lane=receipt["lane"],
                                       receipt_sha256=entry["receipt_sha256"], nonexact_windows=indices))
    combined = Counter()
    for values in by_lane.values():
        combined.update(values)
    report = dict(indexes_sha256=sources, excluded_from_normalized_timing=unknown,
                  scope="Four original TRAIN cohorts. Check native lengths before characterizing control word slips.",
                  pooled=dict(combined), by_lane={lane: dict(count) for lane, count in by_lane.items()},
                  nonexact_windows=mismatches)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"]), sha(output))


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("intake_root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    make(args.intake_root, args.output)
