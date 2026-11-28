"""Join measured container CPU deltas to already independently checked held controls."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path


def read_samples(path):
    samples = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            for name, values in row["groups"].items():
                samples[name].append((row["monotonic_ns"], values["stat"]))
    return samples


def observed_interval_flags(samples, sample_times, begin, end):
    offset = bisect_right(sample_times, begin)
    finish = bisect_right(sample_times, end)
    if offset < 1 or finish >= len(samples):
        return None
    matched = samples[offset-1:finish+1]
    if any(after[0]-before[0] > 300_000_000 for before, after in zip(matched[:-1], matched[1:])):
        return None
    first, last = matched[0][1], matched[-1][1]
    return last["nr_throttled"] > first["nr_throttled"]


def match_controls(held, samples):
    reports = []
    total = Counter()
    for name, entries in sorted(samples.items()):
        receipt_path = held / "intake" / (name + ".json")
        folder = held / "raw" / name
        if not receipt_path.is_file() or not (folder / "archive.sealed").is_file():
            continue
        result = json.loads(receipt_path.read_bytes())
        receipt = result.get("strict_source_receipt")
        if not receipt or not result["physical_endpoint_verified"]:
            continue
        with (folder / "controls.csv").open(newline="") as source:
            cameras = [int(row["camera_ns"]) for row in csv.DictReader(source)]
        with (folder / "applied.csv").open(newline="") as source:
            resumes = [int(row["resume_ns"]) for row in csv.DictReader(source)]
        failed = set(receipt["nonexact_action_indices"])
        sample_times = [t for t, _ in entries]
        counts = Counter()
        discards = []
        for decision, (resume, following_capture) in enumerate(zip(resumes, cameras[1:])):
            key = "nonexact" if decision in failed else "other"
            if following_capture <= resume:
                raise ValueError(f"Reversed time around {name} action {decision}")
            throttle = observed_interval_flags(entries, sample_times, resume, following_capture)
            if throttle is None:
                continue
            counts[key] += 1
            counts[key+"_with_throttle"] += int(throttle)
            if key == "nonexact":
                discards.append(dict(decision=decision, throttle=throttle))
        first, last = entries[0][1], entries[-1][1]
        report = dict(original=name, counts=dict(counts), covered_errors=discards,
                      cpu_quota_periods=last["nr_periods"]-first["nr_periods"],
                      throttled_periods=last["nr_throttled"]-first["nr_throttled"],
                      charged_throttle_seconds=(last["throttled_usec"]-first["throttled_usec"])/1e6)
        reports.append(report)
        total.update(counts)
    return reports, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held", type=Path, required=True)
    p.add_argument("--telemetry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    reports, totals = match_controls(args.held, read_samples(args.telemetry))
    data = dict(totals=dict(totals), observed_originals=reports,
                method="Each nonfinal command's resume-to-next-camera interval is bracketed by actual cgroup samples. Only complete sample coverage with no gap above 300 ms is compared. A recorded increase in cgroup nr_throttled indicates a nearby quota-throttling period. The bracket (~100 ms) exceeds a 60-Hz game tick and does not identify the delayed thread or establish cause.")
    args.out.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(data["totals"]))


if __name__ == "__main__":
    main()
