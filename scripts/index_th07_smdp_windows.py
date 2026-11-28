"""Index every CRC-verified native action slot for offline TH07 duration diagnostics."""

import argparse
import json
from pathlib import Path
from runpy import run_path

import numpy as np

from scripts.th07_native_duration import actor_eligible
from scripts.th07_retail_fragment_intake import OBSERVER, require, sha, slot_words, table


def index_windows(source, trace_path, receipt_path, output):
    receipt = json.loads(receipt_path.read_bytes())
    require(sha(trace_path) == receipt["sha256"] and receipt["role"] == "train",
            "Only receipts bound to a registered original TRAIN trace can be indexed here")
    require(all(sha(source / name) == receipt["native_source_sha256"][name]
                for name in ("native-input.csv", "applied.csv", "events.bin", "archive.sealed", "controls.csv")),
            "Native source changed after independent normalization")
    journal = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"](source)
    require(journal["journal_sha256"] == receipt["archive"]["journal_sha256"],
            "Original CRC journal does not match its normalizer receipt")
    native = table(source / "native-input.csv")
    _, words = slot_words(source, native, receipt["source_kind"])
    controls = table(source / "controls.csv")
    with np.load(trace_path, allow_pickle=False) as trace:
        length = len(trace["action"])
        require(length == receipt["decisions"], "Wrong number of normalized original actions")
        first = receipt["source_start_index"]
        requested = controls["requested"][first:first + length]
        starts = trace["native_tick"]
        ends = trace["native_next_tick"]
        start_stages, end_stages = trace["native_stage"], trace["native_next_stage"]
        prior_lengths = trace["native_ticks_under_command"]
        prior_words = trace["native_words"]
        previously_eligible = trace["native_action_eligible"]
    require(len(requested) == length and np.array_equal(starts, controls["tick"][first:first + length]),
            "Normalized actions no longer align to actual native requests")

    cumulative = [0]
    ticks, stages, recorded = [], [], []
    for i, (start_stage, start, end_stage, end) in enumerate(zip(start_stages, starts, end_stages, ends, strict=True)):
        start_stage, start, end_stage, end = map(int, (start_stage, start, end_stage, end))
        if start_stage == end_stage:
            segments = [(start_stage, start + 1, end + 1)]
        else:
            seam = receipt["native_stage_boundaries"][str(i)]
            require(start_stage == 1 and end_stage == 2 and seam[0] >= start and 0 < seam[1] <= end,
                    "Unrecognized original phase seam")
            segments = [(start_stage, start + 1, seam[0] + 1), (end_stage, 1, end + 1)]
        for stage, begin, stop in segments:
            for tick in range(begin, stop):
                stages.append(stage)
                ticks.append(tick)
                recorded.append(words[stage].get(tick, -1))
        cumulative.append(len(recorded))
        prefix = recorded[cumulative[i]:min(cumulative[i + 1], cumulative[i] + 4)]
        require(cumulative[i + 1] - cumulative[i] == prior_lengths[i]
                and np.array_equal(prefix, prior_words[i, :len(prefix)]),
                "Expanded native window disagrees with the existing qualified trace prefix")
    offsets = np.asarray(cumulative, dtype=np.int64)
    duration = np.diff(offsets)
    observed = np.array([all(word >= 0 for word in recorded[begin:end]) for begin, end in zip(offsets[:-1], offsets[1:])])
    raw_masks = np.asarray(recorded, dtype=np.int16)
    complete = observed & (duration > 0)
    eligible = np.zeros(len(duration), dtype=bool)
    if complete.any():
        eligible[complete] = actor_eligible(requested[complete], raw_masks[np.repeat(complete, duration)], duration[complete])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        np.savez_compressed(destination, offsets=offsets, duration_native_ticks=duration,
                            stage=np.asarray(stages, dtype=np.uint8), tick=np.asarray(ticks, dtype=np.int64),
                            observed_key_mask=raw_masks, requested_key_mask=requested,
                            every_native_word_observed=observed, smdp_actor_eligible=eligible,
                            original_four_tick_actor_eligible=previously_eligible)
    report = dict(seed=receipt["seed"], role=receipt["role"], outcome=receipt["outcome"], trace_sha256=sha(trace_path),
                  prior_receipt_sha256=sha(receipt_path), compressed_index_sha256=sha(output),
                  original_journal_sha256=journal["journal_sha256"], decision_windows=length, native_word_slots=len(recorded),
                  missing_native_word_slots=recorded.count(-1), all_native_words_equal_requested=int(eligible.sum()),
                  wrong_length_but_all_words_equal=int(((duration != 4) & eligible).sum()),
                  qualifies_for_original_fitter=receipt["qualified"],
                  changes_to_original_receipt_or_fitter=False)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw_original", type=Path)
    cli.add_argument("normalized_trace", type=Path)
    cli.add_argument("signed_receipt", type=Path)
    cli.add_argument("sidecar", type=Path)
    args = cli.parse_args()
    print(json.dumps(index_windows(args.raw_original, args.normalized_trace, args.signed_receipt, args.sidecar)))
