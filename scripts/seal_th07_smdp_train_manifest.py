"""Bind the original Letty TRAIN roster to complete, verified native duration sidecars."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.th07_retail_fragment_intake import GAME, PARENT, require, sha


COHORTS = (
    ("letty-nonspell2", "th07-letty-nonspell2-train", tuple(range(21749701, 21749709))),
    ("letty-nonspell2-extension", "th07-letty-nonspell2-extension", tuple(range(21749709, 21749722))),
    ("letty-final", "th07-letty-final-spell-train", tuple(range(21749601, 21749609))),
    ("letty-final-extension", "th07-letty-final-spell-extension", (21749609,)),
)


def read(path):
    return json.loads(path.read_bytes())


def bound_source(path):
    return dict(path=path.resolve().as_posix(), sha256=sha(path))


def relative_file(path, parent):
    return Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()


def check_windows(path, receipt, decisions):
    with np.load(path, allow_pickle=False) as trace:
        duration, offsets, ticks, words, request, observed, eligible = (
            trace[name] for name in ("duration_native_ticks", "offsets", "tick", "observed_key_mask",
                                     "requested_key_mask", "every_native_word_observed", "smdp_actor_eligible"))
    require(len(duration) == decisions == len(request) == len(observed) == len(eligible)
            and len(offsets) == decisions + 1 and offsets[0] == 0 and (duration > 0).all()
            and len(words) == len(ticks) == offsets[-1] == int(duration.sum())
            and np.array_equal(np.diff(offsets), duration) and observed.all() and (words >= 0).all()
            and int(eligible.sum()) == receipt["all_native_words_equal_requested"]
            and np.array_equal(np.logical_and.reduceat(words == np.repeat(request, duration), offsets[:-1]), eligible),
            "Missing or inconsistent native duration slots in prospective manifest")


def seal(output_root, intake_root, duration_root, output):
    rows = []
    cohort_paths = {}
    for intake_name, study_name, seeds in COHORTS:
        raw_study = output_root / study_name
        intake = intake_root / intake_name
        index_path = intake / "intake-index.json"
        registration_path = raw_study / "registration.json"
        index, registry = read(index_path), read(registration_path)
        index_rows = {row["seed"]: row for row in index["rows"]}
        require(sha(registration_path) == index["registration_sha256"]
                and tuple(row["seed"] for row in registry["rows"]) == seeds and tuple(index_rows) == seeds
                and registry["role"] == "train" and registry["policy_sha256"] == PARENT
                and registry["game_sha256"] == GAME and index["complete"] is True and not index["missing"],
                f"Signed original TRAIN enrollment changed in {intake_name}")
        cohort_paths[intake_name] = dict(registration=bound_source(registration_path), intake=bound_source(index_path))
        for planned in registry["rows"]:
            seed = planned["seed"]
            entry = index_rows[seed]
            require(entry["role"] == planned["role"] == "train", "A non-TRAIN observation reached the duration manifest")
            source = raw_study / "raw" / str(seed)
            identity = dict(seed=seed, role="train", scene=entry["scene"], scenario=planned["scenario"],
                            registration_sha256=index["registration_sha256"], source_raw_directory=source.resolve().as_posix())
            if "receipt" not in entry:
                fault = source / "attempt-failed.json"
                require(sha(fault) == entry["original_failure_sha256"], "Original source-failure evidence changed")
                rows.append(dict(identity, admitted=False, original_fault=bound_source(fault),
                                 reason="Original TRAIN endpoint package was incomplete. Physical-only derivatives are ineligible."))
                continue
            normalized_path = intake / entry["file"]
            normalized_receipt_path = intake / entry["receipt"]
            window_path = duration_root / str(seed) / "windows.npz"
            window_receipt_path = window_path.with_suffix(".json")
            normalized, window = read(normalized_receipt_path), read(window_receipt_path)
            original_endpoint = source / "finished-game.json"
            original_endpoint_hash = sha(original_endpoint)
            original_status = read(original_endpoint)
            require(sha(normalized_path) == entry["sha256"] == normalized["sha256"] == window["trace_sha256"]
                    and sha(normalized_receipt_path) == entry["receipt_sha256"] == window["prior_receipt_sha256"]
                    and sha(window_path) == window["compressed_index_sha256"]
                    and normalized["role"] == window["role"] == original_status["role"] == "train"
                    and normalized["seed"] == window["seed"] == original_status["seed"] == seed
                    and normalized["policy_sha256"] == original_status["policy_sha256"] == PARENT
                    and normalized["registration_sha256"] == original_status["registration_sha256"] == index["registration_sha256"]
                    and normalized["game_sha256"] == GAME and normalized["endpoint_sha256"] == original_endpoint_hash
                    and window["original_journal_sha256"] == normalized["archive"]["journal_sha256"],
                    f"Original, normalized and duration receipts disagree for {seed}")
            check_windows(window_path, window, normalized["decisions"])
            observed = normalized["outcome"]
            require(observed in ("game_over", "practice_stage_clear", "native_tick_cutoff")
                    and entry["outcome"] == window["outcome"] == observed and normalized["continues"] == 0,
                    f"Unverified original physical outcome for {seed}")
            evidence = {
                relpath: bound_source(source / relpath)
                for relpath in ("qualification.json", "spell-phase.csv", "archive.sealed", "events.bin",
                                "native-input.csv", "applied.csv", "controls.csv", "policy-trace.npz")
            }
            if observed == "practice_stage_clear":
                names = ("practice-result-state.csv", "native-practice-result.png", "practice-result-visual.json")
                evidence.update({name: bound_source(source / name) for name in names})
                visual = read(source / "practice-result-visual.json")
                name = f"stage-clear-tally-{visual['tally_native_tick']}.png"
                evidence[name] = bound_source(source / name)
                require(visual["stage_clear_verified"] is True
                        and evidence[name]["sha256"] == visual["tally_sha256"], "Positive original tally changed")
            elif observed == "native_tick_cutoff":
                name = original_status["boundary_frame_file"]
                evidence[name] = bound_source(source / name)
            require(all(normalized["native_source_sha256"].get(name, item["sha256"]) == item["sha256"]
                        for name, item in evidence.items())
                    and evidence["spell-phase.csv"]["sha256"] == entry["phase_sha256"],
                    "A bound original endpoint artifact changed after independent verification")
            recovery = entry.get("endpoint_recovery_sha256")
            if recovery is not None:
                evidence["endpoint-recovery.json"] = bound_source(source / "endpoint-recovery.json")
                require(recovery == evidence["endpoint-recovery.json"]["sha256"]
                        and read(source / "endpoint-recovery.json")["original_attempt_sha256"] == sha(source / "attempt-failed.json")
                        and observed == "practice_stage_clear" and "native-practice-result.png" in evidence,
                        "A recovered source lacks the verified same-original physical endpoint")
            rows.append(dict(identity, admitted=True, outcome=observed, native_horizon=normalized["native_horizon"],
                             native_elapsed_ticks=normalized["native_elapsed_ticks"], decisions=normalized["decisions"],
                             actor_eligible_decisions=window["all_native_words_equal_requested"],
                             original_four_tick_qualified=normalized["qualified"],
                             file=relative_file(normalized_path, output.parent), sha256=normalized["sha256"],
                             receipt=relative_file(normalized_receipt_path, output.parent), receipt_sha256=entry["receipt_sha256"],
                             duration_sidecar=relative_file(window_path, output.parent),
                             duration_sha256=window["compressed_index_sha256"],
                             duration_receipt=relative_file(window_receipt_path, output.parent),
                             duration_receipt_sha256=sha(window_receipt_path),
                             normalized_trace=bound_source(normalized_path), normalized_receipt=bound_source(normalized_receipt_path),
                             native_window_index=bound_source(window_path), native_window_receipt=bound_source(window_receipt_path),
                             original_endpoint=bound_source(original_endpoint), source_evidence=evidence))
    manifest = dict(
        schema="TH07 original Letty TRAIN native-duration source manifest v1", role="train", complete=True,
        mode="second-nonspell", duration_model="native-smdp", held_seeds=list(range(21749801, 21749821)),
        registered_originals=len(rows), admitted_originals=sum(row["admitted"] for row in rows),
        actor_eligible_decisions=sum(row.get("actor_eligible_decisions", 0) for row in rows),
        native_clocked_decisions=sum(row.get("decisions", 0) for row in rows), game_sha256=GAME, policy_sha256=PARENT,
        frozen_cohorts=cohort_paths, rows=[row for row in rows if row["admitted"]],
        excluded_originals=[row for row in rows if not row["admitted"]],
    )
    output.parent.mkdir(exist_ok=True, parents=True)
    temp = output.with_suffix(".pending")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(output)
    print(json.dumps({key: manifest[key] for key in ("registered_originals", "admitted_originals", "actor_eligible_decisions",
                                                     "native_clocked_decisions")}), sha(output))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output_root", type=Path)
    cli.add_argument("intake_root", type=Path)
    cli.add_argument("duration_root", type=Path)
    cli.add_argument("manifest", type=Path)
    args = cli.parse_args()
    seal(args.output_root, args.intake_root, args.duration_root, args.manifest)
