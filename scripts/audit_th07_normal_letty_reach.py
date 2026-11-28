"""Independently attest archived Normal Game Start encounters from the original camera."""

import argparse
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from PIL import Image, ImageFilter

from scripts.th07_retail_fragment_intake import (
    GAME, OBSERVER, PARENT, camera_rows, digest_camera, require, sha, table,
)


THRESHOLD = .68
CONSECUTIVE = 3
REFERENCE_SPANS = (("nonspell2", 120, 356), ("final", 120, 512), ("nonspell2", 640, 1300), ("nonspell2", 2320, 3600))
COLS = slice(0, 22)
TAIL = slice(9, 22)
ROWS = slice(3, 6)
SOURCE_SEEDS = tuple(seed for seed in range(21749101, 21749118) if seed != 21749104)


def load(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def features(frames):
    filtered = []
    for frame in frames:
        band = frame[:13, :33]
        smooth = Image.fromarray(band).filter(ImageFilter.GaussianBlur(.78))
        filtered.append(band.astype(float) - np.asarray(smooth, dtype=float))
    band = np.stack(filtered)
    pieces = []
    for columns in (COLS, TAIL):
        flat = band[:, ROWS, columns].reshape(len(band), -1).copy()
        flat -= flat.mean(axis=1, keepdims=True)
        flat /= np.maximum(1., np.linalg.norm(flat, axis=1, keepdims=True))
        pieces.append(flat)
    return pieces


def reference(vectors):
    medians = [np.median(block, axis=0) for block in vectors]
    return [block / np.linalg.norm(block) for block in medians]


def related(scores, ticks, *, threshold=THRESHOLD):
    start = np.logical_and.reduce([scores[offset:len(scores) - CONSECUTIVE + offset + 1] >= threshold
                                   for offset in range(CONSECUTIVE)])
    cadence = np.logical_and.reduce([ticks[offset + 1:len(ticks) - CONSECUTIVE + offset + 2]
                                     - ticks[offset:len(ticks) - CONSECUTIVE + offset + 1] == 4
                                     for offset in range(CONSECUTIVE - 1)])
    return np.flatnonzero(start & cadence)


def match(vectors, templates):
    name_vectors, tail_vectors = vectors
    name_templates, tail_templates = templates
    return np.minimum(name_vectors @ name_templates.T, tail_vectors @ tail_templates.T + .08).max(axis=1)


def calibrate(nonspell, final, easy, destination):
    paths = dict(nonspell2=nonspell / "policy-trace.npz", final=final / "policy-trace.npz",
                 easy=easy / "policy-trace.npz")
    archives = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as source:
            archives[name] = (source["native_tick"], features(source["frames"]))
    # Physical references are native boss phases and the independently photographed Easy HUD.
    visual = load(easy / "boss-evidence.json")
    require(visual["evidence"][0]["tick"] == 6070 and visual["evidence"][1]["tick"] == 6192
            and min(item["similarity"] for item in visual["evidence"]) > .9,
            "The original independent Easy photographic validation changed")
    proof = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"]
    for case in (nonspell, final, easy):
        proof(case)
    matrices = []
    for name, first, last in REFERENCE_SPANS:
        ticks, vectors = archives[name]
        chosen = (ticks >= first) & (ticks <= last)
        matrices.append(reference([vec[chosen] for vec in vectors]))
    templates = [np.stack(group) for group in zip(*matrices, strict=True)]
    easy_ticks, easy_vectors = archives["easy"]
    easy_scores = match(easy_vectors, templates)
    negative = easy_ticks <= 5950
    positive = (easy_ticks >= 6072) & (easy_ticks <= 7270)
    false_hits = related(easy_scores[negative], easy_ticks[negative])
    true_hits = related(easy_scores[positive], easy_ticks[positive])
    require(not len(false_hits) and len(true_hits), "Independent Easy witness is not separable at the fixed detector setting")
    destination.mkdir(parents=True, exist_ok=True)
    bank = destination / "original-letty-camera-template.npz"
    np.savez_compressed(bank, name=templates[0], tail=templates[1])
    receipt = dict(
        template_sha256=sha(bank), fixed_score_threshold=THRESHOLD, adjacent_exact_four_tick_cameras=CONSECUTIVE,
        policy_roi=dict(row_start=ROWS.start, row_stop=ROWS.stop, col_start=COLS.start, col_stop=COLS.stop,
                        tail_col_start=TAIL.start), preparation="Gaussian high-pass (0.78 pixels), centered unit vectors.",
        camera_sources_sha256={name: sha(path) for name, path in paths.items()},
        original_easy_boss_photo_verification_sha256=sha(easy / "boss-evidence.json"),
        original_reference_context=dict(
            nonspell_native_phase_sha256=sha(nonspell / "spell-phase.csv"),
            normal_original_name_screens_sha256=[sha(case / name) for case, name in (
                (nonspell, "original-spell-0540.png"), (final, "original-spell-0120.png"))],
            spans=REFERENCE_SPANS,
        ),
        validation_easy=dict(
            no_Letty_candidate_windows=len(false_hits), no_Letty_camera_rows=int(negative.sum()),
            photographed_positive_camera_rows=int(positive.sum()), positive_consecutive_starts=len(true_hits),
            maximum_negative_single_frame_score=float(easy_scores[negative].max()),
            max_positive_single_frame_score=float(easy_scores[positive].max()),
            first_positive_original_tick=int(easy_ticks[positive][true_hits[0]])),
    )
    write(destination / "camera-template-calibration.json", receipt)


def scan(case, seed, templates):
    verifies = case / "verified.json"
    proof = load(verifies)
    require(proof["seed"] == seed and proof["role"] == "train" and proof["passed"] is True
            and proof["arm"] == "parent" and proof["sha256"] == PARENT and proof["native_rank"] == 1,
            "Expected immutable qualified Normal Game Start parent credit")
    build = dict(row.split()[::-1] for row in (case / "build.sha256").read_text().splitlines())
    require(build["th07.exe"] == GAME, "Original TH07 executable differs")
    archive = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"](case)
    require(archive["journal_sha256"] == proof["archive"]["journal_sha256"], "Normal original journal changed")
    native, applied, controls = (table(case / name) for name in ("native-input.csv", "applied.csv", "controls.csv"))
    require(np.all(native["rank"] == 1) and np.all(native["shot"] == 0) and not np.any(native["continues"])
            and np.all(native["stage"] == 1), "Original rank, shot or Stage 1 metadata changed")
    trace = case / "policy-trace.npz"
    with np.load(trace, allow_pickle=False) as source:
        camera = {key: source[key] for key in ("frames", "native_tick", "camera_ns")}
    if "artifacts_sha256" in proof:
        require(proof["artifacts_sha256"].get("policy-trace.npz", sha(trace)) == sha(trace), "Original camera artifact changed")
    states = camera_rows(native, controls, applied, camera)
    scores = match(features(camera["frames"]), templates)
    registered_hits = related(scores, camera["native_tick"])
    row = dict(seed=seed, role="train", original_verified_sha256=sha(verifies), original_camera_archive_sha256=sha(trace),
               original_native_journal_sha256=archive["journal_sha256"], original_policy_sha256=PARENT,
               terminal_native_tick=proof["stage1_native_ticks"], stage2_reached=proof["stage2"],
               captured_original_cameras=len(scores), max_original_name_score=float(scores.max()),
               verified_first_onset=None, candidate_evidence=None)
    if not len(registered_hits):
        return dict(row, status="unknown_no_distinctive_adjacent_original_name_witness")
    start = int(registered_hits[0])
    candidates = []
    for i in range(start, start + CONSECUTIVE):
        tick = int(camera["native_tick"][i])
        state = states[i]
        require(tick == state["tick"] == controls[i]["tick"] and int(state["stage"]) == 1,
                "Original boss witness did not match the native camera clock")
        candidates.append(dict(original_trace_camera_index=i, native_stage=1, native_tick=tick, original_score=float(scores[i]),
                               original_camera_sha256=digest_camera(camera["frames"][i]),
                               native_source_ns=int(state["ns"]), native_spares=int(state["lives"]),
                               native_bombs=int(state["bombs"]), native_misses=int(state["misses"])))
    return dict(row, status="adjacent_original_name_candidate", candidate_evidence=candidates)


def detect(credits, calibration, output):
    bank = calibration / "original-letty-camera-template.npz"
    frozen = calibration / "camera-template-calibration.json"
    receipt = load(frozen)
    require(sha(bank) == receipt["template_sha256"] and receipt["fixed_score_threshold"] == THRESHOLD,
            "Original template was changed after source calibration")
    with np.load(bank, allow_pickle=False) as source:
        templates = [source["name"], source["tail"]]
    rows = [scan(credits / f"{seed}-parent", seed, templates) for seed in SOURCE_SEEDS]
    write(output, dict(role="train", cohort="Common Normal original Game Start", registered_qualified_originals=len(rows),
                       source_template_calibration_sha256=sha(frozen), rows=rows,
                       review="Adjacent exact-clock camera candidates await visual identity review. A missing name witness leaves reach unknown."))
    print(json.dumps([(r["seed"], r["candidate_evidence"][0]["native_tick"] if r["candidate_evidence"] else None,
                       round(r["max_original_name_score"], 4)) for r in rows]))


def seal_review(candidates, witness_figure, confirmed, destination):
    observed = load(candidates)
    rows = observed["rows"]
    eligible = {row["seed"] for row in rows if row["candidate_evidence"]}
    require(set(confirmed) <= eligible and len(set(confirmed)) == len(confirmed),
            "Reviewed originals do not match their independently scored camera evidence")
    for row in rows:
        witnessed = row["seed"] in confirmed
        row["status"] = "confirmed_visible_letty_name" if witnessed else "unknown_reach"
        row["first_confirmed_witness"] = row["candidate_evidence"][0] if witnessed else None
    witnessed_rows = [row for row in rows if row["first_confirmed_witness"]]
    stocks = [row["first_confirmed_witness"] for row in witnessed_rows]
    report = dict(
        role=observed["role"], game="TH07 1.00b", difficulty="Normal", scope=observed["cohort"],
        boss_identity="Letty Whiterock", physically_complete_originals=len(rows), confirmed_reach=len(stocks),
        reach_unknown=len(rows) - len(stocks), archived_reach_fraction_lower_bound=len(stocks) / len(rows),
        reached_with_no_spare=sum(stock["native_spares"] == 0 for stock in stocks),
        reached_with_no_spare_and_no_bomb=sum(stock["native_spares"] == stock["native_bombs"] == 0 for stock in stocks),
        reached_with_no_spare_and_one_bomb=sum(stock["native_spares"] == 0 and stock["native_bombs"] == 1 for stock in stocks),
        observed_stage2_reach_after_confirmed_letty=sum(row["stage2_reached"] for row in witnessed_rows),
        source_template_calibration_sha256=observed["source_template_calibration_sha256"],
        candidate_receipt_sha256=sha(candidates), witness_figure_sha256=sha(witness_figure),
        rows=rows,
        interpretation=("Three consecutive captured four-tick original cameras support each positive identification. "
                        "A missing distinctive glyph leaves reach unresolved. The first verified camera tick "
                        "bounds the visible encounter without estimating an earlier spawn."),
    )
    write(destination, report)
    print(json.dumps({key: report[key] for key in ("confirmed_reach", "reach_unknown", "archived_reach_fraction_lower_bound",
                                                   "reached_with_no_spare", "reached_with_no_spare_and_no_bomb",
                                                   "observed_stage2_reach_after_confirmed_letty")}), sha(destination))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_subparsers(dest="action", required=True)
    setup = action.add_parser("calibrate")
    for arg in ("nonspell", "final", "easy", "destination"):
        setup.add_argument(arg, type=Path)
    score = action.add_parser("detect")
    for arg in ("credits", "calibration", "output"):
        score.add_argument(arg, type=Path)
    review = action.add_parser("seal-review")
    for arg in ("candidates", "witness_figure", "output"):
        review.add_argument(arg, type=Path)
    review.add_argument("--confirmed", type=int, nargs="*", required=True)
    args = cli.parse_args()
    if args.action == "calibrate":
        calibrate(args.nonspell, args.final, args.easy, args.destination)
    elif args.action == "detect":
        detect(args.credits, args.calibration, args.output)
    else:
        seal_review(args.candidates, args.witness_figure, args.confirmed, args.output)
