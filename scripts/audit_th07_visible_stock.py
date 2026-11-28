"""Audit visible TH07 resource icons against the exact archived original camera and native stock."""

import argparse
from collections import Counter
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from PIL import Image

from scripts.th07_retail_fragment_intake import OBSERVER, camera_rows, require, sha, table


GAMEPLAY_RECT = (32, 16, 416, 464)
SPARES_RECT = (498, 97, 623, 110)
BOMBS_RECT = (498, 114, 623, 127)
STOCK_RECT = (498, 97, 623, 127)
SHOTS = (0, 16, 48)


def reduce_game_camera(rgb):
    x0, y0, x1, y1 = GAMEPLAY_RECT
    pixels = rgb[y0:y1, x0:x1]
    gray = (pixels.astype(np.int32) @ [19595, 38470, 7471] + 32768) >> 16
    boxed_rows = (gray.reshape(112, 4, 96, 4).sum(axis=3) + 2) // 4
    return ((boxed_rows.sum(axis=1) + 2) // 4).astype(np.uint8)


def star_counts(rgb):
    """Reproduce the existing full-X-image resource decoder from visible pixels."""
    require(rgb.shape[0] >= 127 and rgb.shape[1] >= 623 and rgb.shape[2] == 3, "Missing visible game HUD pixels")
    counts, strengths = [], []
    for is_spare, y in ((True, 103), (False, 120)):
        seen = []
        for x in range(504, 617, 16):
            r, _, b = rgb[y - 6:y + 7, x - 6:x + 7].astype(np.int16).transpose(2, 0, 1)
            pixels = (r > 180) & (r - b > 30) if is_spare else (b > 130) & (b - r > 35)
            seen.append(int(pixels.sum()))
        counts.append(next((i for i, pixels in enumerate(seen) if pixels < 24), len(seen)))
        strengths.append(seen)
    return counts, strengths


def aligned_archive(folder, verify):
    proof = verify(folder)
    native, controls, applied = (table(folder / name) for name in ("native-input.csv", "controls.csv", "applied.csv"))
    trace_path = folder / "policy-trace.npz"
    with np.load(trace_path, allow_pickle=False) as loaded:
        trace = {key: loaded[key] for key in ("native_tick", "camera_ns", "frames")}
    states = camera_rows(native, controls, applied, trace)
    return proof, controls, trace, states


def verify_exact_readback(folder, archive):
    proof, controls, trace, normal = archive
    observations = []
    for i in SHOTS:
        image = folder / f"readback-root-{i:02d}.png"
        bounds = folder / f"readback-window-{i:02d}.txt"
        coordinates = dict(line.split("=") for line in bounds.read_text().splitlines())
        require([int(coordinates[key]) for key in ("X", "Y", "WIDTH", "HEIGHT")] == [0, 0, 640, 480],
                "Original readback is not aligned to the certified 640x480 game window")
        with Image.open(image) as source:
            rgb = np.asarray(source.convert("RGB"))
        error = int(np.count_nonzero(reduce_game_camera(rgb) != trace["frames"][i]))
        decoded, strengths = star_counts(rgb)
        native_stock = [int(normal[i][key]) for key in ("lives", "bombs")]
        visual_packet = [int(controls[i][key]) for key in ("spares_hud", "bombs_hud")]
        require(error == 0 and decoded == native_stock == visual_packet, "Original held HUD readback did not match its trace")
        observations.append(dict(decision_index=i, camera_tick=int(trace["native_tick"][i]),
                                 original_camera_ns=int(trace["camera_ns"][i]), stock_rgb=decoded,
                                 signed_visual_packet=visual_packet, independent_native_stock=native_stock,
                                 star_pixel_counts=strengths, reduced_camera_different_pixels=error,
                                 original_png_sha256=sha(image), original_geometry_sha256=sha(bounds)))
    return dict(independent_native_journal_sha256=proof["journal_sha256"], original_trace_sha256=sha(folder / "policy-trace.npz"),
                exact_same_camera_readbacks=observations)


def census(folders, verify):
    by_stocks = Counter()
    comparison = Counter()
    mismatches = []
    journals = []
    for folder in folders:
        proof, controls, trace, native = aligned_archive(folder, verify)
        journals.append(dict(seed=folder.name, original_journal_sha256=proof["journal_sha256"],
                             visual_packet_sha256=sha(folder / "controls.csv"),
                             original_trace_sha256=sha(folder / "policy-trace.npz"), camera_rows=len(controls)))
        visual = np.stack((controls["spares_hud"], controls["bombs_hud"]), axis=1)
        independently_native = np.stack((native["lives"], native["bombs"]), axis=1)
        match = visual == independently_native
        previous_native = np.concatenate((independently_native[:1], independently_native[:-1]))
        native_changed = np.any(independently_native != previous_native, axis=1)
        joint = match.all(axis=1)
        lagged = (~joint) & (visual == previous_native).all(axis=1)
        comparison.update(windows=len(controls), exact_joint=int(match.all(axis=1).sum()),
                          exact_spares=int(match[:, 0].sum()), exact_bombs=int(match[:, 1].sum()),
                          native_stock_change_windows=int(native_changed.sum()),
                          exact_joint_on_stock_change=int((native_changed & joint).sum()),
                          stable_native_stock_windows=int((~native_changed).sum()),
                          exact_joint_with_stable_stock=int(((~native_changed) & joint).sum()),
                          mismatches_equal_previous_sampled_native=int(lagged.sum()))
        for lives, bombs in independently_native:
            by_stocks[f"{lives}/{bombs}"] += 1
        for i in np.flatnonzero(~match.all(axis=1)):
            mismatches.append(dict(seed=folder.name, decision_index=int(i), native_tick=int(trace["native_tick"][i]),
                                   visual_stock=visual[i].tolist(), native_stock=independently_native[i].tolist(),
                                   previous_sampled_native_stock=previous_native[i].tolist()))
    return dict(totals=dict(comparison), native_stock_coverage=dict(sorted(by_stocks.items())),
                sources=journals, mismatch_count=len(mismatches), mismatch_examples=mismatches[:60])


def audit(fixture, output, cold=None):
    verify = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"]
    report = dict(
        gameplay_camera_rect_xyxy=GAMEPLAY_RECT, source_xyxy_half_open=True,
        spare_star_rect_xyxy=SPARES_RECT, bomb_star_rect_xyxy=BOMBS_RECT, combined_stock_rect_xyxy=STOCK_RECT,
        policy_camera_intersects_stock_rect=False,
        original_window_xywh=[0, 0, 640, 480], original_visible_decoder_min_color_pixels_per_star=24,
        exact_readback=verify_exact_readback(fixture, aligned_archive(fixture, verify)),
        prospective_held_readback_check=[
            "Capture the two visible stock rows from the exact same original X image and native tick as the policy crop.",
            "Keep original window geometry, camera clock and independent passive native stock records for audit.",
            "Decode the raw RGB pixels without consulting native stock, then compare to the passive same-clock values.",
            "Report both joint and per-resource accuracy by observed native stock, with a separate tally on a stock-change window.",
            "Verify zero-resource, ordinary-stock and Bomb-spend states. Preserve unreadable or disagreeing frames as explicit observations.",
        ],
    )
    if cold is not None:
        folders = []
        for folder in sorted(cold.iterdir()):
            status = folder / "verified.json"
            if status.is_file():
                record = json.loads(status.read_bytes())
                if record["passed"] and record["role"] == "train" and record["native_rank"] == 1:
                    folders.append(folder)
        require(len(folders) == 16, "Expected the original qualified Normal TRAIN source ledger")
        report["qualified_normal_training_credits"] = census(folders, verify)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report.get("qualified_normal_training_credits", {}).get("totals")), sha(output))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archived_fixture", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--cold", type=Path)
    args = cli.parse_args()
    audit(args.archived_fixture, args.output, args.cold)
