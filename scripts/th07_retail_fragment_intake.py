"""Seal a causal Stage 1 fragment from the retail TH07 1.00b observer-v2 archive."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from PIL import Image
import torch


GAME = "35467eaf8dc7fc85f024f16fb2037255f151cefda33cf4867bc9122aaa2e80ca"
DATA = "5fe8d772191505bf208f966da8070ee0a9e0d0ee8567a8827ef3a362bcb0c287"
PARENT = "a6e1dec1625af50e214f141a4e405795fbdae905d78c4b887e423de0bca6a77e"
FRAME = (112, 96)
HEADS = (9, 2, 2, 2)
OBSERVER = (Path(__file__).resolve().parents[1]
            / "tools/rollout-worker-probe/th07-retail100b/live/observer-v2")
STOCK = {"spares": "lives", "bombs": "bombs", "misses": "misses", "bombs_used": "bombs_used"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest_camera(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def table(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="ascii", ndmin=1)


def frame_pairs(current, previous):
    difference = np.abs(current.astype(np.int16) - previous.astype(np.int16)).astype(np.uint8)
    return torch.from_numpy(np.stack((current, difference), axis=1)).float() / 255.0


def forward(actor, frames, earlier, action):
    taken_logp, critic_values = [], []
    with torch.inference_mode():
        for start in range(0, len(action), 64):
            end = start + 64
            x = frame_pairs(frames[start:end], earlier[start:end])
            y = actor(x)
            require([tuple(t.shape) for t in y] == [(len(x), width) for width in (*HEADS, 1)],
                    "The frozen actor does not have the original four heads and critic")
            chosen = torch.from_numpy(action[start:end])
            logp = torch.stack([head.log_softmax(-1).gather(1, chosen[:, j:j + 1]).squeeze(1)
                                for j, head in enumerate(y[:4])], dim=1)
            taken_logp.append(logp.numpy())
            critic_values.append(y[4][:, 0].numpy())
    return np.concatenate(taken_logp), np.concatenate(critic_values)


def camera_rows(native, controls, applied, trace):
    n = len(trace["native_tick"])
    require(n and len(controls) == n == len(applied), "The camera, control and journal counts disagree")
    ticks = trace["native_tick"]
    ns = trace["camera_ns"]
    require(np.array_equal(ns, controls["camera_ns"]) and np.all(np.diff(ns) > 0)
            and np.array_equal(controls["seq"], np.arange(n))
            and np.array_equal(applied["sequence"], np.arange(n)),
            "The original trace and command identities do not match")
    for raw in (controls["tick"], applied["camera_tick"], applied["applied_tick"], applied["frozen_tick"]):
        require(np.array_equal(ticks, raw), "The camera and actual frozen command clocks diverged")
    require(np.all(applied["freeze_start_ns"] <= ns) and np.all(ns <= applied["apply_ns"])
            and np.all(applied["apply_ns"] <= applied["resume_ns"]), "Invalid original frozen-camera ordering")
    ix = np.searchsorted(native["ns"], applied["apply_ns"], side="right") - 1
    require(np.all(ix >= 0) and np.all(native["ns"][ix] >= ns) and np.array_equal(native["tick"][ix], ticks),
            "A decision lacks the observer's frozen native row after the actual camera")
    states = native[ix]
    require(np.all(states["flags"] & 4) and not np.any(states["retry"]),
            "The actor trace contains a nonplayable camera")
    return states


def slot_words(folder, native, source_kind):
    """Recover consumed words only from repeated native record slots, never from injected keys."""
    live = native[(native["flags"] & 4) != 0]
    require(len(live) > 0 and np.all((live["flags"] & 11) == (1 if source_kind == "practice" else 0)),
            "Native practice/ordinary mode changed or playback was active")
    by_stage = {}
    for stage in np.unique(live["stage"]):
        rows = live[live["stage"] == stage]
        words = by_stage[int(stage)] = {}
        for row in rows:
            tick = int(row["tick"])
            for index, slot in ((tick, row["record_slot"]), (tick - 1, row["previous_slot"])):
                word = int(slot) & 247
                if index >= 0:
                    require(words.setdefault(index, word) == word, "Current and previous native slot readings conflict")
        dump = folder / f"record-stage{stage}.bin"
        if dump.is_file():
            buf = dump.read_bytes()
            require(len(buf) >= 44 and (len(buf) - 44) % 4 == 0, "Truncated native stage-slot dump")
            saved = np.frombuffer(buf, "<u2", offset=44)[::2] & 247
            require(len(saved) > 0 and all(saved[t] == word for t, word in words.items() if t < len(saved)),
                    "Live current/previous slots disagree with the final original stage buffer")
    return live, by_stage


def native_boundary_row(native, endpoint):
    tick, stage = endpoint["boundary_tick"], endpoint["boundary_stage"]
    selected = native[(native["tick"] == tick) & (native["stage"] == stage)
                      & (native["ns"] >= endpoint["boundary_camera_ns"]) & (native["retry"] == 0)
                      & ((native["flags"] & 4) != 0)]
    require(len(selected) > 0, "The separately captured next camera lacks a playable native slot after its timestamp")
    row = selected[0]
    if "native_boundary" in endpoint:
        witness = endpoint["native_boundary"]
        require(int(witness["clock"]) == tick and int(witness["camera_ns"]) == endpoint["boundary_camera_ns"]
                and int(witness["cursor"]) == int(witness["block"]) + 44 + 4 * tick,
                "Next original camera disagrees with its independent clock and record cursor")
        mapping = {"spares": "lives", "previous": "previous_word", "record": "record_slot",
                   "record_previous": "previous_slot"}
        fields = ("stage", "tick", "flags", "spares", "bombs", "misses", "continues",
                  "word", "previous", "record", "record_previous")
        require(all(int(witness[name]) == row[mapping.get(name, name)] for name in fields),
                "Next original camera's current/previous record slots or stock were not corroborated")
    return row


def playable_boundary(folder, native, endpoint):
    path = folder / "boundary-frame.npy"
    require(sha(path) == endpoint["boundary_frame_sha256"], "Exact next original camera bytes changed")
    frame = np.load(path, allow_pickle=False)
    require(frame.shape == FRAME and frame.dtype == np.uint8, "Invalid next original camera")
    if "boundary_pixel_sha256" in endpoint:
        require(digest_camera(frame) == endpoint["boundary_pixel_sha256"], "Raw next-camera pixels changed")
    return native_boundary_row(native, endpoint), frame, endpoint.get("boundary_value"), path


def practice_native_result(folder, native, endpoint):
    path = folder / "practice-result-state.csv"
    screen = folder / "native-practice-result.png"
    visual = json.loads((folder / "practice-result-visual.json").read_bytes())
    with path.open(encoding="ascii", newline="") as source:
        observed = list(csv.DictReader(source))
    require(len(observed) == 1 and observed[0] == endpoint.get("practice_result")
            and visual["verified"] is True and visual["screen"] == "practice_result" and sha(screen) == visual["sha256"],
            "Missing same-original practice result state or screen")
    source = observed[0]
    tick, stage = int(source["last_tick"]), int(source["last_stage"])
    require(int(source["supervisor_address"], 0) == 0x575AA8 and int(source["supervisor_state"]) == 6
            and int(source["stageclock"]) == tick == endpoint["terminal_observed_tick"]
            and stage == 1 and int(source["flags"]) == 5
            and int(source["retry"]) == int(source["continues"]) == 0
            and int(source["last_native_ns"]) < int(source["result_observed_ns"]),
            "Stage 1 practice completion needs the original result state and last playable clock")
    candidate = native[(native["stage"] == stage) & (native["tick"] == tick) & (native["retry"] == 0)
                       & (native["flags"] == 5) & (native["ns"] <= int(source["last_native_ns"]))]
    require(len(candidate) > 0 and not np.any(native["stage"] != 1), "Practice clear lacks a real final Stage 1 state")
    row = candidate[-1]
    mapping = {"spares": "lives", "bombs": "bombs", "misses": "misses", "continues": "continues",
               "consumed_word": "word", "record_word": "record_slot"}
    require(all(int(source[name]) == row[field] for name, field in mapping.items()),
            "The reported practice result disagrees with the original final playable stock or slots")
    return row, path, visual


def verify_stage_clear_tally(folder, visual, row):
    require(visual.get("stage_clear_verified") is True, "Original result screen does not prove Stage Clear")
    trace = folder / "policy-trace.npz"
    tick = visual["tally_native_tick"]
    index = visual["tally_frame_index"]
    picture = folder / f"stage-clear-tally-{tick}.png"
    require(visual["tally_source_npz_sha256"] == sha(trace) and visual["tally_sha256"] == sha(picture)
            and visual["native_result_record_sha256"] == sha(folder / "practice-result-state.csv")
            and 0 < tick <= row["tick"], "Stage Clear evidence changed or followed the actual original endpoint")
    with np.load(trace, allow_pickle=False) as source:
        require(0 <= index < len(source["native_tick"]) and source["native_tick"][index] == tick
                and visual["tally_pixels_sha256"] == digest_camera(source["frames"][index]),
                "Stage Clear tally does not bind an exact camera from this original")
        with Image.open(picture) as tally:
            width, height = tally.size
            scale = width // FRAME[1]
            require(scale > 0 and (height, width) == (FRAME[0] * scale, FRAME[1] * scale)
                    and np.array_equal(np.asarray(tally.convert("L")),
                                       np.repeat(np.repeat(source["frames"][index], scale, axis=0), scale, axis=1)),
                    "Stage Clear image is not an unfiltered enlargement of its original camera")
    return picture


def practice_result_boundary(folder, native, endpoint):
    row, source, visual = practice_native_result(folder, native, endpoint)
    verify_stage_clear_tally(folder, visual, row)
    return row, source


def game_over_boundary(native, spec, first_tick, target):
    row = native[-1]
    starting = native[(native["stage"] == 1) & ((native["flags"] & 4) != 0)][0]
    minimum_misses = int(starting["misses"]) + int(starting["lives"]) + 1
    require(row["stage"] == 1 and row["retry"] > 0 and row["lives"] == 0 and row["misses"] >= minimum_misses,
            "Original Game Over needs a native retry state after exhausting lives")
    require(row["tick"] == spec["terminal_observed_tick"] and row["tick"] > first_tick
            and (target is None or row["tick"] <= target), "Original Game Over clock is outside the fragment")
    return row


def select_endpoint(folder, native, trace, camera, spec, start, horizon):
    first_tick = int(camera[start]["tick"])
    target = first_tick + horizon if horizon is not None else None
    for index in range(start + 1, len(camera)):
        row = camera[index]
        if row["stage"] == 2:
            return index, "stage2_entry", row, trace["frames"][index], trace["value"][index], None
        require(row["stage"] == 1, "Native original jumped across the Stage 1 to Stage 2 seam")
        if target is not None and row["tick"] >= target:
            require(row["tick"] - target < 4, "Original skipped the required playable timeout-boundary camera")
            return index, "native_tick_cutoff", row, trace["frames"][index], trace["value"][index], None
    if spec.get("outcome") in ("stage2_entry", "native_tick_cutoff"):
        row, frame, value, source = playable_boundary(folder, native, spec)
        if spec["outcome"] == "native_tick_cutoff":
            require(row["stage"] == 1 and target is not None and target <= row["tick"] < target + 4,
                    "Unproved native elapsed timeout")
        else:
            require(row["stage"] == 2, "A practice stage clear does not prove a playable Stage 2 transition")
        return len(camera), spec["outcome"], row, frame, value, source
    if spec.get("outcome") == "practice_stage_clear":
        require(spec.get("source_kind") == "practice", "A Stage 1 practice result cannot close Game Start")
        row, source, visual = practice_native_result(folder, native, spec)
        require(row["tick"] > first_tick and (target is None or row["tick"] <= target),
                "Practice result arrived after the registered playable timeout")
        outcome = "practice_stage_clear"
        if visual.get("exhaustion_verified") is True:
            initial = native[(native["stage"] == 1) & ((native["flags"] & 4) != 0)][0]
            require(visual.get("stage_clear_verified") is False and row["lives"] == 0
                    and row["misses"] >= initial["misses"] + initial["lives"] + 1,
                    "Practice exhaustion requires distinct result evidence and the original last-stock loss")
            outcome = "game_over"
        else:
            verify_stage_clear_tally(folder, visual, row)
        return len(camera), outcome, row, np.zeros(FRAME, dtype=np.uint8), 0., source
    require(spec.get("outcome") in ("game-over", "game_over"), "Missing a true Game Over or playable next boundary")
    row = game_over_boundary(native, spec, first_tick, target)
    return len(camera), "game_over", row, np.zeros(FRAME, dtype=np.uint8), 0., None


def native_controls(current, following, requested, slots, losses, outcome, stage1_last):
    words = np.full((len(requested), 4), -1, dtype=np.int64)
    duration = np.empty(len(requested), dtype=np.int64)
    qualifies = np.ones(len(requested), dtype=bool)
    complete = np.ones(len(requested), dtype=bool)
    consumed = np.zeros(len(requested), dtype=bool)
    for i, (before, after, word) in enumerate(zip(current, following, requested)):
        start, end = int(before["tick"]), int(after["tick"])
        crossed = int(after["stage"]) == 2
        same_stage_end = stage1_last if crossed else end
        measured = [slots[1].get(t, -1) for t in range(start + 1, same_stage_end + 1)]
        if crossed:
            measured.extend(slots[2].get(t, -1) for t in range(1, end + 1))
        duration[i] = len(measured)
        words[i, :min(len(measured), 4)] = measured[:4]
        consumed[i] = word in measured
        structural_endpoint = i == len(requested) - 1 and (
            crossed or (outcome in ("game_over", "practice_stage_clear") and len(measured) < 4))
        complete[i] = not structural_endpoint
        qualifies[i] = bool(measured and -1 not in measured and (
            (len(measured) == 4 and all(w == word for w in measured))
            or (structural_endpoint and consumed[i])))
    exact, count = int(np.sum(qualifies & complete)), int(complete.sum())
    fraction = exact / count if count else 0.
    inexact = np.flatnonzero(~qualifies)
    issues = []
    if count == 0 or fraction < .995:
        issues.append("Fewer than 99.5% of complete native windows were exact")
    if not consumed.all():
        issues.append("The original recorder did not consume every requested mask")
    if len(inexact) > 2 or any(not losses[i] for i in inexact):
        issues.append("Nonexact windows exceeded two or occurred outside an observed spare loss")
    return dict(native_words=words, native_ticks_under_command=duration, native_action_eligible=qualifies), dict(
        qualified=not issues, control_exclusions=issues,
        exact_four_tick_fraction=fraction, exact_native_windows=exact,
        original_windows=count, action_masks_with_native_consumption=int(consumed.sum()),
        nonexact_action_indices=inexact.tolist(), unconsumed_action_indices=np.flatnonzero(~consumed).tolist())


def intake_fragment(folder, checkpoint, output, *, endpoint=None, start_index=0, horizon=None,
                    expected_policy_sha=PARENT):
    folder, checkpoint = map(Path, (folder, checkpoint))
    output = Path(output) if output is not None else None
    finish = Path(endpoint) if endpoint else folder / "finished-game.json"
    spec = json.loads(finish.read_bytes())
    policy_sha = sha(checkpoint)
    require(spec.get("sha256", spec.get("policy_sha256")) == policy_sha == expected_policy_sha
            and spec.get("role") in ("train", "held", "engineering"),
            "Native trajectory does not bind its actual actor or provenance role")
    build = dict(line.split()[::-1] for line in (folder / "build.sha256").read_text().splitlines() if line.strip())
    require(build.get("th07.exe") == GAME and build.get("th07.dat") == DATA,
            "Capture does not identify the pinned original retail executable and data")
    source_kind = spec.get("source_kind", "game_start" if (folder / "official-policy.rpy").is_file() else None)
    require(source_kind in ("game_start", "practice"), "Missing explicit original Game Start / practice provenance")
    if source_kind == "practice" and horizon is None:
        horizon = 6000
    require(horizon is None or horizon > 0, "Invalid native horizon")
    archive = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"](folder)
    native, controls, applied = (table(folder / file) for file in ("native-input.csv", "controls.csv", "applied.csv"))
    verification_path = folder / "verified.json"
    verified = json.loads(verification_path.read_bytes()) if verification_path.is_file() else {}
    native_rank = spec.get("native_rank")
    if native_rank is None and spec["role"] in ("engineering", "held") and verified.get("archive"):
        # Published observer-v2 verification above explicitly required rank 1.
        require(verified["archive"]["journal_sha256"] == archive["journal_sha256"],
                "Legacy Normal declaration does not bind the currently observed journal")
        native_rank = 1
    require(type(native_rank) is int and native_rank in (0, 1)
            and (source_kind != "practice" or native_rank == 1),
            "The frozen source must declare Easy or Normal Game Start, or Normal practice")
    require(len(native) and np.all(np.diff(native["ns"]) > 0) and not np.any(native["continues"])
            and np.all(native["rank"] == native_rank) and np.all(native["shot"] == 0),
            "Missing the declared retail rank, Reimu A or zero-Continue evidence")
    trace_path = folder / "policy-trace.npz"
    with np.load(trace_path, allow_pickle=False) as loaded:
        trace = {key: loaded[key] for key in ("frames", "action", "logp", "value", "native_tick", "camera_ns")}
        prior = loaded["predecessor_frame"] if "predecessor_frame" in loaded else None
    n = len(trace["native_tick"])
    frames, acts, lp, value = (trace[key] for key in ("frames", "action", "logp", "value"))
    require(frames.shape == (n, *FRAME) and frames.dtype == np.uint8 and acts.shape == (n, 4)
            and np.issubdtype(acts.dtype, np.integer) and np.all((0 <= acts) & (acts < HEADS))
            and lp.shape == (n, 4) and value.shape == (n,) and np.isfinite(lp).all() and np.isfinite(value).all()
            and 0 <= start_index < n and spec.get("actions", n) == n, "Malformed original camera action trace")
    require(prior is None or (prior.shape == FRAME and prior.dtype == np.uint8), "Invalid captured predecessor camera")
    acts = acts.astype(np.int64, copy=False)
    move, focus, shot, bomb = acts.T
    requested = (np.array([16, 0, 32])[move // 3] | np.array([64, 0, 128])[move % 3]
                 | (4 * focus) | shot | (2 * bomb))
    require(np.array_equal(requested, controls["requested"]) and np.array_equal(requested, applied["requested_word"]),
            "Recorded native requests disagree with the sampled four-head policy actions")
    camera = camera_rows(native, controls, applied, trace)
    require(camera[start_index]["stage"] == 1, "The registered fragment must start in live Stage 1")
    live, slots = slot_words(folder, native, source_kind)
    end, outcome, boundary, image, recorded_value, boundary_source = select_endpoint(
        folder, native, trace, camera, spec, start_index, horizon)
    selected = slice(start_index, end)
    current = camera[selected]
    following = np.r_[current[1:], [boundary]].astype(native.dtype)
    require(len(current) and np.all(current["stage"] == 1) and np.all(np.diff(current["tick"]) > 0),
            "Original Stage 1 camera clock did not advance")
    at_boundary = live[live["ns"] <= boundary["ns"]]
    stage1 = at_boundary[at_boundary["stage"] == 1]
    last_stage1 = int(stage1[-1]["tick"])
    stage_brackets = {"1": {"first_tick": int(stage1[0]["tick"]), "last_tick": last_stage1}}
    stage_boundaries = {}
    if outcome == "stage2_entry":
        stage2 = at_boundary[at_boundary["stage"] == 2]
        require(len(stage2) and (horizon is None or last_stage1 < int(current[0]["tick"]) + horizon)
                and stage2[0]["ns"] > stage1[-1]["ns"],
                "Stage 2 must have a fresh active native seam before the registered timeout")
        first_stage2 = int(stage2[0]["tick"])
        stage_boundaries[str(len(current) - 1)] = [last_stage1, first_stage2]
        stage_brackets["2"] = {"first_tick": first_stage2, "first_playable_tick": int(boundary["tick"])}
    arrays, control = native_controls(current, following, requested[selected], slots,
                                       following["lives"] < current["lives"], outcome, last_stage1)
    if len(current) < 2:
        control["qualified"] = False
        control["control_exclusions"].append("Fewer than two policy decisions")
    if source_kind == "game_start" and "passed" in verified:
        require(verified.get("seed") == spec["seed"] and verified.get("sha256") == policy_sha
                and verified["archive"]["journal_sha256"] == archive["journal_sha256"],
                "The signed full-original qualification does not bind this source")
        original_files = verified.get("artifacts_sha256", {})
        for name in ("controls.csv", "applied.csv", "policy-trace.npz", "native-input.csv", "events.bin", "archive.sealed"):
            if name in original_files:
                require(original_files[name] == sha(folder / name), f"Signed full-original evidence changed: {name}")
        control["full_original_passed"] = verified["passed"] is True
        control["qualified"] &= control["full_original_passed"]
        if not control["full_original_passed"]:
            control["control_exclusions"].append("The independent full original verification failed")
    start_frame = prior if start_index == 0 and prior is not None else frames[max(0, start_index - 1)]
    pairs = np.concatenate((start_frame[None], frames[selected][:-1]))
    actor = torch.jit.load(str(checkpoint), map_location="cpu").eval()
    actual_lp, actual_value = forward(actor, frames[selected], pairs, acts[selected])
    logp_error = float(np.max(np.abs(actual_lp - lp[selected])))
    value_error = float(np.max(np.abs(actual_value - value[selected])))
    require(logp_error <= 8e-4 and value_error <= 3e-4, "Frozen actor log-probability or critic re-forward failed")
    terminal, cap, transition, practice_clear = (np.zeros(len(current), dtype=bool) for _ in range(4))
    terminal[-1] = outcome in ("game_over", "practice_stage_clear")
    cap[-1], transition[-1], practice_clear[-1] = (outcome == name for name in
        ("native_tick_cutoff", "stage2_entry", "practice_stage_clear"))
    bootstrap = 0.
    if not terminal[-1]:
        with torch.inference_mode():
            bootstrap = float(actor(frame_pairs(image[None], frames[end - 1:end]))[-1].item())
        require(np.isfinite(bootstrap) and (recorded_value is None or abs(bootstrap - recorded_value) <= 3e-4),
                "Original boundary critic re-forward failed")
    arrays.update(frames=frames[selected], predecessor_frame=start_frame, action=acts[selected],
                  logp=lp[selected].astype(np.float32), value=value[selected].astype(np.float32),
                  native_stage=current["stage"], native_tick=current["tick"],
                  native_next_stage=following["stage"], native_next_tick=following["tick"],
                  hud_spares=controls["spares_hud"][selected], hud_bombs=controls["bombs_hud"][selected],
                  terminated=terminal, truncated=cap, stage2_reached=transition,
                  practice_stage_clear=practice_clear, practice_clear_reward=practice_clear.astype(np.float32),
                  stage_transition_reward=transition.astype(np.float32), boundary_frame=image,
                  boundary_value=np.float32(bootstrap), boundary_playable=np.bool_(not terminal[-1]))
    for field, source in STOCK.items():
        arrays[field], arrays[f"next_{field}"] = current[source], following[source]
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as target:
            np.savez_compressed(target, **arrays)
    sources = {name: sha(folder / name) for name in (
        "archive.sealed", "events.bin", "native-input.csv", "applied.csv", "controls.csv", "policy-trace.npz", "build.sha256")}
    for path in (*sorted(folder.glob("record-stage*.bin")), *sorted(folder.glob("stage-clear-tally-*.png")),
                 folder / "official-policy.rpy",
                 folder / "native-continue.png", folder / "native-practice-result.png", folder / "native-cutoff.csv",
                 folder / "practice-result-visual.json", folder / "qualification.json",
                 folder / "ui-verified.json", verification_path, boundary_source):
        if path and path.is_file():
            sources[path.name] = sha(path)
    receipt = dict(seed=spec["seed"], policy_sha256=policy_sha, game_sha256=GAME, role=spec["role"],
                   source_kind=source_kind, native_rank=native_rank, outcome=outcome,
                   decisions=len(current),
                   native_horizon=horizon, native_elapsed_ticks=int(arrays["native_ticks_under_command"].sum()),
                   source_start_index=start_index, native_stage_boundaries=stage_boundaries,
                   native_stage_brackets=stage_brackets, continues=0,
                   boundary_camera_sha256=None if terminal[-1] else digest_camera(image), behavior_boundary_value=bootstrap,
                   first_stock=[int(current[0][k]) for k in ("lives", "bombs")],
                   end_stock=[int(boundary[k]) for k in ("lives", "bombs")],
                   native_source_sha256=sources, endpoint_sha256=sha(finish), archive=archive,
                   policy_logp_max_abs_error=logp_error, policy_critic_max_abs_error=value_error, **control)
    receipt.update({name: spec[name] for name in ("host", "lane", "scenario", "registration_sha256") if name in spec})
    if output is not None:
        receipt.update(sha256=sha(output), file=output.name)
        receipt_path = output.with_suffix(".json")
        pending = receipt_path.with_suffix(".pending")
        pending.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        pending.replace(receipt_path)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, help="Optional explicit native cap. Practice defaults to 6000 ticks.")
    parser.add_argument("--expected-policy-sha", default=PARENT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(intake_fragment(args.folder, args.checkpoint, args.output, endpoint=args.endpoint,
                                     start_index=args.start_index, horizon=args.horizon,
                                     expected_policy_sha=args.expected_policy_sha), indent=2))
