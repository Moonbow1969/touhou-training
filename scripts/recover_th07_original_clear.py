"""Verify a physical original clear from sources sealed before a serializer fault."""

import argparse
import csv
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from PIL import Image
import torch

from scripts.th07_retail_fragment_intake import (
    DATA, GAME, OBSERVER, PARENT, camera_rows, digest_camera, forward, native_controls, require, sha, slot_words, table,
)


def recover(study, seed, tick, actor, output):
    folder = study / "raw" / str(seed)
    registration = study / "registration.json"
    roster = json.loads(registration.read_bytes())
    row = next(row for row in roster["rows"] if row["seed"] == seed)
    failure = folder / "attempt-failed.json"
    fault = json.loads(failure.read_bytes())
    log = study / "logs" / f"{seed}.log"
    fatal = "dict() got multiple values for keyword argument 'initial_spares'"
    require(row["role"] == fault["role"] == "train" and fatal in log.read_text(encoding="utf-8")
            and sha(log) == fault["log_sha256"] and row["scenario"] == fault["scenario"]
            and fault["seed"] == seed and sha(actor) == roster["policy_sha256"] == PARENT,
            "Registered source or signed post-rollout failure does not match")
    inputs = ("initial-hashes.txt", "configured-practice.png", "controls.csv", "applied.csv", "native-input.csv",
              "events.bin", "archive.sealed", "policy-trace.npz", "spell-phase.csv", "practice-result-state.csv")
    evidence = {name: sha(folder / name) for name in inputs}
    require(all(fault["lossless_evidence_files"][name] == digest for name, digest in evidence.items()),
            "Original source no longer matches its preserved failure manifest")
    build = dict(line.split()[::-1] for line in (folder / "initial-hashes.txt").read_text().splitlines())
    require(build["th07.exe"] == GAME and build["th07.dat"] == DATA
            and build["/tool/thprac.exe"] == roster["practice_tool_sha256"], "Wrong original retail build")
    archive = run_path(str(OBSERVER / "record_proof.py"))["verify_archive"](folder)
    native = table(folder / "native-input.csv")
    live, slots = slot_words(folder, native, "practice")
    require(np.all(live["stage"] == 1) and np.all(live["rank"] == 1) and np.all(live["shot"] == 0)
            and np.all(live["continues"] == 0) and all(t in slots[1] for t in range(1, 121)),
            "Original stage, difficulty, trainer stock or initial control slots invalid")
    with (folder / "practice-result-state.csv").open(encoding="ascii", newline="") as stream:
        records = list(csv.DictReader(stream))
    require(len(records) == 1, "Did not see a single original result")
    result = records[0]
    end = int(result["last_tick"])
    boundary = live[(live["tick"] == end) & (live["ns"] <= int(result["last_native_ns"]))][-1]
    require(result["supervisor_state"] == "6" and int(result["supervisor_address"], 0) == 0x575AA8
            and result["last_stage"] == "1" and int(result["stageclock"]) == end and result["flags"] == "5"
            and int(result["last_native_ns"]) < int(result["result_observed_ns"])
            and result["retry"] == result["continues"] == "0"
            and all(int(result[x]) == boundary[y] for x, y in (("spares","lives"), ("bombs","bombs"), ("misses","misses"),
                                                             ("consumed_word","word"), ("record_word","record_slot"))),
            "Post-result original clock, stock and native words did not agree")
    phase = table(folder / "spell-phase.csv")
    previous = phase[(phase["supervisor"] == 2) & (phase["active"] == 0) & (phase["boss_present"] != 0)
                     & (phase["boss_sub"] == 39) & (phase["lifetime_callback"] == 2000)
                     & (phase["timer_callback_tick"] == 2400) & (phase["timer_callback_sub"] == 48)]
    active = phase[(phase["supervisor"] == 2) & (phase["active"] != 0) & (phase["spell_index"] == 7)]
    def real(rows):
        return sorted({int(r["stage_tick"]) for r in rows
                       if slots[1].get(int(r["stage_tick"])) == (int(r["native_keys"]) & 247)})
    nonspell = real(previous)
    spell = real(active)
    at_tally = phase[(phase["stage_tick"] == tick) & (phase["finished_stage"] == 2)]
    require(nonspell and spell and nonspell[-1] < spell[0] < tick <= end and len(at_tally)
            and int(boundary["misses"]) <= row["initial_spares"],
            "The original camera is missing the recorded clear progression")

    controls, applied = (table(folder / name) for name in ("controls.csv", "applied.csv"))
    with np.load(folder / "policy-trace.npz", allow_pickle=False) as loaded:
        trace = {key: loaded[key] for key in ("frames", "action", "logp", "value", "camera_ns", "native_tick")}
    camera = camera_rows(native, controls, applied, trace)
    require(tuple(int(camera[0][field]) for field in ("lives", "bombs")) == (row["initial_spares"], row["initial_bombs"])
            and camera[0]["tick"] < end <= camera[0]["tick"] + 4 * row["decision_cap"],
            "Causal original windows do not start from the frozen stock and remain inside the horizon")
    frames, actions = trace["frames"], trace["action"].astype(np.int64, copy=False)
    move, focus, shot, bomb = actions.T
    request = (np.array([16, 0, 32])[move // 3] | np.array([64, 0, 128])[move % 3]
               | (4 * focus) | shot | (2 * bomb))
    require(np.array_equal(request, controls["requested"]) and np.array_equal(request, applied["requested_word"]),
            "Original key requests differ from the full sampled policy")
    next_row = np.r_[camera[1:], [boundary]].astype(native.dtype)
    _, timing = native_controls(camera, next_row, request, slots, next_row["lives"] < camera["lives"],
                                "practice_stage_clear", end)
    torch.set_num_threads(1)
    model = torch.jit.load(str(actor), map_location="cpu").eval()
    logp, values = forward(model, frames, np.r_[frames[:1], frames[:-1]], actions)
    logp_error = float(np.max(np.abs(logp - trace["logp"])))
    value_error = float(np.max(np.abs(values - trace["value"])))
    require(logp_error <= 8e-4 and value_error <= 3e-4, "Archived policy cannot be re-forwarded")

    index = int(np.flatnonzero(trace["native_tick"] == tick).item())
    tally = frames[index]
    output.mkdir(parents=True, exist_ok=True)
    image = output / f"stage-clear-camera-{tick}.png"
    Image.fromarray(tally).resize((6 * tally.shape[1], 6 * tally.shape[0]), Image.Resampling.NEAREST).save(image)
    report = dict(seed=seed, role=row["role"], scenario=row["scenario"], registration_sha256=sha(registration),
                  qualification="excluded; original final qualification and result-screen capture were never issued",
                  outcome="practice_stage_clear", outcome_recovered_from_immutable_sources=True,
                  source_failure_sha256=sha(failure), source_fault_log_sha256=sha(log), source_evidence_sha256=evidence,
                  source_archived_policy_sha256=PARENT, archive=archive, original_result=result,
                  original_first_tick=int(camera[0]["tick"]), native_ticks=end - int(camera[0]["tick"]),
                  first_nonspell_tick=nonspell[0], last_nonspell_tick=nonspell[-1], matched_nonspell_samples=len(nonspell),
                  first_spell_tick=spell[0], last_spell_tick=spell[-1], matched_spell_samples=len(spell),
                  verified_tally_tick=tick, verified_camera_index=index, native_tally_pixels_sha256=digest_camera(tally),
                  stage_clear_camera_png_sha256=sha(image), decisions=len(actions), control_audit=timing,
                  first_stock=[int(camera[0]["lives"]), int(camera[0]["bombs"])],
                  end_stock=[int(boundary["lives"]), int(boundary["bombs"])],
                  native_misses=int(boundary["misses"] - camera[0]["misses"]),
                  native_bomb_units_spent=int(boundary["bombs_used"] - camera[0]["bombs_used"]),
                  reforward_logp_max_absolute_error=logp_error, reforward_critic_max_absolute_error=value_error)
    out = output / "recovered-physical-original.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report), sha(out))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("seed", type=int)
    cli.add_argument("tally_tick", type=int)
    cli.add_argument("actor", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    recover(args.study, args.seed, args.tally_tick, args.actor, args.output)
