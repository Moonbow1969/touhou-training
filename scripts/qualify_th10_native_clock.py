"""Compare native TH10 timer candidates through pause, scripted loops and stage changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(folder):
    clock = folder / "discovered-clocks.csv"
    rows = read(clock)
    if not rows:
        raise ValueError("No original-game candidate clock observations")
    stamp = np.array([int(r["utc_ms"]) for r in rows])
    stage = np.array([int(r["stage"]) for r in rows])
    background = np.array([int(r["stage_background_tick"]) for r in rows])
    if np.any(np.diff(stamp) < 0):
        raise ValueError("Out-of-order native samples")
    with (folder / "pause-events.csv").open() as stream:
        event = {key:int(t) for t,key in csv.reader(stream)}
    plateau = (stamp >= event["paused_sample"]) & (stamp < event["resume_requested"])
    period = np.diff(background)
    same_stage = stage[1:] == stage[:-1]
    advance = (period > 0) & (period < 20) & same_stage
    # The independent stage-background timer has a known -512 boss loop and a reset on Stage 2.
    loop = (period < -100) & same_stage & (stage[1:] == 1)
    transition = (stage[:-1] == 1) & (stage[1:] == 2)
    rank = []
    for key in list(rows[0])[3:]:
        values = np.array([float(r[key]) if r[key] else np.nan for r in rows])
        valid = np.isfinite(values)
        if not valid.all():
            continue
        delta = np.diff(values)
        def summary(mask):
            sample = delta[mask]
            return dict(n=int(len(sample)), minimum=int(sample.min()), maximum=int(sample.max())) if len(sample) else None
        paused = values[plateau]
        if len(paused) < 5 or not np.any(advance):
            continue
        match = float((abs(delta[advance] - period[advance]) <= 1).mean())
        score = dict(field=key, first=int(values[0]), last=int(values[-1]), hold_range=int(np.ptp(paused)),
                     active_increment_match=match, same_stage_decreases=int(np.sum(delta[same_stage] < 0)),
                     stage1_decreases=int(np.sum(delta[same_stage & (stage[1:] == 1)] < 0)),
                     largest_same_stage_increment=int(delta[same_stage].max()),
                     background_loop_increments=summary(loop), stage_change_increments=summary(transition))
        rank.append(score)
    found = [r["field"] for r in rank if r["hold_range"] == 0 and r["active_increment_match"] > .97
             and r["stage1_decreases"] == 0 and r["largest_same_stage_increment"] < 30
             and r["background_loop_increments"] and r["background_loop_increments"]["minimum"] >= 0]
    # This fixed global can be checked across independently launched credits without
    # relying on the address of any dynamically allocated manager.
    fixed = "g472000_b4c"
    fixed_details = None
    if fixed in rows[0]:
        v = np.array([int(r[fixed]) for r in rows])
        fixed_details = dict(field="0x472b4c", first_observed=int(v[0]), last_observed=int(v[-1]),
                             pause_limits=[int(v[plateau].min()),int(v[plateau].max())],
                             before_after_each_stage_change=[[int(v[i]),int(v[i+1])] for i in np.flatnonzero(transition)],
                             before_after_first_background_loop=[[int(v[i]),int(v[i+1]),int(background[i]),int(background[i+1])]
                                                                  for i in np.flatnonzero(loop)[:1]])
        first_transition = np.flatnonzero(transition)
        if len(first_transition):
            transition_index = int(first_transition[0]) + 1
            after = np.flatnonzero((np.arange(len(v)) >= transition_index) & (stamp < stamp[transition_index] + 4000))
            fixed_details["first_four_seconds_after_stage2"] = dict(minimum=int(v[after].min()),maximum=int(v[after].max()),
                decreases=[dict(ms_after_first2=int(stamp[i+1]-stamp[transition_index]),before=int(v[i]),after=int(v[i+1]))
                           for i in np.flatnonzero(np.diff(v)<0) if transition_index-1<=i<after[-1]])
        for phase,start,end in [("right","right_down","right_up"),("left","left_down","left_up")]:
            take = np.flatnonzero((stamp >= event[start]) & (stamp < event[end]))
            if len(take) >= 2:
                a,b=take[0],take[-1]
                fixed_details[phase]=dict(observations=len(take), candidate_increment=int(v[b]-v[a]),
                                          background_increment=int(background[b]-background[a]))
    result = dict(fixed_global=fixed_details,trace_sha256=hashlib.sha256(clock.read_bytes()).hexdigest(), native_samples=len(rows),
                  sample_span_s=float((stamp[-1]-stamp[0])/1000), paused_observations=int(plateau.sum()),
                  same_stage_background_backward_loops=int(loop.sum()), original_stage_transitions=int(transition.sum()),
                  observed_stage_values=sorted(set(map(int,stage))), matched_native_fields=found, candidates=rank,
                  scope="Read-only engineering observations on a single original trial credit. Passing a trace identifies candidate elapsed simulation clocks. Their game-specific semantics and repeatability require an independent credit.")
    (folder / "native-clock-qualification.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    result = run(parser.parse_args().folder)
    print(json.dumps({k:v for k,v in result.items() if k != "candidates"},indent=2))
