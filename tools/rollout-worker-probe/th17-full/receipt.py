"""Seal an original TH17 credit under the camera PPO data contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from original_ui import comparison
from timeline import coordinates, progressed_ticks, event_coordinate, initialized_stock_rows
from score_ranking import inspect as inspect_ranking

BUILD = "3ffa3963003cce6dc02b1dae5adeb4411138772834efdb195a05b9081027aaba"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_rows(path):
    return np.genfromtxt(path, delimiter="\t", names=True, dtype=None, encoding="utf8", ndmin=1)

def terminal_evidence(root, native_stage):
    terminal = root / "terminal.png"
    for screen in ("game_over", "score_ranking"):
        score, valid = comparison(terminal, screen)
        if valid:
            with np.load(Path(__file__).with_name("score-stage-one.npz")) as references:
                classified = inspect_ranking(terminal, references)
            if classified["stage"] is not None and classified["stage"] != native_stage:
                raise ValueError("Highlighted original ranking contradicts the observed stage")
            return screen, score, classified
    raise ValueError("Native terminal photograph did not pass qualification")

def state_events(native, decisions):
    parts = np.zeros((len(decisions), 4), dtype="f4")
    history = []
    entries = {}
    action_end = coordinates(decisions["after_stage"], decisions["after_tick"])
    for prior, current in zip(native, native[1:]):
        stage, tick = int(current["stage"]), int(current["tick"])
        step = min(len(decisions)-1, int(np.searchsorted(action_end, event_coordinate(stage,tick))))
        change = int(current["spares"] - prior["spares"])
        if change:
            parts[step,0] += change
            history.append(dict(step=step, tick=tick, kind="life", change=change, **({"stage":stage} if stage>1 else {})))
        cast = int(prior["bombs"] - current["bombs"])
        if change == 0 and cast > 0:
            parts[step,1] -= .03 * cast
            history.append(dict(step=step, tick=tick, kind="spent_bombs", count=cast, **({"stage":stage} if stage>1 else {})))
        if current["stage"] > prior["stage"]:
            parts[step,2] += 2
            entries[int(current["stage"])] = dict(step=step, tick=tick, spares=int(current["spares"]), bombs=int(current["bombs"]))
    return parts, history, entries

def seal(root, identity):
    rank = identity["difficulty"]
    native = load_rows(root/"native.tsv")
    native, initialization_rows = initialized_stock_rows(native[native["stage"]>=1],identity.get("observer_version",1))
    decisions = load_rows(root/"decisions.tsv")
    n = len(decisions)
    if not len(native) or n<2:
        raise ValueError("Native or behavior acquisition is incomplete")
    first, last = native[0], native[-1]
    if tuple(first[k] for k in ("stage","start_stage","spares","bombs")) != (1,1,2,3) or not 0<=first["tick"]<180:
        raise ValueError("A whole fresh original credit is required")
    if not all(np.all(native[k]==v) for k,v in {"rank":rank,"character":0,"beast":0,"continues":0}.items()):
        raise ValueError("Original rank, character, selected beast or Continue changed")
    progress = progressed_ticks(native["stage"],native["tick"])
    if last["spares"] != -1:
        raise ValueError("Original stage or stock-exhaustion witness is incomplete")
    if decisions[0]["tick"]>=180 or decisions[-1]["spares"]!=-1 or not np.array_equal(decisions["keys"],decisions["observed_keys"]):
        raise ValueError("Original controls or final action are incomplete")
    rank_score, rank_ok = comparison(root/"rank-selected.png", "hard" if rank==2 else "lunatic")
    if not rank_ok or not comparison(root/"rank-initial.png", "normal")[1]:
        raise ValueError("Selected difficulty does not match the original rank menu")
    stage = int(max(native["stage"]))
    terminal, terminal_score, ranking = terminal_evidence(root, stage)
    action = np.stack([decisions[k] for k in ("move","focus","shoot","bomb")], axis=1).astype("i8")
    logp = np.stack([decisions[k] for k in ("logp_move","logp_focus","logp_shoot","logp_bomb")], axis=1).astype("f4")
    value = decisions["value"].astype("f4")
    if not (np.isfinite(logp).all() and np.isfinite(value).all()):
        raise ValueError("Nonfinite sampled outputs")
    frames = np.fromfile(root/"observations.u8", dtype="u1").reshape(-1,112,96)
    if len(frames)!=n or frames[n//2].std()<5:
        raise ValueError("Camera and decisions disagree")
    parts, events, entries = state_events(native, decisions)
    ended = np.zeros(n, dtype=bool)
    ended[-1] = True
    tmp = root/"behavior.pending.npz"
    np.savez_compressed(tmp, frames=frames, action=action, logp=logp, value=value, reward=parts.sum(1), terminated=ended, reward_parts=parts,
        observed_keys=decisions["observed_keys"].astype("u1"), native_tick=decisions["tick"].astype("i4"),
        stock=np.stack((decisions["spares_before"],decisions["bombs_before"]),axis=1).astype("i2"))
    tmp.replace(root/"behavior.npz")
    last_live = native[native["spares"]>=0][-1]
    entry = entries.get(2)
    receipt = dict(**identity, game="TH17", title=17, mode=identity["split"], policy=identity["arm"], actions=n, steps=n,
        difficulty_name="Hard" if rank==2 else "Lunatic", difficulty_verified_native_samples=len(native), native_preinitialization_rows=initialization_rows, native_rank=rank,
        character=0, beast=0, shot=0, continues=0, difficulty_evidence="Two original rank-menu photographs and native rank for complete story credit",
        rank_selection_template_score=rank_score, terminal_reason="game_over", terminal="game_over", terminal_observed=True, terminal_screen=terminal,
        terminal_template_recall=terminal_score, independent_ranking=ranking, maximum_stage=stage, stage2_reached=entry is not None,
        stage2_spare_lives=None if entry is None else entry["spares"],stage2_bombs=None if entry is None else entry["bombs"],stage_entries=entries,
        initial_stock=dict(stage=1,spares=2,bombs=3,continues=0,character=0,beast=0), last_live_spares=int(last_live["spares"]),
        last_live_bombs=int(last_live["bombs"]), remaining_bombs=int(last["bombs"]), remaining_spare_lives=0, remaining_lives=0, terminal_spares=-1,
        first_loss_native_tick=next((e["tick"] for e in events if e["kind"]=="life" and e["change"]<0),None),
        last_stage_native_tick=int(last["tick"]), seconds=float(decisions["observed_ns"][-1]-decisions["onset_ns"][0])/1e9,
        native_ticks_per_second=1000*float(progress[-1])/float(last["ms"]-first["ms"]),
        terminal_score=int(last["score"])*10,verified_rate=1.,action_counts=np.bincount(action[:,0],minlength=9).tolist(),
        key_positive_counts=action[:,1:].sum(0).tolist(), reward_parts=parts.sum(0).tolist(), reward_events=events,
        reward_components=["native life changes including final loss","same-life Bomb spending","first entry into new stage","dense time omitted"],
        source_executable_sha256=BUILD, executable_sha256=BUILD, trajectory="behavior.npz")
    for name, field in {"native.tsv":"native_trace_sha256","decisions.tsv":"decisions_sha256","observations.u8":"frames_sha256",
                        "rank-initial.png":"rank_initial_sha256","rank-selected.png":"rank_selected_sha256","start-hud.png":"start_hud_sha256",
                        "terminal.png":"terminal_observation_sha256","behavior.npz":"trajectory_sha256"}.items():
        receipt[field] = digest(root/name)
    receipt["native_difficulty_trace_sha256"]=receipt["native_trace_sha256"]
    tmp = root/"receipt.pending.json"
    tmp.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    tmp.replace(root/"receipt.json")
    return receipt

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode",type=Path)
    parser.add_argument("--identity",type=Path,required=True)
    a=parser.parse_args()
    r=seal(a.episode,json.loads(a.identity.read_bytes()))
    print(json.dumps({k:r[k] for k in ("actions","difficulty","terminal_screen","stage2_reached","last_live_bombs","trajectory_sha256")},indent=2))


