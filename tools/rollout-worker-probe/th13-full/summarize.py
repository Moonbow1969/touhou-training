"""Aggregate completed TH13 native full credits by the fitted policy and held assignment."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean,median

import numpy as np


def wilson(wins,n):
    z=1.959963984540054
    p=wins/n
    scale=1+z*z/n
    mid=(p+z*z/(2*n))/scale
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/scale
    return [0. if wins==0 else max(0.,mid-half),1. if wins==n else min(1.,mid+half)]


def score(root):
    groups=defaultdict(list)
    for path in sorted(root.glob("*/receipt.json")):
        row=json.loads(path.read_bytes())
        if not row["terminal"] or row["continues"]!=0:raise ValueError(path)
        trace=path.parent/row["file"]
        if hashlib.sha256(trace.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError(trace)
        with np.load(trace,allow_pickle=False) as z:
            if len(z["action"])!=row["steps"] or sum(z["terminated"])!=1 or not z["terminated"][-1]:raise ValueError(trace)
            if str(z["policy_sha256"])!=row["policy_sha256"] or z["continues"].any():raise ValueError(trace)
            components=z["reward_components"]
            expected_reward=components@np.array([-1,1,2,-.03,.001])
            if not np.allclose(expected_reward,z["reward"],atol=1e-6):raise ValueError(f"Reward mismatch: {trace}")
            resource_after=np.column_stack((z["spares"],z["bombs"]))
            resource_gaps=int((resource_after[:-1]!=z["resource_before"][1:]).any(axis=1).sum())
            death_rows=np.flatnonzero(components[:,0]>0)
            bombs_on_losses=z["resource_before"][death_rows,1].astype(int).tolist()
            stage1=z["native_frame"][z["stage"]==1]
            movements=np.bincount(z["action"][:,0],minlength=9)
            final_score=int(z["raw_score"][-1])*10
        state_path=path.parent/"native-terminal.json"
        native_ranking=json.loads(state_path.read_bytes()) if state_path.is_file() else None
        if native_ranking:
            if native_ranking["native_pause_state"]!=1 or native_ranking["continues"]!=0 or native_ranking["spare_lives"]!=-1:raise ValueError(state_path)
            stamp=path.parent/"final-stock.bin"
            if hashlib.sha256(stamp.read_bytes()).hexdigest()!=native_ranking["digest"]:raise ValueError(stamp)
        arms=path.parent.name.split("-")
        cohort="original" if row["role"]!="held" or row["seed"]<733000 else "fresh"
        groups[(row["role"],cohort,arms[1],row["policy_sha256"])].append({
            "seed":row["seed"],"steps":row["steps"],"stage1_last_observed_tick":int(stage1.max()),
            "stage1_first_observed_tick":int(stage1[0]),"resource_changes_between_decisions":resource_gaps,
            "confirmed_life_losses":row["confirmed_losses"],"earned_lives":row["life_earned"],
            "bombs_before_life_losses":bombs_on_losses,
            "movement_histogram_row_major":movements.tolist(),"median_hold_ms":row["median_hold_ms"],
            "wall_seconds":row["wall_seconds"],"arrival":row["stage2"],"entry_spares":row["stage2_entry_spares"],
            "entry_bombs":row["stage2_entry_bombs"],"terminal_active_lives":max(0,1+row["remaining_lives"]),
            "native_raw_terminal_spares":row["remaining_lives"],"last_playable_spares":row["last_playable_spares"],
            "last_playable_bombs":row["last_playable_bombs"],"terminal_stock_bombs":row["remaining_bombs"],
            "effective_bombs":row["effective_bombs"],"actual_native_inputs_matching":row["native_key_match"],
            "sampled_vertical_moves":row["vertical_moves"],"sampled_shoot":row["shoot_steps"],"return":row["return_sum"],
            "display_score":final_score,"late_terminal_screen_saved":(path.parent/"ranking.png").is_file(),
            "ranking_telemetry_verified":bool(native_ranking),
            "ranking_display_score":native_ranking["display_score"] if native_ranking else None,
            "data_sha256":row["sha256"]})
    reports=[]
    for (role,cohort,arm,digest),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r["seed"]);n=len(rows);win=sum(r["arrival"] for r in rows)
        if len({r['seed'] for r in rows})!=n:raise ValueError("Duplicate seed")
        reports.append(dict(role=role,cohort=cohort,arm=arm,policy_sha256=digest,n=n,stage2=win,stage2_fraction=win/n,
                            stage2_wilson_95=wilson(win,n),mean_decisions=mean(r["steps"] for r in rows),
                            median_decisions=median(r["steps"] for r in rows),
                            mean_effective_bombs=mean(r["effective_bombs"] for r in rows),
                            mean_last_playable_bombs=mean(r["last_playable_bombs"] for r in rows),
                            mean_terminal_active_lives=mean(r["terminal_active_lives"] for r in rows),
                            mean_terminal_bomb_stock=mean(r["terminal_stock_bombs"] for r in rows),
                            median_stage1_last_observed_tick=median(r["stage1_last_observed_tick"] for r in rows),
                            mean_stage1_last_observed_tick=mean(r["stage1_last_observed_tick"] for r in rows),
                            mean_last_hud_score=mean(r["display_score"] for r in rows),
                            independently_verified_rankings=sum(r["ranking_telemetry_verified"] for r in rows),
                            confirmed_life_losses=sum(r["confirmed_life_losses"] for r in rows),
                            losses_with_bombs=sum(b>0 for r in rows for b in r["bombs_before_life_losses"]),
                            unresolved_resource_changes=sum(r["resource_changes_between_decisions"] for r in rows),
                            sampled_vertical_moves=sum(r["sampled_vertical_moves"] for r in rows),
                            movement_histogram_row_major=np.sum([r["movement_histogram_row_major"] for r in rows],axis=0).tolist(),
                            realized_inputs=sum(r["actual_native_inputs_matching"] for r in rows),
                            decisions=sum(r["steps"] for r in rows),rows=rows))
    return dict(game="TH13 official Trial 0.01a, Normal Reimu",host="Endeavour",
                clock="Stock stage counter. Four measured game frames per action where the stage clock advances, wall hold guard during time stops.",
                stage2="First native transition into Stage 2. Keep playing same credit after entry. Entry inventory undefined when no entry.",
                terminal="Native spare-lives reaching -1 exhausts the last active life, which is zero remaining lives. Record Bomb stock at the preceding playable decision separately.",
                seed="Matched actor seed per held pair, uncontrolled original RNG.",
                timings="stage1_last_observed_tick is a direct original stage counter; the wall-time denominator includes slower software rendering on six concurrent workers.",
                groups=reports)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.write_text(json.dumps(score(args.root),indent=2)+"\n",encoding="utf-8")
