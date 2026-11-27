"""Reproduce TH13 Trance evidence solely from recorded original inputs, stocks, image windows and policy receipts."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import struct
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scripts.collect_th10_full_action import save
from scripts.audit_th13_policy_suffix import verify as verify_shared
from scripts.run_th10_camera_cohort import digest


def read(p):return json.loads(p.read_bytes())


def seal(study):
    lr=read(study/"human-census-and-head.json")
    physical=read(study/"c-key-engineering.json");assert digest(study/"physical-probe-sensor-source.c")==physical["observer_source_sha256"]
    stock=struct.unpack("<21i",(study/"physical-probe-terminal-stock.bin").read_bytes());assert stock[0]==stock[-1] and stock[8:10]==(2,0)
    with (study/"c-key-original.csv").open() as f:raw=list(csv.DictReader(f))
    pulses=[]
    for p in physical["phases"]:
        a,b=p["begin_tick"]+4,p["end_tick"]-3
        c=Counter(int(r["native_input"]) for r in raw if a<=int(r["native_tick"])<=b)
        expected={"none":0,"c":0xa00,"Shift_L":8,"z":1}[p["key"]]
        assert c[expected]>15 and set(c)<=set((0,expected))
        pulses.append(dict(key=p["key"],native_counts=dict(c)))
    outcomes=[];witnesses=[]
    for file in sorted(study.glob("replay-*/witness.json")):
        w=read(file);assert w["raw_sha256"]==digest(file.parent/"native.csv") and w["exact_full_word_matches"]==w["observed_ticks"]
        ws=dict(source_sha256=w["replay_sha256"],group=w["player_group"],split=w["split"],stage=w["stage"],exact_full_word_matches=w["exact_full_word_matches"])
        witnesses.append(ws)
        with (file.parent/"native.csv").open() as f:d={int(r["native_tick"]):r for r in csv.DictReader(f) if int(r["stage"])==w["stage"]}
        for e in w["events"]:
            t=e["native_tick"];pre,state,after=d[t-1],d[t],d[t+1];bits=int(state["native_input"])&0xa00
            assert int(state["native_input"])==e["source_word"] and min(int(x["spares"]) for x in (pre,state,after))>=0
            assert int(pre["spares"])==int(state["spares"]) and int(pre["bombs"])==int(state["bombs"])
            activated=int(pre["active_candidate"])==0 and int(state["active_candidate"])==2 and int(after["gauge_candidate"])==int(pre["gauge_candidate"])-1
            outcomes.append(dict(**ws,tick=t,special_mask=hex(bits),gauge_before=int(pre["gauge_candidate"]),gauge_on_edge=int(state["gauge_candidate"]),
                         gauge_next=int(after["gauge_candidate"]),active_transitions=[int(r["active_candidate"]) for r in (pre,state,after)],
                         spares=int(state["spares"]),bombs=int(state["bombs"]),manual_activation=activated))
    assert len(outcomes)==6 and sum(r["manual_activation"] for r in outcomes)==3
    pilot=study/"five-head-original";allfive=[]
    for seed in read(pilot/"protocol.json")["registered_seeds"]:
        folder=pilot/f"engineering-five-head-{seed}";r=read(folder/"receipt.json");p=read(folder/"fifth-qualification.json")
        assert digest(folder/"rollout.npz")==r["sha256"] and digest(folder/"five-head-actions.npz")==p["post_replay_npz_sha256"]
        assert r["continues"]==0 and r["outcome"]=="game_over" and r["difficulty"]==2
        with np.load(folder/"five-head-actions.npz") as z:
            assert z["action"].shape==z["logp"].shape==(p["steps"],5) and z["terminated"].sum()==1 and z["terminated"][-1]
        base_audit=verify_shared(folder,study.parents[1]/"four-game-online/policy/policy.ts",prefix=False)
        allfive.append(dict(**p,shared_four_head_rescore=base_audit,stage2_reached=r["stage2_reached"],terminal_spares=max(0,r["remaining_lives"]),terminal_bombs=r["remaining_bombs"]))
    # Independent display from unit-calibrated timer at the previously registered screenshot.
    ref=study/"replay-th12_ud003670-s2"
    with (ref/"native.csv").open() as f:series=[r for r in csv.DictReader(f) if int(r["stage"])==2 and 1180<=int(r["native_tick"])<=1900]
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.9),layout="constrained",width_ratios=(1,1.3))
    im=Image.open(ref/"original-1338.png").crop((3,22,643,502))
    axes[0].imshow(im);axes[0].axis("off");axes[0].set_title("Original game countdown")
    x=[int(r["native_tick"]) for r in series];y=[int(r["gauge_candidate"]) for r in series]
    axes[1].plot(x,y,color="#165c7c",lw=2.5);axes[1].axvline(1239,color="#c95842",ls="--");axes[1].text(1254,555,"Manual input at tick 1,239")
    axes[1].set(xlabel="Original Stage 2 tick",ylabel="Spirit gauge (native units)",ylim=(-22,650));axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("TH13 trial 0.01a   ·   ULUK’s original Easy replay",fontsize=14)
    fig.savefig(study/"native-trance-witness.png",dpi=160);plt.close(fig)
    summary=dict(game_sha256=witnesses and physical["official_trial_sha256"],physical_pulses=pulses,physical_probe_original_rank=stock[8],
        raw_header_correction="In the original physical-pulse CSV the field labeled difficulty is the unqualified 0xb90b0, and score is unqualified 0xb9048. Ignore both. Independent qualified observer terminal stock reports Hard=2 and Reimu=0. Only native input, stage clock, stocks and Trance fields contribute.",
        memory_fields={"sampled_input":"image-base + 0xdf430","native_stage_tick":"image-base + 0xb90b4","stage":"image-base + 0xb90a8",
                       "spirit_gauge":"image-base + 0xb9094; 600 at full, loses one per original tick during verified Trance","trance_active":"image-base + 0xb9098; 0→2 on verified manual and can also change on death"},
        witnesses=witnesses,human_events=outcomes,raw_source_onsets_and_camera_coverage=lr["groups"],held_camera_model=lr["metrics"]["held"],
        head_sha256=lr["weights_sha256"],engineering_full_credits=allfive,
        interpretation="Three full-gauge manual positives and two low-gauge 0x800 non-activations, plus a full-gauge 0x200-only non-activation. These are two independent recorded human games. Five-head live engineering tests do not establish that the timing policy is useful.")
    save(study/"qualification.json",summary)
    archive=study/"trance-native-evidence.zip"
    include_names={"qualification.json","human-census-and-head.json","visual-trance-head.bin","c-key-original.csv","c-key-engineering.json",
        "physical-probe-terminal-stock.bin","physical-probe-native-selected-rank.png","physical-probe-sensor-source.c","native-trance-witness.png",
        "native.csv","witness.json","protocol.json","fifth-qualification.json","fifth-head.csv","native-trance.csv","five-head-actions.npz","rollout.npz","steps.bin",
        "receipt.json","ranking.png","final-stock.bin","events.tsv"}
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(study.rglob("*")):
            if p.relative_to(study).parts[0].startswith("standalone-repeat-"):continue
            if p.is_file() and (p.name in include_names or p.name.startswith(("original-","c-key-control-")) and p.suffix==".png"):
                z.write(p,p.relative_to(study))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    save(study/"archive.json",dict(file=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,raw_policy_archives_preserved=True))
    print(witnesses,[(p["seed"],p["steps"],p["c_decisions"],p["c_native_bit_matches"],p["observed_same_frame_c_full_gauge_onsets"]) for p in allfive],digest(archive),archive.stat().st_size)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("study",type=Path);seal(p.parse_args().study)
