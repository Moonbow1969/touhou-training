"""Match X11 commands to the original TH08 0.02 recording buffer, indexed by stage and native tick."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def game_word(x11):
    return sum(native for bit,native in enumerate((64,128,16,32,4,1,2)) if x11 & (1 << bit))


def read_records(folder):
    paths=sorted(folder.glob("native-record-stage-*.bin"))
    if not paths:
        paths=[folder/"native-record.bin"]
    result={}
    for path in paths:
        stage=int(path.stem.split("-")[-1]) if "-stage-" in path.stem else 1
        data=path.read_bytes()
        count=int.from_bytes(data[:4],"little")
        words=np.frombuffer(data[4:],"<u2")
        if len(words)!=count or stage not in (1,2,3):
            raise ValueError("Native recording extent or stage is invalid")
        result[stage]=words
    return result


def audit(folder,out,marker=False):
    out.mkdir(exist_ok=True,parents=True)
    gates=np.genfromtxt(folder/"gates.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    timing=[json.loads(s) for s in (folder/"timing.jsonl").read_text().splitlines()]
    play=[s for s in timing if s["phase"]=="policy"]
    stage=gates["stage"] if "stage" in gates.dtype.names else np.ones(len(gates),dtype=int)
    assert len(gates)==len(play) and np.array_equal(gates["index"],np.arange(len(gates)))
    expected_build="3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1"
    assert (folder/"build.sha256").read_text().split()[0]==expected_build
    native=read_records(folder)
    decisions=[]
    with np.load(folder/"behavior.npz",allow_pickle=False) as behavior:
        actions=behavior["action"].copy()
    assert actions.shape==(len(play),4)
    final=next((r for r in reversed(timing) if r.get("phase")=="terminal"),None)
    if not final and (folder/"engineering-last-camera.png").exists():
        from PIL import Image
        from scripts.run_th08_trial_camera import stocks
        last=stocks(np.asarray(Image.open(folder/"engineering-last-camera.png").convert("RGB")))
    else:
        last=[final["spares"],final["bombs"]] if final else [None,None]
    for i,(g,r) in enumerate(zip(gates,play)):
        which=int(stage[i]); target=int(g["apply_target"])+1
        w=game_word(int(g["x11_mask"])); seq=native[which]
        assert 0<target and target+4<=len(seq) and int(g["x11_mask"])==r["mask"]
        prior=decisions[i-1]["expected_game_word"] if i and which==stage[i-1] else 0
        boundary_observable=prior!=w
        before_matches=int(seq[target-1])==prior
        all_four=np.all(seq[target:target+4]==w)
        exact_edge=boundary_observable and before_matches and int(seq[target])==w
        following=([play[i+1]["spares"],play[i+1]["bombs"]] if i+1<len(play) else last)
        decisions.append(dict(index=i,stage=which,camera_target=int(g["camera_target"]),camera_seen=int(g["camera_observed"]),
            x11_target=int(g["apply_target"]),x11_seen=int(g["apply_observed"]),native_earliest_scheduled=target,
            x11_mask=int(g["x11_mask"]),expected_game_word=w,prior_game_word=int(seq[target-1]),
            native_word_0=int(seq[target]),native_word_1=int(seq[target+1]),native_word_2=int(seq[target+2]),native_word_3=int(seq[target+3]),
            all_four_native_words_match=bool(all_four),native_boundary_observable=boundary_observable,
            exact_distinguishable_native_edge=bool(exact_edge),stock_lives=r["spares"],stock_bombs=r["bombs"],
            next_lives=following[0],next_bombs=following[1],recorded_action=actions[i].tolist()))
    if marker and any(not r["native_boundary_observable"] for r in decisions):
        raise ValueError("A registered focus boundary is ambiguous")
    with (out/"native-decisions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
    confirmed_edges={d["index"] for d in decisions if d["exact_distinguishable_native_edge"]}
    continuous=[(a,b) for a,b in zip(decisions,decisions[1:]) if a["stage"]==b["stage"]]
    matched_slots=sum(d["all_four_native_words_match"] for d in decisions)
    exact_mask_intervals=sum(a["all_four_native_words_match"] and b["all_four_native_words_match"] and
        b["native_earliest_scheduled"]-a["native_earliest_scheduled"]==4 for a,b in continuous)
    resolved=[(a,b) for a,b in continuous if a["index"] in confirmed_edges and b["index"] in confirmed_edges]
    intervals_with_proven_edges=sum(b["native_earliest_scheduled"]-a["native_earliest_scheduled"]==4 for a,b in resolved)
    camera=[d for d in decisions if d["camera_target"]>=0]
    result=dict(build_sha256=expected_build,engineering_focus_marker=marker,
        stage_clock_validation="Original mode-0 recording word at stage origin + 0x24 + 2*tick; per-stage readback after original input consumption.",
        stages_observed=sorted(set(map(int,stage))),live_stage_transition_verified=bool(np.any(np.diff(stage)!=0)),
        sampled_decisions=len(play),all_four_native_words_match=matched_slots,exact_x11_gate=sum(d["x11_target"]==d["x11_seen"] for d in decisions),
        camera_gates=len(camera),exact_camera_gates=sum(d["camera_target"]==d["camera_seen"] for d in camera),
        distinguishable_edges=sum(d["native_boundary_observable"] for d in decisions),verified_native_edges=len(confirmed_edges),
        adjacent_edge_pairs_observable=sum(a["native_boundary_observable"] and b["native_boundary_observable"] for a,b in continuous),
        exact_adjacent_edge_intervals=intervals_with_proven_edges,within_stage_intervals=len(continuous),
        consecutive_fully_matching_four_tick_intervals=int(exact_mask_intervals),preshaped_or_automatic_actions=False if not marker else None,
        exact_interval_fraction=float(exact_mask_intervals/len(continuous)) if continuous else None,
        first_stock=[play[0]["spares"],play[0]["bombs"]],last_stock=last,
        median_x11_interval_ms=float(np.median(np.diff(gates["x_ack_ns"]))/1e6),
        x11_interval_range_ms=(np.array([min(np.diff(gates["x_ack_ns"])),max(np.diff(gates["x_ack_ns"]))])/1e6).tolist(),
        outcome=json.loads((folder/"receipt.json").read_bytes())["outcome"],
        csv_sha256=sha(out/"native-decisions.csv"))
    (out/"verified-timing.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--engineering-focus-marker",action="store_true")
    a=p.parse_args();print(json.dumps(audit(a.raw,a.out,a.engineering_focus_marker),indent=2))
