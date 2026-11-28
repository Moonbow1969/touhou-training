"""Stratify regular native-input boundary misses and compare independent timing/CPU evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_th08_live_tick4 import game_word, read_records


def inspect(raw):
    t=np.genfromtxt(raw/"gates.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    obs=np.genfromtxt(raw/"clock.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    logs=[json.loads(s) for s in (raw/"timing.jsonl").read_text().splitlines()]
    play=[r for r in logs if r["phase"]=="policy"]
    stages=t["stage"] if "stage" in t.dtype.names else np.ones(len(t),int)
    words=read_records(raw)
    losses=np.flatnonzero(np.diff([r["spares"] for r in play]+[logs[-1]["spares"]])<0)
    casts=np.flatnonzero(np.array([r["mask"] for r in play])&64)
    data=[]
    for i,g in enumerate(t[:-1]):
        stage=int(stages[i]);tick=int(g["apply_target"])+1;buf=words[stage]
        expected=game_word(int(g["x11_mask"]));p=buf[tick:tick+4]
        changed=i==0 or t["x11_mask"][i-1]!=g["x11_mask"]
        missing=np.flatnonzero(p!=expected).tolist()
        if not changed:offset=None
        else:offset=next((j for j in range(-2,4) if buf[tick+j]==expected and buf[tick+j-1]!=expected),None)
        data.append(dict(index=i,stage=stage,tick=tick,expected_mask=expected,mismatch_slots=missing,changed=bool(changed),edge_offset=offset,
            x11_late=bool(g["apply_target"]!=g["apply_observed"]),camera_late=bool(g["camera_target"]>=0 and g["camera_target"]!=g["camera_observed"]),
            near_cast=bool(np.any(abs(casts-i)<=5)),near_spare_loss=bool(np.any(abs(losses-i)<=5)),ack_ns=int(g["x_ack_ns"])))
    groups={}
    for key in ("changed","x11_late","camera_late","near_cast","near_spare_loss"):
        groups[key]={}
        for flag in (False,True):
            selected=[r for r in data if r[key]==flag]
            groups[key][str(flag).lower()]=dict(n=len(selected),mismatches=sum(bool(r["mismatch_slots"]) for r in selected))
    stage1=obs[(obs["stage_index"]==0)&(obs["flags"]==4)&(obs["mode"]==0)&(obs["stable"]==1)]
    for label,miss in (("on_time_changed",False),("late_changed",True)):
        chosen=[r for r in data if r["changed"] and bool(r["mismatch_slots"])==miss and r["stage"]==1]
        a=np.array([r["ack_ns"] for r in chosen],np.int64)
        tt=np.array([r["tick"] for r in chosen],np.int64)
        near=np.searchsorted(stage1["tick"],tt)
        if len(a) and np.max(near)<len(stage1):
            groups[label]=dict(n=len(a),ack_until_next_native_first_observation_ms_p10_p50_p90=
                (np.percentile(stage1["after_ns"][near]-a,[10,50,90])/1e6).tolist())
    return dict(intervals=len(data),mismatches=sum(bool(r["mismatch_slots"]) for r in data),
        original_start_ack_ns=int(t["x_ack_ns"][0]),original_last_ack_ns=int(t["x_ack_ns"][-1]),strata=groups,
        mismatch_slots_histogram={str(k):sum(r["mismatch_slots"]==k for r in data) for k in
            [list(j) for j in sorted(set(tuple(r["mismatch_slots"]) for r in data))]},
        changed_edge_offsets={str(k):sum(r["edge_offset"]==k and r["changed"] for r in data) for k in
            set(r["edge_offset"] for r in data if r["changed"])})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--raw",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.write_text(json.dumps(inspect(a.raw),indent=2)+"\n",encoding="utf-8")
