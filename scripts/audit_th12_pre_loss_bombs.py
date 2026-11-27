"""Compare native Bomb-key timing in matched windows of training-only original TH12 lives."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read_table(path):
    return np.genfromtxt(path,delimiter="\t",names=True,encoding="utf-8",dtype=None,ndmin=1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_window(index, stage, first, stop, spare):
    # Include the previous native key value to identify a rising edge at tick first.
    values=[]
    for tick in range(first-1,stop):
        states=index.get((stage,tick),set())
        if len(states)!=1:
            return None,"missing_or_disagreeing_tick"
        state,stock,x=next(iter(states))
        if state!=spare:
            return None,"another_life_or_transition"
        values.append((stock,x))
    if values[1][0]<=0:
        return None,"no_bomb_in_stock_at_window_entry"
    return dict(initial_bombs=values[1][0], final_bombs=values[-1][0],
                rising_edges=sum(not a[1] and b[1] for a,b in zip(values,values[1:])),
                spent=sum(max(0,a[0]-b[0]) for a,b in zip(values,values[1:]))),None


def audit(study):
    plan=json.loads((study/"camera-protocol.json").read_bytes())
    seeds=plan["train_seeds"]["12"]
    rows=[]
    files={}
    for seed in seeds:
        (source,)=(study/"native12/credits").glob(f"train-six-ppo-{seed}-*.json")
        receipt=json.loads(source.read_bytes())
        raw=study/"native12/raw"/source.stem
        path=raw/"engineering-native.tsv"
        trace=read_table(path)
        live=trace[(trace["stage"]==trace["manager_stage"])&(trace["stage"]>0)&(trace["tick"]>=0)&(trace["mode"]==0)]
        index=defaultdict(set)
        for v in live:
            index[int(v["stage"]),int(v["tick"])].add((int(v["spares"]),int(v["bombs"]),bool(int(v["keys"])&2)))
        started=read_table(raw/"engineering-decisions.tsv")[0]
        deaths=np.flatnonzero(np.diff(live["spares"])==-1)+1
        assert len(deaths)==3 and live[deaths[-1]]["spares"]==-1
        for i in deaths:
            a,b=live[i-1:i+1]
            stage,tick,spares=int(b["stage"]),int(b["tick"]),int(a["spares"])
            row=dict(seed=seed,rank=receipt["difficulty"],stage=stage,death_tick=tick,
                     pre_loss_bombs=int(a["bombs"]),spares_before_loss=spares)
            if (stage,tick-73)<(int(started["apply_stage"]),int(started["apply_tick"])):
                rows.append(dict(**row,reason="before_learned_control"))
                continue
            earlier,err1=edge_window(index,stage,tick-72,tick-48,spares)
            danger,err2=edge_window(index,stage,tick-24,tick,spares)
            rows.append(dict(**row,earlier=earlier,pre_loss=danger,reason=err1 or err2))
        files[str(seed)]=dict(receipt_sha256=sha(source),original_trace_sha256=sha(path),
                              first_decision_witness_sha256=sha(raw/"engineering-decisions.tsv"))
    summary={}
    for rank in (2,3):
        all_events=[r for r in rows if r["rank"]==rank]
        pairs=[v for v in all_events if not v["reason"]]
        ids=sorted({v["seed"] for v in all_events})
        result={}
        for key in ("rising_edges","spent"):
            both=np.array([[sum(v["seed"]==s and v["earlier"][key]>0 for v in pairs),
                            sum(v["seed"]==s and v["pre_loss"][key]>0 for v in pairs),
                            sum(v["seed"]==s for v in pairs)] for s in ids],int)
            rng=np.random.default_rng(1711200+rank)
            totals=both[rng.integers(0,len(ids),(20000,len(ids)))].sum(1)
            boot=(totals[:,1]-totals[:,0])[totals[:,2]>0]/totals[totals[:,2]>0,2]
            n=len(pairs)
            result[key]=dict(eligible_losses=n,early_with_event=int(both[:,0].sum()),pre_loss_with_event=int(both[:,1].sum()),
                             pre_loss_minus_early_rate=float((both[:,1].sum()-both[:,0].sum())/n) if n else None,
                             credit_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist() if n else None)
        summary[rank]=dict(credits=len(ids),losses=len(all_events),holding_any_bomb_at_loss=sum(r["pre_loss_bombs"]>0 for r in all_events),
                           mean_bombs_before_loss=float(np.mean([r["pre_loss_bombs"] for r in all_events])),eligible=len(pairs),
                           excluded=dict(Counter(r["reason"] for r in all_events if r["reason"])),paired=result)
    return dict(scope="Exploratory training-only analysis of 32 genuine 171 originals. Two 24-tick windows in the same native life, ticks death−72 through death−49 and death−24 through death−1. Both require positive Bomb stock at entry and complete unambiguous per-tick telemetry, including the prior tick. Each native X rising edge and each same-life stock decrement is counted.",
                selection="Conditioning on observed deaths excludes threats successfully avoided or bombed. These figures describe resource and key timing among actual losses. Estimating the causal value of removing a Bomb penalty requires the independent paired original experiment.",
                by_rank=summary,events=rows,sources=files)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    result=audit(args.study)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["by_rank"]))
