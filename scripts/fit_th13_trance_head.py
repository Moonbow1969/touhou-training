"""Fit a separate TH13-only C head from actual replay controls and immutable train/held pilots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


def expit(x):
    return 1/(1+np.exp(-np.clip(x,-70,70)))


def log_loss(actual,score):
    score=np.clip(score.astype(np.float64),1e-12,1-1e-12)
    return -np.mean(np.where(actual,np.log(score),np.log1p(-score)))


def average_precision_score(actual,score):
    order=np.argsort(-score,kind="stable");sorted_score=score[order]
    positive=np.cumsum(actual[order]);end=np.flatnonzero(np.r_[np.diff(sorted_score)!=0,True])
    return float(np.sum(np.diff(np.r_[0,positive[end]])*positive[end]/(end+1))/positive[-1])
from replays.inspect_rpy import action_streams
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest


def features(frames):
    n=len(frames);out=np.empty((n,84),dtype=np.float32)
    for i in range(0,n,192):
        current=frames[i:i+192]
        previous=frames[np.maximum(np.arange(i,i+len(current))-1,0)]
        diff=np.abs(current.astype(np.int16)-previous.astype(np.int16))
        out[i:i+len(current),:42]=current.reshape(-1,7,16,6,16).mean((2,4)).reshape(-1,42)/255
        out[i:i+len(current),42:]=diff.reshape(-1,7,16,6,16).mean((2,4)).reshape(-1,42)/255
    return out


def run(workspace):
    workspace=workspace.resolve();d=workspace/"outputs/touhou-vision-261126";src=d/"human-trial13";out=d/"engineering/th13-trance"
    out.mkdir(exist_ok=True,parents=True);catalog=json.loads((src/"frozen-source-splits.json").read_bytes())
    rows=[];data={"train":[],"held":[]}
    for r in catalog["sources"]:
        f=workspace/"work/th13-human-vision/trial-archive"/r["name"];assert digest(f)==r["sha256"]
        for stage,stream in action_streams(f,13,version="0.01a"):
            a=np.frombuffer(stream,"<u2").reshape(-1,3)[:,0]
            onset={bit:np.flatnonzero(((a&bit)!=0)&np.r_[True,(a[:-1]&bit)==0]) for bit in (0x200,0x800)}
            stem=f"th13trial-{f.stem}-s{stage}";p=src/r["split"]/(stem+".npz")
            row=dict(group=r["player_group"],split=r["split"],source=r["name"],stage=stage,source_sha256=r["sha256"],
                     raw_native_ticks=len(a),raw_0x200_onsets=onset[0x200].tolist(),raw_0x800_onsets=onset[0x800].tolist(),
                     raw_0x800_held_ticks=int(np.count_nonzero(a&0x800)),camera_available=p.exists())
            if p.exists():
                with np.load(p) as z:
                    t=z["native_frame"].astype(np.int64);frames=z["obs"]
                    eligible=(t+4<=len(a))&np.r_[False,np.diff(t)==4]
                    time=t[eligible];chunks=z["input_chunk"][eligible]
                    assert np.array_equal(chunks,a[time[:,None]+np.arange(4)])
                    split_onset=np.zeros(len(a),dtype=bool);split_onset[onset[0x800]]=True
                    y=(chunks[:,0]&0x800)!=0
                    rising=split_onset[time[:,None]+np.arange(4)].any(1)
                    covered=np.unique((time[:,None]+np.arange(4))[split_onset[time[:,None]+np.arange(4)]]).tolist()
                    data[r["split"]].append((features(frames)[eligible],y,rising))
                    row.update(camera_sha256=digest(p),eligible_camera_decisions=len(time),upcoming_first_tick_held=int(y.sum()),
                               covered_rising_source_ticks=covered,covered_source_onsets=len(covered))
            rows.append(row)
    pooled={split:tuple(np.concatenate(values) for values in zip(*arrays)) for split,arrays in data.items()}
    x,y,_=pooled["train"];x=x.astype(np.float64);y=y.astype(np.float64)
    mu=x.mean(0);sd=np.maximum(x.std(0),.01);standard=(x-mu)/sd;prevalence=y.mean()
    # Same class likelihood as the witnessed human data. Only slopes receive a fixed weak ridge penalty.
    penalty=.001
    torch.set_num_threads(2)
    xt=torch.from_numpy(standard);yt=torch.from_numpy(y)
    theta=torch.tensor(np.r_[np.log(prevalence/(1-prevalence)),np.zeros(84)],requires_grad=True)
    optimizer=torch.optim.LBFGS([theta],max_iter=140,tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn="strong_wolfe")
    def obj():
        optimizer.zero_grad()
        z=theta[0]+xt@theta[1:]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yt)+.5*penalty*theta[1:].square().sum()
        loss.backward();return loss
    optimizer.step(obj);fit=theta.detach().numpy()
    slope=fit[1:]/sd;weights=np.r_[fit[0]-slope@mu,slope].astype("<f4")
    path=out/"visual-trance-head.bin";path.write_bytes(b"TH13TR01"+weights.tobytes())
    metrics={}
    for split,(xx,yy,rising) in pooled.items():
        score=expit(xx@weights[1:]+weights[0])
        metrics[split]=dict(camera_decisions=len(yy),held_key_labels=int(yy.sum()),held_fraction=float(yy.mean()),
             predicted_fraction=float(score.mean()),negative_log_likelihood=float(log_loss(yy,score)),
             train_prevalence_baseline_nll=float(log_loss(yy,np.full(len(yy),prevalence))),
             held_label_average_precision=float(average_precision_score(yy,score)),upcoming_onset_rows=int(rising.sum()),
             upcoming_onset_average_precision=float(average_precision_score(rising,score)),upcoming_onset_base_rate=float(rising.mean()))
    groups={sp:dict(raw_onsets_0x800=sum(len(r["raw_0x800_onsets"]) for r in rows if r["split"]==sp),
                    raw_onsets_0x200=sum(len(r["raw_0x200_onsets"]) for r in rows if r["split"]==sp),
                    independent_pilots_with_0x800=sorted({r["group"] for r in rows if r["split"]==sp and r["raw_0x800_onsets"]}),
                    camera_covered_0x800_onsets=sum(r.get("covered_source_onsets",0) for r in rows if r["split"]==sp))
            for sp in pooled}
    save(out/"human-census-and-head.json",dict(groups=groups,source_rows=rows,metrics=metrics,weights_sha256=digest(path),
         protocol="Frozen official TH13 trial split; 84 features of the original 96x112 camera and previous four-tick difference. Regularized logistic regression, no class rebalance. Target is whether 0x800 is held on the next native tick. Evaluate early rising onsets independently. Physical C may additionally set 0x200. Group-disjoint held data never used to fit.",
         fit_penalty=penalty,scope="Auxiliary engineering head, independent from the four main action heads."))
    print(groups,metrics,digest(path))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("workspace",type=Path);run(p.parse_args().workspace)
