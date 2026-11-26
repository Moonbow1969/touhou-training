"""Audit full-action human replay predictions conditional on starts, turns and Bomb onset."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts import pretrain_policy as bc
from scripts.train_portable_full_sft import FullActor
from scripts.checkpoint_storage import digest

def score(args):
    torch.set_num_threads(2);model=FullActor.from_bin(args.weights).eval()
    manifest=json.loads(args.manifest.read_bytes())
    by_title={}
    for r in manifest["evaluation_manifest"]:
        p=args.data/r["file"]
        if digest(p)!=r["sha256"]:raise ValueError("Evaluation shard changed")
        with np.load(p) as z:
            frame=z["stage_frame"];chunks=z["expert_action_chunk"]
            ii=np.arange(0,len(frame)-1,2);ii=ii[frame[ii+1]-frame[ii]==4]
            y=bc.label_actions(np.concatenate([chunks[ii],chunks[ii+1]],axis=1))
            x=np.zeros((len(ii),110),np.float32);last=np.zeros_like(y);valid=np.zeros(len(y),bool)
            for lag in (1,2):
                ix=np.flatnonzero(frame[ii][lag:]-frame[ii][:-lag]==8*lag)+lag
                prev=y[ix-lag];off=84+(lag-1)*12
                x[ix,off+prev[:,0]]=1;x[ix,off+9:off+12]=prev[:,1:]
                if lag==1:last[ix]=prev;valid[ix]=True
            x[:,108]=(frame[ii]/(8*120)).clip(0,1);x[:,109]=(frame[ii]/(8*1800)).clip(0,1)
            with torch.no_grad():logits,_=model(torch.from_numpy(x))
            prob=np.concatenate([h.softmax(1).numpy() for h in logits],axis=1)
            guess=np.stack([h.argmax(1).numpy() for h in logits],axis=1)
            by_title.setdefault(r['title'],[]).append((y,last,valid,guess,prob))
    result={}
    for t,shards in by_title.items():
        y,old,ok,g,p=map(np.concatenate,zip(*shards))
        measures={}
        for k,name,mask in [(0,'movement_changed',(old[:,0]!=y[:,0])),
                             (0,'start',(old[:,0]==4)&(y[:,0]!=4)),
                             (1,'focus_changed',old[:,1]!=y[:,1]),(2,'shoot_changed',old[:,2]!=y[:,2]),(3,'bomb_onset',(old[:,3]==0)&(y[:,3]==1))]:
            keep=ok&mask;idx=np.arange(len(y))[keep]
            off=(0,9,11,13)[k]
            measures[name]=dict(n=len(idx),greedy_correct=int(np.count_nonzero(g[keep,k]==y[keep,k])),
                                mean_sample_success=float(np.mean(p[idx,off+y[keep,k]])) if len(idx) else None)
        result[t]=dict(total=len(y),conditional=measures,bomb_positive=int(y[:,3].sum()),
                       model_greedy_bomb_positive=int(g[:,3].sum()),expected_bomb=float(p[:,14].mean()),observed_bomb=float(y[:,3].mean()))
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','data','weights','out'):a.add_argument('--'+k,required=True,type=Path)
    score(a.parse_args())
