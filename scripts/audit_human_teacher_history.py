"""Frozen human replay movement metrics with native motion history intact or masked."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from . import pretrain_policy as bc
from .audit_th10_replay_teacher import TEACHER_SHA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    manifest=json.loads(args.manifest.read_bytes())
    assert digest(args.teacher)==args.expected_sha
    torch.set_num_threads(2)
    teacher=MultiInputActorCriticPolicy(bc.observation_space(True),bc.action_space(),lambda _:3e-4,**bc.network_kwargs(256))
    teacher.load_state_dict(torch.load(args.teacher,weights_only=True,map_location='cpu'));teacher.eval()
    groups={};sources=[]
    for row in manifest['evaluation_manifest']:
        file=next((p for p in args.data.rglob(row['file']) if digest(p)==row['sha256']),None)
        if file is None:raise FileNotFoundError(row)
        b,c,y=bc.read_shard((file,row),True,motion_spans=(4,12,36));y=y[:,0]
        with np.load(file) as z:
            frames=z['stage_frame'];lives=z['context'][:,3];native=bc.movement_direction(z['native'])
            first=bc.movement_direction(z['expert_action_chunk'][:,0])
        prev=np.r_[4,y[:-1]]
        contiguous=np.r_[False,(np.diff(frames)==4)&(np.diff(lives)==0)]
        masks={'all':np.ones(len(y),bool),'contiguous':contiguous,'stable':contiguous&(prev==y),
               'direction_changed':contiguous&(prev!=y),'stationary_to_movement':contiguous&(prev==4)&(y!=4),
               'movement_to_other_movement':contiguous&(prev!=4)&(y!=4)&(prev!=y),
               'movement_to_stationary':contiguous&(prev!=4)&(y==4),
               'native_stationary_to_first_tick_movement':(native==4)&(first!=4)}
        probs={}
        for name in ('intact','masked'):
            ctx=c.copy()
            if name=='masked':ctx[:,-9:]=0
            parts=[]
            with torch.inference_mode():
                for i in range(0,len(y),192):
                    logits=bc.predict_heads(teacher,torch.from_numpy(b[i:i+192]),torch.from_numpy(ctx[i:i+192]))[0]
                    parts.append(logits.log_softmax(1).numpy())
            probs[name]=np.concatenate(parts)
        unit=dict(title=row['title'],replay=row['replay'],stage=row['stage'],sha256=row['sha256'],samples=len(y))
        sources.append(unit)
        for group in (str(row['title']),'all_titles'):
            totals=groups.setdefault(group,{})
            for cond,mask in masks.items():
                if not mask.any():continue
                ix=np.flatnonzero(mask);truth=first[ix] if cond=='native_stationary_to_first_tick_movement' else y[ix]
                sums=totals.setdefault(cond,{})
                for name,lp in probs.items():
                    pred=lp[ix].argmax(1)
                    a=sums.setdefault(name,np.zeros(4,np.float64))
                    a+=[len(ix),(pred==truth).sum(),-lp[ix,truth].sum(dtype=np.float64),(pred!=4).sum()]
    out={}
    for g,metrics in groups.items():
        out[g]={}
        for condition,arms in metrics.items():
            v={name:dict(n=int(a[0]),correct=int(a[1]),accuracy=a[1]/a[0],nll=a[2]/a[0],
                         predicted_moving_rate=a[3]/a[0]) for name,a in arms.items()}
            v['masked_minus_intact']={k:v['masked'][k]-v['intact'][k] for k in ('accuracy','nll')}
            out[g][condition]=v
    report=dict(teacher_sha256=args.expected_sha,split_manifest_sha256=digest(args.manifest),held_shards=sources,
                design="Exact human-replay teacher, identical inputs, labels and weights. Mask sets the previous 4/12/36-native-frame displacement and all three corresponding availability flags to zero. Primary direction changes compare the original human four-tick majority direction to the immediately preceding majority direction with a four-native-tick gap on the same life. Stationary-to-movement is the subset with previous neutral and current nonneutral. Auxiliary first-tick metric defines prior from the instantaneous native key and targets the first expert tick. NLL uses natural log. Masking is an evaluation intervention; the model was trained with its real history.",
                metrics=out)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for k,v in out.items():
        print(k,{name:tuple(round(v[name][a]['accuracy'],4) for a in ['intact','masked']) for name in ['all','direction_changed','stationary_to_movement']})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data','manifest','teacher','out'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--expected-sha',default=TEACHER_SHA)
    run(p.parse_args())
