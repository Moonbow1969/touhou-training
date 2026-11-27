"""Calibrate independent movement samples to temporal persistence in training-player replay inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import sha


class TemperedMovement(torch.nn.Module):
    def __init__(self,actor,temperature):
        super().__init__();self.actor=actor;self.temperature=temperature

    def forward(self,images):
        move,focus,shoot,bomb=self.actor(images)
        return move/self.temperature,focus,shoot,bomb


def stats(rows,temp):
    results=[]
    for name,title,group,logits,truth,cont in rows:
        scores=logits/temp;scores-=scores.max(axis=1,keepdims=True)
        probs=np.exp(scores);probs/=probs.sum(axis=1,keepdims=True)
        redraw=1-(probs[1:]*probs[:-1]).sum(axis=1)
        results.append(dict(source=name,title=title,group=group,pairs=int(cont.sum()),expert=float((truth[1:]!=truth[:-1])[cont].mean()),
            expected_independent_redraw=float(redraw[cont].mean())))
    return results


def calibration_error(groups):
    # Each title receives equal total weight, and each player within title has one vote.
    title_means=[]
    for title in sorted({r['title'] for r in groups}):
        title_rows=[r for r in groups if r['title']==title]
        title_means.append(np.mean([np.mean([abs(r['expected_independent_redraw']-r['expert']) for r in title_rows if r['group']==g])
                                    for g in {r['group'] for r in title_rows}]))
    return float(np.mean(title_means))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train','held'):p.add_argument('--'+key,type=Path,nargs='+',required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();meta=json.loads((a.parent/'report.json').read_bytes());weights=a.parent/'policy.ts'
    if sha(weights)!=meta['torchscript_sha256']:raise ValueError('Frozen policy changed')
    actor=torch.jit.load(str(weights)).eval();device='cuda' if torch.cuda.is_available() else 'cpu';actor.to(device)
    torch.set_num_threads(2)
    def load(split):
        paths=getattr(a,split)
        if {sha(f) for f in paths}!={x['sha256'] for x in meta[split]}:raise ValueError('Must use the original player split')
        rows=[]
        for f in paths:
            r=load_replay(f);x=torch.from_numpy(r['raw']);cont=np.diff(r['native_frame'])==4
            with torch.no_grad():logits=torch.cat([actor(b.to(device).float()/255)[0].cpu() for b in x.split(64)]).numpy()
            rows.append((r['name'],r['title'],r['group'],logits,r['action'][:,0],cont))
        return rows
    train=load('train');grid=np.r_[np.geomspace(.1,1.,81),np.geomspace(1.,2.,17)[1:]]
    errors=[calibration_error(stats(train,t)) for t in grid];temp=float(grid[int(np.argmin(errors))])
    held=load('held');a.out.mkdir(parents=True,exist_ok=False)
    actor.cpu();wrapper=TemperedMovement(actor,temp).eval();fixture=torch.zeros(1,2,112,96)
    with torch.no_grad():
        compiled=torch.jit.trace(wrapper,fixture)
        for head in range(1,4):torch.testing.assert_close(compiled(fixture)[head],actor(fixture)[head],rtol=0,atol=0)
        compiled.save(str(a.out/'policy.ts'))
    report=dict(temperature=temp,parent_sha256=sha(weights),policy_sha256=sha(a.out/'policy.ts'),train=meta['train'],held=meta['held'],
        criterion='Mean absolute consecutive movement switch-rate error, balanced by title and player. Only actions separated by exactly four native ticks.',
        grid=grid.tolist(),train_error=errors,training_at_1=stats(train,1.),training_calibrated=stats(train,temp),
        held_at_1=stats(held,1.),held_calibrated=stats(held,temp))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
