"""Hold current image, expert targets and model fixed while varying genuinely earlier camera frames."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

def evaluate(data, manifest, model_file, device):
    source=json.loads(manifest.read_bytes())
    if source['split']!='held':
        raise ValueError('The diagnostic is scored on frozen held pilots')
    torch.set_num_threads(1)
    model=torch.jit.load(str(model_file),map_location=device).eval()
    columns=('direction','focus','fire','bomb')
    groups=defaultdict(list)
    stage_counts=[]
    with torch.inference_mode():
        for stage in source['qualified']:
            with np.load(data/stage['path'],allow_pickle=False) as archive:
                ticks=archive['native_frame'].ravel().astype(np.int64)
                frames=archive['obs']
                target=archive['action']
                stable=archive['stable'].astype(bool) if 'stable' in archive else np.ones(len(frames),bool)
            index=np.arange(3,len(ticks))
            keep=stable[index]
            for lag in (1,2,3):
                keep &= stable[index-lag] & (ticks[index]-ticks[index-lag]==4*lag)
            index=index[keep]
            stage_counts.append(dict(pilot=stage['player_group'],stage=stage['stage'],count=len(index)))
            for start in range(0,len(index),96):
                ids=index[start:start+96]
                now=torch.from_numpy(frames[ids]).to(device=device,dtype=torch.float32)/255
                y=torch.from_numpy(target[ids]).to(device=device,dtype=torch.int64)
                for lag in (0,1,2,3):
                    then=torch.from_numpy(frames[ids-lag]).to(device=device,dtype=torch.float32)/255
                    heads=model(torch.stack((now,(now-then).abs()),1))[:4]
                    nll=torch.stack([-h.log_softmax(1).gather(1,y[:,j:j+1]).ravel() for j,h in enumerate(heads)],1)
                    correct=torch.stack([h.argmax(1)==y[:,j] for j,h in enumerate(heads)],1)
                    groups[stage['player_group'],lag*4].append((len(ids),nll.sum(0).cpu().numpy(),correct.sum(0).cpu().numpy()))
    def metrics(parts):
        n=sum(k for k,_,_ in parts)
        return dict(rows=n,mean_head_nll=dict(zip(columns,map(float,sum(x for _,x,_ in parts)/n))),
                    head_accuracy=dict(zip(columns,map(float,sum(x for _,_,x in parts)/n))))
    per_pilot={}
    combined={}
    for ticks in (0,4,8,12):
        for pilot in source['distinct_pilots']:
            per_pilot.setdefault(pilot,{})[ticks]=metrics(groups[pilot,ticks])
        combined[ticks]=metrics([x for (_,gap),parts in groups.items() if gap==ticks for x in parts])
    return dict(scope='Offline diagnostic. Earlier frames only; current pixels, expert next-tick/four-tick labels, qualified rows and frozen four-head checkpoint are held fixed. Zero means zero difference plane. Counterfactual image offset changes the model input only. No claims about resulting original game performance.',frozen_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),stages=stage_counts,by_previous_native_tick_offset=combined,by_pilot=per_pilot)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('held_manifest',type=Path);p.add_argument('model',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args()
    doc=evaluate(a.data,a.held_manifest,a.model,a.device)
    a.output.write_text(json.dumps(doc,indent=2)+'\n')
