"""Check the actual native TH12 key word on every tick of two-action visual macro steps."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    return np.genfromtxt(p,names=True,delimiter='\t',encoding='utf8',dtype=None,ndmin=1)
def native_mask(command):
    x=int(command[0])%3-1;y=int(command[0])//3-1
    return (64*(x<0)+128*(x>0)+16*(y<0)+32*(y>0)+8*int(command[1])+int(command[2])+2*int(command[3]))
def x11_mask(command):
    x=int(command[0])%3-1;y=int(command[0])//3-1
    return int(x<0)+2*(x>0)+4*(y<0)+8*(y>0)+16*int(command[1])+32*int(command[2])+64*int(command[3])
def score(path,actor,mode):
    steps=table(path/'subwindow-wire.tsv');sensor=table(path/'engineering-native.tsv')
    actual=table(path/'difficulty-at-reset.tsv')[0]
    assert actual['rank']==actual['recorded_rank']==2 and actual['spares']==2
    assert (path/'diagnostic-end.txt').is_file() and len(steps)==96
    ns={}
    for row in sensor:
        if row['mode']==0 and row['stage']==row['manager_stage']==1:
            ns.setdefault(int(row['tick']),set()).add(int(row['keys'])&251)
    z=np.load(path/'model-wire.npz',allow_pickle=False)
    imgs=z['frames'];a=z['action'];assert a.shape==(96,2,4)
    assert imgs.tobytes()==(path/'subwindow-frames.u8').read_bytes()
    previous=np.concatenate([imgs[:1],imgs[:-1]],0)
    inp=np.stack([imgs,np.abs(imgs.astype('i2')-previous).astype('u1')],axis=1)
    model=torch.jit.load(str(actor),map_location='cpu').eval();torch.set_num_threads(1)
    with torch.inference_mode():
        heads=model(torch.from_numpy(inp).float()/255.);logits=torch.cat(heads,dim=-1).numpy()
        errors=[]
        for slot in (0,1):
            hslot=0 if mode=='four-tick' else slot
            for k,h in enumerate(heads):
                index=torch.from_numpy(a[:,slot,k].copy()).long()[:,None]
                errors.append(float(np.abs(h[:,hslot,:].log_softmax(1).gather(1,index)[:,0].numpy()-z['logp'][:,slot,k]).max()))
    max_logits=float(np.max(np.abs(z['logits']-logits)))
    assert max_logits<.001 and max(errors)<.001
    groups=[]
    for i,s in enumerate(steps):
        t=int(s['image_tick']);targets=[native_mask(command) for command in a[i]]
        req=[x11_mask(command) for command in a[i]]
        if [int(s[k]) for k in ('first_x11','second_x11')]!=req or [int(s[k]) for k in ('first_verified','second_verified')]!=req:
            raise ValueError('The image-model commands differ from native physical X key queries')
        keys=[next(iter(ns[t+j])) if len(ns.get(t+j,()))==1 else None for j in (1,2,3,4)]
        expected=[targets[0],targets[0],targets[1],targets[1]]
        groups.append(dict(group=i,image_native_tick=t,physical_consume_observations=keys,requested_by_tick=expected,
                     exact_match_by_tick=[k==v if k is not None else None for k,v in zip(keys,expected)],
                     second_command_differs=targets[0]!=targets[1],late_first_apply=int(s['first_apply_tick'])-t,
                     late_second_apply=int(s['second_apply_tick'])-(t+2),end_clock_error=int(s['end_tick'])-(t+4)))
    def rate(values):
        values=list(values);return dict(matches=values.count(True),mismatches=values.count(False),unknown=values.count(None),total=len(values))
    hist=lambda vals:{str(int(k)):int(v) for k,v in sorted(Counter(vals).items())}
    return dict(mode=mode,original_role='engineering',rank='Hard',verified_physical_masks=192,groups=len(steps),
        observed_camera_native_gap_histogram=hist(np.diff(steps['image_tick'])),camera_recaptures=int(steps['camera_retries'].sum()),
        desired_half_hold_ticks=2,observed_second_apply_offset_histogram=hist(steps['second_apply_tick']-steps['image_tick']),first_apply_offset_histogram=hist(steps['first_apply_tick']-steps['image_tick']),
        changed_second_commands=sum(g['second_command_differs'] for g in groups),
        original_key_match_by_tick={str(j+1):rate(g['exact_match_by_tick'][j] for g in groups) for j in range(4)},
        changed_second_command_first_native_tick=rate(g['exact_match_by_tick'][2] for g in groups if g['second_command_differs']),
        changed_second_command_second_native_tick=rate(g['exact_match_by_tick'][3] for g in groups if g['second_command_differs']),
        all_requested_native_tick_matches=rate(v for g in groups for v in g['exact_match_by_tick']),
        max_abs_logit_error=max_logits,max_abs_sampled_logp_error=max(errors),policy_sha256=digest(actor),
        cold_start_native={'stored':int(actual['recorded_rank']),'rank':int(actual['rank']),'spares':int(actual['spares'])},
        sources={n:digest(path/n) for n in ('subwindow-wire.tsv','engineering-native.tsv','model-wire.npz','subwindow-frames.u8','difficulty-at-reset.tsv','difficulty-menu-selected.png','diagnostic-end.txt')},
        methodology='Physical X state is queried after both commands. Native keys are separately re-read at original ticks image+1,+2,+3,+4 for every camera. A missing or conflicting native row is unknown. Equal commands give no information about mid-hold switching latency. No score or training use.',
        native_per_group=groups)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--mode',choices=['four-tick','two-subwindows'],required=True)
    a=ap.parse_args();data=score(a.folder,a.actor,a.mode);(a.folder/'native-clock-audit.json').write_text(json.dumps(data,indent=2)+'\n');print({k:v for k,v in data.items() if k not in ('native_per_group','sources')})
