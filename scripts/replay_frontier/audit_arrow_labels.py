"""Measure title-specific conflicting arrow label incidences and score competing resolutions on native displacement."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

if __package__:
    from .audit_frontier import A,D,F,OUT,SCALE,controls,sha
else:
    from audit_frontier import A,D,F,OUT,SCALE,controls,sha


def legacy(w):
    return ((w&32!=0).astype(int)-(w&16!=0)+1)*3+(w&128!=0).astype(int)-(w&64!=0)+1


def add(c,key,value):
    c[key]+=int(np.count_nonzero(value))


def rate(c,num,den):
    return dict(changed=c[num],total=c[den],percent=round(100*c[num]/max(1,c[den]),5))


def paired(c,z):
    w=z['expert_action_chunk']; native=z['native'];n=len(w)
    new=controls(w)%9; old=legacy(w);c['windows']+=n;c['ticks']+=w.size
    changed=new!=old
    add(c,'ticks_changed',changed);add(c,'any_changed',changed.any(1));add(c,'first_changed',changed[:,0])
    # Use exactly pretrain_policy.label_actions's most common composite key and tie break.
    k=w&0xf9;counts=(k[:,:,None]==k[:,None,:]).sum(-1);primary=k[np.arange(n),counts.argmax(1)]
    add(c,'hard_changed',legacy(primary)!=controls(primary)%9)
    soft=(old[:,:,None]==np.arange(9)).sum(1)!=(new[:,:,None]==np.arange(9)).sum(1)
    add(c,'soft_changed',soft.any(1))
    prior=controls(native)%9;first=new[:,0];change=prior!=first
    old_prior=legacy(native)
    old_change=old_prior!=old[:,0]
    add(c,'change',change);add(c,'change_any',change&changed.any(1));add(c,'change_label',change&changed[:,0])
    add(c,'change_prior_or_first',change&((old_prior!=prior)|changed[:,0]))
    add(c,'change_membership',change!=old_change)
    now_starts=np.where(prior==4,first,4);old_starts=np.where(old_prior==4,old[:,0],4)
    add(c,'immediate_start_label_changed',now_starts!=old_starts)


def raw(c,k):
    new=controls(k)%9;old=legacy(k);x=new!=old
    c['ticks']+=len(k);add(c,'ticks_changed',x)
    add(c,'horizontal_conflicts',(k&192)==192);add(c,'vertical_conflicts',(k&48)==48)
    native_change=new[1:]!=new[:-1]
    add(c,'change',native_change)
    add(c,'change_prior_or_first',native_change&(x[1:]|x[:-1]))
    add(c,'change_label',native_change&x[1:])
    n=len(k)//4*4
    c['windows']+=n//4;add(c,'any_changed',x[:n].reshape(-1,4).any(1))


def physical(rid,deltas,bounds):
    metrics={}
    for axis,mask in [('vertical',48),('horizontal',192)]:
        errors=[[] for _ in range(3)]
        for stage in range(1,7):
            with np.load(F/f'th14_ud{rid}-stage{stage}.npz',allow_pickle=False) as z:
                w=z['expert_action_chunk'][:-1]
                actual=z['context'][:,:2]*SCALE
                inc=((np.diff(z['stage_frame'])==4)&(np.diff(z['context'][:,3])==0)&
                     ((w&mask)==mask).any(1)&~(w&2!=0).any(1))
                keys=w[inc]
                for p,priority in enumerate((-1,0,1)):
                    pos=actual[:-1][inc].copy()
                    for k in keys.T:
                        dx=np.where(k&64,-1,(k&128)>0)
                        dy=np.where(k&32,1,-((k&16)>0).astype(int))
                        if axis=='vertical':
                            dy=np.where((k&48)==48,priority,dy)
                        else:
                            dx=np.where((k&192)==192,priority,dx)
                        action=9*((k&8)!=0)+3*(dy+1)+dx+1
                        pos=np.clip(pos+deltas[action],*bounds)
                    errors[p].extend(np.linalg.norm(pos-actual[1:][inc],axis=1).tolist())
        metrics[axis]={str(i-1):dict(rows=len(v),within_one=int(np.sum(np.asarray(v)<=1)),
                                     median=float(np.median(v)) if v else None)
                       for i,v in enumerate(errors)}
    return metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((D/'expanded-corpus.json').read_text(encoding='utf-8'))
    targets=[r for r in manifest if r['title']==14 and r['split']=='train']
    result={};snap=F.parent/'frozen-training-with-hard'
    for group,rows in [('all_train_th14',targets),('normal_reimu_a',[r for r in targets
                                if r['difficulty']=='Normal' and r['shot'].replace(' ','')=='ReimuA'])]:
        by_name={r['name'] for r in rows};r=Counter();p=Counter()
        for row in rows:
            path=D/(Path(row['name']).stem+'-actions.npz')
            with np.load(path,allow_pickle=False) as z:
                for key in z.files:
                    raw(r,z[key])
        identities=set();shards=0
        for path in snap.glob('th14_*-stage*.json'):
            meta=json.loads(path.read_text(encoding='utf-8'))
            if meta['name'] in by_name and meta['split']=='train':
                identities.add(meta['name']);shards+=1
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                    paired(p,z)
        result[group]=dict(raw=dict(identities=len(rows),totals=r,ticks=rate(r,'ticks_changed','ticks'),
            disjoint_windows=rate(r,'any_changed','windows'),change_incoming_label=rate(r,'change_label','change'),
            change_either_boundary=rate(r,'change_prior_or_first','change')),
            paired=dict(identities=len(identities),shards=shards,totals=p,
                four_tick_windows=rate(p,'any_changed','windows'),hard=rate(p,'hard_changed','windows'),
                soft=rate(p,'soft_changed','windows'),first=rate(p,'first_changed','windows'),
                change_incoming_label=rate(p,'change_label','change'),change_either_boundary=rate(p,'change_prior_or_first','change'),
                change_any_future_word=rate(p,'change_any','change'),
                starts=rate(p,'immediate_start_label_changed','windows')))
    with np.load(A/'route.npz',allow_pickle=False) as z:
        physical_checks={rid:physical(rid,z['action_deltas']/4,z['bounds']) for rid in ('0613','05e0','068f')}
    result.update(schema='th14-opposite-arrow-label-audit-v1',reconstruction_motion_calibration='Alice train paired',
                  paired_snapshot=str(snap.relative_to(F.parents[3])),all_raw_source_sha256=sha(D/'expanded-corpus.json'),
                  native_displacement_hypotheses=physical_checks,
                  denominator_definition='A direction-change tick/decision is an effective native direction change from previous input to first upcoming input (includes starts/stops). Either-boundary means either endpoint direction was mislabeled; incoming alone also reported. Raw windows nonoverlapping from stage tick zero; actual paired windows use exact shard indices.')
    (OUT/'arrow-labels.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for group in ('all_train_th14','normal_reimu_a'):
        print(group,'raw',result[group]['raw']['ticks'],result[group]['raw']['change_either_boundary'],'paired',result[group]['paired']['four_tick_windows'],result[group]['paired']['change_either_boundary'],result[group]['paired']['hard'])
    print(physical_checks)


if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
