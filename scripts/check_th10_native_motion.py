"""Measure actual player displacement during fixed held evaluation actions."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.collect_th10_full_action import native_mask,save


def audit(folder):
    r=json.loads((folder/'report.json').read_bytes())
    with gzip.open(folder/'native.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    nt=np.array([int(a['utc_ms']) for a in rows])
    pos=np.array([[float(a['x']),float(a['y'])] for a in rows])
    lives=np.array([int(a['lives']) for a in rows])
    keys=np.array([native_mask(int(a['native_input']))&15 for a in rows])
    with np.load(folder/'behavior.npz') as data:
        direction=data['action'][:,0];clock=data['action_utc_ms']
    table={key:Counter() for key in ('horizontal','vertical','diagonal')}
    for t,code in enumerate(direction):
        dx,dy=code%3-1,code//3-1
        if dx==0 and dy==0:continue
        kind='diagonal' if dx and dy else ('vertical' if dy else 'horizontal')
        out=table[kind];out['requested']+=1
        start=int(np.searchsorted(nt,clock[t]+10))
        end=int(np.searchsorted(nt,(clock[t+1] if t+1<len(clock) else nt[-1])-5))
        expected=(1 if dx<0 else 2 if dx>0 else 0)|(4 if dy<0 else 8 if dy>0 else 0)
        if end-start<2 or np.any(lives[start:end]!=lives[start]):
            continue
        ids=np.flatnonzero(keys[start:end]==expected)+start
        if len(ids)<2 or nt[ids[-1]]-nt[ids[0]]<20:
            continue
        movement=pos[ids[-1]]-pos[ids[0]]
        if np.linalg.norm(movement)>40:
            continue
        out['measured']+=1
        dot=float(np.dot(movement,[dx,dy]))
        out['along_command' if dot>.25 else 'against_command' if dot<-.25 else 'zero_or_perpendicular']+=1
    return dict(seed=r['seed'],decisions=r['steps'],counts=table)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path)
    args=p.parse_args()
    out={}
    for name in ('ppo3','camera_sft','camera_ppo'):
        rows=[audit(x.parent) for x in sorted((args.base/('held-'+name)).glob('*/report.json'))]
        assert [r['seed'] for r in rows]==list(range(754001,754007))
        counts={key:sum((r['counts'][key] for r in rows),Counter()) for key in ('horizontal','vertical','diagonal')}
        out[name]=dict(credits=rows,counts=counts)
    result=dict(protocol='Match original consumed movement bits in at least two native position samples at least 20 ms apart; observe between action +10 ms and next action -5 ms, stable spare count, maximum net displacement 40 native coordinate units. Positive projection above 0.25 native coordinate units counts as movement along command. No edge, invulnerability or stationary frames were removed. Native samples are diagnostics.',arms=out)
    save(args.base/'native-displacement.json',result)
    print(json.dumps({k:v['counts'] for k,v in out.items()},indent=2))
