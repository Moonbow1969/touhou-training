"""Add one-time Stage 1 milestones from an independent native training reference."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from scripts.checkpoint_storage import digest


def original(root,seed):
    matches=list(root.glob(f'train-*-{seed}/receipt.json'))
    if len(matches)!=1:raise ValueError(f'Missing unique original training source {seed}')
    f=matches[0];r=json.loads(f.read_bytes());trace=f.parent/r['file']
    if r['seed']!=seed or r['role']!='train' or r.get('continues')!=0 or digest(trace)!=r['sha256']:
        raise ValueError('Invalid native training evidence')
    with np.load(trace,allow_pickle=False) as z:
        return r,z['stage'].copy(),z['native_frame'].copy(),z['reward'].copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();d=a.root;out=a.out;out.mkdir(exist_ok=True,parents=True)
    ref=json.loads((d/'camera-protocol.json').read_bytes())
    maxima=[]
    for seed in ref['train_seeds']['13']:
        r,stage,ticks,_=original(d/'camera-native13',seed)
        maxima.append(int(ticks[stage==1].max()))
    thresholds=np.quantile(maxima,[.25,.5,.75],method='nearest').astype(int)
    if len(set(thresholds))!=3:raise ValueError('Three distinct native milestones required')
    base=d/'camera-round2';m=base/'camera-batch/credits.json';rows=json.loads(m.read_bytes())
    info=[]
    for row in rows:
        path=m.parent/row['file'];dest=out/row['file']
        if digest(path)!=row['sha256']:raise ValueError('Changed original learner trace')
        if row['title']!=13:
            if not dest.exists():
                try:os.link(path,dest)
                except OSError:shutil.copyfile(path,dest)
            if digest(dest)!=row['sha256']:raise ValueError('Changed control-title trajectory')
            continue
        r,stage,ticks,original_reward=original(base/'native13',row['seed'])
        if r['policy_sha256']!=row['policy_sha256']:raise ValueError('Original camera trace differs from native clock trace')
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if not np.array_equal(original_reward,arrays['reward']):raise ValueError('Original reward mismatch')
        extra=np.zeros_like(original_reward);events=[]
        for threshold in thresholds:
            matches=np.flatnonzero((stage==1)&(ticks>=threshold))
            if len(matches):
                i=int(matches[0]);extra[i]+=.25
                events.append(dict(threshold=int(threshold),decision=i,observed_tick=int(ticks[i]),bonus=.25))
        arrays.update(reward_original=original_reward,reward_progress=extra,reward=original_reward+extra)
        np.savez_compressed(dest,**arrays)
        info.append(dict(seed=row['seed'],original_sha256=row['sha256'],original_utility=float(original_reward.sum()),
                         shaped_utility=float(arrays['reward'].sum()),events=events))
        row['sha256']=digest(dest)
    details=dict(reference='Eight prior SFT-generated TH13 training credits',reference_maximum_stage1_ticks=maxima,
                 reference_seeds=ref['train_seeds']['13'],thresholds=thresholds.tolist(),bonus_per_first_crossing=.25,
                 titles_changed=[13],base_manifest=digest(m),credits=info)
    for f,v in [('credits.json',rows),('shaping.json',details)]:
        (out/f).write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
