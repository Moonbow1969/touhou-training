"""Reproduce native TH08 AWR hazards from the public per-credit receipt and resource JSONs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path,checksum=None):
    data=path.read_bytes()
    if checksum is not None and hashlib.sha256(data).hexdigest()!=checksum:
        raise ValueError(f'Original record changed: {path.name}')
    return json.loads(data)


def result(study):
    records=load(study/'physical-intake.json')['records'];groups={};draws={};rng=np.random.default_rng(80966000)
    for arm in ['awr','uniform','lambda1']:
        subset=[r for r in records if r['arm']==arm];covered=[]
        for r in subset:
            seed=r['seed'];root=study/arm;x=load(root/'resource'/f'{seed}.json',r['resource_sha256']);receipt=load(root/'sealed'/f'{seed}.json',r['receipt_sha256'])
            if receipt['policy_sha256']!=r['actor_sha256'] or receipt['role']!='held':raise ValueError('Original policy identity differs')
            if r['strict_control'] and r['stage1_native_exposure'] is not None:
                deaths=sum(ev['stage']==1 for ev in x['events'])
                covered.append([r['lane'],deaths,r['stage1_native_exposure'],x['losses_with_positive_pre_loss_bomb'],x['losses_with_known_pre_loss_bomb']])
        numbers=np.asarray([v[1:] for v in covered],dtype=int).sum(0);d,e,pos,known=map(int,numbers)
        groups[arm]=dict(physical=len(subset),strict=len(covered),stage1_deaths=d,stage1_exposure=e,deaths_per_1000_ticks=1000*d/e,
                         deaths_with_bomb=pos,deaths_with_known_bomb=known,stage2=sum(r['actual_endpoint']['stage2'] is True for r in subset),
                         exact_seeds=[r['seed'] for r in subset if r['strict_control']])
        counts=np.zeros((20000,2),dtype=float)
        for lane in ['A','B']:
            source=np.array([row[1:3] for row in covered if row[0]==lane],dtype=float)
            if not len(source):raise ValueError('No qualified data on a physical lane')
            indices=rng.integers(0,len(source),(len(counts),len(source)));counts+=source[indices].sum(1)
        draws[arm]=counts[:,0]/counts[:,1]
    per_rate={k:groups['awr']['deaths_per_1000_ticks']/groups[k]['deaths_per_1000_ticks'] for k in ['uniform','lambda1']}
    bands={k:np.quantile(draws['awr']/draws[k],[.025,.975]).tolist() for k in per_rate}
    return dict(groups=groups,observed_rate_ratios=per_rate,stratified_bootstrap_95=bands)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);args=p.parse_args();print(json.dumps(result(args.study),indent=2))
