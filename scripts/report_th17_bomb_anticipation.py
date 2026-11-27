"""Measure learned Bomb probability before losses using complete original training credits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

def describe(row, masks, probabilities):
    out={}
    for name,mask in masks.items():
        out[name]=dict(eligible_decisions=int(mask.sum()),mean_x_probability=float(probabilities[mask].mean()) if mask.any() else None,
                       expected_x_requests=float(probabilities[mask].sum()),sampled_x_requests=int(row['bomb'][mask].sum()))
    return out

def episode(source):
    receipt=json.loads((source/'receipt.json').read_bytes())
    if receipt['split']!='train' or receipt['terminal']!='game_over':raise ValueError(source)
    for f,tag in [('decisions.tsv','decisions_sha256'),('native.tsv','native_trace_sha256')]:
        if hashlib.sha256((source/f).read_bytes()).hexdigest()!=receipt[tag]:raise ValueError(source/f)
    data=np.genfromtxt(source/'decisions.tsv',names=True,delimiter='\t',dtype=None,encoding='utf8')
    native=np.genfromtxt(source/'native.tsv',names=True,delimiter='\t',dtype=None,encoding='utf8')
    native=native[native['stage']>=1]
    probs=np.where(data['bomb']==1,np.exp(data['logp_bomb']),-np.expm1(data['logp_bomb']))
    if not (np.all(np.isfinite(probs)) and np.all((probs>=0)&(probs<=1))):
        raise ValueError('Invalid binary head log probability')
    deaths=[]
    for i in np.flatnonzero(np.diff(native['spares'])<0)+1:
        before,death=native[i-1],native[i]
        if before['bombs']<1:continue
        j=i-1
        while j>0 and native[j-1]['stage']==death['stage'] and native[j-1]['spares']==before['spares']:
            j-=1
        # Eligible decision starts must be in this same life, at the same
        # observed Bomb inventory as just before loss.
        current=(data['tick']>=native[j]['tick'])&(data['stage']==death['stage'])&(data['spares_before']==before['spares'])&(data['bombs_before']==before['bombs'])
        delta=int(death['tick'])-data['tick']
        m={'near':current&(delta>0)&(delta<=120),'earlier':current&(delta>240)&(delta<=360)}
        if not all(v.any() for v in m.values()):continue
        deaths.append(dict(stage=int(death['stage']),loss_tick=int(death['tick']),bombs=int(before['bombs']),**describe(data,m,probs)))
    return dict(seed=receipt['seed'],rank=receipt['difficulty'],arm=receipt['arm'],decisions=receipt['actions'],policy_sha256=receipt['policy_sha256'],
                trajectory_sha256=receipt['trajectory_sha256'],eligible_life_losses=deaths)

def summarize(rows):
    pairs=[]
    for row in rows:
        losses=row['eligible_life_losses']
        if losses:pairs.append([np.mean([e[n]['mean_x_probability'] for e in losses]) for n in ('earlier','near')])
    if not pairs:return dict(credits=len(rows),eligible_credits=0)
    x=np.array(pairs);delta=x[:,1]-x[:,0]
    draws=np.random.default_rng(261126).choice(delta,(20000,len(delta)),replace=True).mean(axis=1)
    events=[e for row in rows for e in row['eligible_life_losses']]
    return dict(credits=len(rows),eligible_credits=len(pairs),eligible_life_losses=len(events),
                mean_earlier_x_probability=float(x[:,0].mean()),mean_near_x_probability=float(x[:,1].mean()),
                near_minus_earlier=float(delta.mean()),credit_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                near_actions=sum(e['near']['eligible_decisions'] for e in events),earlier_actions=sum(e['earlier']['eligible_decisions'] for e in events))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    plan=json.loads((args.native.parent/'camera-protocol.json').read_bytes())
    expected=set(plan['train_seeds']['17'])
    rows=[episode(f.parent) for f in sorted(args.native.glob('train-*/receipt.json'))]
    if len(rows)!=len(expected) or {r['seed'] for r in rows}!=expected or {r['arm'] for r in rows}!={plan['reference_arm']}:
        raise ValueError('Complete preregistered TRAIN cohort is required')
    groups={rank:summarize([r for r in rows if r['rank']==rank]) for rank in (2,3)}
    out=dict(scope='TRAIN only, exploratory. Compare mean probability of the Bomb head in the last 120 native stage ticks to ticks 241–360 before the same observed death, requiring the same stage, life and positive Bomb inventory in both windows. Both windows must have observations. Credit-level means weight eligible losses equally. Bootstrap resamples credits within native difficulty. An observed loss does not identify a preventable death or a correct Bomb label.',by_rank=groups,credits=rows)
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
