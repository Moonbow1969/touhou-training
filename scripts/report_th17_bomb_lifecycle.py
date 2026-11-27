"""Audit actual native Bomb use and inventory just before death on training-only TH17 credits."""
from pathlib import Path
import argparse,json
import numpy as np

def credit(folder):
    r=json.loads((folder/'receipt.json').read_bytes())
    if r['split']!='train':raise ValueError('Use TRAIN credits only')
    z=np.genfromtxt(folder/'native.tsv',names=True,delimiter='\t')
    z=z[z['stage']>=1]
    d=np.genfromtxt(folder/'decisions.tsv',names=True,delimiter='\t')
    held=np.asarray(d['bomb'],dtype=bool)
    rising=held & ~np.r_[False,held[:-1]]
    events=[]
    for i in np.flatnonzero(np.diff(z['spares'])<0)+1:
        previous,after=z[i-1],z[i]
        tick=int(after['total_tick']);window=(d['total_tick']>=tick-120)&(d['total_tick']<tick)
        events.append(dict(death_tick=tick,ended_credit=after['spares']<0,
                           bombs_on_last_playable_tick=int(previous['bombs']),native_post_death_bombs=int(after['bombs']),
                           requested_bomb_controls_in_prior_120_native_ticks=int((window&held).sum()),
                           rising_bomb_edges_in_prior_120_native_ticks=int((window&rising).sum())))
    events=[{**e,'ended_credit':bool(e['ended_credit'])} for e in events]
    return dict(seed=r['seed'],rank=r['difficulty'],policy=r['policy_sha256'],actions=r['actions'],native_same_life_casts=sum(e['count'] for e in r['reward_events'] if e['kind']=='spent_bombs'),
                bomb_controls=int(held.sum()),bomb_rising_edges=int(rising.sum()),edges_on_zero_inventory=int((rising & (d['bombs_before']==0)).sum()),deaths=events)

def report(root):
    rows=[credit(p.parent) for p in sorted(root.glob('train-*/receipt.json'))]
    groups={}
    for rank in (2,3):
        subset=[r for r in rows if r['rank']==rank];deaths=[e for r in subset for e in r['deaths']]
        groups[rank]=dict(credits=len(subset),deaths=len(deaths),deaths_with_stock=sum(e['bombs_on_last_playable_tick']>0 for e in deaths),
                          pre_death_stock_total=sum(e['bombs_on_last_playable_tick'] for e in deaths),
                          deaths_with_stock_and_no_bomb_request_in_prior_120_native_ticks=sum(e['bombs_on_last_playable_tick']>0 and e['requested_bomb_controls_in_prior_120_native_ticks']==0 for e in deaths),
                          native_same_life_bomb_casts=sum(r['native_same_life_casts'] for r in subset),bomb_rising_edges=sum(r['bomb_rising_edges'] for r in subset),
                          requested_edges_at_zero_inventory=sum(r['edges_on_zero_inventory'] for r in subset))
    return dict(scope='Original TH17 Reimu/Wolf full credits, TRAIN only. Every loss is a native decrease in spare lives. A bomb request is a policy-sampled X input. A cast is a measured reduction of Bomb inventory during the same life. The last playable inventory precedes the native death refill. The 120-native-tick window is descriptive, not a deathbomb counterfactual.',by_rank=groups,credits=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    out=report(a.root);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['by_rank'],indent=2))
