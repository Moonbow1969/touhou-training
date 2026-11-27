"""Audit native TH17 Bomb inventory, policy requests and life losses in a complete cohort."""
from pathlib import Path
import argparse,json
import numpy as np

def credit(folder, expected_split="train"):
    r=json.loads((folder/'receipt.json').read_bytes())
    if r['split']!=expected_split:raise ValueError('Unexpected credit split')
    z=np.genfromtxt(folder/'native.tsv',names=True,delimiter='\t')
    z=z[z['stage']>=1]
    d=np.genfromtxt(folder/'decisions.tsv',names=True,delimiter='\t')
    held=np.asarray(d['bomb'],dtype=bool)
    rising=held & ~np.r_[False,held[:-1]]
    events=[]
    for i in np.flatnonzero(np.diff(z['spares'])<0)+1:
        previous,after=z[i-1],z[i]
        tick=int(after['tick']);window=(d['stage']==after['stage'])&(d['tick']>=tick-120)&(d['tick']<tick)
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

def held_report(root):
    protocol=json.loads((root.parent/'camera-protocol.json').read_bytes())
    arms=(protocol['reference_arm'],protocol['candidate'])
    expected={(seed,arm) for arm in arms for seed in protocol['eval_seeds']['17']}
    rows=[]
    for path in sorted(root.glob('held-*/receipt.json')):
        receipt=json.loads(path.read_bytes())
        r=credit(path.parent,'held')
        rows.append(dict(**r,arm=receipt['arm'],lane=receipt['lane'],trajectory_sha256=receipt['trajectory_sha256']))
    if len(rows)!=len(expected) or {(r['seed'],r['arm']) for r in rows}!=expected:
        raise ValueError('Held cohorts must be complete before diagnostic analysis')
    for seed in protocol['eval_seeds']['17']:
        if len({r['lane'] for r in rows if r['seed']==seed})!=1:
            raise ValueError('Unmatched held instance')
    summary={}
    for rank in (2,3):
        summary[rank]={}
        for arm in arms:
            subset=[r for r in rows if r['rank']==rank and r['arm']==arm]
            deaths=[d for r in subset for d in r['deaths']]
            carrying=[d for d in deaths if d['bombs_on_last_playable_tick']>0]
            summary[rank][arm]=dict(credits=len(subset),actions=sum(r['actions'] for r in subset),native_life_losses=len(deaths),
                life_losses_holding_bombs=len(carrying),total_bombs_at_life_loss=sum(d['bombs_on_last_playable_tick'] for d in deaths),
                stock_holding_losses_with_no_request_in_previous_120_ticks=sum(d['requested_bomb_controls_in_prior_120_native_ticks']==0 for d in carrying),
                native_same_life_casts=sum(r['native_same_life_casts'] for r in subset),
                requested_bomb_controls=sum(r['bomb_controls'] for r in subset),rising_edges=sum(r['bomb_rising_edges'] for r in subset),
                edges_on_zero_inventory=sum(r['edges_on_zero_inventory'] for r in subset))
    return dict(scope='Descriptive post-hoc analysis of fully sealed held pairs. Life losses within a credit are dependent. The original observer reports the latest pre-death stock, and a requested X does not guarantee a successful cast. The 120-tick window does not determine whether a particular death was preventable.',by_rank=summary,credits=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=('train','held'),default='train');a=p.parse_args()
    out=(report if a.mode=='train' else held_report)(a.root);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['by_rank'],indent=2))

