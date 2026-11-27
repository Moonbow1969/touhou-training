"""Price confirmed loss of payable TH10 Spirit Strikes at death on frozen PPO experience."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import numpy as np
from scripts.checkpoint_storage import digest


def death_costs(path, actions_ms, life_rewards):
    with gzip.open(path, 'rt') as f:
        rows=list(csv.DictReader(f))
    ts=np.array([int(r['utc_ms']) for r in rows])
    spare=np.array([int(r['lives']) for r in rows])
    power=np.array([int(r['power_units']) for r in rows])
    bomb=np.array([(int(r['native_input'])&2)!=0 for r in rows])
    if np.any(np.diff(ts)<0) or not np.all((power>=0)&(power<=100)):
        raise ValueError('Invalid TH10 native timeline')
    presses=ts[np.flatnonzero(bomb & ~np.r_[False,bomb[:-1]])]
    cost=np.zeros(len(life_rewards),dtype=np.float32)
    events=[]
    for i in np.flatnonzero(np.diff(spare)<0)+1:
        t=ts[i]
        step=int(np.clip(actions_ms.searchsorted(t,side='right')-1,0,len(cost)-1))
        if life_rewards[step]>=0:raise ValueError('Native life drop did not align to packed reward')
        before=int(power[i-1]);post=power[(ts>=t)&(ts<=t+250)&(spare==spare[i])]
        terminal=spare[i]<0
        retained=0 if terminal else int(post.min())
        recent_x=bool(np.any((presses>=t-250)&(presses<=t+250)))
        consistent=bool(ts[i]-ts[i-1]<=80 and (terminal or ts[-1]>=t+200))
        lost=max(0,before//20-retained//20)
        accounted=lost if consistent and not recent_x else 0
        cost[step]-=.03*accounted
        events.append(dict(step=step,utc_ms=int(t),terminal=bool(terminal),spares_before=int(spare[i-1]),
            pre_power_units=before,lowest_post_power_units=None if terminal else retained,whole_casts_lost=lost,
            nearby_native_x_edge=recent_x,time_coverage_ok=consistent,charged_casts=accounted))
    return cost,events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','out','control-model'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    src=a.root/'corrected';manifest=src/'credits.json';control=json.loads(a.control_model.read_bytes())
    if control['manifest_sha256']!=digest(manifest):raise ValueError('Frozen baseline used another manifest')
    rows=json.loads(manifest.read_bytes())
    a.out.mkdir(parents=True,exist_ok=True)
    evidence=[]
    receipts={}
    for f in (a.root/'native10').rglob('report.json'):
        r=json.loads(f.read_bytes())
        receipts[r['seed'],r['policy_sha256']]=f
    for row in rows:
        source=src/row['file'];dest=a.out/row['file']
        if digest(source)!=row['sha256']:raise ValueError('Changed source experience')
        if row['title']!=10:
            if not dest.exists():
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
            if digest(dest)!=row['sha256']:raise ValueError('Other-game trajectory changed')
            continue
        f=receipts[row['seed'],row['policy_sha256']]
        if digest(f)!=row['source_log_sha256']:raise ValueError('Native training receipt changed')
        source_actions=f.parent/'behavior.npz'
        if digest(source_actions)!=row['source_actions_sha256']:raise ValueError('Native sampled actions changed')
        with np.load(source,allow_pickle=False) as original:arrays={k:original[k] for k in original.files}
        with np.load(source_actions,allow_pickle=False) as native:
            np.testing.assert_array_equal(native['action'],arrays['action'])
            np.testing.assert_array_equal(native['reward'],arrays['reward'])
            cost,events=death_costs(f.parent/'native.csv.gz',native['action_utc_ms'],native['reward_life'])
        arrays['reward']=(arrays['reward']+cost).astype(np.float32)
        arrays['reward_power_lost_at_death']=cost
        np.savez_compressed(dest,**arrays)
        row['sha256']=digest(dest)
        evidence.append(dict(seed=row['seed'],native_sha256=digest(f.parent/'native.csv.gz'),events=events,
                             charged_whole_casts=sum(e['charged_casts'] for e in events)))
    (a.out/'credits.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'events.json').write_text(json.dumps(dict(control_manifest=digest(manifest),credits=evidence,
        cost_per_lost_whole_cast=.03,scope='Training-only TH10 Easy. Native Power is measured immediately before the confirmed life decrement. On nonterminal loss count only the decrease in payable whole casts observed within 250 ms. Terminal remaining usable stock expires. Ignore ambiguous nearby X rising edges and incomplete windows. Dropped items can be recovered on a later respawn.'),indent=2)+'\n')


if __name__=='__main__':
    main()
