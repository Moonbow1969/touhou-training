"""Summarize seed- and host-matched pixel policies on a game excluded from their updates."""
import argparse
import json
from pathlib import Path
from statistics import mean
import numpy as np


def summarize(report):
    info=json.loads(report.read_bytes())
    if info['status']!='complete':
        raise ValueError('A closed frozen cohort is required.')
    pairs={}
    parents=set()
    candidates=set()
    for row in info['credits']:
        key=(row['host'],row['seed'])
        arm=row['arm']
        if arm not in ('parent','candidate') or arm in pairs.setdefault(key,{}):
            raise ValueError('Unexpected or duplicated frozen arm.')
        pairs[key][arm]=row
        (parents if arm=='parent' else candidates).add(row['model_sha256'])
    if len(parents)!=1 or len(candidates)!=1 or parents==candidates:
        raise ValueError('Expected two distinct frozen policies.')
    results=[]
    for (host,seed),pair in sorted(pairs.items()):
        a,b=pair['parent'],pair['candidate']
        if a['lane']==b['lane'] or a['game_sha256']!=b['game_sha256']:
            raise ValueError('Incorrect game build or lane allocation.')
        def actions(r):
            source=report.parent/f"{host}-{r['arm']}-{seed}-{r['lane']}"/"behavior.npz"
            with np.load(source,allow_pickle=False) as z:
                return z["action"]
        left,right=actions(a),actions(b)
        n=min(len(left),len(right))
        results.append(dict(host=host,seed=seed,parent_seconds=a['first_loss_restricted_s'],
                            candidate_seconds=b['first_loss_restricted_s'],
                            difference_seconds=b['first_loss_restricted_s']-a['first_loss_restricted_s'],
                            chronological_action_agreement=int((left[:n]==right[:n]).sum()),common_controls=n,
                            parent_hit=a['first_loss_within_limit'],candidate_hit=b['first_loss_within_limit']))
    return dict(measure='Native first spare-life loss restricted to 40 wall seconds',
                parent_raw_sha256=parents.pop(),candidate_raw_sha256=candidates.pop(),pairs=results,
                parent_mean_s=mean(r['parent_seconds'] for r in results),
                candidate_mean_s=mean(r['candidate_seconds'] for r in results),
                mean_difference_s=mean(r['difference_seconds'] for r in results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(summarize(a.report),indent=2)+'\n',encoding='utf-8')
