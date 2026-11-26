"""Score paired whole original-game credits by title and physical host."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
import numpy as np

def wilson(k,n):
    z=1.959963984540054
    center=(k+z*z/2)/(n+z*z)
    half=z*np.sqrt(k*(n-k)/n+z*z/4)/(n+z*z)
    return [max(0.,float(center-half)),min(1.,float(center+half))]

def one(rows):
    reach=[r for r in rows if r['stage2_reached']]
    def average(field,group=rows):
        values=[r[field] for r in group if r.get(field) is not None]
        return mean(values) if values else None
    return dict(n=len(rows),stage2=len(reach),stage2_rate=len(reach)/len(rows),stage2_wilson95=wilson(len(reach),len(rows)),
                total_confirmed_bombs=sum(r['effective_bombs'] for r in rows),
                terminal_game_overs=sum(r['terminal']=='game_over' for r in rows),mean_decisions=average('decisions'),
                mean_seconds=average('wall_seconds'),mean_terminal_spares=average('terminal_spares'),
                mean_terminal_bombs=average('terminal_bombs'),mean_terminal_power=average('terminal_power'),
                mean_stage2_spares=average('stage2_spares',reach),mean_stage2_bombs=average('stage2_bombs',reach),
                mean_stage2_power=average('stage2_power',reach))

def summarize(rows,protocol,candidate):
    if candidate not in ("ppo1","ppo2"):raise ValueError("Unknown comparison.")
    results={}
    for t,seeds in protocol['eval_seeds'].items():
        t=int(t);host={12:'steamdeck' if candidate=='ppo1' else 'entropy'}.get(t)
        group=[r for r in rows if r['title']==t and r['arm'] in ('sft',candidate) and (not host or r['host']==host)]
        by_arm={a:{} for a in ('sft',candidate)}
        for r in group:
            if r['seed'] in by_arm[r['arm']]:raise ValueError("Duplicate eligible credit.")
            by_arm[r['arm']][r['seed']]=r
        if any(set(records)!=set(seeds) for records in by_arm.values()):
            raise ValueError(f"Unfinished TH{t} {candidate} evaluation: "+str({k:len(v) for k,v in by_arm.items()}))
        pairs=[]
        for seed in seeds:
            left=by_arm['sft'][seed];right=by_arm[candidate][seed]
            if left['host']!=right['host'] or left['game_sha256']!=right['game_sha256']:
                raise ValueError("Host/build mismatch within a pair")
            pairs.append(dict(seed=seed,host=left['host'],decision_difference=right['decisions']-left['decisions'],
                              seconds_difference=right['wall_seconds']-left['wall_seconds'],
                              sft_stage2=left['stage2_reached'],candidate_stage2=right['stage2_reached']))
        rng=np.random.default_rng(216720+t)
        differences=np.asarray([p['seconds_difference'] for p in pairs]);ci=np.quantile(differences[rng.integers(len(pairs),size=(20000,len(pairs)))].mean(1),[.025,.975]).tolist()
        results[t]=dict(sft=one(list(by_arm['sft'].values())),candidate=one(list(by_arm[candidate].values())),pairs=pairs,
                        stage2_discordances=dict(sft_only=sum(p['sft_stage2'] and not p['candidate_stage2'] for p in pairs),candidate_only=sum(p['candidate_stage2'] and not p['sft_stage2'] for p in pairs)),
                        mean_paired_wall_difference=float(differences.mean()),mean_paired_wall_difference_bootstrap95=ci)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['credits','protocol','out']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--candidate',choices=['ppo1','ppo2'],required=True)
    a=p.parse_args()
    result=summarize(json.loads(a.credits.read_bytes()),json.loads(a.protocol.read_bytes()),a.candidate)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
