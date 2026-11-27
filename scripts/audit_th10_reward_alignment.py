"""Audit the fixed TH10 training reward on complete native credits without retraining a policy."""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.collect_th10_full_action import save


def ranks(values):
    _,ids,sizes=np.unique(values,return_inverse=True,return_counts=True)
    return (np.cumsum(sizes)-(sizes+1)/2)[ids]


def audit(folder):
    receipts=json.loads((folder/'ppo-receipts.json').read_bytes())
    assert all(r['role']=='train' for r in receipts)
    rows=[]
    for r in receipts:
        d=folder/f"{r['host']}-{r['seed']}";report=json.loads((d/'report.json').read_bytes())
        with np.load(d/'behavior.npz') as b:
            reward=b['reward'].astype(np.float64);life=b['reward_life'];stage=b['reward_stage'];survive=b['reward_survival'];cost=b['reward_bomb']
            assert np.allclose(reward,life+stage+survive+cost,atol=2e-7)
            discounts=np.power(.999,np.arange(len(reward)))
            rows.append(dict(seed=r['seed'],steps=len(reward),duration_s=report['duration_s'],
                casts=len(report['successful_bombs']),stage2=report['stage2_reached'],net_life=float(life.sum()),
                stage_reward=float(stage.sum()),survival=float(survive.sum()),spend_reward=float(cost.sum()),total=float(reward.sum()),
                discounted=float(discounts@reward),without_spend_penalty=float((reward-cost).sum()),discounted_without_spend=float(discounts@(reward-cost))))
    durations=np.array([r['duration_s'] for r in rows]);metrics={}
    for key in ('total','discounted','without_spend_penalty','discounted_without_spend'):
        x=np.array([r[key] for r in rows]);a,b=np.triu_indices(len(x),1)
        dif=durations[a]-durations[b];rew=x[a]-x[b]
        metrics[key]=dict(spearman_vs_survival=float(np.corrcoef(ranks(durations),ranks(x))[0,1]),
            longer_credit_has_lower_return=int((dif*rew<0).sum()),all_credit_pairs=len(a),mean=float(x.mean()),sd=float(x.std(ddof=1)))
    data=dict(credits=len(rows),decisions=sum(r['steps'] for r in rows),stage2_credits=sum(r['stage2'] for r in rows),
        distinct_net_life_rewards=sorted(set(r['net_life'] for r in rows)),actual_spirit_strikes=sum(r['casts'] for r in rows),
        survival_reward_per_second=.001,cost_per_confirmed_spirit_strike=.03,seconds_of_survival_reward_per_spend=30.,
        learner_discount_per_decision=.999,rank_diagnostics=metrics,episodes=rows,
        scope='Training credits only. Endpoint reward sum and start-state discounted Monte Carlo return evaluated on fixed original trajectories. Removing the cost is an offline rescore; the policy and game outcome were unchanged. PPO uses GAE with gamma .999 and lambda .985. These associations do not identify the causal survival effect of using a Bomb.')
    save(folder/'reward-alignment.json',data);print(json.dumps({k:v for k,v in data.items() if k!='episodes'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    audit(p.parse_args().folder)
