"""Measure the reward actually consumed by PPO on the frozen training starts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.checkpoint_storage import digest


def midranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    src = args.root/'corrected/credits.json'
    report = json.loads((args.root/'policy/report.json').read_bytes())
    plan = json.loads((args.root/'fit-protocol.json').read_bytes())
    gamma = plan['ppo']['gamma']
    if digest(src) != report['manifest_sha256']:
        raise ValueError('Training manifest and checkpoint disagree')
    fitted = {(row['title'],row['seed']):row for row in report['credits']}
    cells = defaultdict(list)
    for row in json.loads(src.read_bytes()):
        fit = fitted.pop((row['title'], row['seed']))
        archive = src.parent/row['file']
        if digest(archive) != row['sha256']:
            raise ValueError('Training input bytes changed')
        with np.load(archive, allow_pickle=False) as data:
            reward = data['reward'][:fit['steps']]
        if len(reward) != fit['steps'] or abs(float(reward.sum()) - fit['reward']) > 1e-4:
            raise ValueError('Playable reward differs from fitted input')
        cells[(row['title'], row['difficulty'])].append(dict(seed=row['seed'], decisions=len(reward),
            total_reward=fit['reward'], discounted_return=float(np.dot(reward, gamma**np.arange(len(reward)))),
            positive_reward_sum=float(reward[reward>0].sum()),
            negative_reward_sum=float(reward[reward<0].sum()), positive_rows=int((reward>0).sum()),
            negative_rows=int((reward<0).sum()), zero_rows=int((reward==0).sum()), outcome=row['outcome']))
    if fitted:raise ValueError('Missing fitted games')
    results = {}
    for (game,rank),rows in sorted(cells.items()):
        decisions = midranks([r['decisions'] for r in rows])
        def rho(field):
            values=midranks([r[field] for r in rows])
            return float(np.corrcoef(decisions,values)[0,1]) if min(np.std(decisions),np.std(values))>0 else None
        results[f"TH{game}-"+{2:'Hard',3:'Lunatic'}[rank]] = dict(credits=rows, spearman_decisions_vs_return=rho('total_reward'),
            spearman_decisions_vs_discounted_return=rho('discounted_return'),
            positive_sum=sum(r['positive_reward_sum'] for r in rows), negative_sum=sum(r['negative_reward_sum'] for r in rows))
    args.out.write_text(json.dumps(dict(manifest_sha256=digest(src),policy_sha256=report['policy_sha256'],gamma=gamma,
        measure='Observed raw and discounted returns from the first decision on the original training games. Spearman uses midranks within game and difficulty. Playable policy decision count is a duration proxy and can differ from native Stage 1 ticks. Each cell has only eight games. Rewards of opposite sign in a single row are already netted. Neither return is the normalized per-action PPO advantage.',cells=results), indent=2)+'\n',encoding='utf8')
    for key,res in results.items():print(key, res['spearman_decisions_vs_return'], res['spearman_decisions_vs_discounted_return'], flush=True)


if __name__=='__main__':main()
