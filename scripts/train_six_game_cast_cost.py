"""Ablate confirmed same-life Bomb spending on TH12 and TH17 in the shared six-game PPO update."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records
from scripts.train_five_original_games import freeze


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    data=parser.parse_args().data.resolve()
    root=data/'six-game-bomb-cost';root.mkdir(exist_ok=True)
    original=data/'five-game-generation3'
    six=data/'six-game-online/joint-six'
    parent=data/'five-game-generation2/policy'
    manifest=six/'optimizer-input/credits.json';control=read(six/'policy/report.json')
    if control['manifest_sha256']!=digest(manifest) or control['native_behavior_sha256']!=digest(parent/'policy.ts'):
        raise ValueError('Changed six-game reference source')
    fit=read(six/'fit-protocol.json');previous=read(original/'camera-protocol.json')
    held={str(t):list(range(1740000+100*t+51,1740000+100*t+67)) for t in (12,17)}
    ranks={t:dict(items) for t,items in fit['rank_by_seed'].items()}
    for t,seeds in held.items():ranks[t].update({str(s):2 if s%2 else 3 for s in seeds})
    plan=dict(fit,name='Native same-life Bomb cost in six-game PPO',reference_arm='six-ppo',candidate='six-no-cast-cost',
              eval_seeds=held,rank_by_seed=ranks,games=previous['games'],controls=previous['controls'],ppo=previous['ppo'],
              cast_cost_titles=[12,17],reference_policy=digest(six/'policy/policy.ts'),source_manifest=digest(manifest),
              treatment='On TH12 and TH17 remove the recorded negative reward for an actual same-life Bomb expenditure. Use the same raw camera observations, behavior actions, native log probabilities, critic, optimizer and initial weights as the control. Retain every other event and the original reward on all other games. Recompute actor advantages and critic targets. Actor has no access to reward events or game identity.',
              evaluation='Eight paired ordinary credits per title and difficulty. Same physical host and build within the pair, uncontrolled game RNG. Primary outcome is Stage 2; secondary outcomes are last actual spare lives and Bombs, effective Bombs spent and native progress. Retain technical attempts and unresolved outcomes with evidence; control coverage is reported separately.')
    registration=root/'camera-protocol.json';freeze(registration,plan)
    sources={digest(path):path for t,path in records(original,True,True) if t in (12,17)}
    costs={};audits=[]
    for row in read(manifest):
        title=row['title']
        if title not in plan['cast_cost_titles']:continue
        source=sources[row['source_log_sha256']];receipt=read(source)
        if receipt.get('split',receipt['mode'])!='train' or receipt['seed']!=row['seed'] or row['role']!='train':
            raise ValueError('Wrong native training source')
        trace=source.parent/receipt.get('behavior',receipt.get('trajectory',receipt.get('file','missing')))
        if digest(trace)!=row['source_actions_sha256'] or digest(manifest.parent/row['file'])!=row['sha256']:
            raise ValueError('Changed original inputs or native reward components')
        col={12:2,17:1}[title]
        labels=receipt.get('reward_parts_order',receipt.get('reward_components'))
        if 'same-life' not in labels[col].lower() or 'bomb' not in labels[col].lower():
            raise ValueError('Native component order changed')
        with np.load(trace,allow_pickle=False) as native,np.load(manifest.parent/row['file'],allow_pickle=False) as packed:
            parts=native['reward_parts'];cost=parts[:,col].astype(np.float32)
            np.testing.assert_allclose(parts.sum(1),packed['reward'].reshape(-1),atol=1e-6,rtol=0)
            np.testing.assert_allclose(cost,-.03*np.rint(-cost/.03),atol=1e-6,rtol=0)
            if (cost>0).any() or len(cost)!=len(packed['action']):raise ValueError('Invalid cast cost')
        costs[f"th{title}-{row['seed']}"]=cost
        audits.append(dict(title=title,seed=row['seed'],rank=row['difficulty'],source_receipt=digest(source),
                           source_trajectory=row['source_actions_sha256'],physical_spends=int(round(-float(cost.sum())/.03))))
    if len(costs)!=64 or len(audits)!=64:raise ValueError('Expected 32 complete games in each treated title')
    archive=root/'cast-costs.npz'
    if not archive.exists():np.savez_compressed(archive,**costs)
    with np.load(archive,allow_pickle=False) as z:
        if set(z.files)!=set(costs) or any(not np.array_equal(z[k],v) for k,v in costs.items()):raise ValueError('Frozen costs differ')
    freeze(root/'cast-event-audit.json',audits)
    protocol=root/'fit-protocol.json'
    freeze(protocol,dict(plan,registered=digest(registration),cast_cost_sha256=digest(archive)))
    if not (root/'policy/report.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',str(manifest),
            '--protocol',str(protocol),'--behavior',str(parent/'policy.ts'),'--parent-actor-critic',str(parent/'camera-ppo.pt'),
            '--remove-cast-cost',str(archive),'--out',str(root/'policy')],check=True)
    result=read(root/'policy/report.json')
    if result['steps']!=control['steps'] or result['manifest_sha256']!=control['manifest_sha256'] or result['removed_cast_cost_sha256']!=digest(archive):
        raise ValueError('Cast-cost experiment differs in source population')
    print('Fitted six-title cast-cost ablation',result['steps'],result['policy_sha256'],flush=True)


if __name__=='__main__':
    main()
