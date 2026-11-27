"""Admit complete original TH08 credits and fit pre-registered small and six-game online policies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from scripts.checkpoint_storage import digest
from scripts.qualify_th08_trial_credit import qualify
from scripts.train_five_original_games import freeze


def read(path):
    return json.loads(path.read_bytes())


def wait(path):
    while not path.is_file():
        time.sleep(8)


def hardlink(source,target,sha):
    if digest(source)!=sha:
        raise ValueError(f'Changed optimizer source {source.name}')
    if not target.exists():
        os.link(source,target)
    elif digest(target)!=sha:
        raise ValueError(f'Existing optimizer view changed {target.name}')


def optimize(root, sources, parent, protocol, stem, registration=None):
    sub=root/stem;sub.mkdir(exist_ok=True)
    batch=sub/'optimizer-input';batch.mkdir(exist_ok=True)
    inputs=[];by_title={};ranks={}
    for row,source in sources:
        key=str(row['title']);seed=row['seed']
        if row['policy_sha256']!=protocol['parent_actor'] or row['role']!='train':
            raise ValueError('Mixed or ineligible behavior policy')
        hardlink(source,batch/row['file'],row['sha256'])
        inputs.append(row);by_title.setdefault(key,[]).append(seed);ranks.setdefault(key,{})[str(seed)]=row['difficulty']
    freeze(batch/'credits.json',inputs)
    freeze(sub/'fit-protocol.json',dict(reference='five-ppo2',behavior_sha256=protocol['parent_actor'],
           training_registration=digest(registration or root/'protocol.json'),train_seeds=by_title,rank_by_seed=ranks,
           balanced_title_rank=True,learning_rates=[2e-5,1e-5,5e-6,2.5e-6]))
    if not (sub/'policy/report.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',str(batch/'credits.json'),
                        '--protocol',str(sub/'fit-protocol.json'),'--behavior',str(parent/'policy.ts'),
                        '--parent-actor-critic',str(parent/'camera-ppo.pt'),'--out',str(sub/'policy')],check=True)
    report=read(sub/'policy/report.json')
    if (report['policy_sha256']!=digest(sub/'policy/policy.ts') or
            report['native_behavior_sha256']!=protocol['parent_actor'] or len(report['credits'])!=len(sources)):
        raise ValueError('Unexpected final learner output')
    print('Qualified optimizer',stem,len(inputs),'credits',report['steps'],'decisions',report['policy_sha256'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--selected',action='store_true')
    a=p.parse_args();data=a.data.resolve()
    root=data/'six-game-online';registration=root/'protocol.json';wait(registration)
    plan=read(registration);parent=data/'five-game-generation2/policy'
    if not (plan['parent_actor']==digest(parent/'policy.ts') and plan['parent_actor_critic']==digest(parent/'camera-ppo.pt')):
        raise ValueError('Original behavior lineage differs')
    torch.set_num_threads(2)
    model=torch.jit.load(str(parent/'policy.ts'),map_location='cpu').eval()
    qualified={};train=root/'th08-training';seeds=plan['th08_train_seeds']
    with ThreadPoolExecutor(1) as queue:
        small=None
        while len(qualified)<len(seeds):
            for seed in seeds:
                if a.selected:
                    selected=root/'selected-training'/f'{seed}.json'
                    if seed in qualified or not selected.exists():
                        continue
                    entry=read(selected);row=entry['record']
                    if (entry['slot']!=seed or row['seed']!=entry['selected_seed'] or
                            entry['registration_sha256']!=digest(root/'technical-menu-amendment.json') or
                            row['policy_sha256']!=plan['parent_actor'] or row['difficulty']!=(2 if seed%2 else 3)):
                        raise ValueError('Invalid technical source assignment')
                    qualified[seed]=(row,train/row['file'])
                    print('Verified registered slot',seed,'physical seed',row['seed'],flush=True)
                    continue
                folder=root/'native08'/f'train-five-ppo2-{seed}'
                if seed in qualified or not (folder/'receipt.json').exists():
                    continue
                row,audit=qualify(folder,model,plan['parent_actor'],seed,plan['rank_by_seed'][str(seed)],'train',train)
                evidence=train/f'{seed}-qualification.json'
                if evidence.exists():
                    original=read(evidence)
                    # Batched CPU convolution reductions may differ by a few FP32 units
                    # between processes. Both runs must independently satisfy the fixed tolerances.
                    variability={'max_actor_logp_error','max_critic_error'}
                    stable=lambda a:{k:v for k,v in a.items() if k not in variability}
                    if (original['record']!=row or stable(original['audit'])!=stable(audit)
                            or original['audit']['max_actor_logp_error']>8e-4
                            or original['audit']['max_critic_error']>3e-4):
                        raise ValueError('Recomputed native qualification changed')
                else:
                    freeze(evidence,dict(record=row,audit=audit))
                qualified[seed]=(row,train/row['file'])
                print('Verified original TH08',seed,audit['decisions'],audit['confirmed_deaths'],flush=True)
            first=plan['first_th08_fit']['training']
            if all(s in qualified for s in first) and small is None:
                small=queue.submit(optimize,root,[qualified[s] for s in first],parent,plan,'first-eight')
            if small is not None and small.done():
                small.result()
            if len(qualified)<len(seeds):
                time.sleep(8)
        small.result()
    raw=data/'five-game-generation3/corrected/credits.json'
    wait(data/'five-game-generation3/policy/report.json')
    if digest(data/'five-game-generation3/camera-protocol.json')!=plan['five_title_generation3_protocol']:
        raise ValueError('Five-game collection lineage changed')
    five=read(raw)
    if len(five)!=160 or len(set((r['title'],r['seed']) for r in five))!=160:
        raise ValueError('Missing or duplicate five-title source')
    optimize(root,[(r,raw.parent/r['file']) for r in five]+[qualified[s] for s in seeds],parent,plan,'joint-six')


if __name__=='__main__':
    main()
