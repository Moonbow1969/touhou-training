"""Estimate agreement of TH08 and five-game PPO gradients using only original training data."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.checkpoint_storage import digest
from scripts.measure_gradient_sample_size import moments
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import CameraActorCritic, CameraPolicy, credit


def normalized_group(basis, stats, ids):
    n,total,ss=np.sum(np.asarray(stats)[ids],axis=0)
    mean=total/n
    sd=np.sqrt(max(0.,ss/n-mean*mean))+1e-7
    selected=[basis[j] for j in ids]
    unweighted=torch.stack([x[0] for x in selected]).sum(0)
    weighted=torch.stack([x[1] for x in selected]).sum(0)
    return (weighted-mean*unweighted)/(n*sd)


def cosine(x,y):
    return float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    parent=d/'five-game-generation2/policy'
    source=d/'six-game-online/joint-six/optimizer-input/credits.json'
    root=d/'six-game-retention/initial-gradients';root.mkdir(exist_ok=True)
    plan=dict(train_manifest=digest(source),actor=digest(parent/'policy.ts'),critic=digest(parent/'camera-ppo.pt'),
        method='Use 16 independent training games per title and rank. Calculate the initial normalized joint-action PPO gradient over all playable decisions in each cell. Estimate cross-title agreement by measuring cosine between the TH08 gradient and the mean gradient of the other ten cells. Repeat on 32 deterministic shuffled twofold splits of eight whole games per side per cell, always comparing opposite halves. Report the median and 10th and 90th percentiles across correlated partitions as sensitivity, with no claim of independent test games. Six-game and five-game training share original behavior and per-cell budgets. No held games, entropy or value gradient in this diagnostic.')
    freeze(root/'protocol.json',plan)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    state=torch.load(parent/'camera-ppo.pt',map_location='cpu',weights_only=True)
    model=CameraActorCritic(CameraPolicy())
    if any('policy.spatial_context.' in k for k in state):model.camera.add_spatial_context()
    model.load_state_dict(state);model.eval().to(device)
    groups=defaultdict(list)
    for r in json.loads(source.read_bytes()):groups[(r['title'],r['difficulty'])].append(r)
    if set(groups)!={(t,r) for t in (8,10,11,12,13,17) for r in (2,3)} or any(len(x)!=16 for x in groups.values()):raise ValueError('Expected all twelve original training strata')
    full={};splits={};source_info={}
    for key,rows in sorted(groups.items()):
        title,rank=key;basis=[];stats=[]
        for row in sorted(rows,key=lambda r:r['seed']):
            arrays,_=credit(row,source,model,plan['actor'],device)
            vec,stat=moments(model,arrays,device)
            basis.append(vec);stats.append(stat)
        full[key]=normalized_group(basis,stats,list(range(16)))
        # Fix all partitions before evaluating the two controller arms in original games.
        splits[key]=[]
        for i in range(32):
            ids=np.random.default_rng(1690000+title*100+rank*37+i*2309).permutation(16)
            splits[key].append(tuple(normalized_group(basis,stats,ii) for ii in (ids[:8],ids[8:])))
        source_info[f'TH{title:02}-{rank}']=dict(n=int(sum(s[0] for s in stats)),seeds=sorted(r['seed'] for r in rows),gradient_norm=float(full[key].norm()))
        print('Verified training gradients',key,source_info[f'TH{title:02}-{rank}']['n'],flush=True)
    existing=[k for k in sorted(full) if k[0]!=8]
    def average(vs):return torch.stack(vs).mean(0)
    outcome={}
    for rank in (2,3):
        k=(8,rank);half=[]
        for i in range(32):
            for side in range(2):
                other=average([splits[j][i][1-side] for j in existing])
                half.append(cosine(splits[k][i][side],other))
        outcome[str(rank)]=dict(full_cosine=cosine(full[k],average([full[j] for j in existing])),
            full_cosine_by_existing_cell={f'TH{t:02}-{r}':cosine(full[k],full[(t,r)]) for t,r in existing},
            split_median=float(np.median(half)),split_10_90=np.quantile(half,[.1,.9]).tolist(),
            positive_split_fraction=float(np.mean(np.asarray(half)>0)),split_estimates=half)
    freeze(root/'report.json',dict(registration_sha256=digest(root/'protocol.json'),sources=source_info,th08_by_rank=outcome))


if __name__=='__main__':
    main()
