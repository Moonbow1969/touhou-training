"""Compare policy-gradient directions across disjoint full credits at the frozen behavior model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import CameraActorCritic, CameraPolicy, chosen, credit


def moments(model, arrays, device):
    images, actions, old_logp, advantages = arrays[:4]
    basis = []
    for advantage_weighted in (False, True):
        model.zero_grad(set_to_none=True)
        for start in range(0, len(images), 48):
            part = slice(start, start + 48)
            logits = model(images[part].to(device).float()/255)[:4]
            logp = chosen(logits, actions[part].to(device)).sum(1)
            ratio = (logp - old_logp[part].to(device)).exp()
            weights = advantages[part].to(device) if advantage_weighted else 1.
            (-(ratio * weights).sum()).backward()
        basis.append(torch.cat([p.grad.detach().reshape(-1).cpu()
                                for p in model.camera.parameters() if p.grad is not None]))
    advantages = advantages.double()
    return basis, [len(images), advantages.sum().item(), advantages.square().sum().item()]


def compare(basis, stats, seed):
    vectors = torch.stack(basis).double()
    gram = (vectors @ vectors.T).numpy()
    stats = np.array(stats, dtype=float)
    n = len(stats)
    if n != 16:
        raise ValueError('Each registered title and rank requires sixteen independent credits')

    def coefficients(group):
        total, sum_adv, sum_square = stats[group].sum(0)
        mean = sum_adv / total
        sd = np.sqrt(max(0., sum_square/total - mean**2)) + 1e-7
        w = np.zeros(2*n, dtype=float)
        w[group*2] = -mean/(total*sd)
        w[group*2+1] = 1/(total*sd)
        return w

    groups = [np.random.default_rng(seed + j).permutation(n) for j in range(512)]
    result = {}
    for k in (2,4,8):
        left = np.stack([coefficients(p[:k]) for p in groups])
        right = np.stack([coefficients(p[8:8+k]) for p in groups])
        dot = lambda a,b: ((a @ gram)*b).sum(1)
        norm = np.sqrt(np.maximum(0, dot(left, left)*dot(right, right)))
        if np.any(norm <= 0):
            raise ValueError('Degenerate original policy gradient')
        cosine = np.clip(dot(left,right)/norm,-1.,1.)
        result[str(k)] = dict(partitions=len(cosine), mean=float(cosine.mean()), median=float(np.median(cosine)),
                              positive_fraction=float(np.mean(cosine > 0)),
                              partition_percentiles_10_90=np.quantile(cosine,[.1,.9]).tolist())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','parent','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.protocol.read_bytes())
    parent=digest(a.parent/'policy.ts')
    state_file=a.parent/'camera-ppo.pt'
    if parent!=plan['behavior_sha256'] or digest(state_file)!=plan['parent_actor_critic_sha256']:
        raise ValueError('Changed frozen five-title behavior or critic')
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    state=torch.load(state_file,weights_only=True,map_location='cpu')
    camera=CameraPolicy()
    if any('policy.spatial_context.' in k for k in state):
        camera.add_spatial_context()
    model=CameraActorCritic(camera)
    model.load_state_dict(state)
    model.eval().to(device)
    rows=json.loads(a.manifest.read_bytes())
    groups={}
    for row in rows:
        groups.setdefault((row['title'],row['difficulty']),[]).append(row)
    if set(groups)!={(t,r) for t in (10,11,12,13,17) for r in (2,3)}:
        raise ValueError('Incomplete five-title original population')
    cells={}
    for (title,rank), population in sorted(groups.items()):
        vectors=[]
        stats=[]
        population.sort(key=lambda row:row['seed'])
        for row in population:
            arrays,_=credit(row,a.manifest,model,parent,device)
            weights,m=moments(model,arrays,device)
            vectors.extend(weights)
            stats.append(m)
        cell=compare(vectors,stats,261130+title*7+rank)
        cells[f'TH{title}-{rank}']=dict(seeds=[v['seed'] for v in population], decisions=int(sum(r[0] for r in stats)), comparisons=cell)
        print('Measured disjoint original policy gradients',title,rank,{k:round(v['median'],3) for k,v in cell.items()},flush=True)
    freeze(a.out,dict(protocol_sha256=digest(a.protocol), training_manifest_sha256=digest(a.manifest),
        behavior_sha256=parent, actor_critic_sha256=digest(state_file), cells=cells,
        method='At the original behavior parameters, retain per-credit sums of negative joint log-policy gradients '
               'and their products with original GAE. Combine those sufficient statistics using each group’s own '
               'mean and population standard deviation of advantages. Two groups use disjoint complete credits. '
               'The same 512 reproducibly shuffled partitions are nested across group sizes. Percentiles describe '
               'partition variability within sixteen observed games. They are not confidence intervals. '
               'This measures the direction of the actor loss alone.'))


if __name__=='__main__':
    main()
