"""Check held-credit value prediction before a new multigame PPO update."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.th12_playable_actions import playable_length
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import CameraActorCritic, CameraPolicy
from scripts.train_replay_camera_encoder import channels


def discounted_returns(reward,gamma):
    total=0.;result=np.empty(len(reward))
    for j in range(len(reward)-1,-1,-1):
        total=float(reward[j])+gamma*total;result[j]=total
    return result


def ridge(train,xname,test):
    x=np.concatenate([r[xname] for r in train]);y=np.concatenate([r['y'] for r in train])
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-5);x=(x-mean)/scale
    ybar=y.mean();lam=.02*len(y)
    fit=np.linalg.solve(x.T@x+lam*np.eye(x.shape[1]),x.T@(y-ybar))
    return [(row[xname]-mean)/scale@fit+ybar for row in test]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    raw=a.root/'corrected/credits.json';rows=json.loads(raw.read_bytes());parent=a.parent/'camera-ppo.pt'
    if len(rows)!=160 or any(r['role']!='train' or r['policy_sha256']!=digest(a.parent/'policy.ts') for r in rows):
        raise ValueError('Expected the registered five-title generation-three behavior')
    plan=dict(manifest_sha256=digest(raw),value_weights_sha256=digest(parent),gamma=.999,
            fold='Within each game and rank, shuffle its sixteen whole credits with seed 290811+title*10+rank, then divide into four disjoint folds of four. Refit each baseline on the other twelve credits. Frozen original critic receives the image only. Clock ridge uses log age and five causal time splines; latent ridge adds the frozen shared visual representation. Ridge penalty 0.02 per training row after normalization.',
            target='Discounted observed rewards up to verified original termination; fixed policy, no terminal bootstrap. No G3 evaluation games or learner gradients. Report per-cell held prediction error and paired squared-error differences across complete trajectories. Forecasting uncertainty contains game RNG and future policy actions.')
    freeze(a.out/'protocol.json',plan)
    model=CameraActorCritic(CameraPolicy())
    state=torch.load(parent,weights_only=True,map_location='cpu')
    if any('policy.spatial_context.' in n for n in state):
        model.camera.add_spatial_context()
    model.load_state_dict(state);model.to(device).eval()
    cells={}
    for row in rows:
        f=raw.parent/row['file']
        if digest(f)!=row['sha256']:raise ValueError('Changed training camera')
        with np.load(f,allow_pickle=False) as z:
            n=playable_length(row['title'],z.get('native_action_eligible'),z['reward'].reshape(-1),row.get('postterminal_input_boundary'))
            frames=z['frames'][:n];reward=z['reward'].reshape(-1)[:n];v=z['value'].reshape(-1)[:n]
        batch=torch.from_numpy(channels(frames,np.r_[frames[:1],frames[:-1]]))
        with torch.inference_mode():
            latent=torch.cat([model.camera.policy[:-1](model.camera.spatial(b.to(device).float()/255)).cpu() for b in batch.split(64)])
            restored=model.value(latent.to(device)).cpu().numpy().reshape(-1)
        if np.max(abs(restored-v))>3e-4:
            raise ValueError('Frozen critic does not reproduce native values')
        age=np.arange(len(v),dtype=np.float64)
        clock=np.column_stack([np.log1p(age)]+[np.minimum(age/t,1) for t in (32,64,128,256,512)])
        cells.setdefault((row['title'],row['difficulty']),[]).append(dict(seed=row['seed'],y=discounted_returns(reward,.999),
              stock_critic=v.astype('float64'),clock=clock,vision=np.column_stack((clock,latent.numpy())).astype('float64')))
    results={}
    for (t,rank),group in sorted(cells.items()):
        if len(group)!=16:raise ValueError('Expected sixteen full credits per stratum')
        group=sorted(group,key=lambda r:r['seed']);rng=np.random.default_rng(290811+t*10+rank)
        folds=np.array_split(rng.permutation(16),4);observations=[]
        for fold in folds:
            train=[r for j,r in enumerate(group) if j not in fold];held=[group[j] for j in fold]
            train_mean=np.concatenate([r['y'] for r in train]).mean()
            predictions={'clock':ridge(train,'clock',held),'clock_camera':ridge(train,'vision',held)}
            for j,r in enumerate(held):
                y=r['y'];preds={k:v[j] for k,v in predictions.items()}
                preds.update(native_critic=r['stock_critic'],train_mean=np.full_like(y,train_mean))
                observations.append(dict(seed=r['seed'],n=len(y),return_sum=float(y.sum()),return_square_sum=float(y@y),
                   prediction_bias={k:float(np.mean(v-y)) for k,v in preds.items()},
                   mse={k:float(np.mean((v-y)**2)) for k,v in preds.items()}))
        total=sum(r['n'] for r in observations);mean_return=sum(r['return_sum'] for r in observations)/total
        variance=sum(r['return_square_sum'] for r in observations)/total-mean_return**2
        transitions={k:sum(r['mse'][k]*r['n'] for r in observations)/total for k in observations[0]['mse']}
        percredit={k:float(np.mean([r['mse'][k] for r in observations])) for k in transitions}
        delta=np.array([r['mse']['native_critic']-r['mse']['clock_camera'] for r in observations])
        bs=np.random.default_rng(290812).choice(delta,(10000,len(delta)),replace=True).mean(1)
        results[f'TH{t:02}-{rank}']=dict(credits=16,n=total,mean_return=mean_return,
          native_mean_prediction=mean_return+sum(r['prediction_bias']['native_critic']*r['n'] for r in observations)/total,
          transition_r_squared={k:float(1-v/variance) for k,v in transitions.items()},credit_macro_mse=percredit,
          paired_native_minus_refit_mse=float(np.mean(delta)),paired_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),
          by_original_credit=observations)
    freeze(a.out/'report.json',dict(protocol_sha256=digest(a.out/'protocol.json'),cells=results))


if __name__=='__main__':
    main()
