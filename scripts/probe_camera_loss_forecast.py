"""Test whether a frozen visual policy representation forecasts native stock loss on new credits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.probe_native_loss_risk import ap, fit, loss_targets
from scripts.score_original_camera_run import records
from scripts.train_full_resolution_camera_ppo import CameraActorCritic
from scripts.train_replay_camera_encoder import CameraPolicy, channels, save_json, sha


def episode(title,f,r,encoder,device):
    trace=f.parent/r.get('behavior',r.get('trajectory',r.get('file','missing')))
    recorded=r.get('behavior_sha256',r.get('trajectory_sha256',r.get('sha256')))
    if sha(trace)!=recorded:raise ValueError(f'Changed credit {trace}')
    with np.load(trace,allow_pickle=False) as z:
        _,loss=loss_targets(z,title)
        frames=z['frames'] if 'frames' in z else None
    if frames is None and title==10:
        with np.load(f.parent/'original-pixels.npz',allow_pickle=False) as z:frames=z['obs'][:len(loss)]
    if frames is None and title==12:
        frames=np.fromfile(f.parent.parent/'raw'/r['episode']/'observations.u8',np.uint8).reshape(-1,112,96)
    if frames is None or len(frames)!=len(loss):raise ValueError('Original frames unavailable')
    inputs=torch.from_numpy(channels(frames,np.concatenate((frames[:1],frames[:-1]))))
    with torch.inference_mode():
        latent=torch.cat([encoder.policy[:-1](encoder.spatial(b.to(device).float()/255)).cpu()
                          for b in inputs.split(64)]).numpy().astype(np.float64)
    # Discard the three nearest observations to each future label window and its incomplete tail.
    idx=np.arange(len(loss)-10)
    cum=np.r_[0,np.cumsum(loss)]
    y=((cum[idx+11]-cum[idx+3])>0).astype(float)
    age=np.column_stack((np.log1p(idx),np.minimum(idx/300,1),np.minimum(idx/1500,1)))
    return dict(seed=r['seed'],policy=r['policy_sha256'],source=recorded,y=y,age=age,x=latent[idx],
                centered_screen_control=np.roll(latent,len(latent)//2,axis=0)[idx],events=int(loss.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CameraActorCritic(CameraPolicy())
    model.load_state_dict(torch.load(a.model,weights_only=True,map_location='cpu'));encoder=model.camera.to(device).eval()
    banks={t:{'train':[],'held':[]} for t in (10,12,13)}
    sft_sha=sha(a.root/'camera-sft/policy.ts');ppo1_sha=sha(a.root/'camera-ppo/policy.ts')
    for round2 in (False,True):
        folder=a.root/'camera-round2' if round2 else a.root
        plan=json.loads((folder/'camera-protocol.json').read_bytes())
        for t,f in records(folder,True,round2):
            r=json.loads(f.read_bytes());seed=r.get('seed')
            if seed in plan['train_seeds'][str(t)] and r.get('policy_sha256')==(ppo1_sha if round2 else sft_sha):role='train'
            elif not round2 and seed in plan['eval_seeds'][str(t)] and r.get('policy_sha256') in (sft_sha,ppo1_sha):role='held'
            else:continue
            banks[t][role].append(episode(t,f,r,encoder,device))
    results={}
    for t,bank in banks.items():
        training=bank['train'];held=bank['held']
        if len(training)!=24 or len(held)!=12 or len({x['source'] for x in training+held})!=36:
            raise ValueError(f'Incomplete or repeated original TH{t} data')
        def feature(r,kind):return r['age'] if kind=='clock' else np.column_stack((r['age'],r[kind]))
        scores={}
        for kind in ('clock','x'):
            pred=fit(np.concatenate([feature(r,kind) for r in training]),np.concatenate([r['y'] for r in training]))
            scores[kind]=[ap(r['y'],pred(feature(r,kind))) for r in held]
            if kind=='x':scores['shifted_camera']=[ap(r['y'],pred(feature(r,'centered_screen_control'))) for r in held]
        public=[]
        for i,r in enumerate(held):
            public.append(dict(seed=r['seed'],policy=r['policy'],trace=r['source'],n=len(r['y']),confirmed_losses=r['events'],
                               positive_rate=float(r['y'].mean()),ap={k:float(v[i]) for k,v in scores.items()}))
        delta=[np.mean([row['ap']['x']-row['ap']['clock'] for row in public if row['seed']==s]) for s in sorted({r['seed'] for r in held})]
        ci=np.percentile(np.random.default_rng(241126+t).choice(delta,(20000,len(delta))).mean(1),[2.5,97.5]).tolist()
        results[t]=dict(held=public,train_credits=len(training),train_decisions=sum(len(r['y']) for r in training),
                        mean_positive_rate=float(np.mean([r['positive_rate'] for r in public])),mean_credit_ap={k:float(np.mean(v)) for k,v in scores.items()},
                        mean_image_minus_clock_ap=float(np.mean(delta)),controller_seed_bootstrap95=ci)
    save_json(a.out,dict(horizon_offsets_in_decisions=[3,10],policy=sha(a.model),label='Future native life-stock confirmation. Collision and HUD update timing can differ.',
                         fit='One fixed L2 linear logistic readout per title on 24 original training credits. Held-test AP is computed per full credit; bootstrap resamples six seed pairs.',titles=results))


if __name__=='__main__':main()
