"""Measure whether a frozen six-game behavior encoder exposes impending native life losses."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scripts.checkpoint_storage import digest
from scripts.th12_playable_actions import playable_length
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import CameraActorCritic
from scripts.train_replay_camera_encoder import CameraPolicy,channels


def read(path):return json.loads(path.read_bytes())


def observed_labels(reward):
    losses=np.flatnonzero(reward<-.8)
    if len(losses)!=3 or len(reward)-1-losses[-1]>1 or np.any(reward[losses[-1]+1:]):
        raise ValueError(f'Expected three stock losses followed by at most one empty final observation: {losses.tolist()}')
    indices=np.arange(losses[-1]+1)
    nxt=losses[np.searchsorted(losses,indices)]-indices
    prior=np.searchsorted(losses,indices,side='left')-1
    eligible=(nxt>=4)&((prior<0)|(indices-losses[prior]>16))
    return indices[eligible],(nxt[eligible]<=16).astype(np.float32)


def average_precision(y,p):
    order=np.argsort(-p,kind='stable')
    yy=y[order];pp=p[order];last=np.r_[pp[:-1]!=pp[1:],True]
    tp=np.cumsum(yy)[last];precision=tp/np.arange(1,len(y)+1)[last]
    return float(np.dot(np.diff(tp,prepend=0),precision)/yy.sum())


def main():
    arg=argparse.ArgumentParser(description=__doc__);arg.add_argument('--data',type=Path,required=True)
    d=arg.parse_args().data.resolve()
    root=d/'six-game-risk-probe';root.mkdir(exist_ok=True)
    manifest=d/'six-game-online/joint-six/optimizer-input/credits.json'
    report=read(d/'six-game-online/joint-six/policy/report.json')
    parent=d/'five-game-generation2/policy/camera-ppo.pt'
    if digest(parent)!=report['parent_actor_critic_sha256'] or digest(manifest)!=report['manifest_sha256']:
        raise ValueError('Unverified six-game behavior encoder or training population')
    groups=defaultdict(list)
    for r in read(manifest):groups[r['title'],r['difficulty']].append(r)
    allocation={}
    for (t,k),rows in groups.items():
        ordered=sorted(rows,key=lambda r:hashlib.sha256(f"six-risk-175:{t}:{r['seed']}".encode()).hexdigest())
        if len(ordered)!=16:raise ValueError('Expected sixteen credits in every title and difficulty')
        allocation[f'{t}-{k}']={str(r['seed']):('train' if i<12 else 'test') for i,r in enumerate(ordered)}
    freeze(root/'protocol.json',dict(source=digest(manifest),frozen_encoder=digest(parent),split=allocation,
      objective='Predict an observed native stock loss in 4 through 16 future policy decisions. Exclude shots within three decisions of the next loss or within 16 decisions after the previous one. Camera features use only the current and previous images. Three full stock decrements, including the true terminal, must reconcile in every credit.',
      comparison='Compare a title/rank elapsed-decision baseline estimated from training credits with the same baseline plus a learned visual residual. The residual sees only the frozen behavior encoder representation. Use 12 whole training credits and 4 held credits per cell. Fit a single head with equally sampled title/rank and credit. Freeze 1500 updates, batch size 192, AdamW with lr .001. Primary score is mean per-credit held Brier loss; report average precision and whole-credit bootstrap. This diagnostic does not change policy controls.'))
    torch.manual_seed(175);torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    state=torch.load(parent,map_location='cpu',weights_only=True);cam=CameraPolicy()
    if any('policy.spatial_context.' in k for k in state):cam.add_spatial_context()
    base=CameraActorCritic(cam);base.load_state_dict(state);base.to(device).eval()
    banks={};lookup={};train=[];held=[];maxbin=64
    with torch.inference_mode():
        for cell,rows in sorted(groups.items()):
            counts=np.zeros(maxbin);pos=counts.copy()
            for row in rows:
                key=row['seed'];path=manifest.parent/row['file']
                if row['role']!='train' or digest(path)!=row['sha256']:raise ValueError('Changed probe source')
                with np.load(path,allow_pickle=False) as z:
                    rew=z['reward'].reshape(-1)
                    n=playable_length(cell[0],z.get('native_action_eligible'),rew,row.get('postterminal_input_boundary'))
                    ids,y=observed_labels(rew[:n]);frames=z['frames'][:n]
                    raw=channels(frames[ids],frames[np.maximum(ids-1,0)])
                features=[]
                for x in torch.from_numpy(raw).split(64):
                    features.append(cam.policy[:-1](cam.spatial(x.to(device).float()/255)).cpu())
                bins=np.minimum(ids//32,maxbin-1)
                mode=allocation[f'{cell[0]}-{cell[1]}'][str(key)]
                if mode=='train':
                    counts+=np.bincount(bins,minlength=maxbin);pos+=np.bincount(bins,weights=y,minlength=maxbin);train.append(key)
                else:held.append(key)
                banks[key]=dict(features=torch.cat(features),bins=bins,label=y,cell=cell)
            prior=pos.sum()/counts.sum()
            rate=(pos+64*prior)/(counts+64)
            lookup[cell]=np.log(rate/(1-rate)).astype(np.float32)
            print('Read source cell',cell,flush=True)
    block=[];offset={};clock=[];labels=[];n=0
    for k,b in banks.items():
        block.append(b['features']);clock.append(lookup[b['cell']][b['bins']]);labels.append(b['label'])
        offset[k]=n;n+=len(b['bins'])
    features=torch.cat(block).to(device);clock=torch.from_numpy(np.concatenate(clock)).to(device)
    labels=torch.from_numpy(np.concatenate(labels)).to(device)
    model=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1)).to(device);nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.001);rng=np.random.default_rng(175);logs=[]
    for step in range(1,1501):
        # All 12 strata have the same number of training credits.
        chosen=rng.choice(train,192)
        ids=torch.as_tensor([offset[k]+rng.integers(len(banks[k]['bins'])) for k in chosen],device=device)
        estimate=clock[ids]+model(features[ids]).flatten()
        loss=nn.functional.binary_cross_entropy_with_logits(estimate,labels[ids])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%100==0:logs.append(dict(step=step,loss=float(loss.detach())))
    torch.save(model.cpu().state_dict(),root/'risk-head.pt');model.to(device).eval()
    with torch.no_grad():
        visual=torch.cat([(clock[a:a+512]+model(features[a:a+512]).flatten()).sigmoid().cpu() for a in range(0,n,512)]).numpy()
        baseline=clock.sigmoid().cpu().numpy()
    result=[]
    for k in held:
        b=banks[k];y=b['label'];sl=slice(offset[k],offset[k]+len(y));metrics={}
        for arm,p in (('clock',baseline[sl]),('clock_camera',visual[sl])):
            metrics[arm]=dict(brier=float(np.mean((p-y)**2)),average_precision=average_precision(y,p))
        result.append(dict(seed=k,title=b['cell'][0],difficulty=b['cell'][1],decisions=len(y),positive=int(y.sum()),metrics=metrics))
    def compare(rows):
        v=np.asarray([r['metrics']['clock_camera']['brier']-r['metrics']['clock']['brier'] for r in rows])
        sample=rng.choice(v,(4000,len(v)),replace=True).mean(1)
        return dict(credits=len(v),mean_brier_difference=float(v.mean()),paired_bootstrap95=np.quantile(sample,[.025,.975]).tolist(),
                    mean_average_precision={a:float(np.mean([r['metrics'][a]['average_precision'] for r in rows])) for a in ('clock','clock_camera')})
    per={f'TH{t:02}-{r}':compare([v for v in result if (v['title'],v['difficulty'])==(t,r)]) for t,r in groups}
    # The overall bootstrap preserves the fixed four held credits in every title/rank cell.
    draws=[]
    for t,r in sorted(groups):
        delta=[v['metrics']['clock_camera']['brier']-v['metrics']['clock']['brier'] for v in result if (v['title'],v['difficulty'])==(t,r)]
        draws.append(rng.choice(delta,(4000,4)).mean(1))
    overall=compare(result);overall['paired_bootstrap95']=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist()
    freeze(root/'report.json',dict(protocol=digest(root/'protocol.json'),weights=digest(root/'risk-head.pt'),
        training_credits=len(train),held_credits=len(held),training_decisions=sum(len(banks[k]['bins']) for k in train),
        held_decisions=sum(v['decisions'] for v in result),metrics=overall,by_cell=per,held=result,training_log=logs))
    print('Held credit-level risk accuracy',overall,flush=True)


if __name__=='__main__':
    main()
