"""Forecast future original-game cameras from real decisions on five titles and two ranks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.th12_playable_actions import playable_length
from scripts.train_five_original_games import freeze


class FramePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(2,32,5,stride=2,padding=2),nn.SiLU(),
                                   nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.keys=nn.Sequential(nn.Linear(15,32),nn.SiLU(),nn.Linear(32,32))
        self.transition=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1))
        self.pixels=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,4,3,padding=1),nn.PixelShuffle(2))

    def forward(self, history, actions, *, use_actions):
        state=self.spatial(history)
        result=[]
        for step in range(4):
            # Zero features define the image-only control. Both arms have an identical parameterization.
            act=actions[:,step]
            if not use_actions:
                act=torch.zeros_like(act)
            state=state+.25*self.transition(state+self.keys(act)[:,:,None,None])
            result.append((history[:,-1:]+self.pixels(state)).squeeze(1))
        return torch.stack(result,1)


def sources(root):
    manifest=root/'corrected/credits.json'
    rows=json.loads(manifest.read_bytes())
    cells={}
    for row in rows:
        if row['role']!='train' or row['outcome'] not in ('ending','game_over') or row['difficulty'] not in (2,3):
            raise ValueError('Expected a verified training original')
        cells.setdefault((row['title'],row['difficulty']),[]).append(row)
    if len(cells)!=10 or any(len(rs)!=8 for rs in cells.values()):
        raise ValueError('Expected the complete five-game second-generation training cohort')
    split={}
    for key,rs in sorted(cells.items()):
        ordered=sorted(rs,key=lambda r:hashlib.sha256(f"camera-dynamics-v1/{r['title']}/{r['seed']}".encode()).hexdigest())
        for j,r in enumerate(ordered):
            split[r['seed']]='train' if j<6 else 'test'
    banks={p:[] for p in ('train','test')}
    register=[]
    for row in rows:
        f=manifest.parent/row['file']
        if digest(f)!=row['sha256']:
            raise ValueError('Changed original camera source')
        with np.load(f,allow_pickle=False) as z:
            length=playable_length(row['title'],z.get('native_action_eligible'),z['reward'].reshape(-1),row.get('postterminal_input_boundary'))
            frames=z['frames'][:length]
            if frames.shape!=(length,112,96) or frames.dtype!=np.uint8:
                raise ValueError('Unexpected camera shape')
            down=frames.astype(np.uint16).reshape(length,56,2,48,2).sum((2,4))
            imgs=torch.from_numpy((down/1020.).astype(np.float32))
            action=torch.from_numpy(z['action'][:length].astype(np.int64))
            one=torch.cat([F.one_hot(action[:,k],w) for k,w in enumerate((9,2,2,2))],1).float()
            role=split[row['seed']]
            banks[role].append((imgs,one))
            register.append(dict(seed=row['seed'],title=row['title'],difficulty=row['difficulty'],source_sha256=row['sha256'],
                                 role=role,playable_decisions=length,forecast_starts=length-5))
    return banks,register,manifest


def sample(bank, rng, n):
    episode=rng.integers(len(bank),size=n)
    times=np.array([rng.integers(1,len(bank[i][0])-4) for i in episode])
    return batch(bank,episode,times)


def batch(bank, episode, times):
    frames=torch.stack([bank[i][0][t-1:t+5] for i,t in zip(episode,times)])
    controls=torch.stack([bank[i][1][t:t+4] for i,t in zip(episode,times)])
    return frames,controls


@torch.inference_mode()
def evaluate(model, bank, register, device, use_actions):
    results=[]
    for i,(images,actions) in enumerate(bank):
        # Circularly relabel real action sequences inside the same original credit.
        # Spill neither images nor controls across full-credit boundaries.
        roll=max(1,len(actions)//2)
        total={}
        for start in range(1,len(images)-4,64):
            times=np.arange(start,min(start+64,len(images)-4))
            frames,keys=batch(bank,[i]*len(times),times)
            hist,targets=frames[:,:2].to(device),frames[:,2:].to(device)
            predictions={'network':model(hist,keys.to(device),use_actions=use_actions)}
            if use_actions:
                other=actions[(torch.as_tensor(times)[:,None]+torch.arange(4)[None,:]+roll)%len(actions)]
                predictions['shifted_keys']=model(hist,other.to(device),use_actions=True)
                predictions['repeat_frame']=hist[:,1:2].expand_as(targets)
            for arm,prediction in predictions.items():
                for idx in (0,3):
                    target=targets[:,idx]; anchor=hist[:,1]
                    err=(prediction[:,idx].clamp(0,1)-target).abs()
                    moving=(target-anchor).abs()>10/255
                    val=total.setdefault((arm,idx+1),np.zeros(4))
                    val += [float(err.sum()),err.numel(),float((err*moving).sum()),int(moving.sum())]
        metrics={}
        for (arm,h),(l,n,lm,nm) in total.items():
            metrics.setdefault(arm,{})[str(h)]=dict(mae=float(l/n),moving_mae=float(lm/nm),moving_pixels=int(nm),pixels=int(n))
        results.append(dict(**register[i],metrics=metrics))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    bank,rows,manifest=sources(args.root)
    train=[r for r in rows if r['role']=='train']
    held=[r for r in rows if r['role']=='test']
    plan=dict(source_manifest_sha256=digest(manifest),credits=rows,seed=261131,updates=1600,batch_size=48,
              architecture='Shared two-frame spatial encoder, learned four-step action-conditioned residual transition, one grayscale decoder. Cameras area-averaged 112x96 to 56x48.',
              sampling='Uniform over the 60 complete training trajectories, uniform valid start within each. Both models use the identical schedule and initializer. No held PPO evaluation games used.',
              metric='Primary mean per-credit absolute 4-step camera error on pixels whose target differs from the last observation by over 10/255. Also all pixels and 1-step. Matched frozen controls: independently trained image-only, same model with circularly relabelled keys, and current-frame persistence.',
              interpretation='Recorded future actions supplied explicitly. Their observational association includes the behavior policy response to intervening frames. Forecast accuracy does not measure policy improvement or identify a causal action effect.')
    freeze(args.out/'protocol.json',plan)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    arms={}
    for arm in ('camera_keys','camera_only'):
        torch.manual_seed(plan['seed'])
        rng=np.random.default_rng(plan['seed'])
        model=FramePredictor().to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        logs=[]
        for step in range(1,plan['updates']+1):
            fr,ac=sample(bank['train'],rng,plan['batch_size'])
            fr=fr.to(device);hist=fr[:,:2];target=fr[:,2:]
            out=model(hist,ac.to(device),use_actions=arm=='camera_keys')
            diff=(out-target).abs()
            # Increase the weight on actually moving pixels in the training objective.
            loss=(diff*(1+2*((target-hist[:,1:2]).abs()>10/255))).mean()
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optim.step()
            if step==1 or step%100==0:
                logs.append(dict(step=step,loss=float(loss.detach())))
                print(arm,logs[-1],flush=True)
                (args.out/f'{arm}-training.json').write_text(json.dumps(logs,indent=2)+'\n',encoding='utf8')
        weights=args.out/f'{arm}.pt'
        torch.save(model.state_dict(),weights)
        arms[arm]=dict(sha256=digest(weights),held=evaluate(model.eval(),bank['test'],held,device,arm=='camera_keys'))
    freeze(args.out/'report.json',dict(protocol_sha256=digest(args.out/'protocol.json'),train_credits=len(train),
           test_credits=len(held),device=torch.cuda.get_device_name() if device=='cuda' else device,arms=arms))


if __name__=='__main__':
    main()
