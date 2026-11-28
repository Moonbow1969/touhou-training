"""Fit direction duration and a matched one-decision control on original TH08 Normal."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy
from scripts.train_th08_normal_temporal import history_windows


SEED=814099
BATCH=128


class DirectionDuration(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout=nn.Sequential(nn.Linear(521,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,36))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, features, previous, prior):
        history=torch.cat((features.flatten(1),F.one_hot(previous,9).to(features.dtype)),dim=1)
        return self.readout(history).reshape(-1,9,4)+prior.unsqueeze(-1)


def read(path):
    return json.loads(path.read_bytes())


def source_rows(data, base, records, device):
    encoder=nn.Sequential(base.spatial,*list(base.policy.children())[:-1])
    items=[]
    with torch.inference_mode():
        for row in records:
            path=data/row['path']
            if digest(path)!=row['archive_sha256']:raise ValueError('Original input changed')
            record=load_replay(path)
            ticks=record['native_frame']
            moves=record['action'][:,0]
            indices=history_windows(ticks)
            current=indices[:,-1]
            good=current+3<len(ticks)
            indices=indices[good]
            current=current[good]
            next_steps=current[:,None]+np.arange(4)
            valid=(ticks[next_steps]-ticks[current,None])==np.arange(4)*4
            duration=np.cumprod((moves[next_steps]==moves[current,None])&valid,axis=1).sum(1)
            features=torch.cat([encoder(x.to(device).float()/255).cpu() for x in torch.from_numpy(record['raw']).split(128)])
            bank=features[indices].clone()
            prior=base.policy[-1](bank[:,-1].to(device))[:,:9].cpu()
            items.append(dict(group=row['player_group'],stage=row['stage'],tick=ticks[current],
                              features=bank,prior=prior,y=torch.tensor(moves[current]),
                              previous=torch.tensor(moves[current-1]),duration=torch.tensor(duration)))
    return items


def evaluate(model, rows, chunk, device):
    outcomes=[]
    with torch.inference_mode():
        for row in rows:
            scores=[]
            for hist,prior in zip(row['features'].split(256),row['prior'].split(256)):
                n=len(hist)
                previous=torch.arange(9,device=device).repeat(n)
                prediction=model(hist.to(device).repeat_interleave(9,0),previous,prior.to(device).repeat_interleave(9,0))
                if chunk:
                    best=prediction.flatten(1).argmax(1)
                    values=torch.stack((best//4,best%4+1),dim=-1)
                else:
                    best=prediction.logsumexp(2).argmax(1)
                    values=torch.stack((best,torch.ones_like(best)),dim=-1)
                scores.append(values.reshape(n,9,2).cpu().numpy())
            states=np.concatenate(scores)
            result=[]
            last=4
            remaining=0
            previous_tick=-100
            y=row['y'].numpy()
            human_previous=row['previous'].numpy()
            for tick,table in zip(row['tick'],states):
                if tick!=previous_tick+4:
                    last,remaining=4,0
                if remaining==0:
                    last,remaining=(int(v) for v in table[last])
                result.append(last)
                remaining-=1
                previous_tick=tick
            action=np.asarray(result)
            teacher=states[np.arange(len(y)),human_previous,0]
            change=y!=human_previous
            out=dict(group=row['group'],stage=row['stage'],decisions=len(y),change=int(change.sum()),
                    stay=int((~change).sum()),own_change_correct=int(np.sum((action==y)&change)),
                    own_stay_correct=int(np.sum((action==y)&~change)),
                    teacher_change_correct=int(np.sum((teacher==y)&change)),teacher_stay_correct=int(np.sum((teacher==y)&~change)),
                    predicted_changes=int(np.sum((action[1:]!=action[:-1])&(np.diff(row['tick'])==4))))
            outcomes.append(out)
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data=a.data
    source=data/'th08-trial002-human'
    paths={k:source/f'{k}-qualified-{n:02}-stages-frozen.json' for k,n in (('train',33),('held',12))}
    selected={k:[row for row in read(path)['qualified'] if row['difficulty']=='Normal'] for k,path in paths.items()}
    parent=data/'th08-human-six-title-rehearsal/epoch-10/policy.ts'
    a.out.mkdir(parents=True,exist_ok=False)
    freeze(a.out/'protocol.json',dict(seed=SEED,steps=1200,batch=BATCH,prior=digest(parent),sources={k:digest(v) for k,v in paths.items()},
        contrast='Both arms freeze the same encoder and receive the last four original observations and preceding direction. Both use the same 36 logits, exact sample indices and independent 20% preceding-direction replacement during training. The control marginalizes duration. The treatment trains the joint direction and up-to-four consecutive decision duration.',
        scoring='Simulate each policy on the fixed held human frames using its own previous direction and actual chunk commitment. Reset control history at native discontinuities. Require at least 5pp more correct directions on human change windows and at least 80% correct stay decisions. This is a repeated development set, not original online performance.'))
    base=CameraPolicy()
    base.load_state_dict(torch.jit.load(str(parent),map_location='cpu').state_dict())
    base.eval().requires_grad_(False).to(device)
    train=source_rows(data,base,selected['train'],device)
    held=source_rows(data,base,selected['held'],device)
    bank={k:torch.cat([r[k] for r in train]).to(device) for k in ('features','prior','y','previous','duration')}
    template=DirectionDuration()
    models={k:deepcopy(template).to(device) for k in ('one-step','chunk')}
    opts={k:torch.optim.AdamW(v.parameters(),lr=.0003,weight_decay=.005) for k,v in models.items()}
    rng=torch.Generator().manual_seed(SEED)
    for step in range(1,1201):
        ix=torch.randint(len(bank['y']),(BATCH,),generator=rng).to(device)
        x,p0,y,previous,duration=(bank[k][ix] for k in ('features','prior','y','previous','duration'))
        bad=torch.rand(BATCH,generator=rng).to(device)<.20
        rand=torch.randint(9,(BATCH,),generator=rng).to(device)
        previous=torch.where(bad,rand,previous)
        record=dict(step=step)
        for name,model in models.items():
            raw=model(x,previous,p0)
            scores,target=(raw.flatten(1),y*4+duration-1) if name=='chunk' else (raw.logsumexp(2),y)
            loss=F.cross_entropy(scores,target)
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            opts[name].step()
            record[name]=float(loss.detach())
        if step==1 or step%100==0:
            with (a.out/'loss.jsonl').open('a',encoding='utf8') as file:file.write(json.dumps(record)+'\n')
            print(record,flush=True)
    scores={name:evaluate(m.eval(),held,name=='chunk',device) for name,m in models.items()}
    gross={}
    for name,result in scores.items():
        gross[name]={k:sum(r[f'own_{k}_correct'] for r in result)/sum(r[k] for r in result) for k in ('change','stay')}
        torch.save(models[name].cpu().state_dict(),a.out/f'{name}.pt')
    criterion=gross['chunk']['change']-gross['one-step']['change']>=.05 and gross['chunk']['stay']>=.80
    freeze(a.out/'report.json',dict(device=device,train_windows=len(bank['y']),held_windows=sum(len(r['y']) for r in held),
        presentations_each=BATCH*1200,scores=scores,by_control_state=gross,predeployment_gate_passed=criterion))
    print('FIT COMPLETE',gross,criterion,flush=True)


if __name__=='__main__':main()
