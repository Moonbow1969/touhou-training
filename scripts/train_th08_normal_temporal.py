"""Matched current-only and four-step human imitation fits on original TH08 Normal."""
import argparse
from collections import defaultdict
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


HISTORY=4
WIDTHS=(9,2,2,2)
SEED=814089
BATCH=128


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(512,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,13))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, history, original):
        move,shoot,bomb=self.layers(history.flatten(1)).split((9,2,2),1)
        return original[0]+move,original[1],original[2]+shoot,original[3]+bomb


class Actor(nn.Module):
    def __init__(self, base, readout, history):
        super().__init__()
        self.features=nn.Sequential(base.spatial,*list(base.policy.children())[:-1])
        self.prior=base.policy[-1]
        self.readout=readout
        self.history=history

    def forward(self, image):
        if self.history:
            features=self.features(image.reshape(-1,2,112,96)).reshape(-1,HISTORY,128)
        else:
            features=self.features(image).unsqueeze(1).expand(-1,HISTORY,-1)
        return self.readout(features,self.prior(features[:,-1]).split(WIDTHS,dim=1))


def read(path):
    return json.loads(path.read_bytes())


def history_windows(ticks):
    present=np.arange(HISTORY-1,len(ticks))
    indices=present[:,None]+np.arange(-HISTORY+1,1)
    return indices[np.all(np.diff(ticks[indices],axis=1)==4,axis=1)]


def encode(data, files, base, device):
    features=nn.Sequential(base.spatial,*list(base.policy.children())[:-1])
    parts=[]
    with torch.inference_mode():
        for item in files:
            path=data/item['path']
            if digest(path)!=item['archive_sha256']:raise ValueError('Original replay changed')
            replay=load_replay(path)
            blocks=history_windows(replay['native_frame'])
            bank=torch.cat([features(batch.to(device).float()/255).cpu()
                            for batch in torch.from_numpy(replay['raw']).split(128)])
            current=blocks[:,-1]
            labels=replay['action'][current]
            if np.any(labels[:,1]):raise ValueError('Recheck Normal focus sample support')
            parts.append(dict(group=item['player_group'],stage=item['stage'],path=item['path'],
                              images=bank[blocks].clone(),labels=torch.tensor(labels),
                              changed=torch.tensor(labels[:,0]!=replay['action'][current-1,0])))
            print('Aligned',item['player_group'],item['stage'],len(labels),flush=True)
    return parts


def metrics(parts, base, model, temporal, device):
    by_group=defaultdict(list)
    with torch.inference_mode():
        for item in parts:
            hist=item['images'].to(device)
            y=item['labels'].to(device)
            previous=item['changed'].to(device)
            old=base.policy[-1](hist[:,-1]).split(WIDTHS,dim=1)
            result=model(hist if temporal else hist[:,-1:].expand_as(hist),old)
            direction=result[0].log_softmax(1)
            nll=-direction.gather(1,y[:,:1])[:,0]
            by_group[item['group']].append(dict(stage=item['stage'],n=len(y),move_nll=float(nll.mean()),
                move_correct=int((direction.argmax(1)==y[:,0]).sum()),changed_rows=int(previous.sum()),
                changed_nll_sum=float(nll[previous].sum()),shoot_nll=float(F.cross_entropy(result[2],y[:,2])),
                bomb_nll=float(F.cross_entropy(result[3],y[:,3]))))
    return by_group


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1200)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    source=args.data/'th08-trial002-human'
    pilot=args.data/'th08-human-six-title-rehearsal/epoch-10/policy.ts'
    tables={}
    for split,n in (('train',33),('held',12)):
        folder=source/f'{split}-qualified-{n:02}-stages-frozen.json'
        tables[split]=[r for r in read(folder)['qualified'] if r['difficulty']=='Normal']
    train_groups={row['player_group'] for row in tables['train']}
    test_groups={row['player_group'] for row in tables['held']}
    if len(train_groups)!=5 or len(test_groups)!=2 or train_groups & test_groups:
        raise ValueError('Unexpected original replay player split')
    if any(row['team']!='Ghost Team' or row['game_version']!='0.02_' for rows in tables.values() for row in rows):
        raise ValueError('Original sample edition or team differs')
    args.out.mkdir(parents=True,exist_ok=False)
    freeze(args.out/'protocol.json',dict(edition='TH08 original 0.02 replay',team='Ghost',rank='Normal',
        prior=digest(pilot),previous_frames=[-12,-8,-4,0],updates=args.updates,batch=BATCH,seed=SEED,
        comparison='Frozen base CNN. Train a capacity-matched residual on shared indices and labels. Control repeats current encoding. Treatment takes the four ordered native camera states. Focus reuses base logits because no Normal positive expert focus data exist.',
        metrics='Movement NLL and accuracy on the two frozen held pilots, plus NLL after true direction changes. The same pilots informed previous studies, so these are development checks. Native performance requires separately registered credits.',
        archives={k:[r['path'] for r in v] for k,v in tables.items()}))
    base=CameraPolicy()
    base.load_state_dict(torch.jit.load(str(pilot),map_location='cpu').state_dict())
    base.requires_grad_(False).eval().to(device)
    data={k:encode(args.data,v,base,device) for k,v in tables.items()}
    bank=torch.cat([r['images'] for r in data['train']]).to(device)
    y=torch.cat([r['labels'] for r in data['train']]).to(device)
    with torch.no_grad():
        original=base.policy[-1](bank[:,-1]).detach()
    template=Readout()
    models={name:deepcopy(template).to(device) for name in ('change-free','temporal')}
    optimizers={name:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.005) for name,m in models.items()}
    rng=torch.Generator().manual_seed(SEED+1)
    log=[]
    for step in range(1,args.updates+1):
        indices=torch.randint(len(bank),(BATCH,),generator=rng).to(device)
        hist=bank[indices]
        target=y[indices]
        native=original[indices].split(WIDTHS,dim=1)
        row=dict(step=step)
        for name,model in models.items():
            features=hist if name=='temporal' else hist[:,-1:].expand_as(hist)
            heads=model(features,native)
            loss=F.cross_entropy(heads[0],target[:,0])+.35*F.cross_entropy(heads[2],target[:,2])+.45*F.cross_entropy(heads[3],target[:,3])
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[name].step()
            row[name]=float(loss.detach())
        if step==1 or step%100==0:
            log.append(row)
            print(row,flush=True)
            with (args.out/'loss.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    report=dict(device=device,unique_training_windows=len(bank),presentations_per_arm=args.updates*BATCH,
        held_windows=sum(len(r['labels']) for r in data['held']),metrics={},history=log,checkpoints={})
    for name,model in models.items():
        temporal=name=='temporal'
        report['metrics'][name]=metrics(data['held'],base,model,temporal,device)
        folder=args.out/name
        folder.mkdir()
        actor=Actor(deepcopy(base).cpu(),model.cpu().eval(),temporal)
        fixture=torch.rand(1,8 if temporal else 2,112,96)
        frozen=torch.jit.trace(actor,fixture)
        for actual,expected in zip(frozen(fixture),actor(fixture)):
            torch.testing.assert_close(actual,expected)
        frozen.save(str(folder/'policy.ts'))
        torch.save(model.state_dict(),folder/'readout.pt')
        report['checkpoints'][name]=digest(folder/'policy.ts')
    freeze(args.out/'report.json',report)
    print('FIT COMPLETE',report['checkpoints'],flush=True)


if __name__=='__main__':main()
