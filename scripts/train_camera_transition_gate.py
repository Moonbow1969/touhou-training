"""Learn when to release a movement from human cameras and the preceding movement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.train_human_camera_policy import evaluation, load_replay
from scripts.train_portable_full_sft import WIDTHS
from scripts.train_replay_camera_encoder import CameraPolicy, save_json, sha


class TransitionPolicy(nn.Module):
    def __init__(self,camera):
        super().__init__()
        self.camera=camera
        self.switch=nn.Linear(128,9)
        nn.init.zeros_(self.switch.weight)
        nn.init.zeros_(self.switch.bias)

    def forward(self,image,previous_direction):
        latent=self.camera.policy[:-1](self.camera.spatial(image))
        movement,focus,shot,bomb=self.camera.policy[-1](latent).split(WIDTHS,dim=1)
        previous=previous_direction.long().reshape(-1,1)
        mask=F.one_hot(previous[:,0],num_classes=9).bool()
        switch=self.switch(latent).gather(1,previous)
        alternative=movement.masked_fill(mask,-1e4).log_softmax(1)
        calibrated=torch.where(mask,F.logsigmoid(-switch),F.logsigmoid(switch)+alternative)
        return calibrated,focus,shot,bomb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('sft','human','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--held',nargs='+',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(261128)
    torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    original=json.loads((a.sft/'report.json').read_bytes())
    cam=CameraPolicy();cam.load_state_dict(torch.load(a.sft/'camera-policy.pt',weights_only=True,map_location='cpu'))
    model=TransitionPolicy(cam).to(device);model.camera.eval()
    train=[]
    for item in original['train']:
        path=a.human/'train'/item['name']
        if sha(path)!=item['sha256']:raise ValueError('Training capture changed')
        row=load_replay(path)
        with torch.no_grad():
            latent=torch.cat([model.camera.policy[:-1](model.camera.spatial(b.to(device).float()/255))
                              for b in torch.from_numpy(row['raw']).split(64)])
        train.append((latent,torch.from_numpy(row['prior'][:,0]).to(device),
                      torch.from_numpy(row['action'][:,0]!=row['prior'][:,0]).to(device).float()))
    opt=torch.optim.AdamW(model.switch.parameters(),lr=.002,weight_decay=.03)
    losses=[]
    for step in range(1,601):
        terms=[]
        for latent,prior,target in train:
            ids=torch.randint(len(prior),(64,),device=device)
            switch=model.switch(latent[ids]).gather(1,prior[ids,None])[:,0]
            terms.append(F.binary_cross_entropy_with_logits(switch,target[ids]))
        loss=torch.stack(terms).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step==1 or step%50==0:losses.append(dict(step=step,bce=loss.item()))
    report={}
    with torch.inference_mode():
        for path in a.held:
            row=load_replay(path)
            if row['group'] in {b['group'] for b in original['train']}:raise ValueError('Overlapping player')
            px=torch.from_numpy(row['raw']);prev=torch.from_numpy(row['prior'][:,0])
            captures={}
            for label,image in [('aligned',px),('shifted_screen_only',px.roll(len(px)//2,0))]:
                heads=[model(b.to(device).float()/255,h.to(device)) for b,h in zip(image.split(64),prev.split(64))]
                probs=[torch.cat(values).softmax(1).cpu().numpy() for values in zip(*heads)]
                detail=evaluation(row,probs)
                p_change=1-probs[0][np.arange(len(px)),prev.numpy()]
                target=(row['action'][:,0]!=prev.numpy()).astype(float)
                detail.update(observed_change_rate=float(target.mean()),predicted_change_rate=float(p_change.mean()),
                              change_brier=float(((p_change-target)**2).mean()))
                captures[label]=detail
            report[path.name]={'player':row['group'],'sha256':row['sha256'],'metrics':captures}
    if a.out.is_dir() and any(a.out.iterdir()):raise FileExistsError(a.out)
    a.out.mkdir(exist_ok=True,parents=True);model.cpu().eval()
    fixture=torch.zeros(1,2,112,96);prev=torch.tensor([4])
    script=torch.jit.trace(model,(fixture,prev))
    for direction in range(9):
        previous=torch.tensor([direction])
        for x,y in zip(model(fixture,previous),script(fixture,previous)):torch.testing.assert_close(x,y)
    script.save(str(a.out/'conditional-policy.ts'))
    torch.save(model.state_dict(),a.out/'conditional-policy.pt')
    save_json(a.out/'report.json',dict(sft=original['sft'],train=original['train'],held=report,training=losses,
              trained_parameters=sum(p.numel() for p in model.switch.parameters()),steps=600,
              history='The real previous movement is known to the agent. The first live action starts from neutral (4). Immutable camera features, direction-given-change and three button heads.',
              checkpoint=sha(a.out/'conditional-policy.ts')))


if __name__=='__main__':main()
