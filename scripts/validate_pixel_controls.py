"""Estimate how well held horizontal inputs can be decoded from fresh original-game image pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_native_pixel_dynamics import read,sha256


class HorizontalDecoder(nn.Module):
    def __init__(self, n_titles):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(2,24,5,padding=2),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,48,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((5,4)),nn.Flatten())
        self.out=nn.Sequential(nn.Linear(48*5*4+n_titles+1,96),nn.SiLU(),nn.Linear(96,3))

    def forward(self,img,successor,context):
        diff=successor-img
        embedding=self.visual(torch.cat([img,diff],dim=1))
        return self.out(torch.cat([embedding,context],dim=1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--seed",type=int,default=710)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    fit,hold,titles,rows=read(args.manifest,args.device)
    args.out.mkdir(exist_ok=False)
    results={}
    for mode in ("before_only","before_after"):
        rng=np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        model=HorizontalDecoder(len(titles)).to(args.device)
        opt=torch.optim.AdamW(model.parameters(),lr=.0007)
        history=[]
        for step in range(1,args.steps+1):
            parts=[]
            for title in titles:
                for _ in range(2):
                    group=fit[title][rng.integers(len(fit[title]))]
                    ix=torch.as_tensor(rng.integers(len(group[0]),size=8),device=args.device)
                    parts.append(tuple(values[ix] for values in group))
            img,feat,nxt=map(torch.cat,zip(*parts))
            pred=model(img,img if mode=="before_only" else nxt,feat[:,4:])
            loss=F.cross_entropy(pred,feat[:,0].long()+1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step==1 or step%50==0 or step==args.steps:
                history.append(dict(step=step,nll=loss.item()))
        ckpt=args.out/(mode+".pt")
        torch.save(model.state_dict(),ckpt)
        scores=[]
        model.eval()
        with torch.inference_mode():
            for (title,lane,ep),(img,feat,nxt) in hold:
                y=feat[:,0].long()+1
                pred=model(img,img if mode=="before_only" else nxt,feat[:,4:])
                confusion=torch.bincount(y*3+pred.argmax(1),minlength=9).reshape(3,3)
                scores.append(dict(title=title,lane=lane,episode=ep,n=len(y),confusion=confusion.tolist(),
                    accuracy=float((pred.argmax(1)==y).float().mean()),nll=F.cross_entropy(pred,y).item()))
        results[mode]=dict(sha256=sha256(ckpt),history=history,held_episodes=scores)
    report=dict(task="Decode randomized dx using the observed start or start and successor; no expert labels or returns.",
                classes=["left","neutral","right"],seed=args.seed,steps=args.steps,
                split_sha256=sha256(args.manifest),results=results)
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
