"""Compare original full frames with the actor's 6x7 image grid on a native player-location diagnostic."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def read(folder):
    with np.load(folder/'original-pixels.npz') as z:
        frames=z['obs'][:-1]
    with np.load(folder/'behavior.npz') as z:
        times=z['action_utc_ms']; features=z['features']
    with gzip.open(folder/'native.csv.gz','rt') as f:
        native=np.genfromtxt(f,names=True,delimiter=',')
    stamp=native['utc_ms']
    right=np.searchsorted(stamp,times).clip(0,len(stamp)-1);left=(right-1).clip(0)
    near=np.where(abs(stamp[right]-times)<abs(stamp[left]-times),right,left)
    valid=(abs(stamp[near]-times)<=40)&(native['lives'][near]>=0)
    for when in stamp[1:][np.diff(native['lives'])!=0]:
        valid&=abs(times-when)>650
    pos=np.stack([(native['x'][near]+192)/4,native['y'][near]/4],axis=1)
    valid&=(pos[:,0]>=0)&(pos[:,0]<96)&(pos[:,1]>=0)&(pos[:,1]<112)
    prior=np.concatenate([frames[:1],frames[:-1]])
    raw=np.stack([frames,np.abs(frames.astype(np.int16)-prior).astype(np.uint8)],axis=1)
    return raw[valid],features[valid],pos[valid].astype(np.float32),abs(stamp[near]-times)[valid]


class SpatialProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(2,16,5,stride=2,padding=2),nn.GroupNorm(4,16),nn.SiLU(),
                                  nn.Conv2d(16,24,5,padding=2),nn.GroupNorm(4,24),nn.SiLU(),
                                  nn.Conv2d(24,24,5,dilation=2,padding=4),nn.GroupNorm(4,24),nn.SiLU(),
                                  nn.Conv2d(24,1,1))

    def forward(self,x):return self.layers(x).flatten(1)


def metrics(pred,target):
    distance=np.linalg.norm(pred-target,axis=1)*4
    return dict(n=len(pred),median_original_pixels=float(np.median(distance)),mean_original_pixels=float(distance.mean()),
                within_16_original_pixels=float(np.mean(distance<=16)),within_32_original_pixels=float(np.mean(distance<=32)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,nargs='+',required=True);p.add_argument('--test',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=500);args=p.parse_args()
    if set(args.train)&set(args.test):raise ValueError('A credit appears in both splits')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    train=tuple(np.concatenate(a) for a in zip(*[read(d) for d in args.train]))
    test=tuple(np.concatenate(a) for a in zip(*[read(d) for d in args.test]))
    output=dict(train_credits=[d.name for d in args.train],evaluation_credits=[d.name for d in args.test],
                training_examples=len(train[0]),held_examples=len(test[0]),max_alignment_ms=float(test[3].max()),metrics={})
    output['metrics']['training_median_position']=metrics(np.broadcast_to(np.median(train[2],axis=0),test[2].shape),test[2])
    x=np.column_stack([train[1],np.ones(len(train[1]))]);te=np.column_stack([test[1],np.ones(len(test[1]))])
    w=np.linalg.solve(x.T@x+np.eye(x.shape[1])*.25,x.T@train[2])
    output['metrics']['full_actor_feature_linear_probe']=metrics(te@w,test[2])
    args.out.mkdir(parents=True,exist_ok=False)
    for representation in ('pooled_grid','full_frame'):
        torch.manual_seed(6183)
        def make(data):
            if representation=='full_frame':return torch.from_numpy(data[0]).to(device).float()/255
            grid=torch.from_numpy(data[1][:,:84].copy()).to(device).reshape(-1,2,7,6)
            return F.interpolate(grid,size=(112,96),mode='bilinear',align_corners=False)
        tx,held=make(train),make(test)
        xy=torch.from_numpy(train[2]).to(device)
        model=SpatialProbe().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        yy,xx=torch.meshgrid(torch.arange(56,device=device),torch.arange(48,device=device),indexing='ij')
        for _ in range(args.steps):
            ids=torch.randint(len(xy),(32,),device=device);px=tx[ids];target=xy[ids].clone()/2
            flip=torch.rand(32,device=device)<.5;px[flip]=px[flip].flip(3)
            target[flip,0]=47.5-target[flip,0]
            d=(xx-target[:,0,None,None])**2+(yy-target[:,1,None,None])**2
            soft=(-d.flatten(1)/2).softmax(1)
            loss=-(soft*model(px).log_softmax(1)).sum(1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        with torch.no_grad():index=torch.cat([model(b).argmax(1).cpu() for b in held.split(128)]).numpy()
        pred=np.column_stack((index%48,index//48))*2
        output['metrics'][representation+'_cnn']=metrics(pred,test[2])
        torch.save(model.cpu().state_dict(),args.out/(representation+'.pt'))
    (args.out/'report.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
