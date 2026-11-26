"""Fit a small spatial player detector from real TH10 pixel and position pairs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.probe_full_action_player_vision import read,metrics


class TinyLocator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(2,12,7,stride=2,padding=3)
        self.spatial=nn.Conv2d(12,12,5,padding=4,dilation=2,groups=12)
        self.readout=nn.Conv2d(12,1,1)

    def forward(self,x):return self.readout(F.relu(self.spatial(F.relu(self.stem(x))))).flatten(1)


def targets(f):
    with np.load(f) as z:
        obs=z['obs']; idx=z['past_native_index']; xy=z['player_xy'][idx].copy()
        good=z['valid'].astype(bool)&(z['lives'][idx]>=0)&(xy[:,1]>=0)&(xy[:,1]<448)&(abs(xy[:,0])<192)
        raw=np.stack([obs,np.abs(obs.astype(np.int16)-np.concatenate([obs[:1],obs[:-1]])).astype(np.uint8)],1)
        xy[:,0]+=192
        return raw[good],xy[good]/4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',nargs='+',type=Path,required=True);p.add_argument('--dev',nargs='+',type=Path,required=True)
    p.add_argument('--independent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(6194)
    torch.set_num_threads(2)
    def assemble(folders):
        rows=[read(f) for f in folders]
        return np.concatenate([r[0] for r in rows]),np.concatenate([r[2] for r in rows])
    train=assemble(args.train);held=assemble(args.dev);ind=targets(args.independent)
    x=torch.from_numpy(train[0]).to(device);xy=torch.from_numpy(train[1]).to(device)
    model=TinyLocator().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    yy,xx=torch.meshgrid(torch.arange(56,device=device),torch.arange(48,device=device),indexing='ij')
    for step in range(1600):
        ids=torch.randint(len(x),(40,),device=device);px=x[ids].float()/255;target=xy[ids].clone()/2
        flip=torch.rand(len(ids),device=device)<.5;px[flip]=px[flip].flip(3);target[flip,0]=47.5-target[flip,0]
        gain=.80+.4*torch.rand(len(ids),1,1,1,device=device)
        px*=gain;px[:,0]+=(torch.rand(len(ids),1,1,device=device)-.5)*.08
        d=(xx-target[:,0,None,None])**2+(yy-target[:,1,None,None])**2
        loss=-((-d.flatten(1)/1.4).softmax(1)*model(px).log_softmax(1)).sum(1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    args.out.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        def test(data):
            guess=torch.cat([model(v.to(device).float()/255).argmax(1).cpu() for v in torch.from_numpy(data[0]).split(128)]).numpy()
            return metrics(np.column_stack([guess%48,guess//48])*2,data[1])
        summary={name:test(data) for name,data in [('dev',held),('independent',ind)]}
        # Record the independent inference input needed to check the C implementation.
        pair=ind[0][17];pair.tofile(args.out/'parity-pair.u8')
        index=int(model(torch.from_numpy(pair[None]).to(device).float()/255).argmax(1)[0])
    torch.save(model.cpu().state_dict(),args.out/'detector.pt')
    def emit(name,value):
        floats=','.join(format(float(v),'.9e')+'f' for v in value.detach().flatten())
        return 'static const float '+name+'[] = {'+floats+'};\n'
    params=[('loc_stem_w',model.stem.weight),('loc_stem_b',model.stem.bias),
            ('loc_spatial_w',model.spatial.weight),('loc_spatial_b',model.spatial.bias),
            ('loc_out_w',model.readout.weight),('loc_out_b',model.readout.bias)]
    (args.out/'locator_params.h').write_text('/* Fitted weights for the image-only native localization diagnostic. */\n'+''.join(emit(k,v) for k,v in params))
    report=dict(fit_rows=len(x),training_credits=[f.name for f in args.train],dev_credits=[f.name for f in args.dev],
                independent_credit=args.independent.parent.name,metrics=summary,parity_index=index,steps=1600,
                inputs='96x112 original grayscale and absolute preceding-frame difference. No live game state required at inference.')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
