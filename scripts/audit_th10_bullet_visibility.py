"""Measure pixel-only projectile localization against original-game observations, split by credit."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def credit(path):
    cameras=[]; labels=[]; targets=[]
    with np.load(path) as z:
        valid=z["valid"];index=z["past_native_index"];count=z["bullet_count"];pixel=z["obs"]
        player=z["player_xy"];bullets=z["bullets"]
    for step in np.flatnonzero(valid):
        k=index[step]
        xy=bullets[k,:count[k],:2]+player[k]+np.array([192,0],np.float32)
        xy=xy[(xy[:,0]>=0)&(xy[:,0]<384)&(xy[:,1]>=0)&(xy[:,1]<448)]
        grid=(xy/8).astype(int)
        target=np.zeros((56,48),np.uint8)
        if len(grid):target[grid[:,1],grid[:,0]]=1
        cameras.append(np.stack([pixel[step],pixel[max(0,step-1)]]))
        labels.append(target);targets.append(xy)
    return np.stack(cameras),np.stack(labels),targets


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,12,5,2,2),nn.ReLU(),
                                nn.Conv2d(12,16,3,padding=1),nn.ReLU(),
                                nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))

    def forward(self,cameras):
        x=cameras.float()/255
        return self.conv(torch.cat([x,abs(x[:,:1]-x[:,1:])],1))[:,0]


def candidates(maps):
    kernel=np.ones((3,3),np.uint8)
    maps=np.ascontiguousarray(maps,dtype=np.float32)
    out=[]
    for probs in maps:
        y,x=np.nonzero((probs==cv2.dilate(probs,kernel))&(probs>0))
        v=probs[y,x]
        order=np.argsort(-v,kind="stable")[:96]
        out.append((v[order],np.stack([x[order]*8+4,y[order]*8+4],axis=1)))
    return out


def score(preds,truths,threshold,tolerance=12):
    tp=fp=fn=0
    for (values,coords),truth in zip(preds,truths):
        coords=coords[values>=threshold]
        counts=0
        if len(truth) and len(coords):
            dist=np.square(coords[:,None,:]-truth[None,:,:]).sum(2)
            for row in dist:
                j=row.argmin()
                if row[j]<=tolerance**2:
                    counts+=1;dist[:,j]=np.inf
        tp+=counts;fp+=len(coords)-counts;fn+=len(truth)-counts
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn))


def best_threshold(maps,truth):
    preds=candidates(maps)
    positive=np.concatenate([v for v,c in preds])
    qs=np.quantile(positive,[0,.25,.5,.65,.75,.85,.9,.94,.96,.97,.98,.99,.995,1])
    rows=[(score(preds,truth,t),float(t)) for t in sorted(set(qs))]
    return max(rows,key=lambda pair:pair[0]["f1"])


def run(args):
    torch.set_num_threads(1);torch.manual_seed(10692);rng=np.random.default_rng(10692)
    seeds=list(range(710201,710209))
    inputs={s:credit(next(args.data.glob(f"*-{s}/privileged-targets.npz"))) for s in seeds}
    split={"train":seeds[:5],"validation":seeds[5:6],"test":seeds[6:]}
    def pack(group,skip):
        rows=[inputs[s] for s in split[group]]
        return (torch.from_numpy(np.concatenate([x[::skip] for x,y,t in rows])),
                torch.from_numpy(np.concatenate([y[::skip] for x,y,t in rows])),
                [p for x,y,ts in rows for p in ts[::skip]])
    x,y,_=pack("train",2);vx,vy,vt=pack("validation",1)
    model=Detector();opt=torch.optim.AdamW(model.parameters(),lr=.002)
    best=None;history=[]
    @torch.no_grad()
    def maps(inputs):
        model.eval()
        return torch.cat([model(batch).sigmoid() for batch in inputs.split(24)]).numpy()
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0.
        for idx in np.array_split(rng.permutation(len(x)),(len(x)+23)//24):
            logits=model(x[idx]);target=y[idx].float();prob=logits.sigmoid()
            loss=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")*
                  torch.where(target.bool(),(1-prob)**2*14,prob**2)).mean()
            opt.zero_grad();loss.backward();opt.step();loss_sum+=loss.item()*len(idx)
        if epoch%2==0 or epoch==args.epochs:
            q,threshold=best_threshold(maps(vx),vt)
            history.append(dict(epoch=epoch,loss=loss_sum/len(x),validation=q,threshold=threshold))
            if best is None or q["f1"]>best[0]:
                best=(q["f1"],epoch,threshold,{k:v.detach().clone() for k,v in model.state_dict().items()})
            print(epoch,round(q["f1"],4),flush=True)
    model.load_state_dict(best[3])
    prior=cv2.GaussianBlur(y.float().mean(0).numpy(),(5,5),0)
    vm=np.broadcast_to(prior,(len(vx),*prior.shape))
    bg_v,bg_thresh=best_threshold(vm,vt)
    per_seed={};pooled={k:[] for k in ("learned","fixed_spatial_prior","empty")};truth=[]
    local_pooled={k:[] for k in pooled};local_truth=[];examples=[]
    for seed in split["test"]:
        a,_,target=inputs[seed]
        heat=maps(torch.from_numpy(a));p=candidates(heat)
        with np.load(next(args.data.glob(f"*-{seed}/privileged-targets.npz"))) as labels:
            player=labels["player_xy"][labels["past_native_index"][labels["valid"]]]+np.array([192,0])
        local_truth.extend(t[np.linalg.norm(t-pos,axis=1)<=64] for pos,t in zip(player,target))
        base=candidates(np.broadcast_to(prior,(len(a),*prior.shape)));truth+=target
        empty=[(np.array([]),np.empty((0,2))) for _ in target]
        per_seed[seed]={}
        for name,vals,thr in (("learned",p,best[2]),("fixed_spatial_prior",base,bg_thresh),("empty",empty,1)):
            pooled[name]+=vals
            per_seed[seed][name]=score(vals,target,thr)
            for pos,(v,xy) in zip(player,vals):
                keep=np.linalg.norm(xy-pos,axis=1)<=64
                local_pooled[name].append((v[keep],xy[keep]))
        i=next((j for j,xy in enumerate(target) if len(xy)>=10),None)
        if i is not None:
            values,xy=p[i];examples.append((seed,a[i,0],heat[i],target[i],xy[values>=best[2]]))
    test={name:score(vals,truth,thr) for (name,vals),thr in zip(pooled.items(),[best[2],bg_thresh,1])}
    local_test={name:score(vals,local_truth,thr) for (name,vals),thr in zip(local_pooled.items(),[best[2],bg_thresh,1])}
    report=dict(design="All data are PPO2 training-rollout credits. Model selection and probability thresholds use a separate entire credit. Test uses two untouched entire credits, one on each host. Detection matches each predicted local maximum to the nearest unmatched ground-truth bullet within 12 native pixels, at most 96 predictions. Native labels precede the camera observation; timing residual and grayscale downsampling affect localization.",
       split=split,train_subsample=2,train_frames=len(x),validation_frames=len(vx),test_frames=len(truth),
       candidates="3x3 peaks on an 8-native-pixel grid; at most 96 per frame. A fixed prior estimated from training positions and an empty predictor are controls.",
       parameter_count=sum(p.numel() for p in model.parameters()),train_history=history,selected_epoch=best[1],
       learned_threshold=best[2],fixed_spatial_prior_threshold=bg_thresh,validation_prior=bg_v,test=test,by_test_seed=per_seed,
       test_within_64_native_pixels_of_player=local_test,
       radius_metric="Secondary, same frozen thresholds and match tolerance. Include predictions and true bullet centers within 64 native game pixels of the measured player. Ground truth player coordinates used only at scoring time.")
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    torch.save(model.state_dict(),args.out/"weights.pt")
    if examples:draw_examples(examples,args.out/"examples.png")
    print(json.dumps(test,indent=2),flush=True)


def draw_examples(rows,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(rows),3,figsize=(7.8,4.1*len(rows)),squeeze=False,layout="constrained")
    for (seed,camera,heat,truth,pred),ax in zip(rows,axes):
        for a in ax:
            a.imshow(camera,cmap="gray",extent=(0,384,448,0),vmin=0,vmax=255,interpolation="nearest")
            a.set_xlim(0,384);a.set_ylim(448,0);a.set_xticks([]);a.set_yticks([])
        ax[0].set_title(f"Seed {seed}");ax[1].set_title("Detector response");ax[2].set_title("Native and predicted centers")
        ax[1].imshow(heat,extent=(0,384,448,0),vmin=0,vmax=1,cmap="magma",alpha=.65)
        ax[2].scatter(truth[:,0],truth[:,1],s=44,facecolors="none",edgecolors="#ffdc53",linewidths=1.1,label="Native")
        ax[2].scatter(pred[:,0],pred[:,1],s=22,c="#69f3f2",marker="+",linewidths=1.2,label="Predicted")
    axes[-1,2].legend(loc="lower left",fontsize=8,framealpha=.8)
    fig.suptitle("TH10 held credits · first valid frame with at least 10 native bullets")
    fig.savefig(path,dpi=135);plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--epochs",type=int,default=12)
    run(parser.parse_args())
