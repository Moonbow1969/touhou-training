"""Fit a shared movement and focus policy to original demonstrations using fields observable in TH10."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from scripts import pretrain_policy as bc
from scripts.checkpoint_storage import digest
from scripts.train_structured import BulletSetEncoder


POLICY_KWARGS=dict(features_extractor_class=BulletSetEncoder,net_arch=dict(pi=[64,64],vf=[64,64]))
OBS=gym.spaces.Dict({"bullets":gym.spaces.Box(-1,1,(96,7),np.float32),
                     "context":gym.spaces.Box(-1,1,(10,),np.float32)})
ACT=gym.spaces.MultiDiscrete([9,2])


def shared_live(obs,motion_sign=False):
    context=obs["context"].copy()
    context[[3,6,9]]=0
    recent=context[7:9]*32
    context[7:9]=np.where(np.abs(recent)>.25,np.sign(recent),0) if motion_sign else 0
    return {"bullets":obs["bullets"],"context":context}


def load(rows, data, device, motion_sign):
    groups=defaultdict(list)
    for row in rows:
        path=data/row["file"]
        if digest(path)!=row["sha256"]:
            raise ValueError("An original paired replay shard changed.")
        with np.load(path,allow_pickle=False) as z:
            b=z["bullets"].copy()
            b[:,:,4]=0
            b[:,:,5]=b[:,:,6]
            c=z["context"]
            live=np.zeros_like(c)
            live[:,:3]=c[:,[0,1,3]]
            live[:,4]=z["bullet_count"].clip(0,512)/512
            live[:,5]=c[:,2]
            if motion_sign:
                delta=np.diff(c[:,:2],axis=0)*np.array([192,448],np.float32)
                valid=(np.diff(z["stage_frame"])==4)&(np.diff(c[:,3])==0)
                live[1:,7:9]=np.where(valid[:,None]&(np.abs(delta)>.25),np.sign(delta),0)
            y=bc.label_actions(z["expert_action_chunk"])[:,:2]
            soft=bc.movement_distribution(z["expert_action_chunk"])
            groups[row["title"]].append((b,live,y,soft))
    return {t:tuple(torch.as_tensor(np.concatenate(v),device=device) for v in zip(*items))
            for t,items in groups.items()}


def heads(model, b, c):
    feat=model.extract_features({"bullets":b,"context":c})
    return model.action_net(model.mlp_extractor.forward_actor(feat)).split([9,2],dim=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=2400)
    p.add_argument("--seed",type=int,default=613)
    p.add_argument("--motion-sign",action="store_true",help="Include the observed direction of the last displacement.")
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_bytes())
    fit=manifest["train_manifest"]
    hold=manifest["evaluation_manifest"]
    if set(r["player_group"] for r in fit)&set(r["player_group"] for r in hold):
        raise ValueError("Held player is present during training.")
    train=load(fit,args.data,"cuda",args.motion_sign); test=load(hold,args.data,"cuda",args.motion_sign)
    if set(train)!={13,14,17} or set(test)!=set(train):
        raise ValueError("Expected the fixed three-title replay cohort.")
    args.out.mkdir(exist_ok=False)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    model=MultiInputActorCriticPolicy(OBS,ACT,lambda _: .0003,**POLICY_KWARGS).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0003)
    history=[]
    for step in range(1,args.steps+1):
        groups=[]
        for t,n in zip(sorted(train),rng.multinomial(256,[1/3]*3)):
            idx=torch.randint(len(train[t][0]),(int(n),),device="cuda")
            groups.append([v[idx] for v in train[t]])
        b,c,y,soft=map(torch.cat,zip(*groups))
        move,focus=heads(model,b,c)
        loss=-(soft*move.log_softmax(1)).sum(1).mean()+F.cross_entropy(focus,y[:,1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step==1 or step%200==0:
            history.append(dict(step=step,loss=loss.item()))
    weights=args.out/"policy.pt"
    torch.save(model.state_dict(),weights)
    held={}
    model.eval()
    with torch.inference_mode():
        for t,(b,c,truth,_) in test.items():
            blocks=[heads(model,b[k:k+256],c[k:k+256]) for k in range(0,len(b),256)]
            m,f=map(torch.cat,zip(*blocks))
            held[t]=dict(n=len(b),move_nll=F.cross_entropy(m,truth[:,0]).item(),
                         moving_accuracy=(m.argmax(1)[truth[:,0]!=4]==truth[truth[:,0]!=4,0]).float().mean().item(),
                         focus_accuracy=(f.argmax(1)==truth[:,1]).float().mean().item())
    report=dict(train_titles=sorted(train),observed_titles_exclude=[10],steps=args.steps,seed=args.seed,
                observed_motion_sign=args.motion_sign,
                manifest_sha256=digest(args.manifest),weights_sha256=digest(weights),train_draws=args.steps*256,
                samples={t:len(v[0]) for t,v in train.items()},history=history,held=held,
                observations="Player position, focus, spare lives, active projectile count, relative XY and velocity. Fields unavailable in the TH10 reader are masked on both sides.",
                objective="Human movement distribution and expert focus cross-entropy. Fixed Shoot on, Bomb off in the downstream game.")
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
