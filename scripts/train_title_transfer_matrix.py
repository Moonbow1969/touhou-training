"""Train equal-budget title ablations on grouped original replay states and score fixed external players."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import socket
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from scripts import pretrain_policy as bc
from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


TITLES=(13,14,17)
# These complete replays are reserved for every model in the matrix. TH17
# already has a test split. TH13 and TH14 also reserve a six-stage player.
RESERVED={"th13_ud0bf1.rpy","th14_ud05e0.rpy","th17_ud040c.rpy","th17_ud0448.rpy"}


def groups(path):
    source=bc.index_split(path,"train")+bc.index_split(path,"test")
    people={m["player_group"] for _,m in source if m["replay"] in RESERVED}
    testing=[s for s in source if s[1]["replay"] in RESERVED]
    training=[s for s in bc.index_split(path,"train") if s[1]["player_group"] not in people]
    if {m["replay"] for _,m in testing}!=RESERVED or {m["title"] for _,m in training}!=set(TITLES):
        raise ValueError("Incomplete sealed transfer matrix.")
    if {digest(f) for f,_ in training} & {digest(f) for f,_ in testing}:
        raise ValueError("Duplicate native state data across training and test.")
    return training,testing


def tensorize(records,device):
    by_title=defaultdict(list)
    for shard in records:
        by_title[shard[1]["title"]].append(bc.read_shard(shard,motion_history=True,soft_movement=True,
                                                        motion_spans=(4,12,36),stage_clock=False))
    return {t:tuple(torch.from_numpy(np.concatenate(cols)).to(device) for cols in zip(*parts))
            for t,parts in by_title.items()}


def switch_indices(records, device):
    """Index changes within each contiguous recorded stage, using training labels only."""
    masks=defaultdict(list)
    for path,meta in records:
        with np.load(path,allow_pickle=False) as data:
            moves=bc.label_actions(data["expert_action_chunk"])[:,0]
            adjacent=np.diff(data["stage_frame"])==4
            masks[meta["title"]].append(np.r_[False,adjacent & (moves[1:]!=moves[:-1])])
    return {title:torch.as_tensor(np.flatnonzero(np.concatenate(parts)),device=device)
            for title,parts in masks.items()}


@torch.inference_mode()
def evaluate(policy,fields):
    b,c,truth,_=fields
    actual=[]; loss=0.
    for i in range(0,len(b),512):
        logits=bc.predict_heads(policy,b[i:i+512],c[i:i+512])
        actual.append(torch.stack([h.argmax(1) for h in logits],1))
        loss+=float(F.cross_entropy(logits[0],truth[i:i+512,0],reduction="sum"))
    pred=torch.cat(actual)
    moving=truth[:,0]!=4
    starts=moving&(pred[:,0]!=4)
    return dict(n=len(b),moving=int(moving.sum()),moving_recall=float(starts.sum()/moving.sum()),
                correct_moving_direction=float((pred[moving,0]==truth[moving,0]).float().mean()),
                move_nll=loss/len(b),idle_prediction=float((pred[:,0]==4).float().mean()),
                head_accuracy=(pred==truth).float().mean(0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[613,971])
    p.add_argument("--steps",type=int,default=2400)
    p.add_argument("--switch-share",type=float,default=0.,
                   help="Share of training draws sampled from expert direction changes; the remainder is uniform.")
    p.add_argument("--joint-only",action="store_true",help="Fit only the three-title condition.")
    p.add_argument("--history-dropout",type=float,default=0.,
                   help="Training-only probability of jointly masking all displacement history and availability flags.")
    args=p.parse_args()
    if shutil.disk_usage(args.out.parent).free<8*1024**3 or args.steps<1 or not 0<=args.switch_share<=1 or not 0<=args.history_dropout<=1:
        p.error("Supply a fresh output on a volume with an 8 GiB reserve and a positive budget.")
    args.out.mkdir(exist_ok=False)
    fit,hold=groups(args.data)
    train=tensorize(fit,"cuda"); test=tensorize(hold,"cuda")
    switches=switch_indices(fit,"cuda") if args.switch_share else {}
    manifest=lambda rows:[dict(file=f.name,sha256=digest(f),**meta) for f,meta in rows]
    state=dict(status="running",host=socket.gethostname(),gpu=torch.cuda.get_device_name(),steps_per_arm=args.steps,
        batch=256,encoder_width=256,motion_history=[4,12,36],train_manifest=manifest(fit),
        evaluation_manifest=manifest(hold),selection="final fixed step, independent players shared across arms",
        interpretation="Offline original-replay action prediction. No game clear or on-policy return is measured.",
        form="Equal-probability training titles and uniform decisions within a title. All arms start from scratch.",
        train_samples={t:len(v[0]) for t,v in train.items()},test_samples={t:len(v[0]) for t,v in test.items()},
        seeds=args.seeds,arms=[])
    state["switch_share"]=args.switch_share
    state["history_dropout"]=args.history_dropout
    state["switch_pool_counts"]={t:len(ix) for t,ix in switches.items()}
    if args.switch_share:
        state["form"]="Equal title probability. Within each title mix uniform training transitions with uniform true movement changes."
    path=args.out/"report.json"
    publish_json(path,state)
    for seed in args.seeds:
        for exclude in ((None,) if args.joint_only else (None,14,13,17)):
            titles=[t for t in TITLES if t!=exclude]
            label=f"seed-{seed}-"+("all" if exclude is None else f"without-{exclude}")
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            mask_rng=torch.Generator(device="cuda").manual_seed(8521+seed)
            model=bc.MovementPolicy(bc.observation_space(True),bc.action_space(),lambda _: .0003,
                                    **bc.network_kwargs(256)).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
            start=perf_counter()
            row=dict(label=label,train_titles=titles,history=[],source_initialized="random")
            state["arms"].append(row)
            for step in range(1,args.steps+1):
                # Equal title probability per row. Repeat counts are recorded separately from unique data.
                counts=rng.multinomial(256,[1/len(titles)]*len(titles))
                parts=[]
                for t,n in zip(titles,counts):
                    ix=torch.randint(len(train[t][0]),(int(n),),device="cuda")
                    if args.switch_share:
                        n_switch=rng.binomial(n,args.switch_share)
                        ix[:n_switch]=switches[t][torch.randint(len(switches[t]),(n_switch,),device="cuda")]
                    parts.append([f[ix] for f in train[t]])
                b,c,y,soft=(torch.cat(x) for x in zip(*parts))
                if args.history_dropout:
                    drop=torch.rand(len(c),device="cuda",generator=mask_rng)<args.history_dropout
                    c[drop,-9:]=0.
                heads=bc.predict_heads(model,b,c)
                loss=-(soft*heads[0].log_softmax(1)).sum(1).mean()
                loss+=sum(F.cross_entropy(h,y[:,k]) for k,h in enumerate(heads[1:],1))
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                if step==1 or step%200==0 or step==args.steps:
                    row["history"].append(dict(step=step,draws=step*256,loss=float(loss.detach()),seconds=perf_counter()-start))
                    publish_json(path,state)
            model.eval()
            dest=args.out/(label+".pt")
            torch.save(model.state_dict(),dest)
            row.update(weights=dest.name,sha256=digest(dest),scores={t:evaluate(model,v) for t,v in test.items()})
            publish_json(path,state)
    state["status"]="complete"
    publish_json(path,state)


if __name__=="__main__":
    main()
