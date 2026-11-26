"""Hindsight goal-conditioned human horizontal movement in stock TH14 late Stage 1."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.distributions import MultiCategoricalDistribution

from scripts import evaluate_full_credit as credit
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm, load_arm, output_lock
from scripts.learn_stage1_boss import (TRAIN,HOLD,SPLICE,TEMPLATE,PPO211,PPO397,HASHES,REPO,
                                      evidence,prepare,crossover_arms,predict_heads,ACTION_WIDTHS)

HORIZONS=(4,8,16,32)
NAME="human_waypoint_horizontal"


class Horizontal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(21,48),nn.Tanh(),nn.Linear(48,3))

    def forward(self,ctx,prior,goal):
        prior_horiz=prior[:,:9].log_softmax(1).view(-1,3,3).logsumexp(1)
        # Exclude the real boss X and the fixed stage clock. Supply raw or relabeled goal in its place.
        features=torch.cat((ctx[:,:6],ctx[:,7:10],ctx[:,-12:-3],prior_horiz),1)
        features=features.clone()
        # Preserve all physical context except absolute power; put the horizontal goal in that slot.
        features[:,5]=(goal/96).clamp(-3,3)
        return prior_horiz + self.net(features)


def pairs(stem,cutoff,base,sft,train=False):
    row=prepare(stem,cutoff,base,sft,train)
    path,_,frame,hp,_=evidence(stem,train)
    with np.load(path) as archive:
        ctx=archive["context"]
        valid=(frame>cutoff)&hp
        indexes=np.flatnonzero(valid)
        x=ctx[:,0]*192
        starts=[]
        delta=[]
        for h in HORIZONS:
            i=indexes[indexes+h<len(x)]
            # Disallow jumping over clocks, respawns or non-boss observations.
            okay=(frame[i+h]-frame[i]==4*h) & (ctx[i,3]==ctx[i+h,3])
            prefix=np.r_[0,np.cumsum(~valid)]
            okay &= (prefix[i+h+1]-prefix[i])==0
            i=i[okay]
            starts.extend(np.searchsorted(indexes,i).tolist())
            delta.extend((x[i+h]-x[i]).tolist())
        # Native displacement is independent confirmation of progress or movement the other way.
        ids=indexes[np.minimum(indexes+1,len(x)-1)==indexes+1]
        intact=(frame[ids+1]-frame[ids]==4)&valid[ids+1]&(ctx[ids,3]==ctx[ids+1,3])
        next_dx=np.full(len(indexes),np.nan,dtype=np.float32)
        next_dx[np.searchsorted(indexes,ids[intact])]=x[ids[intact]+1]-x[ids[intact]]
    row.update(hidx=torch.tensor(starts),future_dx=torch.tensor(delta,dtype=torch.float32),next_dx=torch.from_numpy(next_dx))
    return row


def groups(goals):
    return torch.where(goals.abs()<8,0,torch.where(goals.abs()<32,1,2))*2+(goals>=0).long()


def training_rows(rows,offset):
    inputs=[[] for _ in range(5)]
    for row in rows:
        n=len(row["context"])
        for ix,goal,mass in [(torch.arange(n),row["context"][:,6]*384+offset,.25),
                             (row["hidx"],row["future_dx"],.75)]:
            cat=groups(goal)
            present=cat.unique()
            weights=torch.zeros_like(goal)
            for c in present:
                flag=cat==c
                weights[flag]=mass/len(rows)/len(present)/int(flag.sum())
            target=row["targets"][ix,:9].reshape(-1,3,3).sum(1)
            for dest,item in zip(inputs,(row["context"][ix],row["prior"][ix],goal,target,weights)):
                dest.append(item)
    return [torch.cat(g) for g in inputs]


def horizontal(logits):
    return logits[:,:9].softmax(1).reshape(-1,3,3).sum(1)


@torch.inference_mode()
def score(row,model,offset):
    ans={}
    for name,ix,goal in (("boss",torch.arange(len(row["context"])),row["context"][:,6]*384+offset),
                         ("hindsight",row["hidx"],row["future_dx"])):
        y=row["targets"][ix,:9].reshape(-1,3,3).sum(1)
        original=horizontal(row["original"][ix])
        student=model(row["context"][ix],row["prior"][ix],goal).softmax(1)
        observed_step=row["next_dx"][ix]*goal.sign()
        wide=goal.abs()>=32
        in_band=goal.abs()<8
        keep=wide&observed_step.isfinite()
        result=dict(all=len(ix),wide=int(wide.sum()),near=int(in_band.sum()),
                    distinct_wide_observations=int(ix[wide].unique().numel()),
                    observed_progress=int((keep&(observed_step>.05)).sum()),
                    observed_retreat=int((keep&(observed_step<-.05)).sum()),
                    observed_no_progress=int((keep&(observed_step.abs()<=.05)).sum()))
        toward=torch.where(goal>0,2,0)
        away=2-toward
        for key,p in (("composite",original),("candidate",student),("human",y)):
            def directional(flag,idx):
                return float(p[flag,idx[flag]].mean()) if flag.any() else None
            result[key]=dict(wide_toward=directional(wide,toward),wide_wrong=directional(wide,away),
                             near_move=float(1-p[in_band,1].mean()) if in_band.any() else None,
                             toward_on_observed_progress=directional(keep&(observed_step>.05),toward),
                             toward_on_observed_retreat=directional(keep&(observed_step<-.05),toward))
            if key!="human":
                result[key]["horizontal_ce"]=float(-(y*p.clamp_min(1e-10).log()).sum(1).mean())
        ans[name]=result
    return ans


def qualifies(result):
    natural=result["boss"]
    for k,delta in (("boss",.05),("hindsight",.10)):
        p=result[k]["composite"]
        c=result[k]["candidate"]
        if c["wide_toward"]<p["wide_toward"]+delta or c["wide_wrong"]>p["wide_wrong"]+.02:
            return False
    return natural["candidate"]["near_move"]<=natural["human"]["near_move"]+.10


def fit(args):
    supports=[evidence(n,True)[4] for n in TRAIN]
    cutoff=(max(r[0][1] for r in supports)+min(r[1][0] for r in supports))//2
    arms=crossover_arms(args)
    base,sft=arms["ppo397"].policy,load_arm(TEMPLATE).policy
    train=[pairs(n,cutoff,base,sft,True) for n in TRAIN]
    # Approximate the human firing lane by the median signed player-minus-boss displacement.
    offset=float(torch.cat([-r["context"][:,6]*384 for r in train]).median())
    ctx,prior,goal,y,w=training_rows(train,offset)
    torch.manual_seed(397)
    net=Horizontal()
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
    for _ in range(180):
        logits=net(ctx,prior,goal)
        loss=-(w[:,None]*y*logits.log_softmax(1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    hold=[pairs(n,cutoff,base,sft) for n in HOLD]
    scores={r["meta"]["replay"]:score(r,net,offset) for r in hold}
    args.out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(model=net.state_dict(),offset=offset,cutoff=cutoff,ppo_hashes=HASHES,
                    sft_sha256=digest(TEMPLATE)),args.out/"waypoint.pt")
    rep=dict(passed_offline_gate=all(qualifies(s) for s in scores.values()),
             gate="per held-out player: wide approaching probability gain >=.10 on natural future waypoints and >=.05 on the boss; wrong-direction probability increase <=.02 for each; near-boss horizontal activity <= human +.10",
             scores=scores,training=dict(whole_credits=TRAIN,actual_late_boss_rows=sum(len(r["context"]) for r in train),
             hindsight_windows=sum(len(r["hidx"]) for r in train),horizons_decisions=HORIZONS,
             large_hindsight_windows=sum(int((r["future_dx"].abs()>=32).sum()) for r in train),offset=offset,cutoff=cutoff,
             objective="75% hindsight + 25% actual boss goals. Equal credit, sign and distance-bin weights within each source. Four-tick human horizontal cross-entropy. 180 full-batch AdamW steps, lr .003, weight decay .01, seed 397."),
             assumptions="Future human positions indicate reachable goals under demonstrated actions. Conditioning on hindsight does not demonstrate counterfactual safety or on-policy wall recovery. Overlapping windows are not independent.",
             deployment="On positive boss HP beyond training-derived frame cutoff: PPO397 vertical movement marginal and focus/shoot, learned horizontal marginal, PPO211 Bomb. All earlier states retain exact crossover.",
             data=[r["meta"] for r in train+hold],run_sha256=digest(Path(__file__)),candidate_sha256=digest(args.out/"waypoint.pt"),
             checkpoint_sources=HASHES,sealed_test_opened=False,native_used=False)
    (args.out/"results.json").write_text(json.dumps(rep,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({k:rep[k] for k in ("passed_offline_gate","training","scores")}))


class WaypointPolicy:
    def __init__(self,composite,sft,weights):
        if weights["ppo_hashes"]!=HASHES or weights["sft_sha256"]!=digest(TEMPLATE):
            raise ValueError("Mismatched source weights.")
        self.base,self.sft=composite,sft
        self.model=Horizontal().eval()
        self.model.load_state_dict(weights["model"])
        self.offset,self.cutoff=weights["offset"],weights["cutoff"]
        for key in ("motion_spans","stage_clock","retain_recent_motion"):
            setattr(self,key,getattr(composite,key))

    @torch.inference_mode()
    def predict(self,obs,deterministic=False):
        c=obs["context"]
        if c[8]<=0 or c[-2]*14400<=self.cutoff:
            return self.base.predict(obs,deterministic=deterministic)
        tensors,_=self.base.actor.obs_to_tensor(obs)
        t,b=tensors["context"],tensors["bullets"]
        old,focus,shoot,_=predict_heads(self.base.actor,b,t)
        bomb=predict_heads(self.base.bomb,b,t)[3]
        prior=torch.cat(predict_heads(self.sft,b,t)[:3],1)
        h=self.model(t,prior,t[:,6]*384+self.offset).log_softmax(1)
        v=old.log_softmax(1).view(-1,3,3).logsumexp(2)
        direction=(v[:,:,None]+h[:,None,:]).reshape(-1,9)
        logits=torch.cat((direction,focus,shoot,bomb),1)
        d=MultiCategoricalDistribution(list(ACTION_WIDTHS)).proba_distribution(logits)
        return d.get_actions(deterministic=deterministic).cpu().numpy()[0],None


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["fit","evaluate"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--waypoint",type=Path)
    parser.add_argument("--seeds",nargs="+",type=int,default=[1751,1752,1753,1754])
    parser.add_argument("--limit",type=int,default=3300)
    parser.add_argument("--start-frame",type=int,default=30)
    parser.add_argument("--focus-wait",type=int,default=30)
    parser.add_argument("--parakit",type=Path,default=REPO.parent/"parakit")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    args.ppo211,args.ppo397,args.ppo_template=PPO211,PPO397,TEMPLATE
    torch.set_num_threads(2 if args.mode=="fit" else 1)
    if args.mode=="fit":
        return fit(args)
    if not args.waypoint or len(args.seeds)!=len(set(args.seeds)) or any(not 0<=s<2**63 for s in args.seeds):
        parser.error("Supply a waypoint checkpoint and distinct valid sampling seeds.")
    orig=crossover_arms(args)[SPLICE]
    weights=torch.load(args.waypoint,map_location="cpu",weights_only=True)
    candidate=WaypointPolicy(orig.policy,load_arm(TEMPLATE).policy,weights)
    arms={SPLICE:orig,NAME:Arm(candidate,orig.history,dict(kind="hindsight_horizontal",checkpoint_sha256=digest(args.waypoint),
             source_sha256=HASHES,sft_sha256=digest(TEMPLATE),clock_cutoff=weights["cutoff"],goal_offset=weights["offset"]))}
    spec=credit.run_specification(args,arms)
    spec["controller"]["kind"]="pure_frozen_hindsight_waypoint_comparison"
    spec["runtime"]["source_sha256"]["scripts.learn_boss_waypoint"]=digest(Path(__file__))
    if not args.resume:
        args.out.mkdir(parents=True,exist_ok=False)
    with output_lock(args.out):
        credit.run_evaluation(args,arms,args.out/"report.json",spec)


if __name__=="__main__":
    main()
