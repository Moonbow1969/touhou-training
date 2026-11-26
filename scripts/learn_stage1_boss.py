"""Fit or natively evaluate a late Stage 1 human-replay action specialist."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.distributions import MultiCategoricalDistribution

from scripts import evaluate_full_credit as credit
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, HASHES, PPO211, PPO397, REPO, SPLICE, TEMPLATE, crossover_arms
from scripts.evaluate_sft_native import Arm, load_arm, output_lock
from scripts.pretrain_policy import ACTION_WIDTHS, predict_heads, read_shard

TRAIN = ("th14_ud0864","th14_ud05e0","th14_ud0613")
HOLD = ("th14_ud06d6","th14_ud068f")
SNAPSHOT = DATA/"sft-native-data-261123/frozen-training-snapshot"
NAME = "learned_late_boss"


def evidence(stem, train=False):
    path=SNAPSHOT/(stem+"-stage1.npz")
    meta=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
    assert "2" in meta["frames_by_stage"] and meta["difficulty"]=="Normal" and meta["shot"].replace(" ","")=="ReimuA"
    assert not train or meta["split"]=="train"
    with np.load(path) as archive:
        frames=archive["stage_frame"].copy()
        hp=archive["context"][:,8]>0
    runs=[r for r in np.split(np.flatnonzero(hp), np.flatnonzero(np.diff(np.flatnonzero(hp))>1)+1) if len(r)]
    sustained=[r for r in runs if len(r)>=25]
    if len(sustained)<2:
        raise ValueError("Two temporally separated native boss segments were expected.")
    return path,meta,frames,hp,[(int(frames[r[0]]),int(frames[r[-1]])) for r in sustained]


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual=nn.Sequential(nn.Linear(32,48),nn.Tanh(),nn.Linear(48,13))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self,context,prior):
        # Three sets of log probabilities retain the SFT's learned projectile response.
        normalized=torch.cat([p.log_softmax(1) for p in prior.split((9,2,2),1)],1)
        features=torch.cat([context[:,:10], context[:,-12:-3], normalized.clamp_min(-10)/4],1)
        return normalized + self.residual(features)


@torch.inference_mode()
def prepare(stem, cutoff, base, sft, train=False):
    path,meta,frames,hp,phases=evidence(stem,train)
    mask=hp & (frames>cutoff)
    bullets,ctx,labels,movement=read_shard((path,dict(meta,samples=meta["pairs"])),motion_history=True,stage_clock=True,soft_movement=True)
    with np.load(path) as data:
        chunk=data["expert_action_chunk"][mask]
        button=np.stack([((chunk & bit)!=0).mean(1) for bit in (8,1)],1)
    obs={"context":torch.from_numpy(ctx[mask]),"bullets":torch.from_numpy(bullets[mask])}
    orig,prior=[],[]
    for i in range(0,len(obs["context"]),128):
        b,c=obs["bullets"][i:i+128],obs["context"][i:i+128]
        orig.append(torch.cat(predict_heads(base,b,c)[:3],1))
        prior.append(torch.cat(predict_heads(sft,b,c)[:3],1))
    targets=torch.cat((torch.from_numpy(movement[mask]),torch.tensor(button,dtype=torch.float32)),1)
    return {"context":obs["context"],"prior":torch.cat(prior),"original":torch.cat(orig),"targets":targets,
            "meta":{"replay":stem,"player":meta["player"],"split":meta["split"],"paired_sha256":digest(path),
                    "late_boss_decisions":int(mask.sum()),"sustained_hp_positive_frame_ranges":phases,
                    "stage2_exists_in_decoded_header":True,"wall_decisions":int((np.abs(ctx[mask,0])>.9).sum())}}


def losses(logits,y):
    m,f,s=logits.split((9,2,2),1)
    return torch.stack([-(y[:,:9]*m.log_softmax(1)).sum(1),
                        F.binary_cross_entropy_with_logits(f[:,1]-f[:,0],y[:,9],reduction="none"),
                        F.binary_cross_entropy_with_logits(s[:,1]-s[:,0],y[:,10],reduction="none")],1)


@torch.inference_mode()
def score(logits,row):
    target,ctx=row["targets"],row["context"]
    ce=losses(logits,target).mean(0)
    m,f,s=logits.split((9,2,2),1)
    move=m.softmax(1)
    dx=ctx[:,6]*384
    wide=dx.abs()>=32
    towards=(torch.tensor([-1,0,1]*3)[None,:]*dx.sign()[:,None])>0
    def horiz(a):
        return float((a*towards).sum(1)[wide].mean()) if wide.any() else None
    return dict(movement_ce=float(ce[0]),focus_ce=float(ce[1]),shoot_ce=float(ce[2]),mean_head_ce=float(ce.mean()),
                four_tick_direction_accuracy=float((move.argmax(1)==target[:,:9].argmax(1)).float().mean()),
                wide_target_count=int(wide.sum()),wide_toward_probability=horiz(move),
                wide_human_toward_fraction=horiz(target[:,:9]),shoot_probability=float(s.softmax(1)[:,1].mean()),
                human_shoot_fraction=float(target[:,10].mean()), focus_probability=float(f.softmax(1)[:,1].mean()),
                human_focus_fraction=float(target[:,9].mean()))


def fit(args):
    # Select the middle of the observed wave-only gap to isolate late boss states without native phase IDs.
    supports=[evidence(name,True)[4] for name in TRAIN]
    cutoff=(max(r[0][1] for r in supports)+min(r[1][0] for r in supports))//2
    arms=crossover_arms(args)
    base,sft=arms["ppo397"].policy,load_arm(TEMPLATE).policy
    train=[prepare(name,cutoff,base,sft,True) for name in TRAIN]
    torch.manual_seed(397)
    model=Readout()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    ctx,prior,y=(torch.cat([r[k] for r in train]) for k in ("context","prior","targets"))
    # Balance credit lengths, with additional support on misalignment in the human training observations.
    weights=[]
    for r in train:
        v=1.+2*(r["context"][:,6].abs()*384>=32).float()
        weights.append(v/v.sum()/len(train))
    weights=torch.cat(weights)
    for _ in range(160):
        logits=model(ctx,prior)
        ce=losses(logits,y)
        changes=logits-torch.cat([p.log_softmax(1) for p in prior.split((9,2,2),1)],1)
        loss=(weights*(ce[:,0]+.5*ce[:,1]+.5*ce[:,2])).sum()+.003*changes.square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    hold=[prepare(name,cutoff,base,sft) for name in HOLD]
    assert not ({r["meta"]["player"].casefold() for r in train}&{r["meta"]["player"].casefold() for r in hold})
    scores={}
    for row in hold:
        with torch.no_grad():
            candidate=model(row["context"],row["prior"])
        scores[row["meta"]["replay"]]={n:score(v,row) for n,v in
                                      (("composite",row["original"]),("sft",row["prior"]),("candidate",candidate))}
    def passes(m):
        a,p,s=m["candidate"],m["composite"],m["sft"]
        return (a["mean_head_ce"]<p["mean_head_ce"] and a["mean_head_ce"]<=s["mean_head_ce"]+.05
                and a["wide_toward_probability"]>=p["wide_toward_probability"]+.05
                and a["shoot_probability"]>=a["human_shoot_fraction"]-.05)
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    torch.save({"readout":model.state_dict(),"cutoff":cutoff,"ppo_hashes":HASHES,"sft_sha256":digest(TEMPLATE)},out/"specialist.pt")
    report={"gate_passed":all(passes(m) for m in scores.values()),"scores":scores,
            "training":{"credits":list(TRAIN),"rows":len(y),"steps":160,"seed":397,"lr":.003,"weight_decay":.01,
                        "criterion":"four-tick human movement/focus/shoot categorical CE, credit-balanced, 3x for |dx_boss|>=32; residual regularization .003; final fixed update"},
            "routing":{"native_observation":"positive boss HP and transformed Stage 1 clock past training gap midpoint",
                       "frame_cutoff":cutoff,"first_three_heads":"BC specialist on routed states, otherwise exact PPO397",
                       "bomb":"exact PPO211 in all states"},
            "gate":"on each development credit improve mean 3-head CE vs composite, within +0.05 of SFT or better, improve wide-offset toward-boss probability >=.05 vs composite and fire within .05 of human mean or higher",
            "data":[r["meta"] for r in train+hold],"sealed_test_used":False,"native_performance_measured":False,
            "sft_sha256":digest(TEMPLATE),"ppo_hashes":HASHES,"run_sha256":digest(Path(__file__)),
            "candidate_sha256":digest(out/"specialist.pt")}
    (out/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps(report))


class BossPolicy:
    def __init__(self,composite,sft,weights):
        self.composite,self.actor,self.sft=composite,composite.actor,sft
        for k in ("stage_clock","motion_spans","retain_recent_motion"):
            setattr(self,k,getattr(self.actor,k))
        if weights["ppo_hashes"]!=HASHES or weights["sft_sha256"]!=digest(TEMPLATE):
            raise ValueError("Mismatched source checkpoints.")
        self.cutoff=weights["cutoff"]
        self.model=Readout().eval()
        self.model.load_state_dict(weights["readout"])

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        c=observation["context"]
        if c.ndim!=1:
            raise ValueError("This stock credit runner expects one observation per decision.")
        if c[8]<=0 or c[-2]*14400<=self.cutoff:
            return self.composite.predict(observation,deterministic=deterministic)
        obs,_=self.actor.obs_to_tensor(observation)
        prior=torch.cat(predict_heads(self.sft,obs["bullets"],obs["context"])[:3],1)
        logits=self.model(obs["context"],prior)
        bomb=predict_heads(self.composite.bomb,obs["bullets"],obs["context"])[3]
        distribution=MultiCategoricalDistribution(list(ACTION_WIDTHS)).proba_distribution(torch.cat((logits,bomb),1))
        return distribution.get_actions(deterministic=deterministic).cpu().numpy()[0],None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["fit","evaluate"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--specialist",type=Path)
    parser.add_argument("--seeds",nargs="+",type=int,default=[1701,1702,1703,1704])
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
    if not args.specialist or any(not 0<=s<2**63 for s in args.seeds) or len(args.seeds)!=len(set(args.seeds)):
        parser.error("Provide the trained specialist and distinct valid policy seeds.")
    orig=crossover_arms(args)[SPLICE]
    weight=torch.load(args.specialist,map_location="cpu",weights_only=True)
    policy=BossPolicy(orig.policy,load_arm(TEMPLATE).policy,weight)
    arms={SPLICE:orig,NAME:Arm(policy,orig.history,dict(kind="human_stage1_boss_specialist",
            source_sha256=HASHES,sft_sha256=digest(TEMPLATE),specialist_sha256=digest(args.specialist),
            frame_cutoff=weight["cutoff"],bomb_source="ppo211"))}
    spec=credit.run_specification(args,arms)
    spec["controller"]["kind"]="pure_frozen_boss_specialist_comparison"
    spec["runtime"]["source_sha256"]["scripts.learn_stage1_boss"]=digest(Path(__file__))
    if not args.resume:
        args.out.mkdir(parents=True,exist_ok=False)
    with output_lock(args.out):
        credit.run_evaluation(args,arms,args.out/"report.json",spec)


if __name__=="__main__":
    main()
