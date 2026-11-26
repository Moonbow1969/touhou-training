"""Offline privileged phase-route distillation with whole-credit and replay holdouts."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import TEMPLATE
from scripts.evaluate_demo_route_phase import PhaseRoute,phase_knots
from scripts.evaluate_sft_native import load_arm
from scripts.fit_randomized_bomb_value import action_values
from scripts.native_story_transitions import NATIVE_FIELDS
from scripts.pretrain_policy import append_motion_history,append_stage_context,condition_context,movement_direction,predict_heads
from scripts.state_route_student import NativeClock,StateActor,features
from scripts.train_demo_route import SCALE,features as route_features

ROOT=Path(__file__).resolve().parents[3]
WORLD=ROOT/"outputs/touhou-sft-world"
PAIR=WORLD/"sft-native-data-261123/frozen-training-snapshot"
ART=WORLD/"th14-demo-alice-route-261125"
BOMB=ROOT/"outputs/th14-randomized-bomb-value-261125/bomb_value.json"
C={n:i for i,n in enumerate(NATIVE_FIELDS)}
TRAIN_NATIVE=("th14-stock-value-story-261125","th14-horizontal-scout-2051-261125",
              "th14-demo-alice-story-2001-261125","th14-demo-alice-2tick-2101-261125")
HOLD_NATIVE=("th14-demo-alice-phase-2201-261125","th14-demo-alice-phase-2202-261125")
HOLD_HUMAN=("th14_ud06d6","th14_ud068f")


class Teacher:
    def __init__(self):
        report=json.loads((ART/"report.json").read_bytes())
        assert digest(ART/"model.pt")==report["model_sha256"] and digest(ART/"route.npz")==report["route_sha256"]
        self.sft=load_arm(TEMPLATE)
        bomb=json.loads(BOMB.read_bytes())
        self.controller=PhaseRoute(ART/"model.pt",ART/"route.npz",bomb,self.sft,"phase-aware")
        self.controller.knots=phase_knots(PAIR/"th14_ud0613-stage1.npz",report["train_stage_shards"]["1"],self.controller.route["frame_1"])
        self.source=report

    def target(self,stage,refs,ctx,buls,frames,vulnerable,expired):
        route=self.controller.route
        timeline=route[f"frame_{stage}"]
        index=np.searchsorted(timeline,refs,side="right").clip(1,len(timeline))-1
        key=route[f"keys_{stage}"][index,(refs-timeline[index]).clip(0,3).astype(int)]
        xy=np.stack([np.interp(refs+4,timeline,p) for p in route[f"positions_{stage}"].T],1).astype("float32")
        # Rebuild the frozen auxiliary Shoot actor's own transform from contemporaneous state.
        policy=self.sft.policy
        fc=condition_context(ctx,1,"ReimuA")
        if self.sft.history:fc=append_motion_history(fc,frames,policy.motion_spans,policy.retain_recent_motion)
        if policy.stage_clock:fc=append_stage_context(fc,frames,stage)
        fallback=[]
        with torch.inference_mode():
            for i in range(0,len(ctx),192):
                logits=predict_heads(policy,torch.tensor(buls[i:i+192]),torch.tensor(fc[i:i+192]))[2]
                fallback.extend(logits.argmax(1).numpy())
        return xy,(key&8)!=0,(key&1)!=0,np.asarray(fallback),vulnerable,expired


def scene(path,teacher,native):
    with np.load(path) as z:
        ctx=z["context"][:,:10].copy();bul=z["bullets"];n=len(ctx)
        if native:
            b=z["native_before"];a=z["native_after"];executed=z["action"].astype(int)
            frame=b[:,C["stage_frame"]];stage=b[:,C["stage"]];state=b[:,C["player_state"]]
            prev=b[:,C["previous_x"]];stock=b[:,C["bombs"]];maxhp=np.maximum(b[:,C["boss_max_hp"]],0)
            spell=b[:,C["spell_id"]];vul=b[:,C["boss_vulnerable"]]==1
            hit=(state==1)&(a[:,C["player_state"]]==4)
            valid=(b[:,C["snapshot_valid"]]==1)&(stage>=1)&(stage<=6)&(stock>=0)
        else:
            frame=z["stage_frame"];st=int(path.stem.split("-stage")[1]);stage=np.full(n,st)
            state=z["player_state"];stock=np.rint(ctx[:,4]*8).astype(int)
            keys=z["expert_action_chunk"];first=keys[:,0]
            prev=np.r_[0,(keys[:-1,-1]&2)!=0]
            executed=np.stack([movement_direction(first),(first&8)!=0,(first&1)!=0,(first&2)!=0],1)
            maxhp=np.where(ctx[:,8]>0,1,0);spell=np.full(n,-1);vul=ctx[:,8]>0
            if path.stem.startswith("th14_ud0613-stage1"):
                # Train-only source lacks raw phase ID. Its frozen replay knots establish the phase prior.
                for k,(l,r) in teacher.controller.knots.items():
                    active=(frame>=l)&(frame<r)
                    maxhp[active]={"midboss":9000,"nonspell1":11300,"spell4":11300,"nonspell2":11400,"spell8":11400}[k]
                    spell[active]={"spell4":4,"spell8":8}.get(k,-1)
            hit=(state==1)&(np.r_[state[1:],state[-1]]==4);valid=(state>=0)&(stock>=0)
    x=[];cl=[];raw=[];clock=NativeClock();refs=[];ex=[]
    p=teacher.controller
    env=SimpleNamespace(reset_count=1);p.environment=env;p.credit=None;p.credit_stage=None
    for i in range(n):
        s=int(stage[i]); f=int(frame[i]); hp=int(maxhp[i]); sp=int(spell[i])
        if not 1<=s<=6:s=1
        age=clock(s,f,hp,sp)
        v=features(ctx[i],bul[i],s,f,hp,sp,vul[i],age,state[i],prev[i],stock[i]);x.append(v[0]);cl.append(v[1]);raw.append(v[2])
        env.snapshot=SimpleNamespace(stage=s,boss=SimpleNamespace(max_hp=hp) if hp>0 else None,spell_id=None if sp<0 else sp)
        if native:
            refs.append(p.phase_time(f,s,bool(ctx[i,8]>0)));ex.append(p.expired_phase)
        else:
            refs.append(f);ex.append(False)
    outputs=[np.empty((n,2),np.float32),*[np.zeros(n,dtype=np.int64) for _ in range(5)]]
    for s in np.unique(stage[valid]):
        ii=np.flatnonzero((stage==s)&valid)
        targets=teacher.target(s,np.asarray(refs)[ii],ctx[ii],bul[ii],frame[ii],vul[ii]&(maxhp[ii]>40),np.asarray(ex)[ii])
        for out,val in zip(outputs,targets):out[ii]=val
    danger=np.any((bul[:,:,6]>0)&(bul[:,:,5]>0)&(np.hypot(bul[:,:,0]*384,bul[:,:,1]*448)-bul[:,:,4]*32<24),axis=1)
    soon=np.array([hit[i:i+8].any() for i in range(n)])
    all_data=dict(x=np.array(x),clock=np.array(cl),raw=np.array(raw),target=outputs[0],focus=outputs[1],fire=outputs[2],
        fallback=outputs[3],vul=outputs[4],expired=outputs[5],actual=executed,stage=stage,state=state,stock=stock,spell=spell,
        danger=danger,soon=soon,phase=maxhp)
    return {k:v[valid] for k,v in all_data.items()}


def labels(teacher,data,position):
    x=route_features(position,data["target"],data["focus"],data["fire"],teacher.controller.route["bounds"])
    with torch.inference_mode():
        joint,shot=teacher.controller.model(torch.tensor(x))
    idx=joint.argmax(1).numpy();shoot=shot.argmax(1).numpy()
    shoot=np.where(data["vul"] & (data["expired"] | (shoot==0)),data["fallback"],shoot)
    vals,_=action_values(teacher.controller.bomb_model,data["raw"])
    return np.stack([idx%9,idx//9,shoot,vals.argmax(1)],1)


def make_examples(data,teacher,synthetic,rng):
    copies=[]
    for scale in ([None,2,8,28] if synthetic else [None]):
        x=data["x"].copy(); pos=x[:,:2]*SCALE
        if scale is not None:
            pos=np.clip(data["target"]+rng.normal(0,scale,size=pos.shape),*teacher.controller.route["bounds"]).astype("float32")
            change=pos-x[:,:2]*SCALE
            x[:,:2]=pos/SCALE
            # Re-express existing relative positions under the hypothetical player offset.
            x[:,6:8]-=change/[384,448]
            for j in range(3):
                k=10+7*j
                x[:,k:k+2]-=change/[384,448]*x[:,k+6:k+7]
        y=labels(teacher,data,pos)
        copies.append((x,data["clock"],data["raw"],data["target"]/SCALE,y))
    return tuple(np.concatenate(v) for v in zip(*copies))


def scores(net,data,teacher,native,device):
    with torch.inference_mode():
        preds=[]
        for lo in range(0,len(data["x"]),1024):
            inputs=[torch.from_numpy(data[k][lo:lo+1024]).to(device) for k in ("x","clock","raw")]
            y=net(*inputs)
            preds.extend(torch.stack([v.argmax(1) for v in y[:4]],1).cpu().numpy())
    pr=np.array(preds);y=data["actual"]
    target=labels(teacher,data,data["x"][:,:2]*SCALE)
    def view(mask):
        return dict(n=int(mask.sum()),acc=(pr[mask]==y[mask]).mean(0).tolist(),
            moving_agree=float((pr[mask,0]==y[mask,0])[y[mask,0]!=4].mean()) if np.any(y[mask,0]!=4) else None,
            horiz=np.bincount(pr[mask,0]%3,minlength=3).tolist(),observed_horiz=np.bincount(y[mask,0]%3,minlength=3).tolist(),
            shoot_rate=float(pr[mask,2].mean()),actual_shoot_rate=float(y[mask,2].mean()),bomb_press=int(pr[mask,3].sum()),
            actual_bomb_press=int(y[mask,3].sum()),false_bomb_press=int(((pr[mask,3]==1)&(y[mask,3]==0)).sum()),
            missed_bomb=int(((pr[mask,3]==0)&(y[mask,3]==1)).sum()))
    maskset={"all":np.ones(len(y),bool),"near_active_bullet":data["danger"],"within_8_decisions_before_hit":data["soon"],
             "hit_state_with_stock":(data["state"]==4)&(data["stock"]>0),"stock_empty":data["stock"]==0}
    for s in np.unique(data["stage"]):
        maskset[f"stage{s}"]=data["stage"]==s
    if native:
        for s in np.unique(data["spell"][data["vul"]>0]):
            maskset[f"vulnerable_spell_{s}"]=(data["vul"]>0)&(data["spell"]==s)
    return dict(slices={k:view(mask) for k,mask in maskset.items() if mask.any()},
        offline_teacher_vs_recorded_per_head=(target==y).mean(0).tolist() if native else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=3600)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(8371);rng=np.random.default_rng(8371)
    teacher=Teacher();device="cuda" if torch.cuda.is_available() else "cpu"
    files=[]
    for name in TRAIN_NATIVE:
        files.extend(sorted((WORLD/name/"transition-shards").glob("*.npz")))
    files+=sorted(PAIR.glob("th14_ud0613-stage*.npz"))
    data=[];sources=[]
    for file in files:
        native=file.parent.name=="transition-shards"
        sample=scene(file,teacher,native)
        data.append(make_examples(sample,teacher,True,rng))
        sources.append(dict(file=str(file.relative_to(ROOT)),sha256=digest(file),rows=len(sample["x"])))
    arrays=[torch.from_numpy(np.concatenate(a)).to(device) for a in zip(*data)]
    x,clock,raw,goal,y=arrays
    model=StateActor().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=.00012)
    counts=y[:,0].bincount(minlength=9).float()
    moveweight=(counts.mean()/counts).sqrt().clamp(.55,2.2)
    # Bomb supervision spans all feasible native states/stocks, independent of holdout archives.
    rb=np.array([[st,prev,b] for st in range(8) for prev in (0,1) for b in range(9)],np.float32)
    bv,_=action_values(teacher.controller.bomb_model,rb);by=torch.tensor(bv.argmax(1),device=device)
    rb=torch.tensor(rb,device=device)
    for step in range(args.steps):
        ii=torch.randint(len(y),(512,),device=device)
        pred=model(x[ii],clock[ii],raw[ii])
        braw=F.one_hot(rb[:,0].long(),9).float()
        blogit=model.bomb(torch.cat([braw,rb[:,1:2],rb[:,2:3]/8],1))
        loss=(F.cross_entropy(pred[0],y[ii,0],weight=moveweight)+.65*F.cross_entropy(pred[1],y[ii,1])
              +.8*F.cross_entropy(pred[2],y[ii,2])+.4*F.cross_entropy(blogit,by)
              +10*F.smooth_l1_loss(pred[4],goal[ii],beta=.05))
        opt.zero_grad();loss.backward();opt.step();schedule.step()
        if step%600==0: print("step",step,"loss",float(loss.detach()),flush=True)
    saved=args.out/"student.pt";torch.save(dict(state_dict=model.cpu().state_dict()),saved);model.to(device).eval()
    result=dict(train=sources,synthetic_position_draws_per_original=3,device=device,seed=8371,steps=args.steps,
        model_sha256=digest(saved),teacher_weight_sha256=[digest(p) for p in (ART/"model.pt",ART/"route.npz",TEMPLATE,BOMB)],
        runtime_source_sha256=digest(Path(__file__).with_name("state_route_student.py")),trainer_sha256=digest(Path(__file__)),
        inference="Only actor weights; current native context, nearest bullets, stage, raw spell/max HP/vulnerability, elapsed ticks since observable identity change, player state, previous Bomb key, inventory. Four simultaneous greedy heads.",
        prior="Privileged train-time teacher uses Alice th14_ud0613 full six-stage reference and phase alignment plus frozen SFT combat Shoot and learned native Bomb values. Auxiliary waypoint target and synthetic position shifts used only for training. Human shards lack raw phase identity; only Alice Stage 1 uses its frozen knots during training.",
        gate="Each held-out native credit >=70% exact 9-way motion on near-bullet states, >=60% on pre-hit states, all directions represented, spell8 Shoot within 5 percentage points of teacher and no missed/extra Bomb in hit states with stock. Freeze weights before holdout; no reranking.")
    result["native"]={};result["human"]={}
    for name in HOLD_NATIVE:
        candidates=list((WORLD/name/"transition-shards").glob("*.npz"))
        if not candidates:continue
        for path in candidates: result["native"][path.name]=scores(model,scene(path,teacher,True),teacher,True,device)
    for replay in HOLD_HUMAN:
        scenes=[scene(path,teacher,False) for path in sorted(PAIR.glob(replay+"-stage*.npz"))]
        union={k:np.concatenate([s[k] for s in scenes]) for k in scenes[0]}
        result["human"][replay]=scores(model,union,teacher,False,device)
    (args.out/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print("frozen",result["model_sha256"],"heldout",list(result["native"]),flush=True)


if __name__=="__main__":main()
