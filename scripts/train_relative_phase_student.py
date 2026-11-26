"""Train a state-only student on translated native clocks and delayed boss phases."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.distill_state_route import (ART,BOMB,C,HOLD_HUMAN,PAIR,ROOT,SCALE,TEMPLATE,TRAIN_NATIVE,WORLD,Teacher,labels,scene)
from scripts.fit_randomized_bomb_value import action_values
from scripts.relative_phase_student import EventClock,RelativeActor

DEV="credit-002-arm-ppo211_with_value-seed-1902.npz"


def observed(path,teacher):
    native=path.parent.name=="transition-shards"
    d=scene(path,teacher,native)
    if native:
        with np.load(path) as z:b=z["native_before"]
        b=b[(b[:,C["snapshot_valid"]]==1)&(b[:,C["stage"]]>=1)&(b[:,C["stage"]]<=6)&(b[:,C["bombs"]]>=0)]
        frac=np.maximum(b[:,C["boss_hp"]],0)/np.maximum(b[:,C["boss_max_hp"]],1)
    else:
        frac=np.where(d["phase"]>1,np.clip(d["x"][:,8]*2000/np.maximum(1,d["phase"]),0,1),d["x"][:,8])
    clock=EventClock()
    ev=np.array([clock.update(int(stage),int(frame),int(size),int(spell),visible) for stage,frame,size,spell,visible in
        zip(d["stage"],d["clock"][:,0],d["phase"],d["spell"],d["x"][:,8]>0)])
    d.update(event=ev,hp=frac.clip(0,1).astype(np.float32))
    return d


def delayed_tail(d,teacher,rng,extra_range=(4,3000)):
    sets=[]
    for phase,(start,stop) in teacher.controller.knots.items():
        source=np.flatnonzero((d["clock"][:,0]>=start)&(d["clock"][:,0]<stop))
        if not len(source):continue
        size=900;i=source[-1]
        s={k:np.repeat(v[i:i+1],size,axis=0) for k,v in d.items()}
        extra=rng.integers(*extra_range,size=size)
        dt=stop-4-d["clock"][i,0]+extra
        s["event"][:,:2]+=dt[:,None]
        s["clock"][:,0]+=dt
        route=teacher.controller.route;f=route["frame_1"];ref=stop-4
        index=int(np.searchsorted(f,ref,side="right").clip(1,len(f))-1)
        key=int(route["keys_1"][index,int(np.clip(ref-f[index],0,3))])
        s["target"][:]=[np.interp(ref+4,f,v) for v in route["positions_1"].T]
        s["focus"][:]=bool(key&8);s["fire"][:]=bool(key&1)
        s["expired"][:]=1;s["vul"][:]=1
        s["hp"]=rng.uniform(.02,.98,size).astype(np.float32)
        sets.append(s)
    return {k:np.concatenate([s[k] for s in sets]) for k in d}


def examples(d,teacher,rng):
    parts=[]
    for noise in (None,1.5,6,24,80):
        x=d["x"].copy();pos=x[:,:2]*SCALE
        if noise is not None:
            pos=np.clip(d["target"]+rng.normal(0,noise,size=pos.shape),*teacher.controller.route["bounds"]).astype("float32")
            change=pos-x[:,:2]*SCALE;x[:,:2]=pos/SCALE
            x[:,6:8]-=change/[384,448]
            for j in range(3):
                k=10+7*j;x[:,k:k+2]-=change/[384,448]*x[:,k+6:k+7]
        y=labels(teacher,d,pos)
        parts.append((x,d["event"],d["hp"],d["raw"],d["target"]/SCALE,y,
                      np.stack((d["focus"],d["fire"]),1).astype("int64"),d["stage"]))
    return tuple(np.concatenate(v) for v in zip(*parts))


def augment(event,x,generator):
    e=event.clone()
    age=e[:,0]
    # Independent synthetic absolute stage and process-clock origins. The actor receives
    # only independently verified elapsed time. The underlying phase policy is invariant.
    onset=torch.randint(4096,20480,(len(e),),generator=generator,device=e.device)
    global_origin=torch.randint(50000,200000,(len(e),),generator=generator,device=e.device)
    e[:,0]=torch.where(onset%2==0,(onset+age)-onset,(global_origin+age)-global_origin)
    # Vary how long earlier subphases lasted independently of the current subphase age.
    # The enclosing encounter age still cannot be younger than its current subphase.
    eligible=(x[:,31]>.5)&(e[:,1]>e[:,0]+24)&(x[:,37]>0)
    earlier=torch.rand(len(e),device=e.device,generator=generator)*4200
    e[:,1]=torch.where(eligible,age+earlier,e[:,1])
    return e


@torch.inference_mode()
def measure(model,d,teacher,device,*,against_actual,alternative_event=None):
    data=[torch.tensor(v,device=device) for v in (d["x"],d["event"] if alternative_event is None else alternative_event,d["hp"],d["raw"])]
    outputs=[model(*(v[i:i+512] for v in data)) for i in range(0,len(d["x"]),512)]
    joint,shot,bomb,target,pref=(torch.cat(v).cpu().numpy() for v in zip(*outputs))
    j=joint.argmax(1);pred=np.stack((j%9,j//9,shot.argmax(1),bomb.argmax(1)),1)
    expected=d["actual"] if against_actual else labels(teacher,d,d["x"][:,:2]*SCALE)
    err=np.linalg.norm(target*SCALE-d["target"],axis=1)
    def group(m):
        y=expected[m];pr=pred[m]
        return dict(n=int(m.sum()),accuracy=(pr==y).mean(0).tolist(),moving_accuracy=float((pr[y[:,0]!=4,0]==y[y[:,0]!=4,0]).mean()) if (y[:,0]!=4).any() else None,
            horiz=np.bincount(pr[:,0]%3,minlength=3).tolist(),target_horiz=np.bincount(y[:,0]%3,minlength=3).tolist(),
            fire=float(pr[:,2].mean()),target_fire=float(y[:,2].mean()),false_bomb=int(((pr[:,3]==1)&(y[:,3]==0)).sum()),
            missed_bomb=int(((pr[:,3]==0)&(y[:,3]==1)).sum()),median_waypoint_error=float(np.median(err[m])),
            p90_waypoint_error=float(np.quantile(err[m],.9)),idle_reference=float((y[:,0]==4).mean()))
    slices={"all":np.ones(len(j),bool),"near":d["danger"],"pre_hit":d["soon"],"hit_with_stock":(d["state"]==4)&(d["stock"]>0),
            "stage1_spell8":(d["stage"]==1)&(d["spell"]==8)&(d["vul"]==1),"empty_stock":d["stock"]==0}
    for st in np.unique(d["stage"]):slices[f"stage{st}"]=d["stage"]==st
    return {k:group(m) for k,m in slices.items() if m.any()}


def train(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    rng=np.random.default_rng(9127);torch.manual_seed(9127);torch.set_num_threads(2)
    teacher=Teacher();device="cuda" if torch.cuda.is_available() else "cpu"
    gen=torch.Generator(device=device);gen.manual_seed(5491)
    sources=[];parts=[]
    paths=[p for name in TRAIN_NATIVE for p in sorted((WORLD/name/"transition-shards").glob("*.npz"))]
    paths+=sorted(PAIR.glob("th14_ud0613-stage*.npz"))
    for path in paths:
        if path.name==DEV:continue
        d=observed(path,teacher);parts.append(examples(d,teacher,rng))
        sources.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path),n=len(d["x"])))
        if path.stem=="th14_ud0613-stage1":
            synth=delayed_tail(d,teacher,rng);parts.append(examples(synth,teacher,rng))
            tail_test=delayed_tail(d,teacher,np.random.default_rng(17299),extra_range=(3000,4800))
            test_pos=np.clip(tail_test["target"]+np.random.default_rng(17300).normal(0,40,tail_test["target"].shape),
                             *teacher.controller.route["bounds"])
            tail_test["x"][:,:2]=test_pos/SCALE
            tail_n=len(synth["x"])
    arr=[torch.tensor(np.concatenate(v),device=device) for v in zip(*parts)]
    x,event,hp,raw,goal,y,pref,stage=arr
    pools=[torch.nonzero(mask).flatten() for mask in ((stage==1)&(x[:,37]>0),
           (stage==1)&(x[:,37]==0),stage>1,torch.ones_like(stage,dtype=torch.bool))]
    model=RelativeActor().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.00005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=.000045)
    rb=np.array([[s,k,b] for s in range(8) for k in (0,1) for b in range(9)],np.float32)
    by=torch.tensor(action_values(teacher.controller.bomb_model,rb)[0].argmax(1),device=device)
    rb=torch.tensor(rb,device=device)
    for step in range(args.steps):
        ii=torch.cat([pool[torch.randint(len(pool),(n,),generator=gen,device=device)] for pool,n in zip(pools,(224,112,112,64))])
        xx=x[ii];ee=augment(event[ii],xx,gen);frac=hp[ii]
        h,t,pr=model.encode(xx,ee,frac)
        # Mix exact and learned internal waypoints during motor training, keeping
        # the auxiliary regression in control of the timing representation.
        forced=torch.rand(len(ii),1,device=device,generator=gen)<.55
        joint,shot=model.decode(xx,h,torch.where(forced,goal[ii],t.detach()),pr)
        loss=(24*F.smooth_l1_loss(t,goal[ii],beta=.012)+F.cross_entropy(joint,9*y[ii,1]+y[ii,0])
            +.8*F.cross_entropy(shot,y[ii,2])+.65*(F.cross_entropy(pr[:,:2],pref[ii,0])+F.cross_entropy(pr[:,2:],pref[ii,1]))
            +.3*F.cross_entropy(model.bomb_logits(rb),by))
        opt.zero_grad();loss.backward();opt.step();schedule.step()
        if step%2400==0:print("step",step,"loss",float(loss.detach()),flush=True)
    model.cpu();torch.save({"state_dict":model.state_dict()},args.out/"student.pt");model.to(device).eval()
    dev=next(p for p in paths if p.name==DEV);d=observed(dev,teacher)
    alt=d["event"].copy();eligible=(d["stage"]==1)&(d["phase"]>0)&(alt[:,1]>alt[:,0]+24)
    alt[eligible,1]=alt[eligible,0]+rng.uniform(0,5200,int(eligible.sum()))
    result=dict(sources=sources,dev_credit=dev.name,native_student_rollouts=0,steps=args.steps,seed=9127,
        stored_rows=sum(s["n"] for s in sources),synthetic_elapsed_tail_rows=tail_n,train_draws_stored=len(x),
        hash=digest(args.out/"student.pt"),teacher_hashes=[digest(q) for q in (ART/"model.pt",ART/"route.npz",TEMPLATE,BOMB)],
        runtime_hash=digest(Path(__file__).with_name("relative_phase_student.py")),trainer_hash=digest(Path(__file__)),
        holdout_contract="2201 and 2202 are prior diagnostics, permanently excluded here. Reserve 2402 and one later comparable complete phase-aware route credit. Inspect only after weights freeze. Current source credit 1902 PPO211 is excluded from all fit and used for development diagnostics.",
        architecture="No absolute stage or global clock enters the neural network. Generic visible-phase and encounter elapsed times evolve from current state. Inputs include normalized raw HP. One network estimates waypoint/preferences and simultaneous joint movement/focus, Shoot and Bomb; no route is present at deployment.",
        prior="Alice full six-stage reference and earlier native states relabeled by a privileged frozen route tracker. Onset translations are independent on synthetic global and stage clocks. Earlier enclosing-phase durations vary independently, and matching source endpoints label synthetic post-expiry tails.",
        gate="On both untouched native teacher credits: >=70% exact 9-way movement near active bullets and >=60% before an observed hit, three horizontal directions represented without single-mode collapse, spell8 Shoot within 5 percentage points, no erroneous Bomb in player_state 4 with stock. Native student survival remains unmeasured.",
        train_replay={key:measure(model,observed(PAIR/f"th14_ud0613-stage{st}.npz",teacher),teacher,device,against_actual=False)
                        for key,st in (("stage1",1),("stage2",2))},
        independent_earlier_credit_teacher=measure(model,d,teacher,device,against_actual=False),
        counterfactual_shifted_preceding_phase_durations=measure(model,d,teacher,device,against_actual=False,alternative_event=alt),
        synthetic_unseen_delayed_tail_3000to4799=measure(model,tail_test,teacher,device,against_actual=False))
    result["whole_human_replays"]={}
    for name in HOLD_HUMAN:
        dd=[observed(path,teacher) for path in sorted(PAIR.glob(name+"-stage*.npz"))]
        full={k:np.concatenate([v[k] for v in dd]) for k in dd[0]}
        result["whole_human_replays"][name]=measure(model,full,teacher,device,against_actual=True)
    (args.out/"training.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print("frozen",result["hash"],"dev",result["independent_earlier_credit_teacher"]["near"]["accuracy"],flush=True)


def score(args):
    training=json.loads((args.out/"training.json").read_bytes())
    if digest(args.out/"student.pt")!=training["hash"]:raise ValueError("Frozen weights changed.")
    torch.set_num_threads(2);teacher=Teacher();model=RelativeActor().eval()
    model.load_state_dict(torch.load(args.out/"student.pt",weights_only=True)["state_dict"])
    results={}
    for path in args.shards:
        path=path.resolve()
        with np.load(path) as z:meta=json.loads(z["metadata_json"].item())
        seed=meta["trial"]["policy_seed"]
        if seed<2402 or seed in results:raise ValueError("Use a reserved whole comparable credit at or after 2402.")
        d=observed(path,teacher)
        teach=labels(teacher,d,d["x"][:,:2]*SCALE)
        matched=(teach==d["actual"]).mean(0).tolist()
        stage1=d["stage"]==1
        results[seed]=dict(path=str(path.relative_to(ROOT)),hash=digest(path),behavior=meta["trial"]["arm"],
            comparison_to_frozen_teacher_alignment=matched,stage1_teacher_alignment=(teach[stage1]==d["actual"][stage1]).mean(0).tolist(),
            observed_on_teacher_trajectory=measure(model,d,teacher,"cpu",against_actual=True))
    dest=args.out/("reserved-"+"-".join(map(str,results))+".json")
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(dict(weights=training["hash"],student_on_policy_rollouts=0,credits=results),indent=2,allow_nan=False))
    print(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("train","score"));p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=14400);p.add_argument("--shards",type=Path,nargs="+")
    args=p.parse_args();train(args) if args.mode=="train" else score(args)


if __name__=="__main__":main()
