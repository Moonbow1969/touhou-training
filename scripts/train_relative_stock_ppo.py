"""Fine-tune the state-only relative-clock student on full, fresh stock TH14 credits."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from scripts import evaluate_credit_story as story, native_story_restart, relative_phase_student as relative
from scripts import state_route_student, train_anticipatory_contact as contact
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA
from scripts.evaluate_sft_native import Arm, output_lock
from scripts.train_full_policy import publish_json
from scripts import th14_inventory_reward as inventory


class ProjectilePolicy(nn.Module):
    def __init__(self, weights, risk_path):
        super().__init__()
        self.actor=relative.RelativeActor()
        self.actor.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True)["state_dict"])
        self.risk=contact.AnticipatoryContact()
        self.risk.load_state_dict(torch.load(risk_path,map_location="cpu",weights_only=True)["state_dict"])
        self.risk.requires_grad_(False)
        self.move_residual=nn.Sequential(nn.Linear(18+5,64),nn.Tanh(),nn.Linear(64,18))
        nn.init.zeros_(self.move_residual[-1].weight)
        nn.init.zeros_(self.move_residual[-1].bias)
        self.value=nn.Sequential(nn.Linear(128+7,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def heads(self,x,event,hp,raw,danger):
        hidden, target, preference=self.actor.encode(x,event,hp)
        move,shoot=self.actor.decode(x,hidden,target,preference)
        move=move+self.move_residual(torch.cat((danger,x[:,:5]),1))
        value=self.value(torch.cat((hidden.detach(),x[:,:5],hp[:,None],raw[:,2:3]/8),1)).squeeze(1)
        return (move,shoot,self.actor.bomb_logits(raw)),value

    @torch.inference_mode()
    def measured_risk(self,observation):
        c=torch.from_numpy(observation["context"][None])
        b=torch.from_numpy(observation["bullets"][None])
        return self.risk(b,contact.local.local_context(c,0))[0,:18].sigmoid()


def inputs(actor, observation):
    env=actor.environment
    snap=env.snapshot
    if actor.credit!=env.reset_count:
        actor.credit=env.reset_count
        actor.clock=relative.EventClock()
    boss=snap.boss
    maximum=boss.max_hp if boss else 0
    spell=snap.spell_id if snap.spell_id is not None else -1
    event=actor.clock.update(snap.stage,env.last_frame,maximum,spell,observation["context"][8]>0)
    x,_,stock=state_route_student.features(observation["context"],observation["bullets"],snap.stage,env.last_frame,
        maximum,spell,bool(boss and boss.vulnerable),event[0],snap.player_state,int("x" in env.held),env.game.read_game_val("bombs"))
    hp=np.float32(np.clip(boss.hp/max(1,maximum),0,1) if boss else 0.)
    return tuple(torch.as_tensor(a) for a in (x,event,hp,stock))+(actor.model.measured_risk(observation),)


class LiveActor:
    stage_clock=False
    motion_spans=None
    retain_recent_motion=False

    def __init__(self,model,temp,training):
        self.model=model
        self.temp=temp
        self.training=training
        self.credit=self.environment=None
        self.rows=[]

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        row=inputs(self,observation)
        heads,v=self.model.heads(*(x.unsqueeze(0) for x in row))
        dists=[Categorical(logits=h/self.temp) for h in heads]
        actions=torch.stack([d.sample()[0] if self.training else d.logits[0].argmax() for d in dists])
        if self.training:
            logp=sum(d.log_prob(a).item() for d,a in zip(dists,actions))
            self.rows.append((row,actions,logp,v.item()))
        move,shoot,bomb=actions.tolist()
        return np.asarray([move%9,move//9,shoot,bomb],dtype=np.int64),None


class HitRecorder:
    def __init__(self,actor,target,bomb_cost=0.,shape_inventory=False):
        self.actor=actor
        self.target=target
        self.rewards=[]
        self.bootstrap=0.
        self.spells=set()
        self.landmarks=[]
        self.bomb_cost=bomb_cost
        self.bombs_spent=[]
        self.shape_inventory=shape_inventory
        self.inventory=None

    def before_step(self,env,observation):
        if self.shape_inventory and self.inventory is None:
            self.inventory=inventory.InventoryReward(inventory.read_inventory(env.game))
        return env.snapshot.player_state, env.snapshot.stage

    def after_step(self,env,before,action,info,terminated,truncated,deaths,bombs):
        state,stage=before
        snap=env.snapshot
        if bombs<0:
            raise ValueError("The native Bomb-consumption counter decreased inside a credit.")
        self.bombs_spent.append(int(bombs))
        reward=(-.015 if state==1 and snap.player_state==4 else 0.)-self.bomb_cost*bombs
        if self.inventory is not None:
            # Truncation keeps the live potential and bootstraps the shaped critic.
            stock=None if terminated else inventory.read_inventory(env.game)
            reward+=self.inventory.advance(stock,terminal=terminated)
        if self.target=="stage3":
            if stage==1 and info["native_stage"]==2:
                reward+=.5+.03*env.game.read_game_val("lives")+.01*env.game.read_game_val("bombs")
                self.landmarks.append(dict(kind="stage2_entry",decision=len(self.rewards)+1))
            # Credit each distinct Stage 2 spell once, even if a boss later times out.
            if (snap.stage==2 and snap.spell_id is not None and snap.boss is not None and snap.boss.vulnerable
                    and snap.boss.hp>0 and snap.spell_id not in self.spells):
                self.spells.add(snap.spell_id)
                reward+=.06
                self.landmarks.append(dict(kind="first_vulnerable_spell",spell=snap.spell_id,stage=2,
                                           frame=env.last_frame,decision=len(self.rewards)+1))
        self.rewards.append(reward)

    @torch.inference_mode()
    def finish(self,outcome,observation):
        if outcome=="time_limit":
            if observation is None:
                raise ValueError("A truncated credit needs its final stock observation.")
            row=inputs(self.actor,observation)
            self.bootstrap=self.actor.model.heads(*(x.unsqueeze(0) for x in row))[1].item()


def pack(rows,rewards,bootstrap):
    if not rows or len(rows)!=len(rewards):
        raise ValueError("Mismatched action/reward sequence.")
    obs,act,old,v=zip(*rows)
    old_v=np.asarray(v)
    delta=np.asarray(rewards)+np.r_[old_v[1:],bootstrap]-old_v
    adv=np.zeros_like(delta)
    running=0.
    for t in reversed(range(len(adv))):
        running=delta[t]+.995*running
        adv[t]=running
    return (*[torch.stack(part) for part in zip(*obs)],torch.stack(act),torch.tensor(old,dtype=torch.float32),
            torch.tensor(adv,dtype=torch.float32),torch.tensor(adv+old_v,dtype=torch.float32))


def update(model,anchor,optim,batch,temperature):
    x,e,hp,stock,r,actions,oldp,adv,targets=batch
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    entries=[]
    for _ in range(3):
        for ix in torch.randperm(len(x)).split(192):
            features=(x[ix],e[ix],hp[ix],stock[ix],r[ix])
            current,val=model.heads(*features)
            with torch.no_grad():
                fixed=anchor.heads(*features)[0]
            dists=[Categorical(logits=h/temperature) for h in current]
            newp=sum(d.log_prob(a) for d,a in zip(dists,actions[ix].T))
            ratio=(newp-oldp[ix]).exp()
            pi=-torch.minimum(ratio*adv[ix],ratio.clamp(.90,1.10)*adv[ix]).mean()
            vf=.5*(val-targets[ix]).square().mean()
            kl=sum(kl_divergence(d,Categorical(logits=h/temperature)).mean() for d,h in zip(dists,fixed))
            entropy=sum(d.entropy().mean() for d in dists)
            loss=pi+vf+.04*kl-.001*entropy
            optim.zero_grad(set_to_none=True)
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite PPO gradients.")
            optim.step()
            entries.append([float(t.detach()) for t in (pi,vf,kl,entropy,norm,(ratio.detach()-1).abs().gt(.1).float().mean())])
    return dict(zip(("policy_loss","value_loss","anchor_kl","entropy","gradient_norm","clipped_fraction"),
                    np.mean(entries,axis=0).tolist()))


def publish_tensors(path, contents):
    temporary=path.with_suffix(".tmp")
    try:
        torch.save(contents,temporary)
        restored=torch.load(temporary,map_location="cpu",weights_only=True)
        if restored["schema"]!=contents["schema"] or restored["steps"]!=contents["steps"]:
            raise ValueError("Staged native PPO artifact failed verification.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(file=path.name,sha256=digest(path),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student",type=Path,default=DATA.parent/"th14-relative-phase-student-261125/student.pt")
    parser.add_argument("--risk",type=Path,default=DATA.parent/"th14-anticipatory-contact-12tick-261125/model.pt")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--credits-per-update",type=int,default=2)
    parser.add_argument("--updates",type=int,default=1)
    parser.add_argument("--seed",type=int,default=3501)
    parser.add_argument("--temperature",type=float,default=.55)
    parser.add_argument("--bomb-cost",type=float,default=0.,help="Cost per confirmed native Bomb consumed.")
    parser.add_argument("--inventory-shaping",action="store_true",help="Add a centered TH14 inventory potential at gamma=1.")
    init=parser.add_mutually_exclusive_group()
    init.add_argument("--continue-from",type=Path,help="Start a fresh run with a compatible published PPO model and optimizer.")
    init.add_argument("--warm-policy",type=Path,help="Transfer learned actor weights to a fresh objective and critic.")
    parser.add_argument("--target",choices=("stage2","stage3"),default="stage2")
    parser.add_argument("--menu-references",nargs=2,type=Path)
    parser.add_argument("--menu-crop",nargs=4,type=int,action="append")
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if (args.seed<0 or not 1<=args.credits_per_update<=8 or args.updates<1 or not .1<=args.temperature<=2
            or not 0<=args.bomb_cost<=1
            or bool(args.menu_references)!=bool(args.menu_crop) or args.menu_crop and len(args.menu_crop)<2):
        parser.error("Use valid credit budgets, sampling temperature and complete menu references.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if (digest(args.student)!=json.loads(args.student.with_name("training.json").read_bytes())["hash"]
            or digest(args.risk)!=json.loads(args.risk.with_name("report.json").read_bytes())["model_sha256"]):
        raise ValueError("Model fingerprint did not match its training run.")
    model=ProjectilePolicy(args.student,args.risk)
    anchor=deepcopy(model).eval().requires_grad_(False)
    optim=torch.optim.Adam([dict(params=model.actor.parameters(),lr=1e-6),
                            dict(params=model.move_residual.parameters(),lr=3e-4),
                            dict(params=model.value.parameters(),lr=7e-4)],eps=1e-5)
    # Exercise the collection/update contract offline, including exact original logits at initialization.
    x=torch.zeros(8,42); ev=torch.zeros(8,5); hp=torch.zeros(8); raw=torch.ones(8,3); risks=torch.rand(8,18)
    with torch.no_grad():
        current,_=model.heads(x,ev,hp,raw,risks)
        original=model.actor(x,ev,hp,raw)[:3]
        for first,second in zip(current,original):
            torch.testing.assert_close(first,second,atol=0,rtol=0)
    args.limit=story.LIMITS[args.target]; args.start_frame=30; args.focus_wait=90
    args.parakit=Path(__file__).resolve().parents[2]/"parakit"
    spec=dict(student_sha256=digest(args.student),risk_sha256=digest(args.risk),source_sha256=digest(Path(__file__)),
              temperature=args.temperature,seed=args.seed,credits_per_update=args.credits_per_update,updates=args.updates,
              critic="Detached actor latent plus current stock. Separate value gradients.",
              reward=dict(stage2_entry=1.,spare_life_at_entry=.1,bomb_at_entry=.02,native_hit_onset=-.015,
                          gamma=1.,gae_lambda=.995),native_start="fresh Normal Reimu A Stage 1, frame 30, two spares, three Bombs",
              initialization="All three actor logits exactly equal the frozen joint move/focus, Shoot and Bomb student. "
                             "New trainable movement residual sees 18 learned risk scores and five current context scalars.",
              runtime_route=False,geometry=False,automatic_bomb=False,aim_override=False,target=args.target)
    if args.target=="stage3":
        spec["reward"]=dict(stage2_entry=.5,stage2_spare_life=.03,stage2_bomb=.01,
                            first_vulnerable_stage2_spell=.06,stage3_entry=1.5,stage3_spare_life=.1,
                            stage3_bomb=.02,native_hit_onset=-.015,gamma=1.,gae_lambda=.995)
    if args.bomb_cost:
        spec["reward"]["native_bomb_consumed"]=-args.bomb_cost
    if args.inventory_shaping:
        spec["reward"]["inventory_potential"]=dict(life=inventory.LIFE_VALUE,bomb=inventory.BOMB_VALUE,
            life_pieces_per_life=3,bomb_pieces_per_bomb=8,absorbing_terminal=0.,center_on_fresh_credit=True,
            source_sha256=digest(Path(inventory.__file__)))
    starting_steps=0
    parent=args.continue_from or args.warm_policy
    if parent:
        prior=json.loads(parent.with_name("report.json").read_text(encoding="utf-8"))
        payload=torch.load(parent,map_location="cpu",weights_only=True)
        key_fields=(("student_sha256","risk_sha256","temperature","source_sha256","reward") if args.continue_from else
                    ("student_sha256","risk_sha256"))
        if (prior["status"]!="complete" or prior["updates"][-1]["sha256"]!=digest(parent)
                or payload["schema"]!="relative-stock-ppo-v1" or prior["specification"]!=payload["specification"]
                or any(payload["specification"][k]!=spec[k] for k in key_fields)):
            raise ValueError("Prior full-credit PPO checkpoint or optimization contract changed.")
        critic=deepcopy(model.value.state_dict())
        model.load_state_dict(payload["state_dict"])
        if args.continue_from:
            optim.load_state_dict(payload["optimizer"])
            starting_steps=int(payload["steps"])
        else:
            model.value.load_state_dict(critic)
            anchor=deepcopy(model).eval().requires_grad_(False)
            spec["parent_native_steps"]=payload["steps"]
            spec["initialization"]="Warm-started all actor and residual weights from the prior PPO phase. Fresh critic and optimizer."
        spec["parent_checkpoint_sha256"]=prior["updates"][-1]["sha256"]
    if args.verify_only:
        with torch.no_grad():
            current=model.heads(x,ev,hp,raw,risks)[0]
        dummy_actions=torch.stack([h.argmax(1) for h in current],1)
        oldp=sum(Categorical(logits=h/args.temperature).log_prob(a) for h,a in zip(current,dummy_actions.T))
        update(model,anchor,optim,(x,ev,hp,raw,risks,dummy_actions,oldp,torch.arange(8).float(),torch.ones(8)),args.temperature)
        print(json.dumps(spec))
        return
    if not args.out or shutil.disk_usage(args.out.resolve().parent).free<20*1024**3:
        parser.error("Select a fresh output directory with at least 20 GiB free.")
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/"report.json"
    log=dict(status="collecting",specification=spec,episodes=[],updates=[],steps=starting_steps,
             starting_steps=starting_steps)
    menu=story.GameOverMenu(*args.menu_references,args.menu_crop,.035) if args.menu_references else None
    runner=LiveActor(model,args.temperature,True)
    arm=Arm(runner,False,{"source":"state-only student plus zero-initialized learned risk residual"})
    begin=perf_counter()
    with output_lock(args.out):
        try:
            publish_json(path,log)
            with story.native_session(args,{"learner":arm},menu,None) as (env,prep):
                log["preparation"]=prep
                for epoch in range(args.updates):
                    batch=[]
                    for j in range(args.credits_per_update):
                        idx=epoch*args.credits_per_update+j
                        trial=dict(block=epoch+1,order=j+1,arm="online",policy_seed=args.seed+idx)
                        runner.rows=[]
                        rec=HitRecorder(runner,args.target,args.bomb_cost,args.inventory_shaping)
                        def reset(stage,trial):
                            return native_story_restart.reset(stage,trial,start_frame=30,menu=menu)
                        def progress(row):
                            log["in_progress"]=dict(trial=trial,result=row)
                            publish_json(path,log)
                        result=story.story_credit(env,arm,trial,args,reset,progress,transition_recorder=rec)
                        if sum(rec.bombs_spent)!=result["actual_bombs_consumed"]:
                            raise ValueError("The reward ledger disagrees with native Bomb consumption.")
                        if result["target_reached"]:
                            resource=result["stage_records"][-1]["entry"]["resources"]
                            terminal_bonus=1.5 if args.target=="stage3" else 1.
                            rec.rewards[-1]+=terminal_bonus+.1*resource["lives"]+.02*resource["bombs"]
                        batch.append(pack(runner.rows,rec.rewards,rec.bootstrap))
                        buffer_info=publish_tensors(args.out/f"credit-{idx+1:03d}.pt",
                            dict(schema="relative-stock-ppo-on-policy-v1",steps=len(rec.rewards),fields=[
                                "state","event","hp","stock","learned_contact","action","old_log_probability","advantage","value_target"],
                                tensors=batch[-1],observed_rewards=rec.rewards,bombs_spent=rec.bombs_spent,bootstrap=rec.bootstrap))
                        log["steps"]+=len(rec.rewards)
                        log["episodes"].append(dict(trial=trial,result=result,native_return=sum(rec.rewards),
                                                    buffer=buffer_info,landmarks=rec.landmarks,
                                                    native_bomb_penalty=-sum(rec.bombs_spent)*args.bomb_cost,
                                                    inventory_shaping=rec.inventory.report() if rec.inventory else None))
                        log["in_progress"]=None
                        publish_json(path,log)
                    update_metrics=update(model,anchor,optim,tuple(torch.cat(chunk) for chunk in zip(*batch)),args.temperature)
                    checkpoint=args.out/f"policy-update-{epoch+1:03d}.pt"
                    artifact=publish_tensors(checkpoint,dict(schema="relative-stock-ppo-v1",state_dict=model.state_dict(),
                                    optimizer=optim.state_dict(),steps=log["steps"],specification=spec))
                    latest=args.out/"policy-latest.tmp"
                    try:
                        try:
                            os.link(checkpoint,latest)
                        except OSError:
                            shutil.copyfile(checkpoint,latest)
                        latest.replace(args.out/"policy.pt")
                    finally:
                        latest.unlink(missing_ok=True)
                    log["updates"].append(dict(epoch=epoch+1,steps=log["steps"],**artifact,**update_metrics))
                    publish_json(path,log)
            log["status"]="complete"
        except BaseException as error:
            log.update(status="interrupted",error=dict(type=type(error).__name__,detail=str(error)))
            raise
        finally:
            log["seconds"]=perf_counter()-begin
            publish_json(path,log)

if __name__=="__main__":
    main()
