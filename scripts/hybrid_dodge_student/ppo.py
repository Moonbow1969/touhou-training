"""Warm-start joint-movement PPO on explicitly designated future native development credits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical,kl_divergence

from .actor import OUT,ROOT,REPO,AnticipatoryContact,MovementStudent,feature_vectors
from .evaluate import PINNED,WEIGHT
from scripts import evaluate_credit_story as story,native_story_restart,native_story_sensors
from scripts import train_relative_stock_ppo as existing
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm,output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.relative_phase_student import RelativeActor
from scripts.train_full_policy import publish_json

RESERVED=frozenset((2402,2601,3301,3601,4001))
ROLE='development_on_policy_joint_movement'


class JointPolicy(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.incumbent=RelativeActor().eval().requires_grad_(False)
        self.incumbent.load_state_dict(weights['state_dict'])
        self.contact=AnticipatoryContact().eval().requires_grad_(False)
        self.contact.load_state_dict(weights['contact'])
        self.movement=MovementStudent()
        self.movement.load_state_dict(weights['movement'])
        self.critic=nn.Sequential(nn.Linear(70+5,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    @torch.inference_mode()
    def measured_risk(self,observation):
        """Cache raw/learned field features; the common helper supplies the actual stage one-hot separately."""
        b=torch.from_numpy(observation['bullets'][None])
        c=torch.from_numpy(observation['context'][None])
        return feature_vectors(self.contact,b,c,torch.ones(1,dtype=torch.long))[0,:-6]

    def heads(self,x,event,hp,raw,field):
        with torch.no_grad():
            base,shoot,bomb,*_=self.incumbent(x,event,hp,raw)
        vec=torch.cat((field,x[:,31:37]),1)
        move=self.movement.logits(vec,base.argmax(1))
        v=self.critic(torch.cat((vec,x[:,:3],hp[:,None],raw[:,2:3]/8),1)).squeeze(1)
        return (move,shoot,bomb),v


class LiveJoint:
    stage_clock=False
    motion_spans=None
    retain_recent_motion=False

    def __init__(self,model,temperature):
        self.model=model
        self.temp=temperature
        self.environment=self.credit=None
        self.rows=[]

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        row=existing.inputs(self,observation)
        heads,value=self.model.heads(*(v[None] for v in row))
        dist=Categorical(logits=heads[0]/self.temp)
        joint=dist.sample()[0]
        self.rows.append((row,joint,dist.log_prob(joint).item(),value.item()))
        move=int(joint)
        return np.asarray([move%9,move//9,int(heads[1].argmax()),int(heads[2].argmax())],np.int64),None


def update_joint(model,anchor,opt,batch,temperature):
    x,e,hp,stock,field,action,old,adv,y=batch
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    stats=[]
    for epoch in range(2):
        for idx in torch.randperm(len(x)).split(192):
            obs=(x[idx],e[idx],hp[idx],stock[idx],field[idx])
            logits,v=model.heads(*obs);policy=Categorical(logits=logits[0]/temperature)
            with torch.no_grad():
                fixed=Categorical(logits=anchor.heads(*obs)[0][0]/temperature)
            ratio=(policy.log_prob(action[idx])-old[idx]).exp()
            actor=-torch.minimum(ratio*adv[idx],ratio.clamp(.9,1.1)*adv[idx]).mean()
            critic=.5*(v-y[idx]).square().mean()
            kl=kl_divergence(policy,fixed).mean();entropy=policy.entropy().mean()
            opt.zero_grad(set_to_none=True)
            (actor+critic+.12*kl-.0005*entropy).backward()
            norm=nn.utils.clip_grad_norm_(list(model.movement.parameters())+list(model.critic.parameters()),.5)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite joint PPO gradient.')
            opt.step()
            stats.append([float(v.detach()) for v in (actor,critic,kl,entropy,norm)])
    return dict(zip(('policy_loss','value_loss','anchor_kl','entropy','gradient_norm'),np.mean(stats,axis=0).tolist()))


class DevelopmentTrace:
    def __init__(self,actor,archive):
        self.rewards=existing.HitRecorder(actor,'stage3')
        self.archive=archive
        self.sensors=None
        self.reward_components=dict(native_hit=0.,stage2_entry_and_resources=0.,stage2_spell_onsets=0.,
                                    stage3_entry_and_resources=0.)

    def before_step(self,env,obs):
        if self.sensors is None:
            self.sensors=self.archive.start(env)
        return self.rewards.before_step(env,obs),self.sensors.before_step(env,obs)

    def after_step(self,env,before,action,info,terminated,truncated,deaths,bombs):
        data=(action,info,terminated,truncated,deaths,bombs)
        prior_landmarks=len(self.rewards.landmarks)
        self.rewards.after_step(env,before[0],*data)
        hit=-.015 if before[0][0]==1 and env.snapshot.player_state==4 else 0.
        new_landmarks=self.rewards.landmarks[prior_landmarks:]
        spells=.06*sum(l['kind']=='first_vulnerable_spell' for l in new_landmarks)
        self.reward_components['native_hit']+=hit
        self.reward_components['stage2_spell_onsets']+=spells
        if any(l['kind']=='stage2_entry' for l in new_landmarks):
            self.reward_components['stage2_entry_and_resources']+=self.rewards.rewards[-1]-hit-spells
        self.sensors.after_step(env,before[1],*data)

    def finish(self,outcome,obs):
        self.rewards.finish(outcome,obs)
        self.sensors.finish(outcome,obs)


def fresh_optimizer(model):
    return torch.optim.Adam([{'params':model.movement.parameters(),'lr':4e-5},
                             {'params':model.critic.parameters(),'lr':7e-4}],eps=1e-5)


def verify(model,anchor,temperature):
    """Use synthetic data only. Verify identical initial distributions and an actual clipped PPO update."""
    n=96
    x=torch.randn(n,42);x[:,31:37]=torch.nn.functional.one_hot(torch.arange(n)%6,6).float()
    e=torch.zeros(n,5);hp=torch.rand(n);raw=torch.ones(n,3);raw[:,0]=1
    bullets=torch.rand(n,96,7);ctx=torch.randn(n,10).clamp(-1,1)
    vec=feature_vectors(model.contact,bullets,ctx,x[:,31:37].argmax(1)+1).detach()
    args=(x,e,hp,raw,vec[:,:-6])
    with torch.no_grad():
        current,value=model.heads(*args)
        base,shoot,bomb,*_=model.incumbent(x,e,hp,raw)
        reference=model.movement.logits(vec,base.argmax(1))
        torch.testing.assert_close(current[0],reference,atol=0,rtol=0)
        torch.testing.assert_close(current[1],shoot,atol=0,rtol=0)
        torch.testing.assert_close(current[2],bomb,atol=0,rtol=0)
        dist=Categorical(logits=current[0]/temperature);act=dist.sample();old=dist.log_prob(act)
        ratio=(dist.log_prob(act)-old).exp()
    before={k:v.clone() for k,v in model.state_dict().items()}
    update=update_joint(model,anchor,fresh_optimizer(model),(*args,act,old,torch.randn(n),torch.randn(n)),temperature)
    altered=[k for k,t in model.state_dict().items() if not torch.equal(t,before[k])]
    assert any(k.startswith('movement.') for k in altered) and any(k.startswith('critic.') for k in altered)
    assert all(k.startswith(('movement.','critic.')) for k in altered)
    return dict(contract='synthetic offline smoke only; no logged native actions used as PPO',
                exact_initial_joint_logits=n,exact_initial_shoot_and_bomb=n,initial_critic_max=float(value.abs().max()),
                old_ratio_range=[float(ratio.min()),float(ratio.max())],changed_tensor_names=altered,synthetic_update=update)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--seed',type=int,default=4101)
    p.add_argument('--credits',type=int,choices=range(1,9),default=2)
    p.add_argument('--temperature',type=float,default=.55)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    seeds=list(range(a.seed,a.seed+a.credits))
    if min(seeds)<0 or RESERVED.intersection(seeds) or not .1<=a.temperature<=2 or not a.verify_only and a.out is None:
        p.error('Use distinct prospective development seeds, a valid temperature and a fresh output.')
    torch.manual_seed(a.seed);torch.set_num_threads(1)
    for path,wanted in PINNED.items():
        if digest(path)!=wanted:
            raise ValueError('Frozen source mismatch: '+path.name)
    data=torch.load(WEIGHT,weights_only=True,map_location='cpu')
    model=JointPolicy(data);anchor=deepcopy(model).requires_grad_(False).eval()
    if a.verify_only:
        print(json.dumps(dict(weights_sha256=digest(WEIGHT),seeds=seeds,**verify(model,anchor,a.temperature))))
        return
    a.target='stage3';a.limit=story.LIMITS[a.target];a.start_frame=30;a.focus_wait=90
    a.parakit=REPO.parent/'parakit';a.seeds=seeds;a.crossover=[];a.timing_policy=[];a.restart_hook=None
    a.menu_crop=[[39,256,48,20],[39,356,118,20],[15,207,70,24]];a.menu_max_error=.035
    a.game_over_default_reference=ROOT/'work/th14-menu-default-runtime-261125.png'
    a.game_over_retry_reference=ROOT/'work/th14-menu-retry-runtime-261125.png'
    a.transition_budget_mib=128;a.transition_reserve_mib=20480
    runner=LiveJoint(model,a.temperature)
    arm=Arm(runner,False,dict(role=ROLE,source_weights_sha256=digest(WEIGHT),sampled='joint movement/focus',
                         shoot_and_bomb='original greedy frozen',geometry=False))
    arms={'development_joint_ppo':arm}
    spec=story.run_specification(a,arms)
    spec['runtime']['runner_sha256']=digest(Path(__file__))
    spec['runtime']['source_sha256'][existing.__name__]=digest(Path(existing.__file__))
    spec['runtime']['source_sha256'].update({p.name:h for p,h in PINNED.items() if p.suffix=='.py'})
    spec['development']=dict(role=ROLE,temperature=a.temperature,credits=a.credits,updates=1,
         gamma=1.,gae_lambda=.995,clip=.10,epochs=2,entropy=.0005,kl_to_frozen=.12,
         actor_lr=4e-5,critic_lr=7e-4,fresh_optimizer=True,zero_initialized_critic=True,
         reward="Stage3 objective from train_relative_stock_ppo including Stage2 entry, native hit, and unique vulnerable Stage2 spell; no geometry reward.")
    archive=native_story_sensors.configure(a,spec)
    menu=GameOverMenu(a.game_over_default_reference,a.game_over_retry_reference,a.menu_crop,.035)
    a.out.mkdir(parents=True,exist_ok=False)
    log=dict(status='collecting',specification=spec,episodes=[],updates=[],steps=0)
    path=a.out/'report.json';chunks=[]
    with output_lock(a.out):
        try:
            publish_json(path,log)
            with story.native_session(a,arms,menu,None) as (env,prep):
                log['preparation']=prep
                for trial in spec['schedule']:
                    runner.rows=[]
                    rec=DevelopmentTrace(runner,archive)
                    def reset(stage,tr):
                        return native_story_restart.reset(stage,tr,start_frame=30,menu=menu)
                    def progress(row):
                        log['in_progress']=dict(trial=trial,result=row);publish_json(path,log)
                    result=story.story_credit(env,arm,trial,a,reset,progress,transition_recorder=rec)
                    reward=rec.rewards
                    if result['target_reached']:
                        stock=result['stage_records'][-1]['entry']['resources']
                        terminal_bonus=1.5+.1*stock['lives']+.02*stock['bombs']
                        reward.rewards[-1]+=terminal_bonus
                        rec.reward_components['stage3_entry_and_resources']=terminal_bonus
                    if not np.isclose(sum(rec.reward_components.values()),sum(reward.rewards),atol=1e-6):
                        raise ValueError('Native reward and separately measured progress components diverged.')
                    result['transition_shard']=archive.publish(trial,result,rec.sensors)
                    batch=existing.pack(runner.rows,reward.rewards,reward.bootstrap);chunks.append(batch)
                    buf=existing.publish_tensors(a.out/f"credit-{trial['policy_seed']}-onpolicy.pt",
                        dict(schema='th14-hybrid-joint-ppo-buffer-v1',steps=len(runner.rows),tensors=batch,
                             rewards=reward.rewards,bootstrap=reward.bootstrap,source=digest(WEIGHT),
                             policy_seed=trial['policy_seed'],role=ROLE))
                    log['episodes'].append(dict(trial=trial,result=result,onpolicy_buffer=buf,native_return=sum(reward.rewards),
                                                reward_components=rec.reward_components,landmarks=reward.landmarks))
                    log['steps']+=len(runner.rows);log.pop('in_progress',None);publish_json(path,log)
            batch=tuple(torch.cat(v) for v in zip(*chunks))
            stats=update_joint(model,anchor,fresh_optimizer(model),batch,a.temperature)
            output=dict(data,movement=model.movement.state_dict())
            # Inference consumes only weights, so no training scheduler or critic enters deployment.
            candidate=existing.publish_tensors(a.out/'candidate.pt',dict(output,steps=log['steps']))
            critic=existing.publish_tensors(a.out/'critic.pt',dict(schema='th14-hybrid-joint-critic-v1',steps=log['steps'],state_dict=model.critic.state_dict()))
            log['updates']=[dict(candidate=candidate,critic=critic,measurements=stats)]
            log['status']='complete'
        except BaseException as err:
            log.update(status='interrupted',error=str(err));raise
        finally:
            publish_json(path,log)


if __name__=='__main__':
    main()
