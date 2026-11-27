"""Full-resolution, title-balanced PPO initialized from a human camera SFT checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.train_replay_camera_encoder import CameraPolicy, channels, save_json, sha
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_portable_full_sft import WIDTHS
from scripts.train_human_camera_policy import load_replay


class CameraActorCritic(nn.Module):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.value = nn.Linear(128,1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, images):
        latent = self.camera.policy[:-1](self.camera.spatial(images))
        logits = self.camera.policy[-1](latent).split(WIDTHS,dim=1)
        return (*logits, self.value(latent))


class FeedbackActorCritic(CameraActorCritic):
    def __init__(self, camera):
        super().__init__(camera)
        from scripts.train_camera_direction_feedback import DirectionResidual
        self.direction_residual=DirectionResidual()

    def forward(self,images,previous_direction):
        latent=self.camera.policy[:-1](self.camera.spatial(images))
        movement,focus,shot,bomb=self.camera.policy[-1](latent).split(WIDTHS,dim=1)
        return self.direction_residual(latent,movement,previous_direction),focus,shot,bomb,self.value(latent)


def chosen(heads, actions):
    return torch.stack([h.log_softmax(1).gather(1, actions[:,k:k+1])[:,0] for k,h in enumerate(heads)],1)


def logits_on(model, images, device, previous_moves=None):
    with torch.inference_mode():
        output=[]
        for start in range(0,len(images),48):
            block=images[start:start+48].to(device).float()/255
            args=() if previous_moves is None else (previous_moves[start:start+48].to(device),)
            output.append(model(block,*args))
        return tuple(torch.cat(part).cpu() for part in zip(*output))


def credit(record, manifest, model, parent, device, *, direction_history=False, actor_baseline=None):
    if record['role']!='train' or record['policy_sha256']!=parent or record['outcome'] not in ('game_over','ending'):
        raise ValueError('Unverified policy or endpoint')
    path=manifest.parent/record['file']
    if sha(path)!=record['sha256']:raise ValueError('Changed original credit')
    with np.load(path,allow_pickle=False) as z:
        frames=z['frames']; action=z['action'].astype(np.int64); native_lp=z['logp']; native_v=z['value'].reshape(-1)
        reward=z['reward'].reshape(-1);done=z['terminated'].reshape(-1).astype(bool)
        eligibility=z['native_action_eligible'] if 'native_action_eligible' in z else None
        prior=z['previous_move'] if direction_history else None
    n=len(action)
    if not n or reward.shape!=(n,) or done.shape!=(n,) or frames.shape!=(n,112,96) or frames.dtype!=np.uint8 or not done[-1] or done[:-1].any():
        raise ValueError(f'Incomplete original camera sequence: {path.name}')
    if np.any(action<0) or np.any(action>=np.asarray(WIDTHS)) or not np.isfinite(reward).all():
        raise ValueError('Illegal native action or reward')
    if native_lp.shape!=(n,4) or not np.isfinite(native_lp).all() or len(native_v)!=n or not np.isfinite(native_v).all():
        raise ValueError('Invalid native behavior evidence')
    images=torch.from_numpy(channels(frames,np.concatenate([frames[:1], frames[:-1]])))
    actions=torch.from_numpy(action)
    if direction_history and not np.array_equal(prior,np.r_[4,action[:-1,0]]):
        raise ValueError('History-aware PPO requires the exact native prior for each action')
    prior=torch.from_numpy(prior.astype(np.int64)) if direction_history else None
    *heads,values=logits_on(model,images,device,prior)
    logp=chosen(heads,actions)
    err=float((logp-torch.from_numpy(native_lp)).abs().max())
    verr=float((values[:,0]-torch.from_numpy(native_v)).abs().max())
    if err>8e-4 or verr>3e-4:
        raise ValueError(f'Actual camera inference mismatch {path.name}: {err:.6f}/{verr:.6f}')
    from scripts.th12_playable_actions import playable_length
    length=playable_length(record['title'],eligibility,reward,record.get('postterminal_input_boundary'))
    eligibility_audit={}
    if length!=n:
        eligibility_audit=dict(raw_sampled_decisions=n,reforwarded_decisions=n,playable_decisions=length,
                               postterminal_excluded_decisions=n-length,finite_terminal_index=length-1,bootstrap_value=0.)
        images=images[:length];actions=actions[:length];native_lp=native_lp[:length]
        native_v=native_v[:length];reward=reward[:length];heads=[h[:length] for h in heads]
    adv=calculate_gae(reward,native_v,.999,.985)
    returns=native_v+adv
    if actor_baseline is not None:
        if actor_baseline.shape!=(length,) or actor_baseline.dtype!=np.float32 or not np.isfinite(actor_baseline).all():
            raise ValueError('Invalid out-of-fold actor baseline')
        original=adv
        adv=calculate_gae(reward,actor_baseline,.999,.985)
        eligibility_audit.update(native_advantage_sd=float(original.std()),crossfit_advantage_sd=float(adv.std()),
                                 advantage_mean_absolute_change=float(np.mean(abs(adv-original))))
    arrays=(images,actions,torch.from_numpy(native_lp.sum(1)),torch.from_numpy(adv),
            torch.from_numpy(returns),torch.cat(heads,dim=1))
    if direction_history:
        arrays+= (prior[:length],)
    return arrays,dict(title=record['title'],seed=record['seed'],steps=length,reward=float(reward.sum()),**eligibility_audit,
                       max_native_logp_error=err,max_native_value_error=verr)


def replay_anchor(paths, supervised_directory, model, device):
    report=json.loads((supervised_directory/'report.json').read_bytes())
    expected={r['sha256'] for r in report['train']}
    if len(paths)!=len(expected) or {sha(path) for path in paths}!=expected:
        raise ValueError('Replay anchor requires the exact supervised training split')
    by_title={}
    for path in paths:
        row=load_replay(path)
        images=torch.from_numpy(row['raw'])
        *heads,_=logits_on(model,images,device)
        by_title.setdefault(row['title'],[]).append((images,torch.cat(heads,dim=1)))
    return by_title


def anchor_loss(model, by_title, rng, device):
    # Draw a replay uniformly inside each title, independent of the PPO permutation.
    images=[];teachers=[]
    for replays in by_title.values():
        source,logits=replays[int(torch.randint(len(replays),(1,),generator=rng))]
        ids=torch.randint(len(source),(12,),generator=rng)
        images.append(source[ids]);teachers.append(logits[ids])
    *current,_=model(torch.cat(images).to(device).float()/255)
    target=torch.cat(teachers).to(device).split(WIDTHS,dim=1)
    return sum(F.kl_div(h.log_softmax(1),ref.softmax(1),reduction='batchmean') for h,ref in zip(current,target))


def fit(model,data,lr,device,anchor=None,anchor_weight=0.):
    opt=torch.optim.Adam(model.parameters(),lr=lr,eps=1e-5)
    history=[];anchor_rng=torch.Generator().manual_seed(734119)
    for epoch in range(4):
        previous=copy.deepcopy(model.state_dict())
        chunks=max(1,round(sum(len(v[0]) for v in data.values())/144))
        order={t:torch.tensor_split(torch.randperm(len(v[0])),chunks) for t,v in data.items()}
        losses=[];anchor_terms=[]
        for k in range(chunks):
            opt.zero_grad(set_to_none=True)
            for t,(images,a,old,adv,returns,_,*history_input) in data.items():
                ids=order[t][k]
                # Accumulate equal title weights with bounded image minibatch memory.
                for small in ids.split(48):
                    px=images[small].to(device).float()/255
                    *heads,value=model(px,*(prior[small].to(device) for prior in history_input))
                    lp=chosen(heads,a[small].to(device)).sum(1)
                    ratio=(lp-old[small].to(device)).exp()
                    advantage=adv[small].to(device)
                    clip=torch.minimum(ratio*advantage,ratio.clamp(.88,1.12)*advantage)
                    entropy=sum(-(h.softmax(1)*h.log_softmax(1)).sum(1) for h in heads)
                    objective=-clip.mean()+.25*F.smooth_l1_loss(value[:,0],returns[small].to(device))-.002*entropy.mean()
                    weighted=objective*len(small)/len(ids)/len(data)
                    weighted.backward();losses.append(weighted.detach().item())
            if anchor is not None:
                term=anchor_loss(model,anchor,anchor_rng,device)
                (term*anchor_weight).backward()
                anchor_terms.append(term.detach().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
        kl={}
        for t,(images,_,_,_,_,ref,*history_input) in data.items():
            *current,_=logits_on(model,images,device,*history_input)
            kl[t]=max(0.,sum((h.softmax(1)*(h.log_softmax(1)-g.log_softmax(1))).sum(1).mean().item()
                             for h,g in zip(ref.split(WIDTHS,1),current)))
        accepted=max(kl.values())<=.025
        history.append(dict(epoch=epoch+1,lr=lr,loss_per_minibatch=sum(losses)/chunks,
                            per_title_joint_kl=kl,accepted=accepted))
        if anchor_terms:history[-1]['mean_joint_kl_on_training_replay']=float(np.mean(anchor_terms))
        print(history[-1],flush=True)
        if not accepted:
            model.load_state_dict(previous);break
    return history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','protocol','behavior','out'):p.add_argument('--'+key,required=True,type=Path)
    initial=p.add_mutually_exclusive_group(required=True)
    initial.add_argument('--sft',type=Path)
    initial.add_argument('--parent-actor-critic',type=Path)
    p.add_argument('--direction-residual',type=Path,help='Qualified previous-direction movement head fitted on top of the specified SFT camera.')
    p.add_argument('--replay-anchor',nargs='+',type=Path,help='Exact human image training shards for KL to the initial SFT actor')
    p.add_argument('--anchor-weight',type=float,default=0.)
    p.add_argument('--actor-advantage-baselines',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(261127)
    if (a.replay_anchor and (not a.sft or a.anchor_weight<=0)) or (not a.replay_anchor and a.anchor_weight!=0.):
        p.error('A replay anchor needs an SFT parent, its training shards, and a positive weight')
    # Native actors execute full FP32 convolutions on CPU. TF32 can shift a
    # rare action's log probability enough to invalidate the behavior check.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    camera=CameraPolicy()
    state=torch.load(a.sft or a.parent_actor_critic,weights_only=True,map_location='cpu')
    if any('policy.spatial_context.' in name for name in state):
        camera.add_spatial_context()
    if a.sft:camera.load_state_dict(state)
    plan=json.loads(a.protocol.read_bytes())
    feedback=bool(plan.get('previous_direction_input'))
    if a.direction_residual and (not a.sft or not feedback):
        raise ValueError('The registered H1 SFT initialization must match the supplied residual')
    if feedback and a.sft and (not a.direction_residual or sha(a.direction_residual)!=plan['residual_sha256']
                               or sha(a.sft)!=plan['camera_sft_sha256']):
        raise ValueError('Changed supervised direction checkpoint')
    if feedback and a.replay_anchor:
        raise ValueError('The image-only replay anchor has no native previous-direction inputs')
    model=FeedbackActorCritic(camera) if feedback else CameraActorCritic(camera)
    if a.direction_residual:
        model.direction_residual.load_state_dict(torch.jit.load(str(a.direction_residual),map_location='cpu').state_dict())
    if a.parent_actor_critic:model.load_state_dict(state)
    model=model.to(device)
    parent=sha(a.behavior);rows=json.loads(a.manifest.read_bytes())
    if feedback and parent!=plan['behavior_sha256']:
        raise ValueError('Changed history-conditioned behavior')
    expected={(int(t),s) for t,seeds in plan['train_seeds'].items() for s in seeds}
    if {(r['title'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):
        raise ValueError('Missing or duplicate training credit')
    by_title={};quality=[]
    baselines={}
    if a.actor_advantage_baselines or plan.get('actor_advantage_baselines_sha256'):
        if not a.actor_advantage_baselines or sha(a.actor_advantage_baselines)!=plan.get('actor_advantage_baselines_sha256'):
            raise ValueError('Actor baselines differ from the registered plan')
        with np.load(a.actor_advantage_baselines,allow_pickle=False) as archive:
            if set(archive.files)!={f'th{t}-{s}' for t,s in expected} or len(archive.files)!=len(expected):
                raise ValueError('Expected one out-of-fold baseline per training credit')
            baselines=dict(archive)
    for r in rows:
        key=f"th{r['title']}-{r['seed']}"
        arrays,details=credit(r,a.manifest,model,parent,device,direction_history=feedback,actor_baseline=baselines.get(key))
        group=r['title']
        if plan.get('balanced_title_rank'):
            rank=plan['rank_by_seed'][str(group)][str(r['seed'])]
            if r.get('difficulty')!=rank or not r.get('difficulty_evidence_sha256'):
                raise ValueError('Unverified native title/difficulty stratum')
            details['difficulty']=rank
            group=f'TH{group}-'+{2:'Hard',3:'Lunatic'}[rank]
        by_title.setdefault(group,[]).append(arrays);quality.append(details)
    data={}
    sampling={}
    for t,records in by_title.items():
        x,actions,lp,adv,ret,ref,*history_input=[torch.cat(v) for v in zip(*records)]
        if 'matched_decisions_per_stratum' in plan:
            n=plan['matched_decisions_per_stratum']
            if not isinstance(n,int) or not 1<n<=len(x):
                raise ValueError('Invalid prospective matched decision budget')
            # Each full terminal sequence already supplied its actual GAE and returns.
            # Only the optimizer rows are sampled. No artificial terminal is introduced.
            order=torch.randperm(len(x),generator=torch.Generator().manual_seed(2851300))[:n].sort().values
            sampling[str(t)]=dict(available=len(x),selected=n,
                                  sorted_indices_sha256=hashlib.sha256(order.numpy().astype('<i8').tobytes()).hexdigest())
            x,actions,lp,adv,ret,ref=(v[order] for v in (x,actions,lp,adv,ret,ref))
            history_input=[v[order] for v in history_input]
        data[t]=(x,actions,lp,(adv-adv.mean())/(adv.std(unbiased=False)+1e-7),ret,ref,*history_input)
    anchor=replay_anchor(a.replay_anchor,a.sft.parent,model,device) if a.replay_anchor else None
    history=[]
    for lr in plan['learning_rates']:
        torch.manual_seed(261127)
        attempt=fit(model,data,lr,device,anchor,a.anchor_weight);history+=attempt
        if any(v['accepted'] for v in attempt):break
    else:raise RuntimeError('No accepted update after reducing the learning rate')
    a.out.mkdir(parents=True,exist_ok=False)
    model.cpu().eval()
    torch.save(model.state_dict(),a.out/'camera-ppo.pt')
    with torch.no_grad():
        fixture=data[sorted(data)[0]][0][:4].float()/255
        inputs=(fixture,data[sorted(data)[0]][6][:4]) if feedback else (fixture,)
        script=torch.jit.trace(model,tuple(v[:1] for v in inputs))
        for expected_tensor,actual in zip(model(*inputs),script(*inputs)):
            torch.testing.assert_close(expected_tensor,actual)
        script.save(str(a.out/'policy.ts'))
    save_json(a.out/'report.json',dict(parent_sft_sha256=sha(a.sft) if a.sft else None,
              parent_actor_critic_sha256=sha(a.parent_actor_critic) if a.parent_actor_critic else None,native_behavior_sha256=parent,
              policy_sha256=sha(a.out/'policy.ts'),manifest_sha256=sha(a.manifest),steps=sum(r['steps'] for r in quality),
              protocol_sha256=sha(a.protocol),credits=quality,history=history,
              equal_weight_strata=sorted(data),
              **({'actor_advantage_baselines_sha256':sha(a.actor_advantage_baselines)} if baselines else {}),
              **({'previous_direction_input':True,'direction_residual_sha256':sha(a.direction_residual) if a.direction_residual else None} if feedback else {}),
              **({'optimizer_decision_sampling':sampling} if sampling else {}),
              replay_anchor_weight=a.anchor_weight,replay_anchor_training_sha256=[sha(p) for p in (a.replay_anchor or [])],
              native_procedure="All four action heads and the spatial encoder receive gradients. The SFT critic initializes at zero; later generations restore the trained critic. Game-over and ending are full-credit terminals."))


if __name__=='__main__':main()
