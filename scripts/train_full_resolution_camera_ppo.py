"""Full-resolution, title-balanced PPO initialized from a human camera SFT checkpoint."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.train_replay_camera_encoder import CameraPolicy, channels, save_json, sha
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_portable_full_sft import WIDTHS


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


def chosen(heads, actions):
    return torch.stack([h.log_softmax(1).gather(1, actions[:,k:k+1])[:,0] for k,h in enumerate(heads)],1)


def logits_on(model, images, device):
    with torch.inference_mode():
        output=[model(b.to(device).float()/255) for b in images.split(48)]
        return tuple(torch.cat(part).cpu() for part in zip(*output))


def credit(record, manifest, model, parent, device):
    if record['role']!='train' or record['policy_sha256']!=parent or record['outcome'] not in ('game_over','ending'):
        raise ValueError('Unverified policy or endpoint')
    path=manifest.parent/record['file']
    if sha(path)!=record['sha256']:raise ValueError('Changed original credit')
    with np.load(path,allow_pickle=False) as z:
        frames=z['frames']; action=z['action'].astype(np.int64); native_lp=z['logp']; native_v=z['value'].reshape(-1)
        reward=z['reward'].reshape(-1);done=z['terminated'].reshape(-1).astype(bool)
    n=len(action)
    if not n or frames.shape!=(n,112,96) or frames.dtype!=np.uint8 or not done[-1] or done[:-1].any():
        raise ValueError(f'Incomplete original camera sequence: {path.name}')
    if np.any(action<0) or np.any(action>=np.asarray(WIDTHS)) or not np.isfinite(reward).all():
        raise ValueError('Illegal native action or reward')
    if native_lp.shape!=(n,4) or not np.isfinite(native_lp).all() or len(native_v)!=n or not np.isfinite(native_v).all():
        raise ValueError('Invalid native behavior evidence')
    images=torch.from_numpy(channels(frames,np.concatenate([frames[:1], frames[:-1]])))
    actions=torch.from_numpy(action)
    *heads,values=logits_on(model,images,device)
    logp=chosen(heads,actions)
    err=float((logp-torch.from_numpy(native_lp)).abs().max())
    verr=float((values[:,0]-torch.from_numpy(native_v)).abs().max())
    if err>8e-4 or verr>3e-4:
        raise ValueError(f'Actual camera inference mismatch {path.name}: {err:.6f}/{verr:.6f}')
    adv=calculate_gae(reward,native_v,.999,.985)
    arrays=(images,actions,torch.from_numpy(native_lp.sum(1)),torch.from_numpy(adv),
            torch.from_numpy(native_v+adv),torch.cat(heads,dim=1))
    return arrays,dict(title=record['title'],seed=record['seed'],steps=n,reward=float(reward.sum()),
                       max_native_logp_error=err,max_native_value_error=verr)


def fit(model,data,lr,device):
    opt=torch.optim.Adam(model.parameters(),lr=lr,eps=1e-5)
    history=[]
    for epoch in range(4):
        previous=copy.deepcopy(model.state_dict())
        chunks=max(1,round(sum(len(v[0]) for v in data.values())/144))
        order={t:torch.tensor_split(torch.randperm(len(v[0])),chunks) for t,v in data.items()}
        losses=[]
        for k in range(chunks):
            opt.zero_grad(set_to_none=True)
            for t,(images,a,old,adv,returns,_) in data.items():
                ids=order[t][k]
                # Accumulate equal title weights with bounded image minibatch memory.
                for small in ids.split(48):
                    px=images[small].to(device).float()/255
                    *heads,value=model(px);lp=chosen(heads,a[small].to(device)).sum(1)
                    ratio=(lp-old[small].to(device)).exp()
                    advantage=adv[small].to(device)
                    clip=torch.minimum(ratio*advantage,ratio.clamp(.88,1.12)*advantage)
                    entropy=sum(-(h.softmax(1)*h.log_softmax(1)).sum(1) for h in heads)
                    objective=-clip.mean()+.25*F.smooth_l1_loss(value[:,0],returns[small].to(device))-.002*entropy.mean()
                    weighted=objective*len(small)/len(ids)/len(data)
                    weighted.backward();losses.append(weighted.detach().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
        kl={}
        for t,(images,_,_,_,_,ref) in data.items():
            *current,_=logits_on(model,images,device)
            kl[t]=max(0.,sum((h.softmax(1)*(h.log_softmax(1)-g.log_softmax(1))).sum(1).mean().item()
                             for h,g in zip(ref.split(WIDTHS,1),current)))
        accepted=max(kl.values())<=.025
        history.append(dict(epoch=epoch+1,lr=lr,loss_per_minibatch=sum(losses)/chunks,
                            per_title_joint_kl=kl,accepted=accepted))
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
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(261127)
    # Native actors execute full FP32 convolutions on CPU. TF32 can shift a
    # rare action's log probability enough to invalidate the behavior check.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    camera=CameraPolicy()
    if a.sft:camera.load_state_dict(torch.load(a.sft,weights_only=True,map_location='cpu'))
    model=CameraActorCritic(camera)
    if a.parent_actor_critic:model.load_state_dict(torch.load(a.parent_actor_critic,weights_only=True,map_location='cpu'))
    model=model.to(device)
    parent=sha(a.behavior);rows=json.loads(a.manifest.read_bytes());plan=json.loads(a.protocol.read_bytes())
    expected={(int(t),s) for t,seeds in plan['train_seeds'].items() for s in seeds}
    if {(r['title'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):
        raise ValueError('Missing or duplicate training credit')
    by_title={};quality=[]
    for r in rows:
        arrays,details=credit(r,a.manifest,model,parent,device)
        by_title.setdefault(r['title'],[]).append(arrays);quality.append(details)
    data={}
    for t,records in by_title.items():
        x,actions,lp,adv,ret,ref=[torch.cat(v) for v in zip(*records)]
        data[t]=(x,actions,lp,(adv-adv.mean())/(adv.std(unbiased=False)+1e-7),ret,ref)
    history=[]
    for lr in plan['learning_rates']:
        torch.manual_seed(261127)
        attempt=fit(model,data,lr,device);history+=attempt
        if any(v['accepted'] for v in attempt):break
    else:raise RuntimeError('No accepted update after reducing the learning rate')
    a.out.mkdir(parents=True,exist_ok=False)
    model.cpu().eval()
    torch.save(model.state_dict(),a.out/'camera-ppo.pt')
    with torch.no_grad():
        fixture=data[sorted(data)[0]][0][:4].float()/255
        script=torch.jit.trace(model,fixture[:1])
        for expected_tensor,actual in zip(model(fixture),script(fixture)):
            torch.testing.assert_close(expected_tensor,actual)
        script.save(str(a.out/'policy.ts'))
    save_json(a.out/'report.json',dict(parent_sft_sha256=sha(a.sft) if a.sft else None,
              parent_actor_critic_sha256=sha(a.parent_actor_critic) if a.parent_actor_critic else None,native_behavior_sha256=parent,
              policy_sha256=sha(a.out/'policy.ts'),manifest_sha256=sha(a.manifest),steps=sum(r['steps'] for r in quality),
              protocol_sha256=sha(a.protocol),credits=quality,history=history,
              native_procedure="All four action heads and the spatial encoder receive gradients. The SFT critic initializes at zero; later generations restore the trained critic. Game-over and ending are full-credit terminals."))


if __name__=='__main__':main()
