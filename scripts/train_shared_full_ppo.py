"""Fit one synchronous, title-balanced full-action PPO round from complete native credits."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scripts.checkpoint_storage import digest
from scripts.train_portable_full_sft import FullActor,WIDTHS
from scripts.probe_native_loss_risk import loss_targets


def likelihood(model,x,action):
    heads,values=model(x)
    lp=[h.log_softmax(1).gather(1,action[:,k:k+1])[:,0] for k,h in enumerate(heads)]
    entropy=sum(-(h.softmax(1)*h.log_softmax(1)).sum(1) for h in heads)
    return torch.stack(lp,1),values,entropy


def calculate_gae(rewards,values,gamma,lam):
    # Genuine full-credit terminals have zero bootstrap. Respawns retain continuity.
    advantage=np.zeros(len(rewards),np.float32)
    next_value=0.;running=0.
    for k in reversed(range(len(rewards))):
        running=rewards[k]+gamma*next_value-values[k]+gamma*lam*running
        advantage[k]=running;next_value=values[k]
    return advantage


def read_credit(record,manifest,model,parent_sha,gamma,lam):
    path=manifest.parent/record["file"]
    if record["role"]!="train" or record["policy_sha256"]!=parent_sha or digest(path)!=record["sha256"]:
        raise ValueError("Unverified behavior data or non-training credit.")
    if record.get("outcome") not in ("game_over","clear","ending","stage_clear_ending"):
        raise ValueError("Requires an uninterrupted game-over or ending.")
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z["features"],dtype=np.float32)
        a=np.asarray(z["action"])
        lp=np.asarray(z["logp"],dtype=np.float32)
        v=np.asarray(z["value"],dtype=np.float32).reshape(-1)
        rewards=np.asarray(z["reward"],dtype=np.float32).reshape(-1)
        done=np.asarray(z["terminated"],dtype=bool).reshape(-1)
    n=len(x)
    if (x.shape!=(n,110) or not n or a.shape!=(n,4) or lp.shape!=(n,4) or
        len(rewards)!=n or len(v)!=n or len(done)!=n or not done[-1] or np.any(done[:-1]) or
        not np.isfinite(x).all() or not np.isfinite(lp).all() or not np.isfinite(v).all() or not np.isfinite(rewards).all()):
        raise ValueError(f"Invalid or truncated trajectory {path}")
    if np.any(a!=np.floor(a)) or np.any(a<0) or np.any(a>=np.asarray(WIDTHS)):
        raise ValueError("Invalid factored actions.")
    xt=torch.from_numpy(x.copy());at=torch.from_numpy(a.astype(np.int64))
    with torch.no_grad():
        check_lp,check_v,_=likelihood(model,xt,at)
        lp_error=np.max(np.abs(check_lp.numpy()-lp));v_error=np.max(np.abs(check_v.numpy()-v))
    if lp_error>5e-4 or v_error>3e-4:
        raise ValueError(f"On-policy likelihood/value replay failed for {path.name}: {lp_error}/{v_error}")
    adv=calculate_gae(rewards,v,gamma,lam)
    return (xt,at,torch.from_numpy(lp.sum(1)),torch.from_numpy(adv),torch.from_numpy(v+adv)),dict(
        title=record['title'],seed=record['seed'],steps=n,total_reward=float(rewards.sum()),action_counts=[np.bincount(a[:,k].astype(int),minlength=m).tolist() for k,m in enumerate(WIDTHS)],
        max_logp_error=float(lp_error),max_value_error=float(v_error))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","parent","protocol","out"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    protocol=json.loads(args.protocol.read_bytes())
    config=protocol["ppo"]
    rows=json.loads(args.manifest.read_bytes())
    parent_sha=digest(args.parent)
    if Counter((int(r['title']),int(r['seed'])) for r in rows)!=Counter((int(t),s) for t,seeds in protocol['train_seeds'].items() for s in seeds):
        raise ValueError("Incomplete, duplicated or substituted multi-title training block.")
    if len(set(r['sha256'] for r in rows))!=len(rows):raise ValueError("Repeated trajectory.")
    torch.manual_seed(config['seed']);torch.set_num_threads(2)
    model=FullActor.from_bin(args.parent);reference=FullActor.from_bin(args.parent).eval()
    by_title=defaultdict(list);quality=[]
    aux_config=config.get("visual_loss_aux")
    for r in rows:
        arrays,metrics=read_credit(r,args.manifest,reference,parent_sha,config['gamma'],config['gae_lambda'])
        if aux_config:
            with np.load(args.manifest.parent/r["file"],allow_pickle=False) as trace:
                target,losses=loss_targets(trace,int(r["title"]),aux_config["horizon"])
            known=(np.arange(len(target))+aux_config["horizon"]<=len(target))|target
            if not np.any(losses):
                raise ValueError("Missing native loss events for auxiliary training.")
            arrays=(*arrays,torch.tensor(np.where(known,target,-1),dtype=torch.float32))
        by_title[int(r['title'])].append(arrays);quality.append(metrics)
    data={}
    for t,credits in by_title.items():
        x,a,lp,adv,ret,*extra=map(torch.cat,zip(*credits))
        data[t]=(x,a,lp,(adv-adv.mean())/(adv.std(unbiased=False)+1e-7),ret,*extra)
    parameters=list(model.parameters())
    if aux_config:
        # The probe uses no additional RNG draws. The paired PPO comparison keeps
        # the same shuffle, original policy initialization and optimizer settings.
        risk_weights=torch.nn.Parameter(torch.zeros(model.output.in_features))
        risk_bias=torch.nn.Parameter(torch.zeros(()))
        parameters.extend((risk_weights,risk_bias))
        prevalences={t:bank[-1][bank[-1]>=0].mean() for t,bank in data.items()}
    opt=torch.optim.Adam(parameters,lr=config['lr'],eps=1e-5)
    history=[]
    for epoch in range(config['epochs']):
        prior={k:v.clone() for k,v in model.state_dict().items()}
        chunks=min(min(len(v[0]) for v in data.values()),max(1,round(sum(len(v[0]) for v in data.values())/config['batch_size'])))
        order={t:torch.tensor_split(torch.randperm(len(v[0])),chunks) for t,v in data.items()}
        losses=[]
        for k in range(chunks):
            per_title=[]
            for t,v in data.items():
                x,a,lp,adv,ret,*extra=[val[order[t][k]] for val in v]
                now,value,entropy=likelihood(model,x,a)
                ratio=(now.sum(1)-lp).exp()
                surrogate=torch.minimum(ratio*adv,ratio.clamp(1-config['clip'],1+config['clip'])*adv)
                loss=-surrogate.mean()+config['vf']*F.smooth_l1_loss(value,ret)-config['entropy']*entropy.mean()
                if aux_config:
                    screen_only=torch.zeros_like(x)
                    screen_only[:,:84]=x[:,:84]
                    risk=model.encode(screen_only)@risk_weights+risk_bias
                    target=extra[0]
                    prevalence=prevalences[t]
                    valid=target>=0
                    weights=valid*torch.where(target>0,.5/prevalence,.5/(1-prevalence))
                    aux_loss=F.binary_cross_entropy_with_logits(risk,target.clamp_min(0),weight=weights,reduction="sum")
                    loss=loss+aux_config["coefficient"]*aux_loss/valid.sum().clamp_min(1)
                per_title.append(loss)
            loss=torch.stack(per_title).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,.5);opt.step();losses.append(loss.item())
        kl={}
        with torch.no_grad():
            for t,(x,*_) in data.items():
                old,_=reference(x);new,_=model(x)
                kl[t]=max(0.,sum((o.softmax(1)*(o.log_softmax(1)-n.log_softmax(1))).sum(1).mean().item() for o,n in zip(old,new)))
        keep=max(kl.values())<=config['max_title_kl']
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),per_title_joint_kl=kl,accepted=keep))
        if not keep:
            model.load_state_dict(prior);break
    if not any(row["accepted"] for row in history):
        raise RuntimeError("The first PPO epoch exceeded the per-title KL bound.")
    args.out.mkdir(parents=True,exist_ok=False)
    target=args.out/'policy.bin';model.export(target)
    with torch.no_grad():
        visual=dict(input_weight_change=float((model.first.weight[:,:84]-reference.first.weight[:,:84]).norm()),
                    per_title_logit_kl_if_screen_shuffled={})
        for t,(x,*_) in data.items():
            sh=x.clone();sh[:,:84]=x[torch.randperm(len(x)),:84]
            true,_=model(x);perturbed,_=model(sh)
            visual["per_title_logit_kl_if_screen_shuffled"][t]=max(0.,sum((o.softmax(1)*(o.log_softmax(1)-n.log_softmax(1))).sum(1).mean().item() for o,n in zip(true,perturbed)))
    report=dict(algorithm="Full action on-policy clipped PPO, equally weighted title losses, complete-credit GAE",
                parent_sha256=parent_sha,policy_sha256=digest(target),protocol_sha256=digest(args.protocol),manifest_sha256=digest(args.manifest),
                steps=sum(len(v[0]) for v in data.values()),credits=quality,epochs=history,vision=visual,settings=config)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding="utf-8")


if __name__=="__main__":main()
