"""Evaluate the exact privileged human SFT teacher on synchronized TH10 PPO2 training rollouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from . import pretrain_policy as bc


TEACHER_SHA="6357a40fe4d90e2619e9a9b19a3e44fa545f33ff1ae7bb929a3e7cee48cf7b85"
HEADS=("movement","focus","shoot","bomb")


def convert(path):
    with np.load(path) as z:
        valid=z["valid"];k=z["past_native_index"][valid]
        count=z["bullet_count"][k];native=z["bullets"][k];life=z["lives"]
        t=z["t1"].astype(np.float64);xy=z["player_xy"];power=z["power_units"][k]/20
        focus=z["focus"][k]
        ages=z["past_age_ms"][valid]
    mask=np.arange(96)[None,:]<count[:,None]
    b=np.zeros((*native.shape[:2],7),np.float32);b[:,:,:4]=native/[384,448,16,16]
    b[:,:,5]=mask;b[:,:,6]=mask;b=b.clip(-1,1)
    context=np.zeros((len(k),10),np.float32)
    context[:,:2]=xy[k]/[192,448];context[:,2]=focus;context[:,3]=life[k]/8
    context[:,5]=power/4;context[:,9]=count/512
    base=bc.condition_context(context.clip(-1,1),1,"ReimuA")
    hist=np.zeros((len(k),9),np.float32)
    for i,span in enumerate((4,12,36)):
        past=np.searchsorted(t,t[k]-span*(1000/60),side="right")-1
        p=past.clip(min=0);ticks=(t[k]-t[p])*.06
        usable=(past>=0)&(life[k]==life[p])&(ticks<=span+3)&(xy[p,1]<=450)&(xy[k,1]<=450)
        hist[usable,i*3:i*3+2]=((xy[k]-xy[p])/(8*ticks.clip(min=1)[:,None]))[usable].clip(-1,1)
        hist[:,i*3+2]=usable
    return b,np.hstack([base,hist]),count,life[k],ages,power


def compare(model,b,c,b2,c2):
    @torch.no_grad()
    def probs(entities,context):
        parts=[]
        for i in range(0,len(context),96):
            heads=bc.predict_heads(model,torch.from_numpy(entities[i:i+96]),torch.from_numpy(context[i:i+96]))
            parts.append(torch.cat([a.softmax(1) for a in heads],1).numpy())
        return np.split(np.concatenate(parts),[9,11,13],axis=1)
    left=probs(b,c);right=probs(b2,c2)
    return {name:np.column_stack([p.argmax(1)!=q.argmax(1),.5*np.abs(p-q).sum(1)]) for name,p,q in zip(HEADS,left,right)}


def summaries(parts):
    total={}
    for description,rows in parts.items():
        total[description]={}
        for head in HEADS:
            vals=np.concatenate([part[head] for part in rows])
            total[description][head]=dict(n=len(vals),modal_changes=int(vals[:,0].sum()),
                modal_change_fraction=float(vals[:,0].mean()),mean_total_variation=float(vals[:,1].mean()),
                p90_total_variation=float(np.quantile(vals[:,1],.9)))
    return total


def run(args):
    assert hashlib.sha256(args.teacher.read_bytes()).hexdigest()==TEACHER_SHA
    torch.set_num_threads(1)
    model=MultiInputActorCriticPolicy(bc.observation_space(True),bc.action_space(),lambda _:3e-4,**bc.network_kwargs(256))
    model.load_state_dict(torch.load(args.teacher,weights_only=True,map_location="cpu"));model.eval()
    pooled={};by_credit=[];all_age=[]
    for seed in range(710201,710209):
        file=next(args.data.glob(f"*-{seed}/privileged-targets.npz"));b,c,num,life,age,power=convert(file)
        all_age+=age.tolist()
        history=(c[:,-9:].reshape(-1,3,3)[:,:,2]>0).sum(0).tolist()
        p=np.arange(len(num));matched=np.zeros(len(num),bool)
        for group in np.unique(np.stack([life,num],1),axis=0):
            ix=np.flatnonzero((life==group[0])&(num==group[1])&(num>0))
            if len(ix)>=2:p[ix]=np.roll(ix,max(1,len(ix)//2));matched[ix]=True
        changed=(b!=b[p]).any(axis=(1,2));matched&=changed
        eligible=np.flatnonzero(matched);base_b=b[eligible];base_c=c[eligible]
        donors=b[p[eligible]];group={}
        def add(name,src,ctx,dst,ctx2):
            result=compare(model,src,ctx,dst,ctx2);group[name]=[result];pooled.setdefault(name,[]).append(result)
        add("same_life_and_exact_count_shuffle",base_b,base_c,donors,base_c)
        src=base_b.copy();dst=donors.copy()
        src[:,:,4]=src[:,:,6]*4/32;dst[:,:,4]=dst[:,:,6]*4/32
        add("same_shuffle_imputed_radius_4_native_pixels",src,base_c,dst,base_c)
        no_motion=base_c.copy();no_motion[:,-9:]=0
        add("same_shuffle_without_approximate_motion_history",base_b,no_motion,donors,no_motion)
        affordable=base_c.copy();affordable[:,4]=np.floor(power[eligible])/8
        add("same_shuffle_with_affordable_strikes_as_stock_proxy",base_b,affordable,donors,affordable)
        positive=num>0;clear_context=c[positive].copy();clear_context[:,9]=0
        add("remove_all_native_bullets_and_zero_count",b[positive],c[positive],np.zeros_like(b[positive]),clear_context)
        by_credit.append(dict(seed=seed,valid=len(b),with_bullets=int(positive.sum()),exact_count_shuffle_rows=len(eligible),
             max_live_count=int(num.max()),history_valid=history,min_past_age_ms=float(age.min()),max_past_age_ms=float(age.max()),
             per_credit=summaries(group)))
    report=dict(teacher_sha256=TEACHER_SHA,training_titles=[13,14,17],target_title=10,credits=8,
       policy_data="Eight genuine on-policy theta-PPO2 training credits, 710201–8. No held story outcomes used. The teacher did not control these games. This is an input sensitivity audit, without expert action labels on these states.",
       cross_title_mapping={
        "entities":"Native TH10 reader keeps active==1 pool entries, closest 96 by squared relative distance. dx/384, dy/448, vx/16, vy/16. Mask and active flag are one for present rows, zero otherwise. Hitbox radius unobserved and set to zero in the primary audit; 4px sensitivity check. Other titles may represent additional inactive/fading rows.",
        "player_and_resources":"x/192, y/448, focus from game's consumed key bit, spares/8. Display Power/4 clipped to [0,1] matches the teacher's 4.00 reference scale; TH10 can reach 5.00. TH10 pays Power per Bomb and has no separate stock: the teacher stock channel is zero by default; floor(Power)/8 is an explicitly tested proxy.",
        "boss":"Relative boss x, relative boss y and normalized boss HP were unobserved and set to zero, the original replay encoder's no-boss representation. The data covers only stage 1.",
        "counts_and_metadata":"Native bullet count/512, Normal difficulty index 1/4, one-hot Reimu/A. No title encoding or stage-clock channel in this exact teacher. Standard observation bounds apply.",
        "history":"Approximate 4-, 12-, 36-tick same-life displacement using previous complete native reads assuming 60 Hz and 8 pixels/tick normalization. No native stage frame in TH10 archive; compare disabled history and expose availability flags.",
        "alignment":"Last native read completed before the camera timestamp. Rows with no prior native read, over 100ms age, negative spares, or offscreen vertically during respawn are discarded. No future native inputs used.",
        "test":"Derange complete bullet sets within controller credit, same life and exact count by a half-group circular shift. Keep player/context/history fixed. Only nonempty groups with >=2 distinct stored sets contribute. Total variation = half L1 distance between the two head distributions."},
       coverage=dict(total_camera=2511,valid=sum(r['valid'] for r in by_credit),native_bullets_present=sum(r['with_bullets'] for r in by_credit),
         same_life_count_shuffle=sum(r['exact_count_shuffle_rows'] for r in by_credit),history_valid=np.sum([r['history_valid'] for r in by_credit],0).tolist(),
         median_past_age_ms=float(np.median(all_age)),maximum_past_age_ms=max(all_age)),summary=summaries(pooled),per_credit=by_credit)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report['coverage']));print(json.dumps({k:v['movement'] for k,v in report['summary'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
