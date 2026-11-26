"""Calibrate twelve-tick neural contact penalties on historical tapes, then score unseen credit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.distill_state_route import C,ROOT,WORLD
from scripts.relative_phase_student import EventClock,RelativeActor
from scripts.state_route_student import features
from scripts.student_contact_policy import CONTACT_HASH,STUDENT_HASH,adjust,load_risk
from scripts.train_anticipatory_contact import contact_targets
from scripts import train_local_avoidance as local

STUDENT=ROOT/"outputs/th14-relative-phase-student-261125/student.pt"
RISK=ROOT/"outputs/th14-anticipatory-contact-12tick-261125/model.pt"
HISTORICAL=("th14-demo-alice-story-2001-261125","th14-demo-alice-phase-2201-261125","th14-demo-alice-phase-2202-261125")


@torch.inference_mode()
def tape(path,model,contact,device):
    with np.load(path) as z:
        names=("native_before","native_after","action","bullets","context","four_tick_stage_transition","native_life_losses","actual_bombs_consumed")
        b,a,act,bul,ctx,exact,loss,bombs=(z[n] for n in names)
        meta=json.loads(z["metadata_json"].item())
    x=[];e=[];r=[];h=[];ec=EventClock();n=len(b)
    for i,v in enumerate(b):
        stage=int(v[C["stage"]]);f=int(v[C["stage_frame"]]);size=max(0,int(v[C["boss_max_hp"]]));sp=int(v[C["spell_id"]])
        evt=ec.update(stage,f,size,sp,ctx[i,8]>0)
        xx,_,raw=features(ctx[i],bul[i],stage,f,size,sp,v[C["boss_vulnerable"]]==1,evt[0],v[C["player_state"]],v[C["previous_x"]],v[C["bombs"]])
        x.append(xx);e.append(evt);r.append(raw);h.append(np.clip(v[C["boss_hp"]]/max(1,size),0,1))
    inputs=[torch.tensor(np.asarray(v),dtype=torch.float32,device=device) for v in (x,e,h,r)]
    logits=[];risk=[]
    bt=torch.from_numpy(bul).to(device);ct=torch.from_numpy(ctx).to(device)
    for i in range(0,n,256):
        logits.append(model(*(v[i:i+256] for v in inputs))[0].cpu())
        risk.append(contact(bt[i:i+256],local.local_context(ct[i:i+256],0)).sigmoid()[:,:18].cpu())
    logits=torch.cat(logits);risk=torch.cat(risk)
    base=logits.argmax(1).numpy();focus=base//9;move=base%9
    stock=np.asarray(r);bomb_pred=model.bomb_logits(inputs[-1]).argmax(1).cpu().numpy()
    eligible=(b[:,C["snapshot_valid"]]==1)&(b[:,C["player_state"]]==1)&(b[:,C["previous_x"]]==0)&(bomb_pred==0)
    continuity=(b[1:,C["global_tick"]]==a[:-1,C["global_tick"]])&(b[1:,C["stage"]]==b[:-1,C["stage"]])
    full=np.zeros(n,bool)
    full[:-2]=exact[:-2]&exact[1:-1]&exact[2:]&continuity[:-1]&continuity[1:]
    # Projection requires preserved fields independent of player injury / clearing Bombs.
    clean=(loss==0)&(bombs==0)&(b[:,C["player_state"]]==1)&(a[:,C["player_state"]]==1)
    good=full.copy();good[:-2]&=clean[:-2]&clean[1:-1]&clean[2:]
    good&=eligible
    ids=np.flatnonzero(good);geometry=np.full((n,18),np.nan)
    dyn=json.loads(RISK.with_name("report.json").read_bytes())["dynamics"]["ReimuA"]
    speeds=torch.tensor(dyn["speed_units_per_tick"],device=device);bounds=torch.tensor(dyn["bounds"],device=device)
    for i in range(0,len(ids),96):
        index=torch.tensor(ids[i:i+96,None]+np.arange(3),device=device)
        geometry[ids[i:i+96]]=contact_targets(bt[index],ct[index],speeds,bounds).cpu().numpy()
    direction=np.asarray(local._DIRECTIONS)
    pos=ctx[:,:2]*local.PLAYER_SCALE
    displacement=4*speeds.cpu().numpy()[(ctx[:,2]>.5).astype(int),focus,:,None]*direction[None]
    dest=np.clip(pos[:,None]+displacement,*dyn["bounds"])
    hit=(b[:,C["player_state"]]==1)&(a[:,C["player_state"]]==4)
    # Group by actual distinct native onset. A changed prior command only counts when
    # the student's baseline movement/focus equals the recorded sustained action until hit.
    onset=[]
    for i in np.flatnonzero(hit):
        matched=[]
        for j in (i-2,i-1):
            if j<0 or not eligible[j] or not exact[j:i+1].all() or not continuity[j:i].all():continue
            if not (act[j:i+1,0]==move[j]).all() or not (act[j:i+1,1]==focus[j]).all():continue
            matched.append(j)
        fatal=False
        for k in range(i,min(i+12,n)):
            if b[k,C["stage"]]!=b[i,C["stage"]]:break
            if loss[k]>0:fatal=True;break
            if k>i and a[k,C["player_state"]]==1:break
        onset.append((i,matched,fatal))
    return dict(name=path.name,hash=digest(path),trial=meta["trial"],logits=logits,risk=risk,geometry=geometry,
        eligible=eligible,good=good,move=move,focus=focus,dest=dest,stage=b[:,C["stage"]],onset=onset,
        factual_agreement=float(((act[:,0]==move)&(act[:,1]==focus))[eligible].mean()))


def metrics(data,weight,stage=None):
    new,focus=adjust(data["logits"],data["risk"],weight);new=new.numpy();focus=focus.numpy();old=data["move"]
    sel=data["eligible"].copy()
    if stage is not None:sel&=data["stage"]==stage
    good=data["good"]&sel;rows=np.arange(len(sel));changed=(new!=old)&sel
    geo=data["geometry"];prior=geo[rows,9*focus+old];after=geo[rows,9*focus+new]
    quiet=good&(np.nan_to_num(geo,nan=1).max(1)<.10)
    drift=np.linalg.norm(data["dest"][rows,old]-data["dest"][rows,new],axis=1)
    count_onset=matched=hits_changed=fatal_changes=0
    for ix,js,fatal in data["onset"]:
        if stage is not None and data["stage"][ix]!=stage:continue
        count_onset+=1;matched+=bool(js)
        altered=any(changed[j] for j in js);hits_changed+=altered;fatal_changes+=altered and fatal
    return dict(eligible=int(sel.sum()),changed=int(changed.sum()),quiet_rows=int(quiet.sum()),quiet_changed=int(changed[quiet].sum()),
        mean_one_step_displacement=float(drift[sel].mean()) if sel.any() else None,
        projected_removed=int((good&(prior>=.5)&(after<.5)).sum()),projected_added=int((good&(prior<.5)&(after>=.5)).sum()),
        native_onsets=int(count_onset),native_onsets_with_action_matched_prior=int(matched),
        native_onsets_changed_before_contact=int(hits_changed),life_loss_onsets_changed_before_contact=int(fatal_changes))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["calibrate","check"])
    p.add_argument("--out",type=Path,required=True);p.add_argument("--shard",type=Path);args=p.parse_args()
    torch.set_num_threads(2);device="cuda" if torch.cuda.is_available() else "cpu"
    if digest(STUDENT)!=STUDENT_HASH:raise ValueError("Wrong state-only actor.")
    model=RelativeActor().to(device).eval();model.load_state_dict(torch.load(STUDENT,weights_only=True)["state_dict"])
    contact=load_risk(RISK).to(device);config=args.out/"calibration.json"
    if args.mode=="calibrate":
        args.out.mkdir(parents=True,exist_ok=False)
        paths=[next((WORLD/name/"transition-shards").glob("*.npz")) for name in HISTORICAL]
        if any("2601" in path.name for path in paths):raise ValueError("2601 is reserved.")
        tapes=[tape(path,model,contact,device) for path in paths]
        gaps=[]
        for d in tapes:
            f=d["logits"].view(-1,2,9)[torch.arange(len(d["focus"])),torch.tensor(d["focus"])].topk(2).values
            gaps.extend((f[:,0]-f[:,1])[d["eligible"]].numpy())
        scale=float(np.median(gaps));grid=[];selected=None
        for multiplier in (.25,.5,1.,2.,4.,8.,16.,32.,64.):
            weight=multiplier*scale
            rows=[metrics(d,weight) for d in tapes]
            passed=(all(r["quiet_rows"] and r["quiet_changed"]/r["quiet_rows"]<=.005 and r["mean_one_step_displacement"]<=.4 for r in rows)
                    and sum(r["native_onsets_changed_before_contact"]>0 for r in rows)>=2
                    and sum(r["projected_removed"] for r in rows)>sum(r["projected_added"] for r in rows)
                    and sum(r["life_loss_onsets_changed_before_contact"] for r in rows)>=1)
            grid.append(dict(multiplier=multiplier,weight=weight,gate=passed,credits=rows))
            if passed and selected is None:selected=weight
        result=dict(student_hash=STUDENT_HASH,contact_hash=CONTACT_HASH,source_hash=digest(Path(__file__)),
            composition_hash=digest(Path(__file__).with_name("student_contact_policy.py")),
            median_fixed_focus_top_two_margin=scale,development_gate=selected is not None,chosen_weight=selected,
            historical=[dict(name=d["name"],sha256=d["hash"],factual_agreement=d["factual_agreement"]) for d in tapes],grid=grid,
            inference="Keep frozen student focus/Shoot/Bomb. On active non-Bomb steps with modeled risk >=0.10, choose movement maximizing original fixed-focus logits minus weight times learned contact probability.",
            gate="Smallest tested weight with <=0.5% all-action-quiet change and <=0.4 game-unit mean four-tick displacement per credit, a prior change at matched native onsets on at least two credits including one fatal onset, and strictly more removed than added modeled 12-tick contacts. Frozen before scoring 2601.",
            limitations="Contact targets use three future recorded projectile fields, overlapping 12-tick held-action windows and empirical player dynamics. Quiet is defined by all 18 independent field projections <0.10. Native onset preemption counts proposed changes 4 or 8 ticks before the actual hit with matched held historical movement/focus; outcomes under the proposed action remain unobserved.")
        config.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ("median_fixed_focus_top_two_margin","development_gate","chosen_weight")}))
    else:
        result=json.loads(config.read_bytes())
        # A failed development gate fixes the zero-strength comparator. Avoid selecting a sensitivity variant on holdout.
        q=tape(args.shard,model,contact,device);seed=q["trial"]["policy_seed"]
        if seed<2402 or any(row["sha256"]==q["hash"] for row in result["historical"]):raise ValueError("Expected a fresh tape.")
        weight=result["chosen_weight"] or 0.
        record=dict(calibration_hash=digest(config),student_hash=STUDENT_HASH,contact_hash=CONTACT_HASH,seed=seed,
            actual_policy=q["trial"]["arm"],shard_hash=q["hash"],factual_agreement=q["factual_agreement"],weight=weight,
            all=metrics(q,weight),stages={int(s):metrics(q,weight,int(s)) for s in np.unique(q["stage"])})
        dest=args.out/f"check-{seed}.json"
        if dest.exists():raise FileExistsError(dest)
        dest.write_text(json.dumps(record,indent=2,allow_nan=False));print(json.dumps(record))


if __name__=="__main__":main()
