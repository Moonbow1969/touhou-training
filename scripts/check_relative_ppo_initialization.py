"""Reconstruct the original student's own native credit and quantify the PPO sampling policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import relative_phase_student as relative, state_route_student as state
from scripts import train_relative_stock_ppo as training
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA
from scripts.train_full_policy import publish_json


def check(args):
    torch.set_num_threads(1)
    model=training.ProjectilePolicy(args.student,args.risk).eval()
    challenger=None
    if args.checkpoint:
        record=json.loads(args.checkpoint.with_name("report.json").read_text(encoding="utf-8"))
        if record["status"]!="complete" or record["updates"][-1]["sha256"]!=digest(args.checkpoint):
            raise ValueError("Incomplete or mismatched updated actor.")
        challenger=training.ProjectilePolicy(args.student,args.risk).eval()
        challenger.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=True)["state_dict"])
    with np.load(args.credit,allow_pickle=False) as data:
        meta=json.loads(str(data["metadata_json"]))
        cols={k:i for i,k in enumerate(meta["native_field_order"])}
        native,ctx,bullets,actual=(data[k] for k in ("native_before","context","bullets","action"))
    clock=relative.EventClock()
    tensors=[]
    for row,c,b in zip(native,ctx,bullets):
        st,frame,ptr,size,spell,vulnerability,hp,pstate,xheld,stock=(row[cols[k]] for k in
            ("stage","stage_frame","boss_pointer","boss_max_hp","spell_id","boss_vulnerable","boss_hp",
             "player_state","previous_x","bombs"))
        maximum=size if ptr>0 else 0
        event=clock.update(st,frame,maximum,spell,c[8]>0)
        x,_,raw=state.features(c,b,st,frame,maximum,spell,vulnerability==1,event[0],pstate,xheld,stock)
        frac=np.float32(np.clip(hp/max(1,maximum),0,1) if ptr>0 else 0.)
        tensors.append((x,event,frac,raw))
    x,e,hp,r=(torch.as_tensor(np.stack(a)) for a in zip(*tensors))
    ps=[]; updated=[]; danger_rows=[]
    with torch.inference_mode():
        for i in range(0,len(x),256):
            ix=slice(i,i+256)
            field=torch.from_numpy(bullets[ix])
            c=torch.from_numpy(ctx[ix])
            risk=model.risk(field,training.contact.local.local_context(c,0))[:,:18].sigmoid()
            heads,_=model.heads(x[ix],e[ix],hp[ix],r[ix],risk)
            ps.append([torch.softmax(h/args.temperature,1).numpy() for h in heads])
            if challenger:
                heads,_=challenger.heads(x[ix],e[ix],hp[ix],r[ix],risk)
                updated.append([torch.softmax(h/args.temperature,1).numpy() for h in heads])
                danger_rows.append(risk.numpy())
    joint,shoot,bomb=(np.concatenate(group) for group in zip(*ps))
    native_joint=(actual[:,0].astype(int)+9*actual[:,1].astype(int))
    reconstructed=np.column_stack((joint.argmax(1)%9,joint.argmax(1)//9,shoot.argmax(1),bomb.argmax(1)))
    if not np.array_equal(reconstructed,actual):
        raise ValueError(f"Native inputs did not reproduce {np.any(reconstructed!=actual,axis=1).sum()} decisions.")
    near=(native[:,cols["player_state"]]==4)&(native[:,cols["bombs"]]>0)
    output=dict(schema="th14-original-student-to-pure-ppo-initialization-v1",credit_sha256=digest(args.credit),
                student_sha256=digest(args.student),risk_sha256=digest(args.risk),verified_four_heads=len(actual),
                temperature=args.temperature,expected_joint_action_agreement=float(joint[np.arange(len(actual)),native_joint].mean()),
                expected_shoot_agreement=float(shoot[np.arange(len(actual)),actual[:,2]].mean()),
                expected_bomb_agreement=float(bomb[np.arange(len(actual)),actual[:,3]].mean()),
                hit_with_stock_decisions=int(near.sum()),mean_press_probability_on_hit_with_stock=float(bomb[near,1].mean()),
                greedy_source="Exact recorded stock credit; sample agreement is conditional on these same observations.")
    if challenger:
        new=[np.concatenate(a) for a in zip(*updated)]
        danger=np.concatenate(danger_rows)
        data={}
        for name,base,after in zip(("move_focus","shoot","bomb"),(joint,shoot,bomb),new):
            changed=base.argmax(1)!=after.argmax(1)
            p,q=np.clip(base,1e-20,1),np.clip(after,1e-20,1)
            data[name]=dict(changed=int(changed.sum()),mean_forward_kl=float((p*np.log(p/q)).sum(1).mean()),
                total_variation=float(.5*np.abs(p-q).sum(1).mean()))
        data["projected_contact"]=dict(expected_before=float((joint*danger).sum(1).mean()),
                                      expected_after=float((new[0]*danger).sum(1).mean()))
        output["updated_checkpoint_sha256"]=digest(args.checkpoint)
        output["frozen_state_change"]=data
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credit",type=Path,required=True)
    parser.add_argument("--student",type=Path,default=DATA.parent/"th14-relative-phase-student-261125/student.pt")
    parser.add_argument("--risk",type=Path,default=DATA.parent/"th14-anticipatory-contact-12tick-261125/model.pt")
    parser.add_argument("--temperature",type=float,default=.55)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=check(args)
    publish_json(args.out,result)
    print(json.dumps(result))
