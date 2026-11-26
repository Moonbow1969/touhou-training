"""Roll a frozen student and teacher through a learned horizontal/vertical motion model."""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.distill_state_route import ART,ROOT,SCALE,Teacher,WORLD,labels
from scripts.relative_phase_student import RelativeActor
from scripts.train_relative_phase_student import DEV,observed


@torch.inference_mode()
def main():
    torch.set_num_threads(2);teacher=Teacher()
    out=ROOT/"outputs/th14-relative-phase-student-261125"
    model=RelativeActor().eval();model.load_state_dict(torch.load(out/"student.pt",weights_only=True)["state_dict"])
    dev=next((WORLD/"th14-stock-value-story-261125/transition-shards").glob(DEV))
    data=observed(dev,teacher);ticks=data["clock"][:,0];phase=data["phase"];spell=data["spell"]
    horizon=16
    starts=np.array([i for i in range(0,len(ticks)-horizon,8) if ticks[i+horizon]-ticks[i]==4*horizon
        and np.all(phase[i:i+horizon+1]==phase[i]) and np.all(spell[i:i+horizon+1]==spell[i])
        and np.all(data["state"][i:i+horizon+1]==1)])
    route=teacher.controller.route
    bounds=route["bounds"];delta=route["action_deltas"]
    initial=data["x"][starts,:2]*SCALE
    start_error=np.linalg.norm(initial-data["target"][starts],axis=1)
    result={}
    for actor in ("student","privileged_teacher"):
        pos=initial.copy()
        for t in range(horizon):
            ii=starts+t
            sub={k:v[ii] for k,v in data.items()}
            x=sub["x"].copy();difference=pos-x[:,:2]*SCALE
            x[:,:2]=pos/SCALE;x[:,6:8]-=difference/[384,448]
            for j in range(3):
                k=10+7*j;x[:,k:k+2]-=difference/[384,448]*x[:,k+6:k+7]
            if actor=="student":
                j=model(torch.tensor(x),torch.tensor(sub["event"]),torch.tensor(sub["hp"]),torch.tensor(sub["raw"]))[0].argmax(1).numpy()
            else:
                act=labels(teacher,sub,pos);j=act[:,0]+9*act[:,1]
            pos=np.clip(pos+delta[j],*bounds)
        end_error=np.linalg.norm(pos-data["target"][starts+horizon-1],axis=1)
        result[actor]={}
        for name,m in (("all",np.ones(len(starts),bool)),("initially_far",start_error>=64),("initially_near",start_error<24)):
            if m.any():
                result[actor][name]=dict(n=int(m.sum()),median_start=float(np.median(start_error[m])),
                    median_end=float(np.median(end_error[m])),reached_within_16_units=float((end_error[m]<=16).mean()))
    payload=dict(kind="simulated self-driven action sequences only",horizon_decisions=horizon,source=DEV,
        uncertainty="Per-action displacement learned from the training demonstration. Exogenous phase, bullets and time follow the held-out recorded credit. Does not model changed enemy trajectories, contact, HP, Bombs, speed variation or invulnerability under the alternative actions. Windows overlap. No native student survival measurements.",results=result)
    (out/"simulated-rollout.json").write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps(result))


if __name__=="__main__":main()
