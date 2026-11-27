"""Audit a behavioral-policy suffix against the original timestamps, controls and frozen network."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.collect_th10_full_action import save
from scripts.read_th13_trial_ranking import classify, load_templates
from scripts.run_th10_camera_cohort import digest
from PIL import Image


@torch.inference_mode()
def verify(folder, policy, *, prefix=True,expected_rank=2):
    r=json.loads((folder/"receipt.json").read_bytes())
    p=folder/"rollout.npz"
    assert digest(policy)==r["policy_sha256"] and digest(p)==r["sha256"] and r["continues"]==0
    torch.set_num_threads(1)
    model=torch.jit.load(str(policy),map_location="cpu").eval()
    with np.load(p) as b:
        frames=torch.from_numpy(b["frames"]).float()/255
        diff=torch.zeros_like(frames)
        diff[1:]=(frames[1:]-frames[:-1]).abs()
        inputs=torch.stack((frames,diff),1)
        a=torch.tensor(b["action"],dtype=torch.long)
        selected=[]; values=[]
        for i in range(0,len(inputs),32):
            heads=model(inputs[i:i+32])
            selected.append(torch.stack([h.log_softmax(1).gather(1,a[i:i+32,j,None])[:,0] for j,h in enumerate(heads[:4])],1))
            values.append(heads[4].reshape(-1) if len(heads)>4 else torch.zeros(len(a[i:i+32])))
        lp=float(np.max(np.abs(torch.cat(selected).numpy()-b["logp"])))
        value=float(np.max(np.abs(torch.cat(values).numpy()-b["value"])))
        assert max(lp,value)<3e-4 and b["terminated"].sum()==1 and b["terminated"][-1]
        n=len(frames)
    raw=np.fromfile(folder/"steps.bin",dtype="<f4").reshape(-1,143)
    assert len(raw)==n and np.all(raw[:,141]==expected_rank) and not raw[:,132].any()
    assert raw[-1,128]==-1 and r["native_key_match"]/n>=.98
    image=folder/"ranking.png"
    with Image.open(image) as im:ranking=classify(im,load_templates())
    assert ranking["readout"] in (1,2) and ((ranking["readout"]==2)==r["stage2"])
    out=dict(seed=r["seed"],steps=n,policy_sha256=r["policy_sha256"],rollout_sha256=r["sha256"],role=r["role"],
             file="rollout.npz",first_native_before_tick=int(raw[0,134]),last_native_after_tick=int(raw[-1,135]),
             max_logp_abs_error=lp,max_value_abs_error=value,rank_verified_steps=n,true_native_key_matches=r["native_key_match"],
             original_ranking=ranking,original_ranking_sha256=digest(image),only_policy_actions=True,prefix_decisions_in_train=0)
    if prefix:
        with (folder/"prefix.csv").open() as f: rows=list(csv.DictReader(f))
        assert 0<=int(rows[0]["stage_tick"])<60 and int(rows[0]["spares"])==int(rows[0]["bombs"])==2
        assert 3000<=int(rows[-1]["stage_tick"])<=3006 and 3000<=raw[0,134]<=3046
        assert int(rows[-1]["spares"])==raw[0,127] and int(rows[-1]["bombs"])==raw[0,129]
        same=sum(row["human_key_after_tick"]==row["native_key"] for row in rows)
        assert same/len(rows)>.98
        out.update(prefix_rows=len(rows),prefix_raw_native_input_matches=same,handoff_spares=int(raw[0,127]),handoff_bombs=int(raw[0,129]))
    return out
