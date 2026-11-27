"""Apply the independently validated original TH11 stage register to entire archived game credits."""
from pathlib import Path
import json
import numpy as np
from scripts.run_th10_camera_cohort import digest

def read_credit(folder):
    r=json.loads((folder/"receipt.json").read_bytes());p=folder/"trajectory.npz"
    assert r["full_credit"] and r["continues"]==0 and r["terminal"] and digest(p)==r["trajectory_sha256"]
    assert r["assets"]["asset_sha256"]["th11.exe"]=="360f542349a08707de5c788ef5a950d5d3ad0fd727a23e74b6813db40dfe77a9"
    with np.load(p) as tr:a=tr["native"]
    assert a.ndim==2 and a.shape[1]==12 and len(a)==r["native_samples"]
    t=a[:,0].astype(np.int64);stage=a[:,7];rank=a[:,6]
    assert np.all(np.diff(t)>=0) and t[-1]>t[0] and np.all(rank==r["difficulty"]) and np.all(a[:,8]==0)
    assert list(a[0,[3,4,5,7]])==[0,0,2,1] and a[-1,5]==-1 and int(stage.max())<=3
    assert np.all((np.diff(stage)>=0)&(np.diff(stage)<=1))
    hits=np.flatnonzero(stage==2);gate=len(hits)>0
    if gate:
        if len(np.unique(t[hits]))<3:raise ValueError("Too few independent native Stage 2 samples")
        i=int(hits[0]);entry=dict(row=i,original_ms=int(t[i]-t[0]),spares=int(a[i,5]),power_units=int(a[i,2]),native_score=int(a[i,1])*10)
    else:entry=None
    return dict(seed=r["seed"],arm=r["arm"],difficulty=r["difficulty"],stage2_reached=gate,first_stage2=entry,raw_stage_values=np.unique(stage).tolist(),
                native_max_sample_spacing_ms=int(np.diff(t).max()),trajectory_sha256=digest(p),endpoint_sha256=r["evidence_sha256"]["endpoint.png"])
