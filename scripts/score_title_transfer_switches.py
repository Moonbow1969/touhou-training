"""Score frozen multi-title SFT policies at true movement starts and direction changes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import pretrain_policy as bc
from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    manifest=json.loads((args.run/"report.json").read_bytes())
    if manifest["status"]!="complete":
        raise ValueError("Score only frozen matrix arms.")
    records={}
    for row in manifest["evaluation_manifest"]:
        path=args.data/row["file"]
        if digest(path)!=row["sha256"]:
            raise ValueError("An evaluation shard changed.")
        b,c,y=bc.read_shard((path,row),True,motion_spans=(4,12,36))
        with np.load(path,allow_pickle=False) as z:
            frame=z["stage_frame"]
        prev=np.r_[4,y[:-1,0]]
        adjacent=np.r_[False,np.diff(frame)==4]
        groups=dict(all=adjacent, moving=adjacent&(y[:,0]!=4), starts=adjacent&(prev==4)&(y[:,0]!=4),
                    turns=adjacent&(prev!=4)&(y[:,0]!=4)&(prev!=y[:,0]),
                    stops=adjacent&(prev!=4)&(y[:,0]==4), persistence=adjacent&(y[:,0]==prev))
        records[row["file"]]=(row["title"],torch.tensor(b,device="cuda"),torch.tensor(c,device="cuda"),y[:,0],prev,groups)
    scores={}
    for arm in manifest["arms"]:
        checkpoint=args.run/arm["weights"]
        if digest(checkpoint)!=arm["sha256"]:
            raise ValueError("Checkpoint fingerprint mismatch.")
        model=bc.MovementPolicy(bc.observation_space(True),bc.action_space(),lambda _: .0003,
                                **bc.network_kwargs(manifest["encoder_width"])).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True))
        arm_counts={}
        for t,b,c,y,prev,groups in records.values():
            counts=arm_counts.setdefault(t,{k:np.zeros(3,dtype=int) for k in groups})
            pred=torch.cat([bc.predict_heads(model,b[i:i+512],c[i:i+512])[0].argmax(1) for i in range(0,len(y),512)]).cpu().numpy()
            for k,mask in groups.items():
                counts[k]+=[mask.sum(),(pred[mask]==y[mask]).sum(),(prev[mask]==y[mask]).sum()]
        scores[arm["label"]]={t:{k:dict(n=int(v[0]),correct=int(v[1]),previous_action_correct=int(v[2]),
            accuracy=float(v[1]/v[0]) if v[0] else None) for k,v in bins.items()} for t,bins in arm_counts.items()}
    publish_json(args.out,dict(matrix_sha256=digest(args.run/"report.json"),scores=scores,
        scope="Only consecutive native four-frame pairs within one stage. Persistence repeats the preceding expert action."))


if __name__=="__main__":
    main()
