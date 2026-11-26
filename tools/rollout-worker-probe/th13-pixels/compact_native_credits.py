"""Export the exact inputs to the 255-parameter actor, without retaining whole screenshots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compact(source, destination):
    manifest=json.loads(source.read_bytes())
    destination.mkdir(parents=True,exist_ok=False)
    packed=[]
    for i,row in enumerate(manifest,1):
        raw=source.parent/row["behavior"]
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("One signed native behavior archive changed.")
        with np.load(raw,allow_pickle=False) as trace:
            screen=trace["obs"][:-1]
            if screen.dtype!=np.uint8 or screen.shape[1:]!=(112,96):
                raise ValueError("Expected the native grayscale actor crop.")
            prior=np.concatenate([screen[:1],screen[:-1]],axis=0)
            delta=np.abs(screen.astype(np.int16)-prior).astype(np.uint8)
            def pool(a):
                return a.reshape(-1,7,16,6,16).sum(axis=(2,4),dtype=np.uint16).reshape(-1,42)
            grid=np.concatenate([pool(screen),pool(delta)],axis=1)
            target=destination/f"th{row['title']}-credit-{i:02d}.npz"
            np.savez_compressed(target,grid_sums=grid,action=trace["action" if "action" in trace else "action_index"],
                                prob=trace["prob" if "prob" in trace else "probabilities"],logp=trace["logp"])
        packed.append({**row,"behavior":target.name,"raw_behavior_sha256":row["sha256"],
                       "sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination/"credits.json").write_text(json.dumps(packed,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    compact(args.source,args.destination)
