"""Evaluate both frozen pixel diagnostics on episodes reserved on a second physical host."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_native_pixel_dynamics import Dynamics,evaluate,load_episode,sha256
from validate_pixel_controls import HorizontalDecoder


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forward",type=Path,required=True)
    p.add_argument("--inverse",type=Path,required=True)
    p.add_argument("--episodes",type=Path,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r=json.loads((args.forward/"report.json").read_bytes())
    inv=json.loads((args.inverse/"report.json").read_bytes())
    titles=r["titles"]
    seen={row["sha256"] for row in r["split"]}
    episodes=set()
    hold=[]
    sources=[]
    for path in args.episodes:
        digest=sha256(path)
        with np.load(path,allow_pickle=False) as z:
            md=json.loads(str(z["metadata_json"]))
        if digest in seen or md["episode_id"] in episodes or md["split"]!="external_test":
            raise ValueError("Duplicate or unreserved external episode.")
        seen.add(digest)
        episodes.add(md["episode_id"])
        title=int(md["title"].removeprefix("TH"))
        key=title,md["host"]+"-"+md["lane"],md["episode_id"]
        hold.append((key,load_episode(path,title,titles,"cpu")))
        sources.append(dict(episode=md["episode_id"],sha256=digest))
    models={}
    for mode,fingerprint in r["weights_sha256"].items():
        ckpt=args.forward/(mode+".pt")
        if sha256(ckpt)!=fingerprint:
            raise ValueError("Changed forward-model checkpoint.")
        model=Dynamics(len(titles))
        model.load_state_dict(torch.load(ckpt,weights_only=True,map_location="cpu"))
        models[mode]=model.eval()
    scores=evaluate(models,hold,r["seed"])
    accuracy={}
    for mode,row in inv["results"].items():
        ckpt=args.inverse/(mode+".pt")
        if sha256(ckpt)!=row["sha256"]:
            raise ValueError("Changed inverse-model checkpoint.")
        model=HorizontalDecoder(len(titles)).eval()
        model.load_state_dict(torch.load(ckpt,weights_only=True,map_location="cpu"))
        by_credit=[]
        for key,(img,feat,nxt) in hold:
            y=feat[:,0].long()+1
            logits=model(img,img if mode=="before_only" else nxt,feat[:,4:])
            by_credit.append(dict(episode=key[2],n=len(y),correct=int((logits.argmax(1)==y).sum()),
                                  nll=F.cross_entropy(logits,y).item()))
        accuracy[mode]=by_credit
    result=dict(sources=sources,dynamics=scores,controls=accuracy,scope="Frozen second-host external episodes.")
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
