"""Independently reforward every scored camera/action through its actually frozen CPU actor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(study,base):
    torch.set_num_threads(1)
    cfg=json.loads((study/"protocol.json").read_text())
    cohort=json.loads((study/"complete.json").read_text())
    out={}
    for arm,checksum in cfg["models"].items():
        model_file=base/f"locator-bc/fit/71281-{arm}/policy.ts"
        assert sha(model_file)==checksum
        actor=torch.jit.load(str(model_file),map_location="cpu").eval()
        for pair in cohort["all_pairs"]:
            if not pair["valid"]:continue
            folder=study/pair["arms"][arm]["path"]
            with np.load(folder/"policy-trace.npz")as z:
                x=torch.from_numpy(z["frames"])
                a=torch.from_numpy(z["action"]).long()
                stored_logp=torch.from_numpy(z["logp"]); stored_val=torch.from_numpy(z["value"]).flatten()
            logerr,valerr,reproduced=0.,0.,0
            g=torch.Generator().manual_seed(pair["seed"])
            with torch.inference_mode():
                for i in range(0,len(x),64):
                    gray=x[i:i+64].float()/255
                    prev=torch.cat((x[max(0,i-1):max(0,i-1)+1],x[i:i+len(gray)-1])).float()/255
                    pred=actor(torch.stack((gray,(gray-prev).abs()),1)); aa=a[i:i+len(gray)]
                    lp=torch.stack([h.log_softmax(1).gather(1,aa[:,j:j+1]).flatten()for j,h in enumerate(pred[:4])],1)
                    logerr=max(logerr,float((lp-stored_logp[i:i+len(gray)]).abs().max()))
                    valerr=max(valerr,float((pred[4].flatten()-stored_val[i:i+len(gray)]).abs().max()))
                    probs=[h.softmax(1)for h in pred[:4]]
                    for row in range(len(aa)):
                        draws=torch.tensor([torch.multinomial(h[row:row+1],1,generator=g).item() for h in probs])
                        reproduced+=int(torch.equal(aa[row],draws))
            assert logerr<5e-5 and valerr<5e-5 and reproduced==len(a),(folder,logerr,valerr,reproduced,len(a))
            out[folder.name]=dict(policy_sha256=checksum,decisions=len(a),all_four_draws_reproduced=reproduced,max_logp_abs=logerr,max_value_abs=valerr)
    (study/"reforward.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--study",type=Path,required=True);p.add_argument("--base",type=Path,required=True)
    a=p.parse_args();audit(a.study,a.base)
