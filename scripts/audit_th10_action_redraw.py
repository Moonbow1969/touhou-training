"""Measure how changes in the screen and independent action sampling affect movement persistence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.collect_th10_full_action import save


@torch.inference_mode()
def main(base,reference,candidate,out):
    torch.set_num_threads(2)
    rows=json.loads((base/'ppo-receipts.json').read_bytes())
    if len(rows)!=8 or any(r['role']!='train' for r in rows):raise ValueError('Use the designated training cohort.')
    result={}
    for name,file in [('original',reference),('corrected',candidate)]:
        model=torch.jit.load(str(file),map_location='cpu').eval();blocks=[]
        for r in rows:
            folder=base/f"momentum-{r['seed']}"
            with np.load(folder/'original-pixels.npz') as z:x=torch.from_numpy(z['obs'][:-1]).float()/255
            with np.load(folder/'behavior.npz') as z:acted=torch.from_numpy(z['action'][:,0].copy()).long()
            diff=torch.zeros_like(x);diff[1:]=(x[1:]-x[:-1]).abs()
            frames=torch.stack((x,diff),1)
            p=torch.cat([model(batch)[0].softmax(-1) for batch in frames.split(32)])
            mode=p.argmax(1)
            blocks.append(dict(seed=r['seed'],pairs=len(p)-1,
                modal_switch=(mode[1:]!=mode[:-1]).float().mean().item(),
                expected_independent_redraw_switch=(1-(p[1:]*p[:-1]).sum(1)).mean().item(),
                expected_same_screen_two_draw_switch=(1-p[1:].square().sum(1)).mean().item(),
                expected_switch_given_real_previous_action=(1-p[1:].gather(1,acted[:-1,None])).mean().item(),
                entropy_nats=(-(p[1:]*p[1:].log()).sum(1)).mean().item(),
                mean_neutral_probability=p[1:,4].mean().item(),mean_max_probability=p[1:].max(1).values.mean().item()))
        weights=np.array([b['pairs'] for b in blocks])
        pooled={key:float(np.average([b[key] for b in blocks],weights=weights)) for key in blocks[0] if key not in ('seed','pairs')}
        result[name]=dict(policy_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),adjacent_pairs=int(weights.sum()),
                          paired_fixed_observations=pooled,credits=blocks)
    result['scope']='The exact same eight isolated TH10 camera SFT training credits are used for both policies. For frozen adjacent observations the probability of changing when drawing two categorical actions independently is 1 minus the dot product of their probability vectors. On the same screen it is 1 minus the sum of squared probabilities. These are offline fixed-state counterfactuals, not replacements for the native closed-loop evaluation.'
    save(out,result);print(json.dumps({k:v['paired_fixed_observations'] for k,v in result.items() if isinstance(v,dict)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','reference','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.base,a.reference,a.candidate,a.out)
