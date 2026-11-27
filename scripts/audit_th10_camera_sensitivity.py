"""Measure camera sensitivity on training-only native TH10 credits at fixed action histories."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.collect_th10_full_action import save
from scripts.train_portable_full_sft import FullActor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(out,original,altered):
    for i,(a,b) in enumerate(zip(original,altered)):
        p,q=a.log_softmax(1),b.log_softmax(1)
        m=torch.logaddexp(p,q)-np.log(2)
        js=((p.exp()*(p-m)).sum(1)+(q.exp()*(q-m)).sum(1))*.5
        out[i].append(torch.stack([(p.argmax(1)!=q.argmax(1)).float(),js,
                                   .5*(p.exp()-q.exp()).abs().sum(1)],1))


@torch.inference_mode()
def main(args):
    torch.set_num_threads(2)
    camera=torch.jit.load(str(args.camera),map_location='cpu').eval()
    portable=FullActor.from_bin(args.portable).eval()
    receipts=json.loads((args.data/'ppo-receipts.json').read_bytes())
    assert len(receipts)==8 and {r['seed'] for r in receipts}==set(range(752101,752109))
    registry=defaultdict(lambda:defaultdict(list));rows=[]
    for r in receipts:
        assert r['role']=='train' and r['title']==10 and r['policy_sha256']==sha(args.camera)
        folder=args.data/f"momentum-{r['seed']}"
        assert sha(folder/'original-pixels.npz')==r['pixel_sha256'] and sha(folder/'behavior.npz')==r['sha256']
        with np.load(folder/'original-pixels.npz') as z:frames=torch.from_numpy(z['obs'][:-1]).float()/255
        with np.load(folder/'behavior.npz') as z:x=torch.from_numpy(z['features'])
        diff=torch.zeros_like(frames);diff[1:]=(frames[1:]-frames[:-1]).abs()
        pixels=torch.stack((frames,diff),1)
        n=len(x);offset=int(np.random.default_rng(r['seed']).integers(n//3,n*2//3))
        rows.append(dict(seed=r['seed'],n=n,one_third_to_two_thirds_circular_shift=offset))
        def direct(v):
            logits=[camera(t)[:4] for t in v.split(24)]
            return [torch.cat(items) for items in zip(*logits)]
        orig=direct(pixels)
        # This policy has no history input; rolling already computed logits is exactly rolling complete image pairs.
        compare(registry['camera_sft_shift'],orig,[v.roll(offset,0) for v in orig])
        pixels[:,1].zero_()
        compare(registry['camera_sft_zero_difference'],orig,direct(pixels))
        orig_portable=portable(x)[0]
        shuffled=x.clone();shuffled[:,:84]=x[:,:84].roll(offset,0)
        compare(registry['portable_ppo3_shift'],orig_portable,portable(shuffled)[0])
        masked=x.clone();masked[:,42:84]=0
        compare(registry['portable_ppo3_zero_difference'],orig_portable,portable(masked)[0])
    totals={}
    for policy,metrics in registry.items():
        totals[policy]={}
        for k,name in enumerate(['move','focus','shoot','bomb']):
            flips,js,tv=torch.cat(metrics[k]).mean(0).tolist()
            totals[policy][name]=dict(modal_action_switch_rate=flips,mean_JS_nats=js,mean_total_variation=tv)
    result=dict(dataset='Eight fresh native TH10 training credits sampled by direct camera SFT, full life stock. No held credits used.',
                n=sum(r['n'] for r in rows),seeds=rows,portable_sha256=sha(args.portable),camera_sha256=sha(args.camera),
                interventions='Within each credit shift complete camera observations by seeded 1/3–2/3 credit; the portable input keeps actual previous actions/time. Separately, zero only the image-difference channel for both models.',
                metrics=totals)
    save(args.data/'camera-sensitivity.json',result);print(json.dumps(totals,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','camera','portable'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
