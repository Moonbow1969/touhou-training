"""Fit one global Bomb-head intercept by maximum likelihood on genuine training windows."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy, save_json


def scores(logit, truth, bias):
    p=1/(1+np.exp(-logit-bias))
    return dict(n=len(p),expert_positive=int(truth.sum()),expected_sampled_windows=float(p.sum()),
                brier=float(np.mean((p-truth)**2)),nll=float(np.mean(np.logaddexp(0,logit+bias)-truth*(logit+bias))),
                expected_on_expert_positive=float(p[truth.astype(bool)].sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','out','protocol'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--held',nargs='+',type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(2)
    original=json.loads((a.parent/'report.json').read_bytes())
    plan=json.loads(a.protocol.read_bytes())
    state=a.parent/'camera-policy.pt'
    base=a.parent/'policy.ts'
    bank=a.parent/'training-demonstrations.npz'
    if digest(state)!=original['sft'] or digest(base)!=plan['parent_sha256'] or digest(base)!=original['torchscript_sha256'] or digest(bank)!=original['demo_sha256']:
        raise ValueError('Parent model or associated training predictions changed')
    with np.load(bank,allow_pickle=False) as data:
        prob=data['teacher_probs'][:,-1].astype(np.float64)
        labels=data['action'][:,3].astype(np.float64)
        replay=data['replay_index']
    lengths=[r['decisions'] for r in original['train']]
    if len(labels)!=sum(lengths) or not np.array_equal(replay,np.repeat(np.arange(len(lengths)),lengths)) or np.any((prob<=0)|(prob>=1)):
        raise ValueError('Incomplete training predictions')
    if set(np.unique(labels))!={0.,1.}:raise ValueError('Both genuine human Bomb actions are required')
    logit=np.log(prob)-np.log1p(-prob)
    lo,hi=-60.,60.
    for _ in range(75):
        mid=(lo+hi)/2
        if (1/(1+np.exp(-logit-mid))).mean()>labels.mean():hi=mid
        else:lo=mid
    bias=(lo+hi)/2
    model=CameraPolicy()
    if original['spatial_attention']:model.add_spatial_context()
    model.load_state_dict(torch.load(state,weights_only=True,map_location='cpu'))
    model.eval()
    parent=torch.jit.load(str(base),map_location='cpu').eval()
    example=torch.rand(4,2,112,96,generator=torch.Generator().manual_seed(7120))
    with torch.no_grad():
        model.policy[-1].bias[-1].add_(bias)
        compiled=torch.jit.trace(model,example[:1])
        before=parent(example);after=compiled(example)
        for i in range(3):torch.testing.assert_close(after[i],before[i],rtol=0,atol=0)
        torch.testing.assert_close(after[3][:,0],before[3][:,0],rtol=0,atol=0)
        torch.testing.assert_close(after[3][:,1],before[3][:,1]+bias)
    if {digest(f) for f in a.held}!={r['sha256'] for r in original['held']}:
        raise ValueError('Held replay list differs from parent')
    train_groups={r['group'] for r in original['train']}
    a.out.mkdir(parents=True,exist_ok=False)
    weights=a.out/'camera-policy.pt'
    torch.save(model.state_dict(),weights)
    compiled.save(str(a.out/'policy.ts'))
    report=dict(parent_sha256=digest(base),protocol_sha256=digest(a.protocol),training_predictions_sha256=digest(bank),
        sft=digest(weights),torchscript_sha256=digest(a.out/'policy.ts'),positive_logit_bias=bias,
        train_before=scores(logit,labels,0),train_after=scores(logit,labels,bias))
    save_json(a.out/'report.json',report)
    held_logits=[];held_y=[];records=[]
    with torch.inference_mode():
        for path in a.held:
            r=load_replay(path)
            if r['group'] in train_groups:raise ValueError('Held pilot appeared in training')
            x=torch.from_numpy(r['raw'])
            heads=torch.cat([parent(px.float()/255)[3] for px in x.split(64)])
            l=(heads[:,1]-heads[:,0]).numpy().astype(np.float64)
            y=r['action'][:,3].astype(np.float64)
            held_logits.append(l);held_y.append(y)
            records.append(dict(player=r['group'],title=r['title'],sha256=r['sha256'],before=scores(l,y,0),after=scores(l,y,bias)))
    l=np.concatenate(held_logits);y=np.concatenate(held_y)
    save_json(a.out/'held-bomb.json',dict(before=scores(l,y,0),after=scores(l,y,bias),replays=records))


if __name__=='__main__':
    main()
