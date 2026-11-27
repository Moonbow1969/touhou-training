"""Fit one learned Bomb-onset logit from genuine human cameras, keeping all other controls fixed."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from scripts.score_human_bomb_onsets import onset_labels
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy,save_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--train',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();source=json.loads((a.parent/'report.json').read_bytes())
    if {r['sha256'] for r in source['train']}!={sha(path) for path in a.train} or len(source['train'])!=len(a.train):
        raise ValueError('Onset readout requires the identical supervised training split')
    model=CameraPolicy();weights=a.parent/'camera-policy.pt'
    if sha(weights)!=source['sft']:raise ValueError('Changed starting model')
    model.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'));model.eval()
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(3);device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    features=[];labels=[];keys=[];examples=[];pilots=defaultdict(set);pilot_counts=Counter()
    with torch.no_grad():
        for path in a.train:
            row=load_replay(path);target,edges=onset_labels(path);key=(row['title'],row['group'])
            if len(target)!=len(row['raw']):raise ValueError('Onset windows differ from policy observations')
            px=torch.from_numpy(row['raw'])
            latent=torch.cat([model.policy[:-1](model.spatial(batch.to(device).float()/255)).cpu() for batch in px.split(64)])
            features.append(latent);labels.append(torch.from_numpy(target));keys.append(key);pilots[key[0]].add(key[1])
            pilot_counts[key]+=len(target)
            examples.append(dict(source=path.name,player=key[1],title=key[0],true_edges=edges,positive_windows=int(target.sum()),windows=len(target)))
            if len(features)==1:fixture=px[:8].float()/255
        model.cpu()
        before=tuple(x.clone() for x in model(fixture))
    x=torch.cat(features).double();y=torch.cat(labels).double()
    sample_weight=torch.cat([torch.full((len(target),),1/(len(pilots)*len(pilots[key[0]])*pilot_counts[key]),dtype=torch.float64)
                             for key,target in zip(keys,labels)])
    head=model.policy[-1]
    w0=(head.weight[-1]-head.weight[-2]).detach().clone().double()
    w=w0.clone().requires_grad_();bias=(head.bias[-1]-head.bias[-2]).detach().clone().double().requires_grad_()
    opt=torch.optim.LBFGS([w,bias],lr=1.,max_iter=70,line_search_fn='strong_wolfe')
    uplift=20.;penalty=.0003;history=[]
    def objective():
        opt.zero_grad()
        # Fit case-weighted logits, correcting the 20:1 sampling odds for export.
        logit=x@w+bias+math.log(uplift)
        loss=(sample_weight*F.binary_cross_entropy_with_logits(logit,y,pos_weight=y.new_tensor(uplift),reduction='none')).sum()
        loss=loss+penalty*(w-w0).square().sum()
        loss.backward();history.append(float(loss.detach()));return loss
    opt.step(objective)
    with torch.no_grad():
        head.weight[-1].copy_(head.weight[-2]+w.float());head.bias[-1].copy_(head.bias[-2]+bias.float())
        deployed=model(fixture)
        for left,right in zip(before[:3],deployed[:3]):torch.testing.assert_close(left,right,rtol=0,atol=0)
        torch.testing.assert_close(deployed[-1][:,1]-deployed[-1][:,0],(features[0][:8].double()@w+bias).float(),rtol=1e-4,atol=2e-4)
        a.out.mkdir(parents=True,exist_ok=False)
        out=a.out/'camera-policy.pt';torch.save(model.state_dict(),out)
        jit=torch.jit.trace(model,fixture[:1]);jit.save(str(a.out/'policy.ts'))
        for left,right in zip(jit(fixture),deployed):torch.testing.assert_close(left,right)
    save_json(a.out/'report.json',dict(train=source['train'],held=source['held'],sft=sha(out),torchscript_sha256=sha(a.out/'policy.ts'),
        parent_supervised=source['sft'],objective='Full-batch title/player-balanced human X-key onset logistic readout on frozen 128D vision features.',
        positive_weight=uplift,export_log_odds_correction=math.log(uplift),squared_weight_displacement_penalty=penalty,
        optimization_loss=history,examples=examples,windows=len(y),positive_windows=int(y.sum()),untouched_heads=['movement','focus','shoot']))


if __name__=='__main__':
    main()
