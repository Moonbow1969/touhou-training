"""Measure human action persistence and camera-policy sampling on the same real observations."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import channels, save_json, sha


def profile(model,raw,actions,prior,device):
    with torch.inference_mode():
        batches=[model(block.to(device).float()/255) for block in torch.from_numpy(raw).split(64)]
        probs=[torch.cat(values).softmax(1).cpu().numpy() for values in zip(*batches)][:4]
    move=probs[0];keep=move[np.arange(len(prior)),prior[:,0]]
    return dict(n=len(raw),movement_changed=int((actions[:,0]!=prior[:,0]).sum()),
                movement_change_rate=float(np.mean(actions[:,0]!=prior[:,0])),
                expected_sampled_change_rate=float((1-keep).mean()),
                greedy_change_rate=float(np.mean(move.argmax(1)!=prior[:,0])),
                mean_direction_entropy_nats=float(np.mean(-(move*np.log(np.maximum(move,1e-9))).sum(1))),
                observed_neutral_rate=float((actions[:,0]==4).mean()),mean_neutral_probability=float(move[:,4].mean()),
                observed_bomb_rate=float(actions[:,3].mean()),mean_bomb_probability=float(probs[3][:,1].mean()),
                greedy_bomb_rate=float((probs[3].argmax(1)==1).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','manifest','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--human',type=Path,nargs='+',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=torch.jit.load(str(a.model),map_location=device).eval()
    results={'model_sha256':sha(a.model),'observed_human':{},'sampled_native':{}}
    for f in a.human:
        row=load_replay(f)
        results['observed_human'][f.name]={'group':row['group'],'split':f.parent.name,
            **profile(model,row['raw'],row['action'],row['prior'],device)}
    for r in json.loads(a.manifest.read_bytes()):
        if r['policy_sha256']!=results['model_sha256']:raise ValueError('Wrong behavior')
        f=a.manifest.parent/r['file']
        if sha(f)!=r['sha256']:raise ValueError('Changed recording')
        with np.load(f,allow_pickle=False) as z:
            frames=z['frames'];y=z['action']
        results['sampled_native'][r['file']]={'title':r['title'],'seed':r['seed'],
            **profile(model,channels(frames[1:],frames[:-1]),y[1:],y[:-1],device)}
    save_json(a.out,results)


if __name__=='__main__':main()
