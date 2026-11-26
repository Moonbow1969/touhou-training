"""Fit and evaluate one-step action-conditioned dynamics on fresh multi-title pixel rollouts."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Dynamics(nn.Module):
    def __init__(self, titles):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(1, 24, 5, padding=2), nn.SiLU(),
                                    nn.Conv2d(24, 32, 3, padding=1), nn.SiLU())
        self.conditions = nn.Sequential(nn.Linear(titles + 5, 48), nn.SiLU(), nn.Linear(48, 32))
        self.decode = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, img, features):
        latent = self.encode(img) + self.conditions(features)[:, :, None, None]
        return img + self.decode(latent)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode(path, title, titles, device):
    key=path.name
    with np.load(path, allow_pickle=False) as z:
        pix, act, ns = z["obs"], z["actions"], z["monotonic_ns"]
        if (pix.dtype != np.uint8 or pix.shape != (len(act)+1,112,96)
                or act.dtype != np.int8 or act.shape != (len(act),4)):
            raise ValueError(f"Invalid observation or action dimensions: {key}")
        if ns.shape != (len(pix),) or np.any(np.diff(ns.astype(np.int64))<=0):
            raise ValueError(f"Invalid monotonic clock: {key}")
        if (np.abs(act[:,:2])>1).any() or not np.isin(act[:,2:], [0,1]).all():
            raise ValueError(f"Invalid action: {key}")
        if len(act) == 0:
            raise ValueError(f"Empty episode: {key}")
        # Average-pool before transfer to keep the dataset small on the learner.
        small = pix.reshape(len(pix),56,2,48,2).mean((2,4),dtype=np.float32) / 255.
        img = torch.as_tensor(small[:,None], device=device)
        code = np.zeros((len(act),len(titles)+5), dtype=np.float32)
        code[:,:4] = act
        code[:,4] = np.log1p(np.diff(ns).astype(np.float64)*1.e-9)
        code[:,5+titles.index(title)] = 1.
        return (img[:-1],torch.as_tensor(code,device=device),img[1:])


def read(manifest, device):
    rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
    titles = sorted({r["title"] for r in rows})
    if len(titles) < 2:
        raise ValueError("At least two playable titles are required.")
    fit = defaultdict(list)
    hold = []
    files, episodes = set(), set()
    for row in rows:
        path = manifest.parent / row["file"]
        digest = sha256(path)
        key = row["title"], row["lane"], row["episode"]
        if digest in files or key in episodes or row["split"] not in {"train", "test"}:
            raise ValueError(f"Duplicate episode or invalid split: {key}")
        files.add(digest)
        episodes.add(key)
        if digest != row["sha256"]:
            raise ValueError(f"Source fingerprint changed: {key}")
        item=load_episode(path,row["title"],titles,device)
        if row["split"] == "train":
            fit[row["title"]].append(item)
        else:
            hold.append((key,item))
    if set(fit) != set(titles) or {key[0] for key,_ in hold} != set(titles):
        raise ValueError("Every title must supply independent training and evaluation episodes.")
    return fit, hold, titles, rows


@torch.inference_mode()
def evaluate(models, hold, seed):
    result=[]
    rng=np.random.default_rng(seed)
    for (title,lane,episode),(img,feat,truth) in hold:
        shuffled=feat.clone()
        shuffled[:,:4]=feat[torch.as_tensor(rng.permutation(len(feat)),device=feat.device),:4]
        blind=feat.clone()
        blind[:,:4]=0
        totals=defaultdict(lambda: np.zeros(2))
        for start in range(0,len(img),64):
            sl=slice(start,start+64)
            current=img[sl]
            guesses={"copy":current,"action":models["action"](current,feat[sl]),
                     "time_title_only":models["time_title_only"](current,blind[sl]),
                     "shuffled_action":models["action"](current,shuffled[sl])}
            for name,guess in guesses.items():
                delta=(guess-truth[sl]).square()
                totals[name]+=np.array([delta.mean().item(),delta[:,:,28:,:].mean().item()])*len(delta)
        result.append(dict(title=title,lane=lane,episode=episode,transitions=len(img),
            mse={name:dict(full=float(pair[0]/len(img)),bottom_half=float(pair[1]/len(img))) for name,pair in totals.items()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--seed",type=int,default=709)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    if args.steps<1:
        p.error("The training budget must be positive.")
    fit,hold,titles,rows=read(args.manifest,args.device)
    args.out.mkdir(exist_ok=False)
    models={}
    learning={}
    fingerprints={}
    for mode in ("action","time_title_only"):
        rng=np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        net=Dynamics(len(titles)).to(args.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0006)
        loss_history=[]
        for step in range(1,args.steps+1):
            parts=[]
            # Equal title share, then uniform independent episodes within the title.
            for title in titles:
                for _ in range(2):
                    run=fit[title][rng.integers(len(fit[title]))]
                    ix=torch.as_tensor(rng.integers(len(run[0]),size=8),device=args.device)
                    parts.append(tuple(t[ix] for t in run))
            img,feat,truth=map(torch.cat,zip(*parts))
            if mode=="time_title_only":
                feat[:,:4]=0
            loss=F.mse_loss(net(img,feat),truth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step==1 or step%50==0 or step==args.steps:
                loss_history.append(dict(step=step,mse=loss.item()))
        models[mode]=net.eval()
        learning[mode]=loss_history
        ckpt=args.out/(mode+".pt")
        torch.save(net.state_dict(),ckpt)
        fingerprints[mode]=sha256(ckpt)
    scores=evaluate(models,hold,args.seed)
    report=dict(titles=titles,seed=args.seed,steps_per_arm=args.steps,training_history=learning,
                weights_sha256=fingerprints,split=rows,held_episodes=scores,
                task="One-step pixel dynamics at measured wall-clock intervals. Episode is the reporting unit.")
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
