"""Learn spatial features from native training credits and a four-head policy from aligned human replay cameras."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.train_multititle_locator import labeled_bank
from scripts.train_native_pixel_locator import targets
from scripts.probe_full_action_player_vision import metrics
from scripts.train_portable_full_sft import WIDTHS


class SpatialContext(nn.Module):
    """Residual self-attention over the policy's 7 by 6 spatial feature grid."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(24, 96)
        self.position = nn.Parameter(torch.randn(1, 42, 96) * .02)
        self.norm = nn.LayerNorm(96)
        self.qkv = nn.Linear(96, 288)
        self.mix = nn.Linear(96, 96)
        self.ff = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 192), nn.GELU(), nn.Linear(192, 96))
        self.output = nn.Linear(96, 24)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, grid):
        tokens = self.input(grid.flatten(2).transpose(1, 2)) + self.position
        q, k, v = self.qkv(self.norm(tokens)).reshape(-1, 42, 3, 4, 24).permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * (24 ** -.5)).softmax(-1)
        tokens = tokens + self.mix((attention @ v).transpose(1, 2).reshape(-1, 42, 96))
        residual = self.output(tokens + self.ff(tokens))
        return grid + residual.transpose(1, 2).reshape_as(grid)


class CameraPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(2, 20, 5, stride=2, padding=2), nn.GroupNorm(4, 20), nn.SiLU(),
                                     nn.Conv2d(20, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv2d(24, 24, 5, padding=4, dilation=2), nn.GroupNorm(4, 24), nn.SiLU())
        self.player = nn.Conv2d(24, 1, 1)
        self.policy = nn.Sequential(nn.Conv2d(24, 24, 5, stride=2, padding=2), nn.SiLU(),
                                    nn.AdaptiveAvgPool2d((7, 6)), nn.Flatten(), nn.Linear(24*42, 128),
                                    nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, sum(WIDTHS)))

    def forward(self, x):
        return self.policy(self.spatial(x)).split(WIDTHS, dim=1)

    def add_spatial_context(self):
        if 'spatial_context' in self.policy._modules:
            raise ValueError('Spatial attention already installed')
        layers = list(self.policy.named_children())
        with torch.random.fork_rng(devices=[]):
            context = SpatialContext()
        self.policy = nn.Sequential(OrderedDict(layers[:3] + [('spatial_context', context)] + layers[3:]))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def channels(obs, prev):
    return np.stack([obs, np.abs(obs.astype(np.int16)-prev).astype(np.uint8)], axis=1)


def load_locator_inputs(paths, title):
    parts = [targets(p) if title == 10 else labeled_bank(p, title) for p in paths]
    return tuple(np.concatenate(items) for items in zip(*parts))


def pretrain(a, device):
    groups = [(title, getattr(a, f"fit{title}"), getattr(a, f"dev{title}")) for title in (10, 12, 13)]
    train, dev = {}, {}
    provenance = {}
    for title, fit, held in groups:
        if {sha(p) for p in fit} & {sha(p) for p in held}:
            raise ValueError("Native credits occur in both splits")
        train[title] = tuple(torch.from_numpy(x).to(device) for x in load_locator_inputs(fit,title))
        dev[title] = load_locator_inputs(held,title)
        provenance[title] = {label: [{"file":p.parent.name+"/"+p.name, "sha256":sha(p)} for p in items]
                             for label,items in (("train",fit),("validation",held))}
    model = CameraPolicy().to(device)
    optimizer = torch.optim.AdamW(list(model.spatial.parameters())+list(model.player.parameters()), lr=.001)
    yy, xx = torch.meshgrid(torch.arange(56,device=device),torch.arange(48,device=device),indexing="ij")
    history = []
    for step in range(1, a.steps+1):
        x, xy = [], []
        for images, coords in train.values():
            ii = torch.randint(len(coords), (12,), device=device)
            x.append(images[ii]);xy.append(coords[ii])
        images = torch.cat(x).float()/255
        coords = torch.cat(xy)/2
        flip = torch.rand(len(coords), device=device)<.5
        images[flip] = images[flip].flip(3)
        coords[flip,0] = 47.5-coords[flip,0]
        images *= .85 + .3*torch.rand(len(coords),1,1,1,device=device)
        dist = (xx-coords[:,0,None,None]).square()+(yy-coords[:,1,None,None]).square()
        logits = model.player(model.spatial(images)).flatten(1)
        loss = -((-dist.flatten(1)/1.4).softmax(1)*logits.log_softmax(1)).sum(1).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step==1 or step%100==0:
            history.append({"step":step,"loss":loss.item()})
            with (a.out/"loss.jsonl").open("a") as log:log.write(json.dumps(history[-1])+"\n")
    results = {}
    with torch.no_grad():
        for title, (raw, xy) in dev.items():
            coords = []
            for batch in torch.from_numpy(raw).split(64):
                idx = model.player(model.spatial(batch.to(device).float()/255)).flatten(1).argmax(1).cpu().numpy()
                coords.append(np.column_stack([idx%48,idx//48])*2)
            results[title] = metrics(np.concatenate(coords),xy)
    torch.save(model.cpu().state_dict(), a.out/"encoder.pt")
    save_json(a.out/"report.json",dict(steps=a.steps,seed=a.seed,train_rows={k:len(v[0]) for k,v in train.items()},
                                      split=provenance,held_player_location=results,history=history))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=26261)
    for title in (10,12,13):
        for side in ("fit","dev"):
            parser.add_argument(f"--{side}{title}", type=Path, nargs="+", required=True)
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    pretrain(a,"cuda" if torch.cuda.is_available() else "cpu")
