"""Compare two fixed learned four-head policies on exactly the same retail cameras."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.original_camera_heads import policy_heads
from scripts.th08_camera_history import reconstruct
from scripts.train_five_original_games import freeze


def on_camera(actor, frames, device):
    results = []
    with torch.inference_mode():
        for start in range(0,len(frames),128):
            observed = reconstruct(frames,start,min(start+128,len(frames)),1)
            volume = torch.from_numpy(observed).to(device).float().div_(255)
            heads,_ = policy_heads(actor(volume),len(observed))
            p = [h.softmax(1) for h in heads]
            results.append(torch.stack((1-p[0][:,4],p[1][:,1],p[2][:,1],p[3][:,1])).T.cpu())
    return torch.cat(results).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source","release","parent","sft","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    release = json.loads(args.release.read_bytes())
    if digest(args.source) != release["sha256"]:
        raise ValueError("Changed qualifying original source ZIP")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    models = {label:torch.jit.load(str(path),map_location=device).eval()
              for label,path in (("parent",args.parent),("sft",args.sft))}
    output = []
    with ZipFile(args.source) as archived:
        for case in release["subjects"]:
            with np.load(io.BytesIO(archived.read(f"{case}/policy-trace.npz")),allow_pickle=False) as data:
                images = data["frames"]
                ticks = data["native_tick"]
            physical = list(csv.DictReader(io.StringIO(archived.read(f"{case}/controls.csv").decode())))
            if list(ticks) != [int(row["tick"]) for row in physical]:
                raise ValueError(f"Native stock rows misalign in {case}")
            stock = np.asarray([int(row["bombs_hud"]) for row in physical])
            masks = {"all":np.ones(len(images),dtype=bool),"bomb_stock_available":stock>0}
            probabilities = {label:on_camera(model,images,device) for label,model in models.items()}
            for label,mask in masks.items():
                if mask.any():
                    output.append(dict(case=case,group=label,observations=int(mask.sum()),
                        arms={name:dict(zip(("movement","focus","shoot","bomb"),p[mask].mean(0).astype(float).tolist()))
                              for name,p in probabilities.items()}))
    freeze(args.out,dict(source_sha256=release["sha256"],parent_sha256=digest(args.parent),sft_sha256=digest(args.sft),
          scope="Two fixed original engineering credits. Both policies receive identical observations, including states produced by the other policy. Camera probabilities are descriptive and do not estimate an intervention effect.",
          per_source=output))
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
