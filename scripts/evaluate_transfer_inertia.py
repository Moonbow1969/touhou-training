"""Measure whether a replay-trained policy predicts movement changes beyond inertia."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.pretrain_transfer_move_focus import ACT, OBS, POLICY_KWARGS, heads, load
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


@torch.inference_mode()
def evaluate(args):
    manifest = json.loads(args.manifest.read_bytes())
    data = load(manifest["evaluation_manifest"], args.data, "cuda", True)
    output = {"manifest_sha256": digest(args.manifest), "models": {}}
    for folder in args.models:
        weights = folder / "policy.pt"
        policy = MultiInputActorCriticPolicy(OBS,ACT,lambda _: .0003,**POLICY_KWARGS).cuda().eval()
        policy.load_state_dict(torch.load(weights, weights_only=True, map_location="cuda"))
        titles = {}
        for title, (bullets, context, labels, _) in data.items():
            logits = torch.cat([heads(policy,bullets[k:k+256],context[k:k+256])[0]
                                for k in range(0,len(context),256)])
            inertia = ((context[:,8].long()+1)*3 + context[:,7].long()+1)
            truth = labels[:,0]
            sections = {}
            for label, mask in [("all", torch.ones_like(truth, dtype=torch.bool)),
                                ("held", inertia==truth),
                                ("start_or_unobserved_prior", (inertia==4)&(truth!=4)),
                                ("turn", (inertia!=4)&(truth!=4)&(truth!=inertia)),
                                ("stop", (inertia!=4)&(truth==4))]:
                n = int(mask.sum())
                sections[label] = {"n": n,
                    "correct": int((logits.argmax(1)[mask]==truth[mask]).sum()),
                    "persistence_correct": int((inertia[mask]==truth[mask]).sum()),
                    "nll": F.cross_entropy(logits[mask],truth[mask]).item() if n else None}
            titles[str(title)] = sections
        output["models"][folder.name] = {"sha256": digest(weights), "titles": titles}
    args.out.write_text(json.dumps(output,indent=2)+"\n", encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--models",required=True,type=Path,nargs="+")
    parser.add_argument("--out",required=True,type=Path)
    evaluate(parser.parse_args())
