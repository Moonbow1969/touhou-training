"""Refit a paired PPO update on the same fresh-policy stock buffers at two native Bomb prices."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from scripts import train_relative_stock_ppo as train
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA
from scripts.train_full_policy import publish_json


def resource_cost(batch, episode):
    raw, action = batch[3].numpy(), batch[5].numpy()
    result = episode["result"]
    last = result["stage_records"][-1]["entry"]["resources"]["bombs"]
    after = np.r_[raw[1:,2], last]
    spent = np.maximum(raw[:,2] - after, 0).astype(int)
    ledger = [e for s in result["stage_records"] for e in s.get("resource_events", []) if e["bombs_used"]]
    actual = [(int(raw[i,0]), bool(raw[i,1]), int(action[i,2]), int(raw[i,2]), int(after[i]), int(spent[i]))
              for i in np.flatnonzero(spent)]
    confirmed = [(e["player_state"], e["previous_x_key"], e["bomb_command"], e["stock_before"],
                  e["stock_after"], e["bombs_used"]) for e in ledger]
    if actual != confirmed or spent.sum() != result["actual_bombs_consumed"]:
        raise ValueError("Historical stock drops are not individually corroborated by the native resource ledger.")
    return spent


def discounted_cost(spent, coefficient):
    seq = np.empty(len(spent), np.float32)
    running = 0.
    for i in range(len(spent)-1, -1, -1):
        running = float(spent[i]) + .995*running
        seq[i] = coefficient*running
    return torch.from_numpy(seq)


@torch.inference_mode()
def bomb_measurements(model, batches, temp):
    raw = torch.cat([batch[3] for batch in batches])
    p = (model.actor.bomb_logits(raw)/temp).softmax(1)[:,1]
    masks = dict(hit_with_stock=(raw[:,0]==4)&(raw[:,2]>0)&(raw[:,1]==0),
                 ordinary_with_stock=(raw[:,0]==1)&(raw[:,2]>0)&(raw[:,1]==0))
    return {name:dict(n=int(mask.sum()), mean_press_probability=float(p[mask].mean()),
                      greedy_presses=int((p[mask]>.5).sum())) for name,mask in masks.items() if mask.any()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cost", type=float, default=.005)
    args = p.parse_args()
    if not 0 < args.cost < .1:
        p.error("Choose a positive cost below a spare life's original Stage 2 entry value.")
    report = json.loads((args.training/"report.json").read_bytes())
    spec = report["specification"]
    if report["status"] != "complete" or len(report["updates"]) != 1 or report.get("starting_steps",0) or len(report["episodes"])!=2:
        raise ValueError("This paired comparison requires the first frozen-policy two-credit rollout round.")
    student = DATA.parent/"th14-relative-phase-student-261125/student.pt"
    risk = DATA.parent/"th14-anticipatory-contact-12tick-261125/model.pt"
    if digest(student)!=spec["student_sha256"] or digest(risk)!=spec["risk_sha256"]:
        raise ValueError("The sampling actor differs from the archived initialization.")
    torch.set_num_threads(1)
    torch.manual_seed(spec["seed"])
    base = train.ProjectilePolicy(student,risk)
    temp = spec["temperature"]
    batches, spends, provenance = [], [], []
    for ep in report["episodes"]:
        f = args.training/ep["buffer"]["file"]
        if digest(f)!=ep["buffer"]["sha256"]:
            raise ValueError("Changed native sampling data.")
        payload = torch.load(f,weights_only=True,map_location="cpu")
        batch = payload["tensors"]
        with torch.no_grad():
            # Match the native single-observation GEMV path. A batched GEMM changes
            # the last few digits for the trained high-magnitude movement logits.
            checked=[]
            for i in range(len(batch[5])):
                heads,v = base.heads(*(col[i:i+1] for col in batch[:5]))
                value=sum(Categorical(logits=h/temp).log_prob(action).item() for h,action in zip(heads,batch[5][i]))
                checked.append(value)
                if v.item()!=0:
                    raise ValueError("Archive does not match the zero-initialized critic.")
            torch.testing.assert_close(torch.tensor(checked),batch[6],atol=1e-6,rtol=1e-6)
        spent = resource_cost(batch,ep)
        batches.append(batch)
        spends.append(spent)
        provenance.append(dict(seed=ep["trial"]["policy_seed"], steps=len(spent), bombs=int(spent.sum()),
            native_return=ep["native_return"], repriced_return=ep["native_return"]-args.cost*int(spent.sum()),
            buffer_sha256=ep["buffer"]["sha256"]))
    args.out.mkdir(parents=True,exist_ok=False)
    result = dict(credits=provenance, baseline=bomb_measurements(base,batches,temp),
        fixed_minibatch_seed=44311, cost=args.cost, variants={},
        estimand="One paired PPO update on logged on-policy samples. Separate frozen gameplay tests measure survival.")
    for name,cost in (("free",0.),("charged",args.cost)):
        dest=args.out/name
        dest.mkdir()
        model,anchor=deepcopy(base),deepcopy(base).eval().requires_grad_(False)
        opt=torch.optim.Adam([dict(params=model.actor.parameters(),lr=1e-6),
                              dict(params=model.move_residual.parameters(),lr=3e-4),
                              dict(params=model.value.parameters(),lr=7e-4)],eps=1e-5)
        parts=[]
        for batch,spent in zip(batches,spends):
            charge=discounted_cost(spent,cost)
            parts.append((*batch[:-2],batch[-2]-charge,batch[-1]-charge))
        torch.manual_seed(result["fixed_minibatch_seed"])
        stats=train.update(model,anchor,opt,tuple(torch.cat(values) for values in zip(*parts)),temp)
        variant=deepcopy(spec)
        variant["reward"]["native_bomb_consumed"]=-cost
        variant["paired_refit"]=dict(source_report_sha256=digest(args.training/"report.json"),
                                     same_behavior_verified=True,minibatch_seed=result["fixed_minibatch_seed"])
        artifact=train.publish_tensors(dest/"policy.pt",dict(schema="relative-stock-ppo-v1",steps=report["steps"],
            specification=variant,state_dict=model.state_dict(),optimizer=opt.state_dict()))
        publish_json(dest/"report.json",dict(status="complete",steps=report["steps"],specification=variant,
            updates=[dict(steps=report["steps"],**artifact,**stats)],sampling_credits=provenance))
        result["variants"][name]=dict(checkpoint_sha256=artifact["sha256"],bomb=bomb_measurements(model,batches,temp),
                                      update=stats)
    publish_json(args.out/"comparison.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
