"""Learn 12-tick action risks from human replay fields and test against held-action native injuries."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts import train_local_avoidance as local
from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json
from scripts.train_world import average_precision


class AnticipatoryContact(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(12,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(), nn.Linear(64,19))
        self.point[-1].bias.data.fill_(-3.)

    def forward(self, bullets, context):
        scaled = bullets[:,:,:6] * bullets.new_tensor((6.,7.,2.,2.,4.,1.))
        inputs = torch.cat((scaled,context[:,None].expand(-1,bullets.shape[1],-1)), dim=-1)
        mask = (bullets[:,:,6] > 0) & (bullets[:,:,5] > 0) & (bullets[:,:,4] > 0)
        return self.point(inputs).masked_fill(~mask[:,:,None],-14.).amax(1)


@torch.no_grad()
def contact_targets(fields, contexts, shot_speed, bounds):
    # Re-anchor the exogenous field at ticks 0, 4 and 8. Test collisions immediately
    # after the projectile tick and before the player's corresponding movement tick.
    device = fields.device
    p0 = contexts[:,0,:2]*contexts.new_tensor(local.PLAYER_SCALE)
    previous_focus = (contexts[:,0,2] > .5).long()
    directions = contexts.new_tensor(local._DIRECTIONS)
    focus = torch.arange(2, device=device)
    first = shot_speed[previous_focus][...,None]*directions
    steady = (shot_speed[focus,focus][...,None]*directions)[None]
    smallest = torch.full((len(fields),2,9), 10000., device=device)
    ticks = torch.arange(4, device=device, dtype=fields.dtype)
    for block in range(3):
        raw = fields[:,block]*fields.new_tensor(local.BULLET_SCALE)
        known = (raw[:,:,5]>0) & (raw[:,:,6]>0) & (raw[:,:,4]>0)
        observed_xy = contexts[:,block,:2]*contexts.new_tensor(local.PLAYER_SCALE)
        centers = observed_xy[:,None,None] + raw[:,None,:,:2] + (ticks[None,:,None,None]+1)*raw[:,None,:,2:4]
        elapsed = ticks + block*4
        paths = p0[:,None,None,None,:] + first[:,:,:,None,:]*elapsed.clamp(max=4)[None,None,None,:,None]
        paths = (paths + steady[:,:,:,None,:]*(elapsed-4).clamp(min=0)[None,None,None,:,None]).clamp(bounds[0],bounds[1])
        gap = torch.linalg.vector_norm(centers[:,None,None] - paths[:,:,:,:,None],dim=-1)-raw[:,None,None,None,:,4]-1
        smallest = torch.minimum(smallest,gap.masked_fill(~known[:,None,None,None],10000.).amin((3,4)))
    return (-smallest/2.5).sigmoid().flatten(1)


def replay_tensors(paths, speeds, bounds, device):
    chunks = []
    for path, meta in paths:
        with np.load(path, allow_pickle=False) as z:
            bullet, ctx, frame, state = [z[k] for k in ("bullets","context","stage_frame","player_state")]
            words = z["expert_action_chunk"]
            valid = (np.diff(frame)==4) & (state[:-1]==1) & (state[1:]==1)
            valid &= ((words[:-1] & 2)==0).all(1) & (ctx[:-1,4]==ctx[1:,4])
        idx = np.flatnonzero(valid[:-2] & valid[1:-1] & valid[2:])
        shot = local.SHOTS.index(meta["shot"].replace(" ",""))
        for start in range(0,len(idx),192):
            ix = idx[start:start+192,None]+np.arange(3)
            b = torch.from_numpy(bullet[ix]).to(device)
            c = torch.from_numpy(ctx[ix]).to(device)
            y = contact_targets(b,c,speeds[shot],bounds[shot])
            chunks.append((b[:,0].to(torch.float16).cpu(),local.local_context(c[:,0],shot).cpu(),y.cpu()))
    return tuple(torch.cat(part).to(device) for part in zip(*chunks))


def ap(y, pred):
    return average_precision(y, pred) if y.any() else None


@torch.inference_mode()
def native_test(model, paths, device):
    collections, totals = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            col = {k:i for i,k in enumerate(meta["native_field_order"])}
            before, after, actions, exact = (z[k] for k in ("native_before","native_after","action","four_tick_stage_transition"))
            bullets, context = z["bullets"], z["context"]
        if meta["task"]["shot"].replace(" ","") != "ReimuA":
            raise ValueError("Native measurement uses the Reimu A decoder.")
        subset, truth = [], []
        for i in range(len(actions)-2):
            for j in range(i,i+3):
                if (not exact[j] or before[j,col["player_state"]] != 1 or before[j,col["previous_x"]]
                        or actions[j,3] or not np.array_equal(actions[j,:2],actions[i,:2])
                        or j>i and before[j,col["global_tick"]] != after[j-1,col["global_tick"]]):
                    break
                if after[j,col["player_state"]] == 4 or j == i+2:
                    subset.append(i)
                    truth.append(after[j,col["player_state"]]==4)
                    break
        ids = np.asarray(subset,dtype=int)
        out = []
        for ix in np.array_split(ids, max(1,(len(ids)+511)//512)):
            b = torch.from_numpy(bullets[ix]).to(device)
            c = local.local_context(torch.from_numpy(context[ix]).to(device),0)
            out.append(model(b,c).sigmoid().cpu().numpy())
        pred = np.concatenate(out)
        selection = actions[ids,1]*9+actions[ids,0]
        score = pred[np.arange(len(ids)),selection]
        average = pred[:,:18].reshape(-1,2,9)[np.arange(len(ids)),actions[ids,1]].mean(1)
        truth = np.asarray(truth)
        totals.append([truth,score,average,pred[:,18]])
        collections.append(dict(file=path.name,sha256=digest(path),n=len(ids),hit_windows=int(truth.sum()),
                                action_ap=ap(truth,score), same_focus_average_ap=ap(truth,average),
                                agnostic_ap=ap(truth,pred[:,18])))
    y,a,f,g = (np.concatenate(x) for x in zip(*totals))
    return dict(n=len(y),hit_windows=int(y.sum()),action_ap=ap(y,a),same_focus_average_ap=ap(y,f),
                agnostic_ap=ap(y,g),per_credit=collections)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--native",nargs="+",type=Path,required=True,help="Directories containing original-game credit tapes.")
    args=parser.parse_args()
    if args.epochs<1:
        parser.error("Use positive epochs.")
    torch.manual_seed(3019)
    torch.set_num_threads(3)
    paths=list(local.replay_index(local.REPLAYS))
    train=[x for x in paths if x[1]["split"]=="train"]
    dev=[x for x in paths if x[1]["split"]=="validation"]
    if not train or not dev or {x[1]["sha256"] for x in train}&{x[1]["sha256"] for x in dev}:
        raise ValueError("Expected separate train and validation replay identities.")
    args.out.mkdir(parents=True,exist_ok=False)
    native=sorted({p for d in args.native for p in d.glob("credit-*.npz")})
    if not native:
        raise ValueError("No native credits.")
    initial=perf_counter()
    s, bounds, fit = local.fit_dynamics(train)
    device=torch.device(args.device)
    s,bounds = [torch.from_numpy(arr).to(device) for arr in (s,bounds)]
    report=dict(status="preparing",seed=3019,horizon_native_ticks=12,labels="18 held focus/move choices; three observed fields. "
        "Projectile updates first. Circular visible projectiles and empirical held-command speed only.",
        native_endpoint="Same command sustained until a native active-to-hit onset or full horizon. Overlapping windows share hits.",
        training=[dict(file=p.name,sha256=digest(p)) for p,_ in train], validation=[dict(file=p.name,sha256=digest(p)) for p,_ in dev],
        source_sha256=digest(Path(__file__)),dynamics=fit,epochs=[])
    reportpath=args.out/"report.json"
    publish_json(reportpath,report)
    b,c,y=replay_tensors(train,s,bounds,device)
    v,vc,vy=replay_tensors(dev,s,bounds,device)
    model=AnticipatoryContact().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    report.update(status="training",training_rows=len(y),validation_rows=len(vy))
    best=float("inf")
    for epoch in range(args.epochs):
        model.train()
        total=0.
        for ix in torch.randperm(len(b),device=device).split(384):
            pred=model(b[ix].float(),c[ix])
            target=torch.cat((y[ix],y[ix].mean(1,keepdim=True)),1)
            loss=F.binary_cross_entropy_with_logits(pred,target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total+=loss.detach().item()*len(ix)
        model.eval()
        with torch.no_grad():
            pred=torch.cat([model(v[i:i+512].float(),vc[i:i+512]) for i in range(0,len(v),512)])
            held=float(F.binary_cross_entropy_with_logits(pred[:,:18],vy).item())
        report["epochs"].append(dict(epoch=epoch+1,train_bce=total/len(y),validation_action_bce=held))
        if held<best:
            best=held
            torch.save(dict(schema="th14-learned-contact-12tick-v1",state_dict=model.state_dict()),args.out/"model.pt")
            report["selected_epoch"]=epoch+1
        publish_json(reportpath,report)
    model.load_state_dict(torch.load(args.out/"model.pt",map_location=device,weights_only=True)["state_dict"])
    report.update(status="complete",model_sha256=digest(args.out/"model.pt"),
                  native=native_test(model,native,device),seconds=perf_counter()-initial)
    publish_json(reportpath,report)
    print(json.dumps({k:v for k,v in report["native"].items() if k!="per_credit"}))

if __name__=="__main__":
    main()
