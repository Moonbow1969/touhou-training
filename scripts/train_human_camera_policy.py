"""Fit four camera-only action heads from chronologically aligned original human replays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.pretrain_policy import movement_direction
from scripts.train_portable_full_sft import FullActor, WIDTHS
from scripts.train_replay_camera_encoder import CameraPolicy, channels, save_json, sha


def camera_labels(words, focus_bit=8):
    """Decode the next tick at a camera boundary; retain short Bomb taps inside the window."""
    first = words[:, 0]
    return np.column_stack((movement_direction(first), (first & focus_bit) != 0,
                            (first & 1) != 0, ((words & 2) != 0).any(1))).astype(np.int64)


def load_replay(path, include_eight_tick_camera=False, include_player_position=False):
    metadata = json.loads(path.with_suffix(".json").read_bytes())
    title=str(metadata["title"]).upper()
    focus_bit={"TH10":4,"TH12":8,"TH13":8}.get(title)
    if focus_bit is None or metadata.get("focus_bit",focus_bit)!=focus_bit:
        raise ValueError(f"Unverified original input schema for {title}")
    with np.load(path, allow_pickle=False) as z:
        obs = z["obs"]
        frame = z["native_frame"].reshape(-1).astype(np.int64)
        chunk = z["input_chunk"]
        actions = z["action"].astype(np.int64)
        past = z["past_input_chunk"] if "past_input_chunk" in z else None
        valid = z["stable"].astype(bool) if "stable" in z else np.ones(len(obs), dtype=bool)
        if include_player_position:
            native_xy=z['xy'].astype(np.float32)
    if obs.dtype != np.uint8 or obs.shape != (len(frame),112,96) or chunk.shape != (len(frame),4):
        raise ValueError(f"Camera/control layout differs: {path}")
    if not np.array_equal(actions, camera_labels(chunk,focus_bit)):
        raise ValueError(f"Decoded replay controls disagree with stored action heads: {path}")
    idx = np.arange(len(frame))
    continuous = (idx >= 1) & (frame - np.roll(frame,1) == 4)
    if not continuous.any():
        raise ValueError(f"No four-tick camera differences in {path}")
    pair = channels(obs, np.roll(obs,1,axis=0))
    feat = np.zeros((len(frame),110), dtype=np.float32)
    # This is the same 16x16 spatial averaging used by the C rollout actor.
    feat[:,:84] = pair.reshape(-1,2,7,16,6,16).mean(axis=(3,5)).reshape(len(frame),84)/255
    if past is not None:
        if past.shape!=(len(frame),8):
            raise ValueError("Expected eight strictly preceding decoded control words")
        last, older = camera_labels(past[:,4:],focus_bit), camera_labels(past[:,:4],focus_bit)
    else:
        last, older = np.roll(actions,1,axis=0), np.roll(actions,2,axis=0)
        continuous &= (idx>=2) & (frame-np.roll(frame,2)==8)
    for lag, previous in enumerate((last,older)):
        offset=84+12*lag
        feat[idx,offset+previous[:,0]]=1
        feat[:,offset+9:offset+12]=previous[:,1:]
    feat[:,108]=np.clip(frame/4/120,0,1);feat[:,109]=np.clip(frame/4/1800,0,1)
    keep=valid & continuous
    if not np.any(keep):
        raise ValueError(f"No usable causal observations in {path}")
    player=metadata.get("player_group",metadata.get("player"))
    if not player:
        raise ValueError("A replay needs an identified player group")
    result={"name":path.name,"group":str(player).casefold(),"title":title,"sha256":sha(path),
            "raw":pair[keep],"features":feat[keep],"action":actions[keep],"prior":last[keep],"native_frame":frame[keep]}
    if include_player_position:
        if native_xy.shape!=(len(frame),2):raise ValueError('Missing native player coordinates')
        native_xy[:,0]+=192
        result['player_xy']=native_xy[keep]/4.
    if include_eight_tick_camera:
        # The two native four-tick intervals must both exist and be playable.
        eight=(idx>=2)&(frame-np.roll(frame,2)==8)&np.roll(valid,1)&np.roll(valid,2)
        use_previous=np.where(eight,idx-2,np.maximum(idx-1,0))
        result["raw8"]=channels(obs[keep],obs[use_previous[keep]])
        result["valid8"]=eight[keep]
    return result


def pool(rows, device):
    return {key: torch.from_numpy(np.concatenate([r[key] for r in rows])).to(device)
            for key in ("raw","features","action","prior")}


def evaluation(r, probs):
    truth=r["action"];prior=r["prior"];guess=np.stack([v.argmax(1) for v in probs],1)
    cond={"all":np.ones(len(truth),bool),"change":truth[:,0]!=prior[:,0],
          "start":(prior[:,0]==4)&(truth[:,0]!=4),"turn":(prior[:,0]!=4)&(truth[:,0]!=4)&(prior[:,0]!=truth[:,0])}
    result={}
    for name,keep in cond.items():
        ids=np.flatnonzero(keep)
        result[name]={"n":len(ids),"correct":int((guess[ids,0]==truth[ids,0]).sum()),
                      "movement_nll":float(-np.log(np.maximum(probs[0][ids,truth[ids,0]],1e-9)).mean()) if len(ids) else None}
    result["head_accuracy"]=(guess==truth).mean(0).tolist()
    result["bomb_onsets"]=int(((prior[:,3]==0)&(truth[:,3]==1)).sum())
    result["movement_confusion_true_by_predicted"]=np.bincount(truth[:,0]*9+guess[:,0],minlength=81).reshape(9,9).tolist()
    result["bomb"]={"positives":int(truth[:,3].sum()),"predicted_positive_argmax":int(guess[:,3].sum()),
                    "mean_p_on_bomb_rows":float(probs[3][truth[:,3]==1,1].mean()) if truth[:,3].any() else None,
                    "mean_p_on_other_rows":float(probs[3][truth[:,3]==0,1].mean()) if (truth[:,3]==0).any() else None}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("train","held"):
        parser.add_argument("--"+key,nargs="+",type=Path,required=True)
    for key in ("encoder","reference","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--steps",type=int,default=800)
    parser.add_argument("--natural-movement-frequency",action="store_true",
                        help="Importance-correct movement cross entropy for the transition-enriched sampler.")
    parser.add_argument("--balance-titles",action="store_true",help="Give each original title equal sampling mass.")
    parser.add_argument("--mixed-difference-lags",action="store_true",help="Mix four- and eight-tick camera differences while preserving the same future action labels.")
    parser.add_argument("--player-location-weight",type=float,default=0.,help="Auxiliary original player-location prediction during SFT; runtime inputs remain camera only.")
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(261126)
    if args.player_location_weight<0:parser.error('Auxiliary weight must be nonnegative')
    device="cuda" if torch.cuda.is_available() else "cpu"
    train=[load_replay(p,include_eight_tick_camera=args.mixed_difference_lags,include_player_position=args.player_location_weight>0) for p in args.train];held=[load_replay(p) for p in args.held]
    if {r["group"] for r in train}&{r["group"] for r in held}:
        raise ValueError("Player leakage across replay partitions")
    if {r["sha256"] for r in train}&{r["sha256"] for r in held}:
        raise ValueError("Repeated replay file across partitions")
    args.out.mkdir(parents=True,exist_ok=False)
    model=CameraPolicy().to(device)
    model.load_state_dict(torch.load(args.encoder,weights_only=True,map_location=device))
    reference=FullActor.from_bin(args.reference).to(device).eval()
    opt=torch.optim.AdamW([{"params":model.spatial.parameters(),"lr":.00015},
                           {"params":model.policy.parameters(),"lr":.0005}],weight_decay=.001)
    data=pool(train,device)
    if args.player_location_weight:
        opt.add_param_group({"params":model.player.parameters(),"lr":.00015})
        positions=torch.from_numpy(np.concatenate([r['player_xy'] for r in train])).to(device)/2
        gy,gx=torch.meshgrid(torch.arange(56,device=device),torch.arange(48,device=device),indexing='ij')
    if args.mixed_difference_lags:
        alternate=torch.from_numpy(np.concatenate([r["raw8"] for r in train])).to(device)
        lag_rng=torch.Generator(device=device).manual_seed(182116)
    # Give each replay equal sampling weight; half of its draws are movement transitions when available.
    chunks=[];offset=0
    move_weight=torch.ones(len(data["action"]),device=device)
    for row in train:
        ids=torch.arange(offset,offset+len(row["action"]),device=device);offset+=len(ids)
        changes=ids[data["action"][ids,0]!=data["prior"][ids,0]]
        if len(changes) and args.natural_movement_frequency:
            # Within a replay the actual marginal proposal mixes 1/2 uniform
            # rows and 1/2 uniform changes. Undo this tilt for movement only.
            move_weight[ids]=2.
            move_weight[changes]=2./(1.+len(ids)/len(changes))
        chunks.append((ids,changes if len(changes) else ids))
    by_title={}
    for i,row in enumerate(train):by_title.setdefault(row["title"],[]).append(i)
    titles=sorted(by_title)
    for step in range(1,args.steps+1):
        chosen=[]
        for i in range(8):
            if args.balance_titles:
                indices=by_title[titles[int(torch.randint(len(titles),(1,)))]]
                replay=indices[int(torch.randint(len(indices),(1,)))]
            else:
                replay=int(torch.randint(len(chunks),(1,)))
            bank=chunks[replay][i%2]
            chosen.append(bank[torch.randint(len(bank),(8,),device=device)])
        ids=torch.cat(chosen)
        px=data["raw"][ids].float()/255
        if args.mixed_difference_lags:
            longer=torch.rand(len(ids),device=device,generator=lag_rng)<.5
            px[longer]=alternate[ids[longer]].float()/255
        y=data["action"][ids].clone()
        flip=torch.rand(len(ids),device=device)<.5
        px[flip]=px[flip].flip(3);v=y[flip,0];y[flip,0]=v//3*3+(2-v%3)
        px*=.85+.3*torch.rand(len(ids),1,1,1,device=device)
        if args.player_location_weight:
            spatial=model.spatial(px);heads=model.policy(spatial).split(WIDTHS,dim=1)
        else:
            heads=model(px)
        per_move=F.cross_entropy(heads[0],y[:,0],label_smoothing=.025,reduction="none")
        loss=(move_weight[ids]*per_move).mean()+sum(F.cross_entropy(heads[k],y[:,k],label_smoothing=.025) for k in range(1,4))
        if args.player_location_weight:
            xy=positions[ids].clone()
            valid=(xy[:,0]>=0)&(xy[:,0]<48)&(xy[:,1]>=0)&(xy[:,1]<56)
            xy[flip,0]=47.5-xy[flip,0];xy=xy[valid]
            if len(xy):
                dist=(gx-xy[:,0,None,None]).square()+(gy-xy[:,1,None,None]).square()
                target=(-dist.flatten(1)/1.4).softmax(1)
                predicted=model.player(spatial[valid]).flatten(1).log_softmax(1)
                loss=loss-args.player_location_weight*(target*predicted).sum(1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step==1 or step%100==0:
            with (args.out/"loss.jsonl").open("a") as f:f.write(json.dumps({"step":step,"loss":loss.item()})+"\n")
    model.eval()
    def predict(raw):
        return [np.concatenate(part) for part in zip(*[[h.softmax(1).cpu().numpy() for h in model(batch.to(device).float()/255)]
                  for batch in torch.from_numpy(raw).split(48)])]
    report={}
    with torch.inference_mode():
        for row in held:
            probs=predict(row["raw"])
            hist=[x.softmax(1).cpu().numpy() for x in reference(torch.from_numpy(row["features"]).to(device))[0]]
            shifted=[np.roll(q,len(q)//2,axis=0) for q in probs]
            report[row["name"]]={"visual_sft":evaluation(row,probs),"within_replay_screen_shift":evaluation(row,shifted),
                                 "frozen_ppo3":evaluation(row,hist)}
        demo_probs=np.concatenate([np.concatenate(predict(row["raw"]),axis=1) for row in train])
    weights=args.out/"camera-policy.pt";model.cpu()
    torch.save(model.state_dict(),weights)
    example=torch.from_numpy(held[0]["raw"][:1]).float()/255
    with torch.no_grad():
        compiled=torch.jit.trace(model,example)
        verification=torch.from_numpy(held[0]["raw"][:4]).float()/255
        for actual,exported in zip(model(verification),compiled(verification)):
            torch.testing.assert_close(actual,exported)
        compiled.save(str(args.out/"policy.ts"))
    np.savez_compressed(args.out/"training-demonstrations.npz",features=data["features"].cpu().numpy(),
                        action=data["action"].cpu().numpy().astype(np.int8),prior=data["prior"].cpu().numpy().astype(np.int8),
                        teacher_probs=demo_probs,replay_index=np.repeat(np.arange(len(train)),[len(r["action"]) for r in train]))
    manifest=lambda rows:[{**{k:r[k] for k in ("name","group","title","sha256")},"decisions":len(r["action"])} for r in rows]
    save_json(args.out/"report.json",{"train":manifest(train),"held":manifest(held),"encoder":sha(args.encoder),
              "natural_movement_frequency":args.natural_movement_frequency,
              "title_balanced":args.balance_titles,
              "player_location_weight":args.player_location_weight,
              "camera_difference_native_ticks":[4,8] if args.mixed_difference_lags else [4],
              "sft":sha(weights),"steps":args.steps,"label_smoothing":.025,"metrics":report,
              "torchscript_sha256":sha(args.out/"policy.ts"),"demo_sha256":sha(args.out/"training-demonstrations.npz")})


if __name__=="__main__":
    main()
