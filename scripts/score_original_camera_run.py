"""Index completed native full-credit receipts, their original endpoints and matched metrics."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.checkpoint_storage import digest
from scripts.score_native_full_actions import summarize


def records(root, camera, round2=False):
    if round2:
        yield from ((10,f) for f in (root/"native10").rglob("report.json"))
        yield from ((12,f) for f in (root/"native12/credits").glob("*.json"))
        yield from ((13,f) for f in (root/"native13").glob("*/receipt.json"))
        return
    for p in root.glob("*10*"):
        if p.is_dir() and ("camera" in p.name)==camera:
            yield from ((10,f) for f in p.rglob("report.json"))
    for d in ([root/"camera-native12"] if camera else root.glob("native12*")):
        yield from ((12,f) for f in (d/"credits").glob("*.json"))
    d=root/("camera-native13" if camera else "native13")
    yield from ((13,f) for f in d.glob("*/receipt.json"))


def verify(title, f, r, trace, arm, out):
    if title==10:
        photo=f.parent/"endpoint.png";steps=r["steps"];seconds=r["duration_s"];casts=len(r["successful_bombs"])
        lives=r["final_lives"];bombs=None;power=r["final_power"];build=r["game_sha256"]
        entry=(r.get("stage2_lives"),None,r.get("stage2_power"))
    elif title==12:
        raw=f.parent.parent/"raw"/r["episode"]
        photo=raw/"terminal-at-control.ppm"
        if not photo.exists():photo=raw/"endpoint.png"
        steps=r["actions"];seconds=r["seconds"];lives=r["remaining_spare_lives"]
        bombs=r["remaining_bombs"];power=None;build=r["game_exe_sha256"]
        casts=sum(e["prior"]-e["current"] for e in r["event_history"] if e["event"]=="bombs" and e.get("cost",0)<0)
        entry=(r.get("stage2_remaining_spare_lives"),r.get("stage2_remaining_bombs"),None)
    else:
        photo=f.parent/r["ranking_photo"];steps=r["steps"];seconds=r["wall_seconds"];build=r["game_sha256"]
        lives=r["remaining_lives"];bombs=r["remaining_bombs"];power=None;casts=r["effective_bombs"]
        entry=(r.get("stage2_entry_spares"),r.get("stage2_entry_bombs"),None)
    if r.get("continues",0)!=0 or not r.get("terminal",True) or not photo.is_file():
        raise ValueError(f"Missing matched original terminal: {f.name}")
    source_sha=r.get("behavior_sha256",r.get("trajectory_sha256",r.get("sha256")))
    if digest(trace)!=source_sha:raise ValueError("Changed native trace")
    host=r["host"].lower().replace("steam deck","steamdeck")
    if host=="deck":host="steamdeck"
    name=f"th{title}-{host}-{r['seed']}-{arm}";image=out/"endpoints"/(name+".jpg")
    with Image.open(photo) as im:
        im=im.convert("RGB");im.thumbnail((640,480));im.save(image,quality=82)
    stage1_tick=None
    with np.load(trace,allow_pickle=False) as z:
        y=z["action"];done=z["terminated"].astype(bool)
        if len(y)!=steps or not done[-1] or done[:-1].any():raise ValueError("Invalid full-credit action trace")
        counts=[np.bincount(y[:,k].astype(int),minlength=n).tolist() for k,n in enumerate((9,2,2,2))]
        if "stage" in z and "native_frame" in z:
            s1=z["native_frame"][z["stage"]==1]
            if len(s1):stage1_tick=int(s1.max())
        keys=["action","logp","value","reward","terminated","seconds","stage","spares","bombs","stock_hud","onset_ns","native_frame",
              "reward_parts","reward_components","action_utc_ms","native_input_matches","native_keys","keys","observed_keys"]
        target=out/"action-traces"/(name+".npz");np.savez_compressed(target,**{k:z[k] for k in keys if k in z})
    outcome=r.get("outcome",r.get("native_outcome","")).lower()
    if not any(x in outcome for x in ("final_life","game_over","ranking","clear","ending")):
        raise ValueError("Unrecognized end state")
    return dict(title=title,arm=arm,seed=r["seed"],host=host,game_sha256=build,policy_sha256=r["policy_sha256"],
                terminal="game_over" if any(x in outcome for x in ("final_life","game_over","ranking")) else "ending",
                stage2_reached=r["stage2_reached"],stage2_spares=entry[0],stage2_bombs=entry[1],stage2_power=entry[2],
                terminal_spares=max(0,int(lives)),terminal_bombs=bombs,terminal_power=power,effective_bombs=casts,
                decisions=steps,wall_seconds=seconds,action_counts=counts,latest_observed_stage1_native_tick=stage1_tick,original_trace_sha256=source_sha,
                native_log_sha256=digest(f),original_endpoint_sha256=digest(photo),
                endpoint=image.relative_to(out).as_posix(),action_trace=target.relative_to(out).as_posix(),
                compact_trace_sha256=digest(target))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","protocol","parent","candidate","out"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--camera",action="store_true")
    p.add_argument("--round2",action="store_true")
    p.add_argument("--titles",nargs="+",type=int,choices=(10,12,13))
    p.add_argument("--arm-names",nargs=2,metavar=("REFERENCE","CANDIDATE"))
    args=p.parse_args()
    arms=args.arm_names or (("camera-sft","camera-ppo") if args.camera else ("ppo3","ppo4"))
    hashes={digest(k):n for k,n in zip((args.parent,args.candidate),arms)}
    protocol=json.loads(args.protocol.read_bytes());protocol["reference_arm"]=arms[0]
    if args.titles:protocol["eval_seeds"]={k:v for k,v in protocol["eval_seeds"].items() if int(k) in args.titles}
    wanted={(int(t),s,a) for t,seeds in protocol["eval_seeds"].items() for s in seeds for a in arms}
    found={}
    for t,f in records(args.root,args.camera,args.round2):
        r=json.loads(f.read_bytes()); arm=hashes.get(r.get("policy_sha256"))
        key=(t,r.get("seed"),arm)
        if key not in wanted:continue
        if t==13 and not r.get("ranking_photo"):continue
        trace=f.parent/r.get("behavior",r.get("trajectory",r.get("file","missing")))
        if key in found:raise ValueError(f"Duplicate native credit {key}")
        found[key]=(t,f,r,trace,arm)
    if found.keys()!=wanted:
        raise SystemExit(f"Waiting for verified whole credits {len(found)}/{len(wanted)} {collections.Counter((t,a) for t,s,a in found)}")
    for k in ("endpoints","action-traces"):(args.out/k).mkdir(parents=True,exist_ok=True)
    rows=[verify(*found[k],args.out) for k in sorted(found)]
    result=summarize(rows,protocol,arms[1])
    (args.out/"credits.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.out/(arms[1]+".json")).write_text(json.dumps(result,indent=2)+"\n")
    print(len(rows),"verified credits",sum(r["stage2_reached"] for r in rows),"Stage 2",flush=True)


if __name__=="__main__":main()
