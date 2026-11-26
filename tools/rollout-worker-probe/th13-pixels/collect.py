"""Collect TH13 image/action episodes from two already-open private displays, starting paused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np
from PIL import Image


def checked(args, **kwargs):
    return subprocess.run(args, capture_output=True, check=True, **kwargs)


def read_hud(image):
    image = np.asarray(image)  # root window screenshot, title window at +3,+22
    counts = {}
    for name,y in [("spare_lives",131),("bombs",169)]:
        counts[name] = sum(float(np.percentile(image[y-3:y+4,x-3:x+4].min(2),85))>130 for x in range(525,625,13))
    mask=image[87:107,547:629].min(2)>160
    yy,xx=np.where(mask)
    counts["score_zero"]=bool(len(xx)==52 and xx.min()==68 and xx.max()==75)
    counts["opening_background_fraction"]=round(float((image[62:332,43:413].max(2)>50).mean()),4)
    return counts


def lane(args,worker):
    opts=["-o","BatchMode=yes","-o","StrictHostKeyChecking=yes"]
    if args.known_hosts:opts+=["-o","UserKnownHostsFile="+str(args.known_hosts)]
    dest=args.output/worker
    dest.mkdir(parents=True,exist_ok=True)
    def exec_lane(command):
        remote_args=["docker","exec","-e","DISPLAY=:66",args.container_prefix+worker,"sh","-ec",command]
        return checked(["ssh",*opts,args.ssh,shlex.join(remote_args)])
    def fetch(remote_name):
        return checked(["ssh",*opts,args.ssh,"cat "+shlex.quote(f"{args.remote_root}/{worker}/{remote_name}")]).stdout
    for ep in range(args.first,args.last+1):
        tag=f"px-{ep:02d}"
        seed=130000 + args.seed_offset + (0 if worker=="a" else 1000) + ep
        rng=np.random.default_rng(seed)
        directions=np.resize([-1,-1,0,0,1,1],args.decisions)
        shots=np.resize([0,1],args.decisions)
        rng.shuffle(directions);rng.shuffle(shots)
        actions=np.zeros((args.decisions,4),dtype=np.int8)
        actions[:,0]=directions;actions[:,2]=shots
        schedule="\n".join(" ".join(str(v) for v in row) for row in actions)
        for attempt in range(1,4):
            # R is the documented native Retry from the pause menu. Qualify the result from HUD pixels.
            exec_lane(f"mkdir -p /capture/{tag}; cd /capture/{tag}; xdotool keydown r; sleep .18; xdotool keyup r; sleep 3.9; scrot --overwrite startup.png")
            startup=Image.open(io.BytesIO(fetch(f"{tag}/startup.png"))).convert("RGB")
            start=read_hud(startup)
            if start["score_zero"] and start["spare_lives"]==2 and start["bombs"]==2 and start["opening_background_fraction"]<.30:
                break
            # May be in a brief fade or still playing following an earlier missed pause.
            exec_lane(f"sleep .4; scrot --overwrite /capture/{tag}/startup.png")
            startup=Image.open(io.BytesIO(fetch(f"{tag}/startup.png"))).convert("RGB")
            start=read_hud(startup)
            if start["score_zero"] and start["spare_lives"]==2 and start["bombs"]==2 and start["opening_background_fraction"]<.30:
                break
            exec_lane("xdotool keydown Escape; sleep .15; xdotool keyup Escape; sleep .8")
        else:
            raise RuntimeError(f"{worker}/{ep}: no certified fresh Stage 1 start: {start}")
        command=f"""cd /capture/{tag}
printf '%s\n' {shlex.quote(schedule)} | /tmp/pixel-capture {args.hold_ms}
scrot --overwrite horizon.png
xdotool keydown Escape; sleep .18; xdotool keyup Escape; sleep .75
tar -czf /capture/{tag}.tgz pixels.u8 wall.ns onset.ns horizon.png
"""
        exec_lane(command)
        with tarfile.open(fileobj=io.BytesIO(fetch(tag+".tgz")),mode="r:gz") as tar:
            def buf(name):return tar.extractfile(name).read()
            obs=np.frombuffer(buf("pixels.u8"),np.uint8).reshape(-1,112,96)
            wall=np.fromstring(buf("wall.ns").decode(),np.int64,sep=" ")
            onsets=np.fromstring(buf("onset.ns").decode(),np.int64,sep=" ")
            horizon=Image.open(io.BytesIO(buf("horizon.png"))).convert("RGB")
        assert len(obs)==len(wall)==len(actions)+1 and len(onsets)==len(actions)
        assert np.all((wall[:-1]<onsets)&(onsets<wall[1:])) and np.any(obs[1:]!=obs[:-1])
        split=args.split_override or ("train" if ep<=8 else "dev" if ep<=10 else "test")
        end=read_hud(horizon)
        md=dict(episode_id=f"th13-{args.host_label.lower()}-{worker}-{ep:02d}",title="TH13",build="official Trial 0.01a",difficulty="Normal",
                character="Hakurei Reimu",host=args.host_label,lane=worker,controller_seed=seed,
                split=split,reset="Stage 1 native R from Pause, onset HUD validated",crop_xywh=[35,38,384,448],
                native_clock_qualified=False,action_fields=["dx","dy","shoot","focus"],action_hold_ms_requested=args.hold_ms,
                clock="CLOCK_MONOTONIC, XGetImage return and XSync after requested key changes",
                obs_format="grayscale RGB 77:150:29 area-averaged 4x4 to 96x112",start_stock=start,end_stock=end,
                terminal="externally truncated time horizon",reset_attempt=attempt,
                executable_sha256="09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6")
        path=dest/(tag+".npz")
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,obs=obs,actions=actions,monotonic_ns=wall,action_onset_ns=onsets,metadata_json=json.dumps(md))
        tmp.replace(path)
        for name,im in [("start",startup),("endpoint",horizon)]:im.crop((3,22,643,502)).save(dest/f"{tag}-{name}.webp",lossless=True,method=2)
        print(worker,ep,split,len(actions),start,end,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh",required=True);p.add_argument("--known-hosts",type=Path)
    p.add_argument("--container-prefix",required=True);p.add_argument("--remote-root",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--lanes",default="ab")
    p.add_argument("--first",type=int,default=1);p.add_argument("--last",type=int,default=12)
    p.add_argument("--decisions",type=int,default=80);p.add_argument("--hold-ms",type=int,default=90)
    p.add_argument("--host-label",default="Endeavour");p.add_argument("--seed-offset",type=int,default=0)
    p.add_argument("--split-override",choices=["external_test"])
    args=p.parse_args()
    with ThreadPoolExecutor(len(args.lanes)) as pool:list(pool.map(lambda w:lane(args,w),args.lanes))

if __name__=="__main__":main()
