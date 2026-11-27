"""Record an original-replay natural TH11 Stage 1→2 transition and its separate image timeline."""
import argparse
import json
from pathlib import Path
import shlex
import time
import numpy as np
from PIL import Image,ImageDraw
import io,tarfile
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.th11_trial_lane import Lane

def run(root,connection):
    lane=Lane(root,connection,"positive-i-hard-s1");offsets=[];latest=None
    sample="""import os,struct,sys
p=sys.argv[1];s=os.stat(p);n=s.st_size//64
with open(p,'rb') as f:
 f.seek((n-1)*64);v=struct.unpack('<16i',f.read(64))
print(s.st_mtime_ns//1000000,n,*v)
"""
    for i in range(120):
        result=list(map(int,lane.remote(shlex.join(["python3","-c",sample,lane.dir+"/witness.bin"])).split()))
        utc,n,*r=result;offsets.append(utc-(r[0]%2**32));latest=r
        print("original",n,"stage",r[7],r[12],"clock",r[13],r[14],flush=True)
        if r[7]==2 and r[13]>=1350:break
        time.sleep(5)
    else:raise RuntimeError(f"No verified natural Stage 2 reached; last {latest}")
    # Preserve the exact measured bytes through this positive; visual frames can continue.
    lane.inside("touch /capture/stop-stock; sleep .12")
    raw=lane.remote("cat "+lane.dir+"/witness.bin")
    assert len(raw)%64==0
    (lane.out/"native-transition.bin").write_bytes(raw)
    rows=np.frombuffer(raw,"<i4").reshape(-1,16);valid=rows[:,6]==2
    index=np.flatnonzero(valid & (rows[:,7]==2))[0];start=np.flatnonzero(valid&(rows[:,7]==1))[0]
    assert (rows[start:index,7]==1).all() and (rows[index:,7]==2).all()
    assert (rows[start:,6]==2).all() and (rows[start:,8]==0).all()
    assert (rows[index:index+100,12]==2).all() or (rows[index:index+100,12]==1).all()
    delta=np.array(offsets)-np.median(offsets);assert np.ptp(delta)<120
    utc=(rows[index,0].item()%2**32)+np.median(offsets)
    t=np.array([int(v) for v in lane.remote("cat "+lane.dir+"/photo-ms.txt").split()])
    ids=np.where((t>=utc-18000)&(t<=utc+24000))[0]
    assert len(ids)>=12
    files=[f"frame-{i:03d}.jpg" for i in ids]
    packet=lane.remote("tar -czf - -C "+lane.dir+" "+" ".join(files))
    photo=lane.out/"transition";photo.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(packet),mode="r:gz") as ar:
        for f in files:(photo/f).write_bytes(ar.extractfile(f).read())
    sheet=Image.new("RGB",(4*256,((len(ids)+3)//4)*212),"#171e22");draw=ImageDraw.Draw(sheet)
    picture=[]
    for k,i in enumerate(ids):
        f=photo/f"frame-{i:03d}.jpg";im=Image.open(f).convert("RGB").crop((3,22,643,502)).resize((256,192))
        x,y=k%4*256,k//4*212;sheet.paste(im,(x,y))
        sec=(t[i]-utc)/1000;draw.text((x+4,y+193),f"{f.name}   {sec:+.2f} s",fill="white")
        picture.append(dict(file="transition/"+f.name,offset_seconds=round(sec,3),sha256=digest(f)))
    sheet.save(lane.out/"transition-contact.png")
    (lane.out/"witness.c").write_bytes(lane.remote("cat "+lane.dir+"/witness.c"))
    evidence=dict(index=int(index),last_stage1=rows[index-1].tolist(),first_stage2=rows[index].tolist(),
        max_stage1_clock=int(rows[start:index,13].max()),first_stage2_min_clock=int(rows[index:index+150,13].min()),record_count=len(rows),
        stage1_count=int(index-start),stage2_count=int(len(rows)-index),stage2_clock_at_last_row=int(rows[-1,13]),
        utc_offset_range_ms=np.ptp(delta).item(),median_guest_win32_tick_to_host_unix_ms=float(np.median(offsets)),
        witness_sha256=digest(lane.out/"native-transition.bin"),witness_source_sha256=digest(lane.out/"witness.c"),
        photos=picture)
    save(lane.out/"native-transition.json",evidence)
    print("Witness ready",evidence["first_stage2"],flush=True)
    lane.remote("docker stop --timeout 2 "+lane.name)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("workers",type=Path);a=p.parse_args()
    run(a.root.resolve(),a.workers.resolve())
