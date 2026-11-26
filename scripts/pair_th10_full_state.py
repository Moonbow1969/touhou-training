"""Match training-only TH10 original pixels to measured, timestamped native privileged targets."""
import csv
import hashlib
import io
import json
import tarfile

import numpy as np


def export(archive,folder):
    with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as z:
        rich=z.extractfile("rich.bin").read()
        decisions=list(csv.DictReader(io.StringIO(z.extractfile("action.csv").read().decode())))
    if rich[:8]!=b'TH10OBS1':raise ValueError("Incorrect native observation stream.")
    names=["t0","t1","stage","difficulty","lives","power_units","score","native_input","bullet_count","player_xy","bullets"]
    formats=["<u8","<u8","<i4","<i4","<i4","<i4","<i4","<i4","<i4",("<f4",(2,)),("<f4",(96,4))]
    arr=np.frombuffer(rich[8:],dtype=np.dtype(list(zip(names,formats))))
    with np.load(folder/"original-pixels.npz") as z:obs=z["obs"][:-1];frame_mono=z["monotonic_ns"][:-1]
    with np.load(folder/"behavior.npz") as z:action=z["action"];features=z["features"]
    frame_utc=np.array([int(rec["utc_ms"]) for rec in decisions])-(np.array([int(rec["mono_ns"]) for rec in decisions])-frame_mono)/1e6
    times=(arr["t0"].astype(np.float64)+arr["t1"])/2
    pos=np.searchsorted(arr["t1"],frame_utc,side="right")-1
    prev=np.clip(pos,0,len(arr)-1)
    adj=np.searchsorted(times,frame_utc,side="left").clip(0,len(arr)-1)
    left=(adj-1).clip(min=0)
    nearest=np.where(abs(frame_utc-times[left])<=abs(frame_utc-times[adj]),left,adj)
    age=frame_utc-arr["t1"][prev]
    valid=(pos>=0)&(age>=0)&(age<=100)&(arr["lives"][prev]>=0)&(arr["player_xy"][prev,1]<=450)
    if valid.sum()<100:raise ValueError("Insufficient synchronized active observations.")
    out=folder/"privileged-targets.npz"
    raw={name:arr[name] for name in names}
    np.savez_compressed(out,obs=obs,features=features,action=action,frame_utc_ms=frame_utc,
        past_native_index=pos,nearest_native_index=nearest,valid=valid,past_age_ms=age,
        focus=((arr["native_input"]&4)!=0).astype(np.int8),**raw)
    report=dict(rows=len(obs),native_samples=len(arr),past_valid=int(valid.sum()),
        past_age_median_ms=float(np.median(age[valid])),past_age_max_ms=float(max(age[valid])),
        nearest_max_abs_ms=float(np.max(np.abs(times[nearest]-frame_utc))),
        past_native_index_description="Last entire native sample completed by camera capture. A negative index has no prior observation; use valid mask. Native rows are separately indexed.",
        coordinate_system="384x448 playfield centered at X=0, Y=0 at top, position native game pixels. bullets[...,0:2] relative to player; bullets[...,2:4] native velocity pixels/tick. Nearest 96 live bullets sorted by squared distance; total native pool count provided.",
        features="110-D shared pixel policy input, sampled four original actions. No ground truth reached actor.",
        sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (folder/"privileged-targets.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
