"""Archive original stage/clock evidence without the game binaries or redundant frames."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def summarize(folder):
    with (folder/"candidate-clocks.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (folder/"pause-events.csv").open() as stream:
        events = {k:int(v) for v,k in csv.reader(stream)}
    utc = np.array([int(r["utc_ms"]) for r in rows])
    stage = np.array([int(r["stage"]) for r in rows])
    background = np.array([int(r["p475448_o3c"] or 0) for r in rows])
    phases = {}
    for tag,start,end in [("before","active_before","right_down"),
                          ("right","right_down","right_up"),
                          ("left","left_down","left_up"),
                          ("pause","paused_sample","resume_requested")]:
        idx = np.flatnonzero((utc >= events[start]) & (utc < events[end]))
        assert len(idx) >= 5
        first,last = idx[0],idx[-1]
        phases[tag] = dict(samples=len(idx),seconds=(utc[last]-utc[first]).item()/1000,
                           from_background_tick=int(background[first]),to_background_tick=int(background[last]),
                           x_start=float(rows[first]["x"]),x_end=float(rows[last]["x"]))
    reset = np.flatnonzero((stage[:-1]==1)&(stage[1:]==2))
    backwards = np.flatnonzero((np.diff(background)<-100)&(stage[:-1]==1)&(stage[1:]==1))
    first2 = rows[int(reset[0])+1] if len(reset) else None
    record = dict(witness=folder.name,original_game_sha256=json.loads((folder/"protocol.json").read_bytes())["game_sha256"],
                  observations=len(rows),native_window_s=float((utc[-1]-utc[0])/1000),
                  phases=phases,stage1_background_backward_jumps=len(backwards),
                  example_backward_jump=[[int(background[k]),int(background[k+1])] for k in backwards[:2]],
                  first_stage2=dict(utc_ms=int(first2["utc_ms"]),spares=int(first2["spares"])) if first2 else None)
    if (folder/"native-clock-qualification.json").is_file():
        q = json.loads((folder/"native-clock-qualification.json").read_bytes())
        assert q["trace_sha256"]==hashlib.sha256((folder/"discovered-clocks.csv").read_bytes()).hexdigest()
        record.update(native_fields_matching=q["matched_native_fields"],discovery_original_stage_transitions=q["original_stage_transitions"])
    sample=folder/"original-pixels.npz"
    with np.load(sample) as image:
        assert image["obs"].shape==(len(image["utc_ms"]),112,96)
        record.update(original_camera_shape=list(image["obs"].shape),original_camera_sha256=hashlib.sha256(sample.read_bytes()).hexdigest())
    return record


def package(root, folders):
    out=root/"original-clock-evidence.zip";tmp=out.with_suffix(".inprogress")
    records=[]
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for fold in folders:
            folder=root/fold
            records.append(summarize(folder))
            for n in ("protocol.json","pause-events.csv","candidate-clocks.csv","discovered-clocks.csv","clock-discovery.log",
                      "native-clock-qualification.json","native.csv","diagnostic.csv","opening.csv",
                      "opening.png","paused.png","resumed.png","stage2-marker","stage2-title-00.png",
                      "stage2-title-10.png","endpoint.png"):
                if (folder/n).is_file():z.write(folder/n,f"{fold}/{n}")
        result=dict(scope="Separate engineering controls on the original TH10 Easy trial. These credits use geometry to reach the boss and establish timing readback; exclude them from learned-policy performance.",witnesses=records)
        data=json.dumps(result,indent=2)+"\n"
        z.writestr("qualification.json",data)
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(out)
    (root/"qualification.json").write_text(data)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("folders",nargs="+")
    args=p.parse_args();package(args.root,args.folders)
