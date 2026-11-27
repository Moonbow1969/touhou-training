"""Seal the 16 training and 32 held TH11 original records and their independent endpoint evidence."""
import argparse
import json
from pathlib import Path
import shutil
import time
import zipfile
from scripts.qualified_th11_stage2 import read_credit
from scripts.audit_th11_original_scores import audit,read
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.th11_trial_lane import SRC

def seal(study,role,expected):
    native=study/"native11";files=[native/f"{role}-{arm}-{seed}/receipt.json" for arm,seed in expected]
    while not all(f.is_file() for f in files):time.sleep(8)
    spec=json.loads((SRC/"score-glyphs.json").read_bytes());tokens={v:k for k,vs in spec["glyphs"].items() for v in vs}
    rollups=[];replay_sha=digest(study/"stage2-qualification.json")
    archive=study/f"th11-{role}-native-records.zip";tmp=archive.with_suffix(".inprogress")
    import numpy as np
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for file,(arm,seed) in zip(files,expected):
            folder=file.parent;r=json.loads(file.read_bytes());s=read_credit(folder)
            assert (r["arm"],r["seed"],r["role"])==(arm,seed,role)
            hud=read(folder/"endpoint.png",tokens);settled=np.fromfile(folder/"native.bin","<i4").reshape(-1,12)[-1]
            assert "?" not in hud and int(hud)==int(settled[1])*10
            rollups.append(dict(**s,settled_score=int(hud),score_at_fatal=r["score"],settled_power_units=int(settled[2]),
                fatal_power_units=r["power_at_last_loss_units"],spares=0,effective_casts=r["effective_confirmed_casts"],
                n=r["steps"],native_key_matches=r["native_input_matches"],first_fatal_ms=r["native_life_losses"][0]["original_ms"],
                duration_ms=r["native_life_losses"][-1]["original_ms"],policy_sha256=r["policy_sha256"],receipt_sha256=digest(file)))
            for name in ["receipt.json","trajectory.npz","native.bin","behavior.csv","camera.ns","endpoint.png","selected-rank.png","build.json"]:
                z.write(folder/name,f"{folder.name}/{name}",compress_type=zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED)
        report=study/f"th11-{role}-qualified-native.json"
        save(report,dict(credits=rollups,stage2_qualification_sha256=replay_sha));z.write(report,report.name)
        z.write(study/"th11-host-allocation.json","host-allocation.json")
        z.write(study/"camera-protocol.json","camera-protocol.json")
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(archive)
    dst=Path("results/2026-11-26-original-camera-training")/study.name;dst.mkdir(exist_ok=True,parents=True)
    for v in [archive,report]:shutil.copy2(v,dst/v.name)
    print(role,len(rollups),archive.stat().st_size,digest(archive),flush=True)
def run(study):
    p=json.loads((study/"camera-protocol.json").read_bytes());ref=p["reference_arm"];alt=p["candidate"]
    seal(study,"train",[(ref,k) for k in p["train_seeds"]["11"]])
    seal(study,"held",[(arm,k) for k in p["eval_seeds"]["11"] for arm in [ref,alt]])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("study",type=Path);a=p.parse_args();run(a.study)
