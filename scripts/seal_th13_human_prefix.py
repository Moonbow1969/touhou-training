"""Seal a human-prefix original using verified controls, image rerun and native coordinates."""
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
from scripts.audit_th13_policy_suffix import verify
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest


def seal(d):
    study=d/"engineering/th13-live-midstage-handoff"
    passed=study/"live-1271303"; folder=passed/"engineering-humanprefix-visual-1271303"
    validation=verify(folder,d/"four-game-online/policy/policy.ts")
    human=d/"human-trial13/train/th13trial-th12_ud003550-s1.npz"
    hist=json.loads((study/"taraka2-hard-reimu-s1-input.json").read_bytes())
    assert digest(human)==hist["previous_camera_archive_sha256"]
    with np.load(human) as arr:
        human_xy={int(t):pt for t,pt in zip(arr["native_frame"],arr["xy"])}
    with (folder/"prefix.csv").open() as f:prefix=list(csv.DictReader(f))
    start=int(prefix[0]["monotonic_ns"]);end=int(prefix[-1]["monotonic_ns"])
    with (passed/"prefix-coordinates.csv").open() as f:
        live={int(row["native_stage_tick"]):[float(row["x"]),float(row["y"])] for row in csv.DictReader(f)
              if start<=int(row["monotonic_ns"])<=end+30000000 and int(row["native_stage_tick"])<=3000 and int(row["stage"])==1}
    common=sorted(set(human_xy)&set(live)); assert 3000 in common
    distance=[float(np.linalg.norm(np.subtract(live[t],human_xy[t]))) for t in common]
    validation["same_native_tick_coordinates"]=dict(exact_pairs=len(common),max_tick_mismatch=0,
           recorded_human_at_3000=human_xy[3000].tolist(),live_at_3000=live[3000],handoff_distance_pixels=distance[-1],
           median_distance_pixels=float(np.median(distance)),max_distance_pixels=float(max(distance)),
           live_source_sha256=digest(passed/"prefix-coordinates.csv"),original_source_sha256=digest(human))
    save(passed/"post-handoff-validation.json",validation)
    archive=study/"original-handoff-evidence.zip"
    keep={"protocol.json","qualification.json","technical-outcome.json","prefix.csv","prefix-opening.png","post-handoff-validation.json",
          "prefix-coordinates.csv","handoff-original.png","steps.bin","receipt.json","events.tsv","rollout.npz","ranking.png","selected-difficulty.png",
          "observation-labels.bin","final-stock.bin","actor.log"}
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in sorted(study.rglob("*")):
            if p.is_file() and (p.parent==study and p.name.startswith("taraka2-") or p.parent!=study and p.name in keep):
                z.write(p,p.relative_to(study))
    out=dict(scope="Engineering-only original human-prefix handoff; excluded from gradient updates and ordinary-opening evaluation.",
             attempts=[json.loads((study/path).read_bytes()) for path in [
                 "live-1271301/technical-outcome.json","live-1271302/qualification.json","live-1271303/qualification.json"]],
             neural_suffix_validation=validation,raw_archive=archive.name,raw_sha256=digest(archive),raw_bytes=archive.stat().st_size)
    save(study/"qualification.json",out)
    print(json.dumps(validation),digest(archive),archive.stat().st_size)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);seal(p.parse_args().root)
