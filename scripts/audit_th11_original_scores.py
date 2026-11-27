"""Independently read the Game Over HUD and audit TH11 fatal-frame versus settled scores."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.th11_trial_lane import SRC

def read(path,templates):
    pixels=np.asarray(Image.open(path).convert("RGB")).min(-1)>190
    tokens=[]
    for k in range(9):
        left=526+12*k;patch=pixels[96:108,left:left+12]
        key=np.packbits(patch.ravel()).tobytes().hex()
        tokens.append(templates.get(key,"?"))
    return "".join(tokens)

def audit(study,watch=False):
    p=SRC/"score-glyphs.json";spec=json.loads(p.read_bytes())
    templates={v:digit for digit,values in spec["glyphs"].items() for v in values}
    assert len(templates)==10
    native=study/"native11";rows={}
    while True:
        for name in native.glob("*/receipt.json"):
            folder=name.parent
            if folder.name in rows:continue
            receipt=json.loads(name.read_bytes())
            pic=folder/"endpoint.png";data=folder/"native.bin"
            assert digest(pic)==receipt["evidence_sha256"]["endpoint.png"] and digest(data)==receipt["evidence_sha256"]["native.bin"]
            raw=np.fromfile(data,"<i4").reshape(-1,12)
            end=raw[-1];assert end[5]==-1 and end[8]==0
            text=read(pic,templates);visible=None if "?" in text else int(text)
            rows[folder.name]=dict(arm=receipt["arm"],role=receipt["role"],seed=receipt["seed"],rank=receipt["difficulty"],
                fatal_score=receipt["score"],settled_score=int(end[1])*10,hud_score=visible,hud_digits=text,
                matches=visible is not None and visible==int(end[1])*10,final_power_units=int(end[2]),final_graze=int(end[10]),
                terminal_spares=0,source_receipt_sha256=digest(name),original_hud_sha256=digest(pic),native_sha256=digest(data))
        result=dict(count=len(rows),unreadable=sum(v["hud_score"] is None for v in rows.values()),
            mismatches=sum(v["matches"] is False for v in rows.values()),different_from_fatal=sum(v["settled_score"]!=v["fatal_score"] for v in rows.values()),
            glyph_template_sha256=digest(p),comparison="Exact binary digit masks; independently labeled originals. Unrecognized digit remains unknown.",rows=rows)
        save(study/"th11-original-hud-scores.json",result)
        if not watch or len(rows)==48:return result
        time.sleep(12)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("study",type=Path);p.add_argument("--watch",action="store_true")
    a=p.parse_args();r=audit(a.study,a.watch);print({k:v for k,v in r.items() if k!="rows"})
