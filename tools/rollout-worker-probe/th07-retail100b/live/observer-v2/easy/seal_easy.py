"""Summarize all attempted Easy credits. Select documentary images with a human original HUD template."""
from pathlib import Path
import csv, hashlib, json, sys, zipfile
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
base=Path(sys.argv[1]); template=Path(sys.argv[2])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def filtered(im):
    roi=im.convert("L").crop((36,30,119,45))
    a=np.asarray(roi,float)-np.asarray(roi.filter(ImageFilter.GaussianBlur(1.3)),float)
    return a.ravel()/max(np.linalg.norm(a),1)
def ids(p):
    return [(line.split(",")[0],float(line.split(",")[1])) for line in p.read_text().splitlines()]
def collect(case):
    v=json.loads((case/"verified.json").read_bytes())
    native=np.genfromtxt(case/"native-input.csv",names=True,delimiter=",",dtype=None)
    assert set(native['rank'])=={0} and set(native['shot'])=={0} and set(native['continues'])=={0}
    assert (native[0]['lives'],native[0]['bombs'])==(2,3)
    vec=np.load(template,allow_pickle=False)
    candidates=[]; pictures=[]; witness=None
    stamps=ids(case/"boss-frame-times.csv")
    for name,seconds in stamps:
        f=case/"boss-frames"/(name+".jpg")
        with Image.open(f) as img:
            similarity=float(filtered(img) @ vec)
            pictures.append((name,img.crop((30,26,160,51)).copy()))
        j=np.abs(native["ns"]-round(seconds*1e9)).argmin(); row=native[j]
        candidates.append(dict(original=f.name,sha256=sha(f),similarity=similarity,stage=int(row['stage']),tick=int(row['tick']),
                               spares=int(row['lives']),bombs=int(row['bombs']),delta_to_nearest_native_ms=(seconds*1e9-int(row['ns']))/1e6))
        if witness is None and len(candidates)>=2 and min(a["similarity"] for a in candidates[-2:])>.70:
            witness=candidates[-2:]
    proof=case/'boss-evidence.json'
    proof.write_text(json.dumps(dict(template_sha256=sha(template),evidence=witness,observations=candidates,supplemental_source="The monotonic stamp precedes scrot; nearest original tick and stock are contextual. Main-boss identity is read from two adjacent photographs."),indent=2)+"\n")
    zname=case/"boss-observations.zip"
    with zipfile.ZipFile(zname,"w") as z:
        for f in [proof,case/"boss-frame-times.csv",*(case/"boss-frames").glob("*.jpg")]:z.write(f,f.relative_to(case))
    width=6*145
    sheet=Image.new("RGB",(width,((len(pictures)+5)//6)*41),(28,33,41));draw=ImageDraw.Draw(sheet)
    for i,(n,im) in enumerate(pictures):
        x,y=(i%6)*145,(i//6)*41;draw.text((x,y),n,fill="white");sheet.paste(im,(x,y+13))
    sheet.save(base/(case.name+"-hud.jpg"),quality=94)
    subset=("seed","passed","stage1_native_ticks","stage1_deaths","actual_bombs","stage2","terminal_stock","native_rank","terminal_continues","original_saved_score",
        "exact_fraction","skipped_decisions","stale","observer_p999_upper_ms","max_synchronous_observer_ms","wall_seconds","stage1_native_ticks_per_wall_second")
    row={k:v[k] for k in subset}
    row.update(boss_original_pair=witness,boss_seen=bool(witness),supplementary_images=len(pictures),boss_bundle_sha256=sha(zname),verified_sha256=sha(case/"verified.json"))
    return row
reg=base/"record/protocol.json"
p=json.loads(reg.read_bytes());rows=[collect(base/"record"/(str(q["seed"])+"-parent")) for q in p["lanes"]]
valid=[q for q in rows if q["passed"]]
def pooled(group):
    ticks=sum(x["stage1_native_ticks"] for x in group);deaths=sum(x["stage1_deaths"] for x in group)
    return dict(credits=len(group),stage1_ticks=ticks,stage1_deaths=deaths,deaths_per_10000_ticks=deaths*10000/ticks if ticks else None,
        boss=sum(q["boss_seen"] for q in group),stage2=sum(q["stage2"] for q in group),bombs=sum(q["actual_bombs"] for q in group))
a=pooled(rows);v=pooled(valid); gate=p["continuation_gate"]; bosses=sum(bool(q["boss_seen"] or q["stage2"]) for q in rows)
allow=bool(len(valid)>=p["quality"]["minimum_valid_credits"] and v["stage1_deaths"]<=gate["max_stage1_deaths_per_tick"]*v["stage1_ticks"] and bosses>=gate["minimum_boss_or_further"])
result=dict(protocol=sha(reg),credits=rows,strict=v,all_attempts_descriptive=a,gate_cleared=allow,required_min_valid=p["quality"]["minimum_valid_credits"],
          max_deaths_per_10000_ticks=gate["max_stage1_deaths_per_tick"]*10000,required_boss=gate["minimum_boss_or_further"],
          boss_reference="Letty Whiterock full original HUD name versus Cirno/empty in human original. For each positive inspect both unmodified sampled original photographs and the per-credit contact sheet.")
(base/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("strict","all_attempts_descriptive","gate_cleared")}))
