"""Seal the registered original TH07 comparison as one independently downloadable bundle."""
import argparse
import hashlib
import json
import statistics
import zipfile
from pathlib import Path
import numpy as np
from read_scorecard import scorecard

def sha(p):
    with p.open("rb")as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def load(p):return json.loads(p.read_text())

def seal(study,base):
    complete=load(study/"complete.json")
    cfg=load(study/"protocol.json")
    references=load(study/"tool/reference-scorecards.json")
    included=[study/"matched-native-2130751.mp4", study/"matched-film.json", study/"reforward.json",study/"complete.json",study/"protocol.json",study/"qualification.json",study/"dispatcher.log"]
    qualified=load(study/"qualification.json")
    relevant=[]
    for q in qualified["reports"]:
        folder=study/q["path"]
        assert sha(folder/"verified.json")==q["report_sha"]
        for f in ["verified.json","native-input.csv","applied.csv","policy-trace.npz","official-policy.rpy","full-stock-opening.png","settled-scorecard.png","settled-continue.png"]:
            included.append(folder/f)
    for pair in complete["all_pairs"]:
        included.append(study/f"pair-{pair['seed']}.json")
        for arm,arm_record in pair["arms"].items():
            folder=study/arm_record["path"]
            for name in ("controls.csv","native-input.csv","applied.csv","policy-trace.npz","official-policy.rpy","receipt.pending-visual.json",
                         "verified.json","native-credit.mp4","full-stock-opening.png","settled-continue.png","settled-scorecard.png",
                         "menu-1.png","menu-2.png","menu-3.png","runner.log","build.sha256"):
                if (folder/name).is_file():included.append(folder/name)
            if not pair["valid"]:continue
            r=load(folder/"verified.json")
            assert r["passed"] and r["policy_sha256"]==cfg["models"][arm]
            for n,h in r["evidence"].items():assert h==sha(folder/n)
            im=scorecard(folder/"settled-scorecard.png",base/"online",references)
            for k,(number,similarity,margin) in im.items():
                assert similarity>.94 and margin>.1,(folder,k,number,similarity,margin)
            assert im["score"][0]==r["game_score"] and im["continue"][0]==0 and im["miss"][0]==len(r["losses"])
            raw=np.genfromtxt(folder/"controls.csv",delimiter=",",names=True,dtype=None)
            t=raw["native_tick"];starts=np.flatnonzero(np.diff(t)<0)+1
            if r["stage2"]: assert len(starts)>0
            entry=({k:int(raw[k][starts[0]]) for k in ("native_tick","spares","bombs")} if r["stage2"] else None)
            # Last physical fatal has a camera upper bracket; conservative count uses last HUD pre-Continue.
            relevant.append(dict(seed=pair["seed"],arm=arm,lane=pair["lane"],first=pair["order"][0]==arm,stage2=r["stage2"],
                stage2_entry=entry,life_losses=r["losses"],final_hud=r["stock_at_final_camera"],actual_bombs=im["bombs"][0],actual_misses=im["miss"][0],
                actual_continues=im["continue"][0],score=r["game_score"],original_progress_pct=im["progress_pct"][0],
                movement_switch_rate=float(np.mean(np.diff(raw["move"])!=0)),
                commanded_neutral_fraction=float(np.mean(raw["move"]==4)),
                commanded_left_fraction=float(np.mean((raw["move"]%3)==0)),commanded_right_fraction=float(np.mean((raw["move"]%3)==2)),scorecard_digit_min_similarity=min(x[1]for x in im.values()),
                stage1_ticks=r["stage1_native_ticks"],stage1_deaths=r["stage1_deaths"],decisions=r["decisions"],exact=r["exact_fraction"],
                fully_stale=len(r["stale_indices"]),terminal_partial=r["incomplete_terminal_decisions"],freeze_ms=r["freeze_ms_p50_p95_p99_max"],
                active_seconds=r["elapsed_active_real_seconds"],container_boot_epoch=(folder/"build.sha256").stat().st_mtime,endpoint_epoch=(folder/"settled-scorecard.png").stat().st_mtime,
                positive_bomb_last_hud_at_stage1_death=sum(x["stage"]==1 and x["bombs_prior_hud"]>0 for x in r["losses"])))
    groups={}
    for arm in cfg["models"]:
        a=[r for r in relevant if r["arm"]==arm]
        groups[arm]=dict(valid=len(a),stage1_ticks=sum(r["stage1_ticks"]for r in a),stage1_deaths=sum(r["stage1_deaths"]for r in a),
                         reached_stage2=sum(r["stage2"]for r in a),bomb_positive_last_hud=sum(r["positive_bomb_last_hud_at_stage1_death"]for r in a),
                         original_progress_pct_mean=statistics.mean(r["original_progress_pct"]for r in a),
                         genuine_bomb_activations=sum(r["actual_bombs"]for r in a),score_mean=statistics.mean(r["score"]for r in a),
                         observed_stage1_median=statistics.median(r["stage1_ticks"]for r in a),minimum_exact_fraction=min(r["exact"]for r in a),
                         native_ticks_per_active_second=sum(r["stage1_ticks"]for r in a)/sum(r["active_seconds"]for r in a) if not any(r["stage2"]for r in a)else None)
    photos=[(study/p["arms"][arm]["path"]/"settled-scorecard.png")for p in complete["all_pairs"]if p["valid"]for arm in cfg["models"]]
    from PIL import Image,ImageDraw
    panel=Image.new("RGB",(740,170*(len(photos)//2)),"#14161d")
    draw=ImageDraw.Draw(panel)
    for i,path in enumerate(photos):
        x,y=(i%2)*370,(i//2)*170
        tile=Image.open(path).crop((42,105,426,310)).resize((288,154))
        panel.paste(tile,(x+40,y+15));draw.text((x+4,y),path.parent.name,fill="white")
    panel.save(study/"scorecards.jpg",quality=91)
    analysis=dict(summary=complete,group=groups,credits=relevant,
        note="Previous displayed Bomb stock is a camera proxy for resources before a loss. Six-seed pairing holds physical lane and controller seed fixed. Original game RNG is uncontrolled.")
    write(study/"scored.json",analysis)
    included.extend([study/"scored.json",study/"scorecards.jpg"])
    included.extend(p for p in (study/"tool").iterdir() if p.suffix in (".py",".c",".json",".bin"))
    included.extend([study/"run_pairs.py",study/"run_credit.py",study/"watchdog-fixture.log",study/"tool/README.md"])
    target=study/"th07-strict-four-tick-locator-credits.zip";tmp=target.with_suffix(".partial")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for p in sorted(set(included)):
            z.write(p,p.relative_to(study),compress_type=zipfile.ZIP_STORED if p.suffix in (".png",".npz",".mp4",".jpg") else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(tmp)as z:assert z.testzip()is None
    tmp.replace(target)
    write(study/"archive.json",dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),entries=len(set(included))))
    print(groups,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--study",type=Path,required=True);p.add_argument("--base",type=Path,required=True);a=p.parse_args();seal(a.study.resolve(),a.base.resolve())
